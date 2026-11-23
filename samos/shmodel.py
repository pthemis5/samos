import numpy as np


class SHModel:
    """
    Time-dependent spherical harmonic model of an internal magnetic field.

    The coefficients are piecewise linear in time: at epoch i they are
    G[i], H[i], and they change with the secular variation dG[i], dH[i]
    until epoch i+1 (or until `valid_until` for the last epoch).

    Coefficients are stored flat, with (n, m) ordered as
    (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...  (np.tril_indices order).

    Units: r [km], theta = colatitude [deg], phi = longitude [deg], B [nT].
    Geocentric spherical components: Br (up), Btheta (south), Bphi (east).
    """

    def __init__(self, G, H, dG, dH, epochs, N_degree, valid_until, a = 6371.2):
        self.G  = np.asarray(G,  dtype = float)     # (n_epochs, n_coeffs) [nT]
        self.H  = np.asarray(H,  dtype = float)
        self.dG = np.asarray(dG, dtype = float)     # (n_epochs, n_coeffs) [nT/yr]
        self.dH = np.asarray(dH, dtype = float)
        self.epochs = np.asarray(epochs, dtype = float)   # decimal years
        self.N_degree = int(N_degree)
        self.valid_until = float(valid_until)
        self.a = float(a)                           # reference radius [km]

        # degree and order of every flat coefficient, computed once
        self.n_idx, self.m_idx = np.tril_indices(self.N_degree + 1)
        self.n_coeffs = len(self.n_idx)
        assert self.G.shape == (len(self.epochs), self.n_coeffs)

        # slice of the flat array belonging to each degree n
        starts = np.arange(self.N_degree + 2) * np.arange(1, self.N_degree + 3) // 2
        self._degree_slices = [slice(starts[n], starts[n + 1]) for n in range(self.N_degree + 1)]

    # ------------------------------------------------------------------ readers
    @classmethod
    def read_sch_igrf(cls, filename = "/project/samos/data/IGRF14.shc"):
        """
        Read an IGRF .shc file. The last (predictive) column is not kept as an
        epoch; it only defines the secular variation of the final interval.
        """
        with open(filename, 'r') as f:
            lines = [line for line in f if not line.startswith('#') and line.strip()]

        # header: N_MIN, N_MAX, NTIMES, SP_ORDER, N_STEPS, ...
        N_degree, n_times = map(int, lines[0].split()[1:3])
        all_epochs = np.array(lines[1].split(), dtype = float)
        data = np.array([line.split() for line in lines[2:]], dtype = float)

        n = data[:, 0].astype(int)
        m = data[:, 1].astype(int)
        coeffs = data[:, 2:].T                              # (n_times, rows)
        assert coeffs.shape[0] == n_times == len(all_epochs)

        idx = n * (n + 1) // 2 + np.abs(m)                  # flat (tril) position
        n_coeffs = (N_degree + 1) * (N_degree + 2) // 2
        G = np.zeros((n_times, n_coeffs))
        H = np.zeros((n_times, n_coeffs))
        G[:, idx[m >= 0]] = coeffs[:, m >= 0]
        H[:, idx[m <  0]] = coeffs[:, m <  0]

        dt = np.diff(all_epochs)[:, None]
        dG = np.diff(G, axis = 0) / dt
        dH = np.diff(H, axis = 0) / dt

        return cls(G[:-1], H[:-1], dG, dH, all_epochs[:-1], N_degree,
                   valid_until = all_epochs[-1])

    @classmethod
    def read_pygeodyn(cls, filename, realisation = None, sv = 'diff'):
        """
        Read the main field of a pygeodyn reanalysis (HDF5 file with an
        'analysed' group, e.g. 2023_gji_gnm_GOVO.hdf5).

        Parameters
        ----------
        filename : str or Path
        realisation : None or int
            None -> ensemble mean (default); int -> that ensemble member.
        sv : 'diff' or 'file'
            'diff' (default): secular variation from finite differences between
                consecutive analyses, so the field is continuous and passes exactly
                through every analysis (same construction as read_sch_igrf).
            'file': the SV stored in the file at each analysis (the filter's own
                SV). The field then jumps slightly at each analysis time.

        The model is valid from the first to the last analysis time.
        Requires h5py.
        """
        import h5py                                         # optional dependency

        if sv not in ('diff', 'file'):
            raise ValueError("sv must be 'diff' or 'file'")
        sel = slice(None) if realisation is None else int(realisation)

        with h5py.File(filename, 'r') as f:
            group = f['analysed']
            N_degree = int(f.attrs['Lb'])
            times = group['times'][()].astype(float)
            MF = group['MF'][sel].astype(float)             # (Nt, K) or (n_real, Nt, K)
            SV = group['SV'][sel].astype(float) if sv == 'file' else None

        if realisation is None:                             # ensemble mean
            MF = MF.mean(axis = 0)
            SV = SV.mean(axis = 0) if SV is not None else None

        # analyses are at month starts, stored as rounded float32 decimal years;
        # recover the exact dates and express them in this class's time convention
        months = np.round((times - 1970) * 12).astype(int)
        dates = np.datetime64('1970-01', 'M') + months
        epochs = cls._decimal_year(dates)

        def to_flat(c):
            pos, is_h = cls._pygeodyn_index(N_degree)
            assert c.shape[-1] == len(pos), 'unexpected number of coefficients'
            G = np.zeros((c.shape[0], (N_degree + 1) * (N_degree + 2) // 2))
            H = np.zeros_like(G)
            G[:, pos[~is_h]] = c[:, ~is_h]
            H[:, pos[is_h]] = c[:, is_h]
            return G, H

        G, H = to_flat(MF)
        if sv == 'file':
            dG, dH = to_flat(SV)
        else:
            dt = np.diff(epochs)[:, None]
            dG = np.diff(G, axis = 0) / dt
            dH = np.diff(H, axis = 0) / dt
            # last epoch only used at t = epochs[-1] (dt = 0); repeat the last slope
            dG, dH = np.vstack((dG, dG[-1:])), np.vstack((dH, dH[-1:]))

        return cls(G, H, dG, dH, epochs, N_degree, valid_until = epochs[-1])

    @staticmethod
    def _pygeodyn_index(N_degree):
        """
        Flat (tril) position and is-h flag of each coefficient in the pygeodyn
        ordering g10, g11, h11, g20, g21, h21, g22, h22, ...
        """
        pos, is_h = [], []
        for n in range(1, N_degree + 1):
            for m in range(n + 1):
                pos.append(n * (n + 1) // 2 + m); is_h.append(False)
                if m > 0:
                    pos.append(n * (n + 1) // 2 + m); is_h.append(True)
        return np.array(pos), np.array(is_h)

    # ------------------------------------------------------------------ time
    @staticmethod
    def _decimal_year(dates):
        """datetime / np.datetime64 / str (scalar or array) -> decimal year (array)."""
        t = np.atleast_1d(np.asarray(dates, dtype = 'datetime64[ns]'))
        year = t.astype('datetime64[Y]')
        start = year.astype('datetime64[ns]')
        end = (year + 1).astype('datetime64[ns]')
        return year.astype(int) + 1970 + (t - start) / (end - start)

    def _coeffs_flat(self, dates):
        """Interpolated coefficients for all dates at once -> g, h of shape (Nt, n_coeffs)."""
        t = self._decimal_year(dates)
        if np.any(t < self.epochs[0]) or np.any(t > self.valid_until):
            raise ValueError(f"dates must be within {self.epochs[0]} - {self.valid_until}")
        i = np.clip(np.searchsorted(self.epochs, t, side = 'right') - 1, 0, len(self.epochs) - 1)
        dt = (t - self.epochs[i])[:, None]
        return self.G[i] + self.dG[i] * dt, self.H[i] + self.dH[i] * dt

    def get_gauss_coeffs(self, date):
        """
        Gauss coefficients at `date` as lower-triangular arrays garr, harr.
        Scalar date -> (N+1, N+1); array of Nt dates -> (Nt, N+1, N+1).
        """
        g, h = self._coeffs_flat(date)
        garr = np.zeros((len(g), self.N_degree + 1, self.N_degree + 1))
        harr = np.zeros_like(garr)
        garr[:, self.n_idx, self.m_idx] = g
        harr[:, self.n_idx, self.m_idx] = h
        if np.ndim(date) == 0:
            return garr[0], harr[0]
        return garr, harr

    # ------------------------------------------------------------------ basis pieces
    def _legendre(self, theta):
        """
        Schmidt semi-normalised P_n^m(cos theta), dP/dtheta, and P/sin(theta),
        via recursion (stable for high degree, no factorials).
        theta: colatitude [deg], shape (K,). Returns three arrays (K, n_coeffs).
        """
        th = np.radians(np.asarray(theta, dtype = float))
        c, s = np.cos(th), np.sin(th)
        N = self.N_degree
        P  = np.zeros((N + 1, N + 1, th.size))
        dP = np.zeros_like(P)
        P[0, 0] = 1.0

        for m in range(N + 1):
            if m > 0:   # diagonal term P_m^m from P_{m-1}^{m-1}
                f = 1.0 if m == 1 else np.sqrt((2 * m - 1) / (2 * m))
                P[m, m]  = f * s * P[m - 1, m - 1]
                dP[m, m] = f * (c * P[m - 1, m - 1] + s * dP[m - 1, m - 1])
            for n in range(m + 1, N + 1):
                k1 = (2 * n - 1) / np.sqrt(n * n - m * m)
                k2 = np.sqrt(((n - 1) ** 2 - m * m) / (n * n - m * m))
                P[n, m]  = k1 * c * P[n - 1, m] - k2 * P[n - 2, m]
                dP[n, m] = k1 * (c * dP[n - 1, m] - s * P[n - 1, m]) - k2 * dP[n - 2, m]

        P  = P[self.n_idx, self.m_idx].T                    # (K, n_coeffs)
        dP = dP[self.n_idx, self.m_idx].T

        # P / sin(theta), with the analytic limit at the poles:
        # m = 1 -> dP/dtheta / cos(theta), m >= 2 -> 0 (m = 0 is multiplied by m = 0 anyway)
        Ps = np.zeros_like(P)
        pole = np.isclose(s, 0.0, atol = 1e-10)
        Ps[~pole] = P[~pole] / s[~pole, None]
        m1 = self.m_idx == 1
        Ps[np.ix_(pole, m1)] = dP[np.ix_(pole, m1)] / c[pole, None]
        return P, dP, Ps

    def _trig(self, phi):
        """cos(m phi), sin(m phi) -> two arrays (K, n_coeffs)."""
        mphi = np.radians(np.asarray(phi, dtype = float))[:, None] * np.arange(self.N_degree + 1)
        return np.cos(mphi)[:, self.m_idx], np.sin(mphi)[:, self.m_idx]

    def _radial(self, r):
        """(a/r)^(n+2) per degree -> (K, N+1)."""
        return (self.a / np.asarray(r, dtype = float))[:, None] ** (np.arange(self.N_degree + 1) + 2)

    # ------------------------------------------------------------------ point basis
    def basis(self, r, theta, phi):
        """
        Design matrices for N points (broadcastable r, theta, phi).
        B_component = M_component @ [g, h]. Reuse the result for repeated
        evaluations at the same points (e.g. a time series at fixed locations).
        """
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        shape = r.shape
        P, dP, Ps = self._legendre(theta.ravel())
        cos, sin = self._trig(phi.ravel())
        R = self._radial(r.ravel())[:, self.n_idx]          # (K, n_coeffs)

        Rr = (self.n_idx + 1) * R * P                       # Br  weights
        Rt = -R * dP                                        # Btheta weights
        Rp = self.m_idx * R * Ps                            # Bphi weights
        return {
            'shape': shape,
            'Mr': np.hstack((Rr * cos, Rr * sin)),          # (K, 2 n_coeffs)
            'Mt': np.hstack((Rt * cos, Rt * sin)),
            'Mp': np.hstack((Rp * sin, -Rp * cos)),
        }

    def _coeff_vector(self, dates):
        g, h = self._coeffs_flat(dates)
        return np.hstack((g, h))                            # (Nt, 2 n_coeffs)

    # ------------------------------------------------------------------ field
    def calculate_field(self, date, r = None, theta = None, phi = None, basis = None):
        """
        Field at one date and N points. Returns Br, Btheta, Bphi with the
        broadcast shape of (r, theta, phi). Pass `basis` to skip recomputing it.
        """
        if basis is None:
            basis = self.basis(r, theta, phi)
        q = self._coeff_vector(date)[0]                     # (2 n_coeffs,)
        return tuple((basis[M] @ q).reshape(basis['shape']) for M in ('Mr', 'Mt', 'Mp'))

    def calculate_field_points(self, dates, r = None, theta = None, phi = None, basis = None):
        """
        Field where every point has its own date: dates, r, theta, phi are
        broadcastable arrays. Coefficients are interpolated for all points in
        one vectorised step, then contracted row by row.
        """
        if basis is None:
            basis = self.basis(r, theta, phi)
        dates = np.broadcast_to(np.asarray(dates, dtype = 'datetime64[ns]'), basis['shape']).ravel()
        Q = self._coeff_vector(dates)                       # (K, 2 n_coeffs)
        return tuple(np.einsum('kc,kc->k', basis[M], Q).reshape(basis['shape'])
                     for M in ('Mr', 'Mt', 'Mp'))

    def calculate_field_grid(self, dates, r, theta, phi):
        """
        Field on the full grid dates x r x theta x phi (all 1D).
        Returns Br, Btheta, Bphi of shape (Nt, Nr, Ntheta, Nphi).

        Each factor is computed once on its own axis: Legendre functions per
        theta, trig per phi, (a/r)^(n+2) per r, coefficients per date.
        For several radii the field is split per degree, S_n(t, theta, phi),
        so each extra radius costs only a weighted sum over N+1 degrees.
        """
        dates = np.atleast_1d(np.asarray(dates, dtype = 'datetime64[ns]'))
        r, theta, phi = (np.atleast_1d(np.asarray(x, dtype = float)) for x in (r, theta, phi))

        g, h = self._coeffs_flat(dates)                     # (Nt, nc)
        P, dP, Ps = self._legendre(theta)                   # (Nth, nc)
        cos, sin = self._trig(phi)                          # (Nph, nc)
        R = self._radial(r)                                 # (Nr, N+1)
        n1 = self.n_idx + 1

        # combine coefficients with longitude: (Nt, nc, Nph)
        A = (g[:, None, :] * cos + h[:, None, :] * sin).transpose(0, 2, 1)
        C = (g[:, None, :] * sin - h[:, None, :] * cos).transpose(0, 2, 1)

        lat_r = n1 * P              # (Nth, nc), the (n+1) factor folded in
        lat_t = -dP
        lat_p = self.m_idx * Ps

        if len(r) == 1:
            # single radius: fold (a/r)^(n+2) into the latitude factor
            Rc = R[0, self.n_idx]
            out = [lat_r * Rc @ A, lat_t * Rc @ A, lat_p * Rc @ C]          # (Nt, Nth, Nph)
            return tuple(o[:, None] for o in out)

        # several radii: per-degree angular fields S_n, then weight by radius
        out = []
        for lat, X in ((lat_r, A), (lat_t, A), (lat_p, C)):
            S = np.stack([lat[:, sl] @ X[:, sl, :] for sl in self._degree_slices])  # (N+1, Nt, Nth, Nph)
            out.append(np.einsum('rn,ntqp->trqp', R, S))
        return tuple(out)
