r"""Penalised Whittle fit of :math:`\log S(t,f) = A_c(f) + B_s(f/\rho) - \log\rho`.

* ``c`` indexes the channel (vehicle, or sensor placement), ``s`` the road
  segment, :math:`\rho` the speed and :math:`\nu = f/\rho` spatial frequency
  in cycles per metre.
* With :math:`K`-taper periodograms :math:`I \approx S\,\chi^2_{2K}/2K`, the
  negative log-likelihood :math:`\sum K[u + I e^{-u}]` with :math:`u=\log S` is
  convex in :math:`u`. :math:`u` is affine in the B-spline coefficients, so for
  fixed speeds and segment labels the fit is a convex problem (a Gamma GLM with
  log link).
* Identifiability: differences between channels are identified whenever
  channels share segments. The split between the *common* channel and the
  road needs the same segments seen at more than one speed; otherwise it rests
  on the smoothness penalties. ``tests/test_scr.py`` checks both cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import signal
from scipy.interpolate import BSpline
from scipy.optimize import minimize

__all__ = ["bspline_basis", "multitaper_frames", "SCRData", "SCRFit", "fit_scr"]


def _knots(lo: float, hi: float, n_basis: int, degree: int = 3) -> np.ndarray:
    n_inner = n_basis - degree - 1
    if n_inner < 0:
        raise ValueError("n_basis must be at least degree + 1")
    inner = np.linspace(lo, hi, n_inner + 2)[1:-1]
    return np.r_[[lo] * (degree + 1), inner, [hi] * (degree + 1)]


def bspline_basis(x: np.ndarray, lo: float, hi: float, n_basis: int, degree: int = 3) -> sp.csr_matrix:
    """Clamped cubic B-spline design matrix on [lo, hi]; x is clipped into the interval."""
    t = _knots(lo, hi, n_basis, degree)
    x = np.clip(np.asarray(x, float), lo, hi - 1e-9 * (hi - lo))
    return BSpline.design_matrix(x, t, degree).tocsr()


def multitaper_frames(x: np.ndarray, fs: float, nperseg: int = 256, hop: int = 128, nw: float = 2.0,
                      k: int = 3, f_lo: float = 0.5, f_hi: float = 20.0):
    """K-taper periodograms of consecutive frames.

    Returns (f, I, centre_index, K) with I of shape (n_frames, n_freq) in
    one-sided PSD units.
    """
    x = np.asarray(x, float)
    tapers = signal.windows.dpss(nperseg, nw, k)  # (k, nperseg), unit energy
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = (f >= f_lo) & (f <= f_hi)
    starts = np.arange(0, len(x) - nperseg + 1, hop)
    I = np.empty((len(starts), int(keep.sum())))
    for n, s0 in enumerate(starts):
        seg = x[s0:s0 + nperseg]
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(tapers * seg[None, :], axis=1)) ** 2  # (k, nfreq)
        I[n] = (2.0 / fs) * spec.mean(0)[keep]
    return f[keep], I, starts + nperseg // 2, k


@dataclass
class SCRData:
    """Frames for an SCR fit: periodograms, speeds, channel and segment labels."""

    f: np.ndarray  # (J,)
    I: np.ndarray  # (N, J)
    rho: np.ndarray  # (N,)
    channel: np.ndarray  # (N,) ints 0..C-1
    segment: np.ndarray  # (N,) ints 0..S-1
    k_tapers: int = 3
    segment_edges: list = field(default_factory=list)  # pairs (s, s') penalised for smoothness along theta


@dataclass
class SCRFit:
    alpha: np.ndarray  # (C, M_A) channel coefficients
    beta: np.ndarray  # (S, M_B) segment coefficients
    logf_range: tuple
    lognu_range: tuple
    n_basis_a: int
    n_basis_b: int
    objective: float
    converged: bool
    message: str

    def A(self, f: np.ndarray, channel: int) -> np.ndarray:
        return bspline_basis(np.log(f), *self.logf_range, self.n_basis_a) @ self.alpha[channel]

    def B(self, nu: np.ndarray, segment: int) -> np.ndarray:
        return bspline_basis(np.log(nu), *self.lognu_range, self.n_basis_b) @ self.beta[segment]


def _second_diff(m: int) -> sp.csr_matrix:
    return sp.diags([np.ones(m - 2), -2 * np.ones(m - 2), np.ones(m - 2)], [0, 1, 2], shape=(m - 2, m)).tocsr()


def fit_scr(data: SCRData, n_basis_a: int = 12, n_basis_b: int = 10, nu_range: tuple = (0.01, 5.0),
            lam_a: float = 10.0, lam_b: float = 10.0, lam_theta: float = 1.0, lam_gauge: float = 1e4,
            maxiter: int = 3000) -> SCRFit:
    """Fit channel and segment spectra by penalised Whittle likelihood (L-BFGS on a convex objective).

    Gauge: the mean coefficient of channel 0 is pinned to zero (B-splines are a
    partition of unity, so this pins the mean level of A_0), because (A + c, B - c)
    fit equally well.
    """
    I = np.asarray(data.I, float)
    N, J = I.shape
    C = int(data.channel.max()) + 1
    S = int(data.segment.max()) + 1
    logf = np.log(data.f)
    logf_range = (float(logf.min()), float(logf.max()))
    lognu_range = (float(np.log(nu_range[0])), float(np.log(nu_range[1])))
    # channel block: row (n, j) gets Phi_A[j] in block channel[n]
    Phi = bspline_basis(logf, *logf_range, n_basis_a).toarray()  # (J, M_A)
    jj, mm = np.nonzero(Phi)
    rows = (np.arange(N)[:, None] * J + jj[None, :]).ravel()
    cols = (data.channel[:, None] * n_basis_a + mm[None, :]).ravel()
    vals = np.tile(Phi[jj, mm], N)
    XA = sp.csr_matrix((vals, (rows, cols)), shape=(N * J, C * n_basis_a))
    # segment block: row (n, j) gets Psi(log f_j - log rho_n) in block segment[n]
    lognu = (logf[None, :] - np.log(data.rho)[:, None]).ravel()
    Psi = bspline_basis(lognu, *lognu_range, n_basis_b).tocoo()
    XB = sp.csr_matrix((Psi.data, (Psi.row, Psi.col + data.segment[Psi.row // J] * n_basis_b)),
                       shape=(N * J, S * n_basis_b))
    X = sp.hstack([XA, XB]).tocsr()
    offset = np.repeat(-np.log(data.rho), J)
    y = I.ravel()
    K = float(data.k_tapers)
    # penalties
    DA = _second_diff(n_basis_a)
    DB = _second_diff(n_basis_b)
    P = sp.block_diag([lam_a * sp.kron(sp.eye(C), DA.T @ DA), lam_b * sp.kron(sp.eye(S), DB.T @ DB)]).tocsr()
    if data.segment_edges and lam_theta > 0:
        e = np.asarray(data.segment_edges, int)
        Lr = np.r_[np.arange(len(e)), np.arange(len(e))]
        Lc = np.r_[e[:, 0], e[:, 1]]
        Lv = np.r_[np.ones(len(e)), -np.ones(len(e))]
        Dseg = sp.csr_matrix((Lv, (Lr, Lc)), shape=(len(e), S))
        Lap = (Dseg.T @ Dseg).tocsr()
        P = P + sp.block_diag([sp.csr_matrix((C * n_basis_a, C * n_basis_a)),
                               lam_theta * sp.kron(Lap, sp.eye(n_basis_b))]).tocsr()
    g = np.zeros(C * n_basis_a + S * n_basis_b)
    g[:n_basis_a] = 1.0 / n_basis_a
    scale = 1.0 / (N * J)

    def fun(w):
        u = np.clip(X @ w + offset, -60, 60)
        r = y * np.exp(-u)
        Pw = P @ w
        gw = g @ w
        val = (K * np.sum(u + r) + 0.5 * w @ Pw + 0.5 * lam_gauge * gw**2) * scale
        grad = (X.T @ (K * (1.0 - r)) + Pw + lam_gauge * gw * g) * scale
        return val, grad

    # start: channels at 0, segments at the mean of log I + log rho (B-splines sum to 1)
    w0 = np.zeros(X.shape[1])
    w0[C * n_basis_a:] = float(np.mean(np.log(np.maximum(y, 1e-30)) - offset))
    res = minimize(fun, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-12, "gtol": 1e-8})
    w = res.x
    return SCRFit(alpha=w[:C * n_basis_a].reshape(C, n_basis_a), beta=w[C * n_basis_a:].reshape(S, n_basis_b),
                  logf_range=logf_range, lognu_range=lognu_range, n_basis_a=n_basis_a, n_basis_b=n_basis_b,
                  objective=float(res.fun), converged=bool(res.success), message=str(res.message))
