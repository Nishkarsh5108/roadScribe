r"""Map-informed PELT (MI-PELT, METHODOLOGY §4.3).

Objective

.. math::
   \min_{m,\tau_{1:m}}\sum_{i=1}^{m+1}C(\mathbf y_{\tau_{i-1}+1:\tau_i})+\sum_{i=1}^{m}\beta(\tau_i),

with the Gaussian mean-shift cost under a shared diagonal covariance and a
location-dependent penalty :math:`\beta(\tau)=\beta_0(1-\lambda\pi(\tau))` that
drops near OpenStreetMap transitions.

Recursion: :math:`F(0)=0`, :math:`G(t)=F(t)+\beta(t)\mathbf 1[t>0]`,
:math:`F(s)=\min_{t\in\mathcal R_s,\,s-t\ge\ell_{min}}\{G(t)+C(\mathbf y_{t+1:s})\}`.

**Corrected pruning.** A candidate :math:`t` is discarded at step :math:`s`
when :math:`G(t)+C(\mathbf y_{t+1:s})\ge G(s)`. Because the mean-shift cost is
superadditive this is safe for every :math:`T\ge s+\ell_{min}` (Lemma, §4.3).
The textbook condition :math:`F(t)+C\ge F(s)` drops the penalties and is only
equivalent when :math:`\beta` is constant; with cheaper change points at map
transitions it discards valid candidates.

**Minimum segment length.** With :math:`\ell_{min}>1` a pruned :math:`t` may
still be the optimal last change point for :math:`T<s+\ell_{min}`, where
:math:`s` is not yet admissible. Pruned candidates are therefore kept until
:math:`T=s+\ell_{min}-1` and only then removed, which keeps the result equal
to exhaustive optimal partitioning under the same length constraint.

Index convention: a change point :math:`\tau\in\{1,\dots,n-1\}` means a new
segment starts at 0-based row :math:`\tau`; ``beta`` has length ``n + 1``
and ``beta[tau]`` is the price of a change point at :math:`\tau`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..config import MIPELTConfig

__all__ = [
    "MeanShiftCost",
    "PeltResult",
    "base_penalty",
    "changepoint_positions",
    "map_prior",
    "location_penalty",
    "mipelt",
    "optimal_partitioning",
    "segment",
]

Pruning = Literal["corrected", "textbook", "none"]
_NEVER = np.iinfo(np.int64).max


class MeanShiftCost:
    r"""C(y_{a:b}) = sum_j ||y_j - mean||^2_{D^{-1}} in O(1) from cumulative sums.

    ``D`` is the shared diagonal covariance: ``None`` (identity, for
    robust-standardised features), an array of per-dimension variances, or
    ``"auto"`` for a robust noise-variance estimate from first differences,
    var_k = (1.4826 * MAD(diff y_k))^2 / 2, which is insensitive to the mean
    shifts themselves.
    """

    def __init__(self, Y: np.ndarray, D: np.ndarray | str | None = None):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2 or len(Y) < 1:
            raise ValueError("Y must be (n, d) with n >= 1")
        if not np.all(np.isfinite(Y)):
            raise ValueError("Y contains non-finite values")
        if isinstance(D, str):
            if D != "auto":
                raise ValueError("D must be None, an array, or 'auto'")
            dy = np.diff(Y, axis=0)
            mad = np.median(np.abs(dy - np.median(dy, axis=0)), axis=0)
            D = np.maximum((1.4826 * mad) ** 2 / 2.0, 1e-12)
        if D is not None:
            D = np.asarray(D, dtype=float)
            if D.shape != (Y.shape[1],) or np.any(D <= 0):
                raise ValueError("D must be a positive vector with one variance per feature")
            Y = Y / np.sqrt(D)
        self.D = D
        self.n, self.d = Y.shape
        self._c1 = np.vstack([np.zeros(self.d), np.cumsum(Y, axis=0)])
        self._c2 = np.concatenate([[0.0], np.cumsum(np.sum(Y**2, axis=1))])

    def __call__(self, t: int, s: int) -> float:
        """Cost of rows t..s-1 (the methodology's y_{t+1:s})."""
        m = s - t
        if m <= 0:
            raise ValueError("empty segment")
        d1 = self._c1[s] - self._c1[t]
        return float(self._c2[s] - self._c2[t] - d1 @ d1 / m)

    def many(self, ts: np.ndarray, s: int) -> np.ndarray:
        """Vectorised cost of rows t..s-1 for every t in ``ts`` (all < s)."""
        ts = np.asarray(ts, dtype=np.int64)
        m = (s - ts).astype(float)
        d1 = self._c1[s][None, :] - self._c1[ts]
        return self._c2[s] - self._c2[ts] - np.einsum("ij,ij->i", d1, d1) / m


@dataclass(frozen=True)
class PeltResult:
    changepoints: tuple[int, ...]  # sorted, each in 1..n-1
    cost: float  # objective value F(n)
    F: np.ndarray  # (n+1,) optimal cost of y_{1:s}
    n_evaluations: int  # number of cost evaluations in the minimisation
    max_candidates: int  # largest candidate set seen

    @property
    def n_segments(self) -> int:
        return len(self.changepoints) + 1


def base_penalty(c: float, d: int, n: int) -> float:
    """beta_0 = c d ln n."""
    return float(c * d * np.log(n))


def changepoint_positions(s_grid: np.ndarray) -> np.ndarray:
    """Along-track position of change point tau = 0..n (midpoint between rows tau-1 and tau)."""
    s_grid = np.asarray(s_grid, dtype=float)
    pos = np.empty(len(s_grid) + 1)
    pos[0], pos[-1] = s_grid[0], s_grid[-1]
    pos[1:-1] = 0.5 * (s_grid[:-1] + s_grid[1:])
    return pos


def map_prior(
    s_pos: np.ndarray, s_b: Sequence[float], w_b: Sequence[float], sigma_b: float = 15.0
) -> np.ndarray:
    """pi(tau) = max_b w_b exp(-(s_tau - s_b)^2 / (2 sigma_b^2)); 0 without transitions."""
    s_pos = np.asarray(s_pos, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    w_b = np.asarray(w_b, dtype=float)
    if s_b.size == 0:
        return np.zeros_like(s_pos)
    if s_b.shape != w_b.shape:
        raise ValueError("s_b and w_b must have the same length")
    k = w_b[None, :] * np.exp(-((s_pos[:, None] - s_b[None, :]) ** 2) / (2.0 * sigma_b**2))
    return k.max(axis=1)


def location_penalty(
    s_grid: np.ndarray,
    s_b: Sequence[float],
    w_b: Sequence[float],
    beta0: float,
    lam: float,
    sigma_b: float = 15.0,
) -> np.ndarray:
    """beta(tau) = beta_0 (1 - lambda pi(tau)) for tau = 0..n (beta[0] is never charged)."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must be in [0, 1)")
    pi = map_prior(changepoint_positions(s_grid), s_b, w_b, sigma_b)
    return beta0 * (1.0 - lam * pi)


def _as_cost(Y_or_cost) -> MeanShiftCost:
    return Y_or_cost if isinstance(Y_or_cost, MeanShiftCost) else MeanShiftCost(Y_or_cost)


def _as_beta(beta, n: int) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    if b.ndim == 0:
        return np.full(n + 1, float(b))
    if b.shape != (n + 1,):
        raise ValueError(f"beta must be a scalar or have shape ({n + 1},), got {b.shape}")
    return b


def _backtrack(last: np.ndarray, n: int) -> tuple[int, ...]:
    cps = []
    s = n
    while s > 0:
        t = int(last[s])
        if t > 0:
            cps.append(t)
        s = t
    return tuple(sorted(cps))


def mipelt(
    Y_or_cost,
    beta: float | np.ndarray,
    min_size: int = 1,
    pruning: Pruning = "corrected",
) -> PeltResult:
    """Exact penalised segmentation with PELT pruning.

    ``pruning="corrected"`` is the rule of the §4.3 lemma and returns the
    optimal partition for any location-dependent penalty. ``"textbook"`` is
    the classical rule F(t) + C >= F(s), kept for the ablation that shows it
    is sub-optimal under location-dependent penalties. ``"none"`` disables
    pruning (optimal partitioning).
    """
    cost = _as_cost(Y_or_cost)
    n = cost.n
    beta = _as_beta(beta, n)
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if n < min_size:
        raise ValueError(f"series of length {n} is shorter than min_size={min_size}")
    if pruning not in ("corrected", "textbook", "none"):
        raise ValueError(f"unknown pruning rule {pruning!r}")

    F = np.full(n + 1, np.inf)
    G = np.full(n + 1, np.inf)
    F[0] = G[0] = 0.0
    last = np.full(n + 1, -1, dtype=np.int64)
    cands = np.array([0], dtype=np.int64)
    expiry = np.array([_NEVER], dtype=np.int64)  # last T for which the candidate may be used
    n_eval = 0
    max_c = 1
    for s in range(1, n + 1):
        adm = s - cands >= min_size
        if adm.any():
            ts = cands[adm]
            vals = G[ts] + cost.many(ts, s)
            n_eval += len(ts)
            k = int(np.argmin(vals))
            F[s] = vals[k]
            last[s] = ts[k]
        G[s] = F[s] + beta[s] if s < n else F[s]
        if pruning != "none" and np.isfinite(G[s]) and s < n:
            vals_all = G[cands] + cost.many(cands, s)
            if pruning == "corrected":
                prune = vals_all >= G[s]
            else:  # textbook: compare F(t) + C with F(s), i.e. drop both penalties
                off = np.where(cands > 0, beta[cands], beta[s])
                prune = vals_all - off >= G[s] - beta[s]
            expiry[prune] = np.minimum(expiry[prune], s + min_size - 1)
        keep = expiry >= s + 1
        cands = np.append(cands[keep], s)
        expiry = np.append(expiry[keep], _NEVER)
        max_c = max(max_c, len(cands))
    if not np.isfinite(F[n]):
        raise ValueError("no feasible segmentation")
    return PeltResult(_backtrack(last, n), float(F[n]), F, n_eval, max_c)


def optimal_partitioning(Y_or_cost, beta: float | np.ndarray, min_size: int = 1) -> PeltResult:
    """Exhaustive O(n^2) optimal partitioning; the reference that MI-PELT must match."""
    cost = _as_cost(Y_or_cost)
    n = cost.n
    beta = _as_beta(beta, n)
    F = np.full(n + 1, np.inf)
    G = np.full(n + 1, np.inf)
    F[0] = G[0] = 0.0
    last = np.full(n + 1, -1, dtype=np.int64)
    n_eval = 0
    for s in range(1, n + 1):
        ts = np.arange(0, s - min_size + 1, dtype=np.int64)
        if len(ts):
            vals = G[ts] + cost.many(ts, s)
            n_eval += len(ts)
            k = int(np.argmin(vals))
            F[s] = vals[k]
            last[s] = ts[k]
        G[s] = F[s] + beta[s] if s < n else F[s]
    if not np.isfinite(F[n]):
        raise ValueError("no feasible segmentation")
    return PeltResult(_backtrack(last, n), float(F[n]), F, n_eval, n + 1)


def segment(
    Y: np.ndarray,
    s_grid: np.ndarray,
    s_b: Sequence[float],
    w_b: Sequence[float],
    cfg: MIPELTConfig | None = None,
    D: np.ndarray | str | None = None,
) -> tuple[PeltResult, np.ndarray]:
    """Run MI-PELT on a distance-domain feature matrix with the §4.3 penalty.

    Returns the result and the penalty vector used. ``lam=0`` is plain PELT.
    """
    cfg = cfg or MIPELTConfig()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, d = Y.shape
    beta0 = base_penalty(cfg.c, d, n)
    beta = location_penalty(s_grid, s_b, w_b, beta0, cfg.lam, cfg.sigma_b_m)
    return mipelt(MeanShiftCost(Y, D), beta, cfg.min_size, cfg.pruning), beta
