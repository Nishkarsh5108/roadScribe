"""Agreement and variance-share statistics for paired recordings (PREREG_P1.md)."""
from __future__ import annotations

import numpy as np

__all__ = ["weighted_kappa", "variance_shares", "block_indices", "fit_cutpoints", "apply_cutpoints"]


def weighted_kappa(a, b, n_classes: int) -> float:
    """Quadratic-weighted Cohen's kappa for integer labels 0..n_classes-1 (NaN if undefined)."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    O = np.zeros((n_classes, n_classes))
    np.add.at(O, (a, b), 1.0)
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((n_classes, n_classes))
    W = (i - j) ** 2 / (n_classes - 1) ** 2
    den = float((W * E).sum())
    return float("nan") if den <= 0 else float(1.0 - (W * O).sum() / den)


def _demean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(x, float, copy=True)
    _, inv = np.unique(groups, return_inverse=True)
    if out.ndim == 1:
        out -= (np.bincount(inv, out) / np.bincount(inv))[inv]
    else:
        cnt = np.bincount(inv)
        for c in range(out.shape[1]):
            out[:, c] -= (np.bincount(inv, out[:, c]) / cnt)[inv]
    return out


def _ssr(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return float(y @ y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def variance_shares(y, location, vehicle, log_v) -> dict:
    """Type-II sums of squares for y ~ location FE + vehicle FE + log v, as shares of the total.

    Returns shares for location, vehicle, speed and residual, plus the
    within-location slope of y on log v (vehicle FE included).
    """
    y = np.asarray(y, float)
    location = np.asarray(location)
    vehicle = np.asarray(vehicle)
    lv = np.asarray(log_v, float)
    vehs = np.unique(vehicle)
    D = (vehicle[:, None] == vehs[None, 1:]).astype(float)  # vehicle dummies, first as base
    sst = float(((y - y.mean()) ** 2).sum())
    # models with location FE: demean within location
    yd, Dd, lvd = _demean(y, location), _demean(D, location), _demean(lv, location)
    full = _ssr(yd, np.c_[Dd, lvd])
    no_veh = _ssr(yd, lvd[:, None])
    no_speed = _ssr(yd, Dd)
    # model without location FE (with intercept)
    yc = y - y.mean()
    Dc = D - D.mean(0)
    lvc = lv - lv.mean()
    no_loc = _ssr(yc, np.c_[Dc, lvc])
    beta, *_ = np.linalg.lstsq(np.c_[Dd, lvd], yd, rcond=None)
    return dict(
        location=(no_loc - full) / sst,
        vehicle=(no_veh - full) / sst,
        speed=(no_speed - full) / sst,
        residual=full / sst,
        slope_log_v_within=float(beta[-1]),
    )


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block resample of positions 0..n-1 (non-overlapping blocks, drawn with replacement)."""
    starts = np.arange(0, n, block)
    pick = rng.choice(starts, size=len(starts), replace=True)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in pick])


def fit_cutpoints(x, y, n_grid: int = 60) -> tuple[float, float]:
    """Two cut-points maximising accuracy of class = (x>=c1) + (x>=c2) for labels 0/1/2."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    grid = np.unique(np.quantile(x, np.linspace(0, 1, n_grid)))
    best, arg = -1.0, (grid[0], grid[-1])
    for i, c1 in enumerate(grid):
        for c2 in grid[i:]:
            acc = float(np.mean(((x >= c1).astype(int) + (x >= c2).astype(int)) == y))
            if acc > best:
                best, arg = acc, (float(c1), float(c2))
    return arg


def apply_cutpoints(x, cuts: tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, float)
    return (x >= cuts[0]).astype(int) + (x >= cuts[1]).astype(int)
