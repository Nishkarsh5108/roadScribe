"""Per-stretch features on a common route coordinate (definitions fixed in PREREG_P1.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from ..io.pvs import bump_mask, surface_labels
from ..layer1.iso2631 import comfort_class

__all__ = ["quality_left", "stretch_table"]


def quality_left(labels: pd.DataFrame) -> np.ndarray:
    """Per-sample left-side quality: 0 good, 1 regular, 2 bad, NaN if unlabelled."""
    q = np.full(len(labels), np.nan)
    q[labels["good_road_left"].to_numpy() == 1] = 0
    q[labels["regular_road_left"].to_numpy() == 1] = 1
    q[labels["bad_road_left"].to_numpy() == 1] = 2
    return q


def _mode(values: np.ndarray):
    vals, counts = np.unique(values, return_counts=True)
    return vals[int(np.argmax(counts))]


def _band_energy(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    nper = int(min(256, len(x)))
    if nper < 32:
        return float("nan")
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=nper, noverlap=nper // 2)
    m = (f >= f_lo) & (f <= f_hi)
    if m.sum() < 2:
        return float("nan")
    return float(np.sum(p[m]) * (f[1] - f[0]))


def stretch_table(
    theta: np.ndarray,
    valid: np.ndarray,
    v: np.ndarray,
    aw: np.ndarray,
    az_raw: np.ndarray,
    fs: float,
    labels: pd.DataFrame | None,
    bin_m: float = 100.0,
    coverage_min: float = 0.7,
    gamma: float = 0.35,
    v_ref: float = 10.0,
) -> pd.DataFrame:
    """One row per stretch with enough valid coverage.

    ``aw`` holds the ISO-weighted vehicle axes (n,3); ``az_raw`` the unweighted
    gravity-free vertical acceleration; ``v`` per-sample speed in m/s.
    """
    theta = np.asarray(theta, float)
    stretch = np.floor(theta / bin_m).astype(int)
    q = quality_left(labels) if labels is not None else None
    surf = surface_labels(labels) if labels is not None else None
    bump = bump_mask(labels) if labels is not None else None
    rows = []
    for b in np.unique(stretch[valid]):
        idx = np.flatnonzero(valid & (stretch == b))
        cells = np.unique(np.floor(theta[idx]).astype(int))
        coverage = len(cells) / bin_m
        if coverage < coverage_min:
            continue
        v_mean = float(np.mean(v[idx]))
        a_wx, a_wy, a_wz = np.sqrt(np.mean(aw[idx] ** 2, axis=0))
        a_v = float(np.sqrt(a_wx**2 + a_wy**2 + a_wz**2))
        a_dec = float(a_wz * (v_ref / v_mean) ** gamma) if v_mean > 0 else float("nan")
        e = _band_energy(az_raw[idx], fs, max(0.3, 0.1 * v_mean), min(0.45 * fs, 1.0 * v_mean))
        row = dict(stretch=int(b), n=len(idx), coverage=coverage, v_mean=v_mean,
                   a_wx=float(a_wx), a_wy=float(a_wy), a_wz=float(a_wz), a_v=a_v,
                   felt_class=comfort_class(a_v), a_dec=a_dec,
                   road_class=comfort_class(a_dec) if np.isfinite(a_dec) else -1,
                   r2=float(np.log(e)) if np.isfinite(e) and e > 0 else float("nan"))
        if labels is not None:
            qq = q[idx][np.isfinite(q[idx])]
            row["gt_quality"] = float(_mode(qq)) if len(qq) else float("nan")
            row["gt_surface"] = str(_mode(surf[idx]))
            row["gt_bump"] = bool(bump[idx].any())
        rows.append(row)
    return pd.DataFrame(rows)
