"""Common route coordinate for repeated traversals of one route (pre-registered in PREREG_P1.md).

PVS routes are loops driven in the same direction by three vehicles, so a plain
nearest-point projection would confuse the start with the end. Fixes are
therefore matched in time order to the nearest reference point inside a
forward window around the previous match.
"""
from __future__ import annotations

import numpy as np

__all__ = ["enu", "densify", "match_forward", "sample_theta"]

EARTH_R = 6371000.0


def enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Local east/north metres (equirectangular; fine over ~20 km)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


def densify(xy: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at ``step`` metres; returns (points (M,2), arc length (M,))."""
    xy = np.asarray(xy, float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.r_[True, seg > 1e-6]  # drop repeated points (stops)
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    grid = np.arange(0.0, s[-1] + 1e-9, step)
    pts = np.c_[np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])]
    return pts, grid


def match_forward(
    track_xy: np.ndarray,
    ref_pts: np.ndarray,
    ref_s: np.ndarray,
    back_m: float = 30.0,
    fwd_m: float = 300.0,
    start_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """theta and distance to the reference for each fix, matched in time order."""
    track_xy = np.asarray(track_xy, float)
    theta = np.empty(len(track_xy))
    dist = np.empty(len(track_xy))
    p = float(start_s)
    for i, q in enumerate(track_xy):
        lo = int(np.searchsorted(ref_s, p - back_m, "left"))
        hi = int(np.searchsorted(ref_s, p + fwd_m, "right"))
        lo, hi = max(lo, 0), max(min(hi, len(ref_s)), lo + 1)
        d = np.hypot(ref_pts[lo:hi, 0] - q[0], ref_pts[lo:hi, 1] - q[1])
        k = int(np.argmin(d))
        theta[i], dist[i] = ref_s[lo + k], d[k]
        p = theta[i]
    return theta, dist


def sample_theta(
    t_samples: np.ndarray, t_fix: np.ndarray, theta_fix: np.ndarray, dist_fix: np.ndarray, max_dist_m: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """theta at sample times (linear in time) and validity (both bracketing fixes within max_dist_m)."""
    t_fix = np.asarray(t_fix, float)
    theta = np.interp(t_samples, t_fix, theta_fix)
    j = np.clip(np.searchsorted(t_fix, t_samples, "right"), 1, len(t_fix) - 1)
    ok_fix = np.asarray(dist_fix) <= max_dist_m
    valid = ok_fix[j - 1] & ok_fix[j] & (t_samples >= t_fix[0]) & (t_samples <= t_fix[-1])
    return theta, valid
