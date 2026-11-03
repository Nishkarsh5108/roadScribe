"""GNSS helpers: PCHIP speed, longitudinal acceleration, course rate (§3.0)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

__all__ = [
    "GnssFixes",
    "clean_fixes",
    "speed_and_alon",
    "haversine_m",
    "bearing_deg",
    "course_rate_deg_s",
    "along_track_distance",
]

EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class GnssFixes:
    """Unique, time-sorted GNSS fixes with finite speed."""

    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray
    course_deg: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.t)


def clean_fixes(gnss: pd.DataFrame, v_max: float = 70.0) -> GnssFixes:
    """Sort, de-duplicate and drop implausible fixes (NaN or |v| > ``v_max`` m/s).

    PVS repeats each 1 Hz fix on every IMU row; this keeps one row per fix.
    """
    g = gnss.sort_values("t").drop_duplicates("t")
    ok = np.isfinite(g["speed_mps"].to_numpy(float)) & np.isfinite(g["t"].to_numpy(float))
    ok &= (g["speed_mps"].to_numpy(float) >= 0) & (g["speed_mps"].to_numpy(float) <= v_max)
    g = g[ok]
    if len(g) < 2:
        raise ValueError("need at least two valid GNSS fixes")
    course = g["course_deg"].to_numpy(float) if "course_deg" in g.columns else None
    return GnssFixes(
        t=g["t"].to_numpy(float),
        lat=g["lat"].to_numpy(float),
        lon=g["lon"].to_numpy(float),
        speed=g["speed_mps"].to_numpy(float),
        course_deg=course,
    )


def speed_and_alon(t: np.ndarray, tg: np.ndarray, vg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Speed interpolated to ``t`` with PCHIP, and its time derivative a_lon^G.

    Outside the GNSS time span the speed is held at the end value and
    a_lon is 0. Speed is clipped at 0 (PCHIP cannot overshoot, but noisy fixes
    can be slightly negative after cleaning thresholds).
    """
    p = PchipInterpolator(tg, vg, extrapolate=False)
    dp = p.derivative()
    tc = np.clip(t, tg[0], tg[-1])
    v = np.maximum(p(tc), 0.0)
    a = dp(tc)
    a[(t < tg[0]) | (t > tg[-1])] = 0.0
    return v, a


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y))


def course_rate_deg_s(fixes: GnssFixes, v_min: float = 3.0) -> np.ndarray:
    """GNSS course rate at each fix (deg/s, clockwise positive, NaN when slow).

    Uses the receiver course if present, otherwise bearings between
    consecutive fixes. Course is clockwise from north, so a left turn
    (positive vehicle yaw rate, z up) has a *negative* course rate.
    """
    t = fixes.t
    if fixes.course_deg is not None and np.isfinite(fixes.course_deg).sum() >= 2:
        course = fixes.course_deg
        tc = t
    else:
        course = bearing_deg(fixes.lat[:-1], fixes.lon[:-1], fixes.lat[1:], fixes.lon[1:])
        tc = 0.5 * (t[:-1] + t[1:])
    if len(tc) < 2:
        return np.full(len(t), np.nan)
    dc = (np.diff(course) + 180.0) % 360.0 - 180.0
    rate_mid = dc / np.diff(tc)
    t_mid = 0.5 * (tc[:-1] + tc[1:])
    ok = np.isfinite(rate_mid)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    rate = np.interp(t, t_mid[ok], rate_mid[ok])
    rate[fixes.speed < v_min] = np.nan
    return rate


def along_track_distance(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """s(t) = integral of v dt (trapezoid), starting at 0. Used before map matching."""
    s = np.zeros(len(t))
    s[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))
    return s
