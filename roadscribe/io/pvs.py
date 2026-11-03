"""PVS (Passive Vehicular Sensors) loader: ONE logger plus labels (METHODOLOGY §8.1).

* Never average the left and right loggers: they have separate clocks, offset
  by up to 0.54 s, and averaging x(t) with x(t - delta) is a comb filter.
  Use the left logger; its dashboard IMU is the phone proxy and its
  suspension IMUs are privileged references.
* Do not use ``dataset_gps_mpu_mid.csv`` (10 Hz decimation of the L/R average).
* PVS gyroscopes log deg/s; :class:`~roadscribe.schemas.RawTrip` holds rad/s.
* PVS sensors are fixed IMU modules, not smartphones.

The dataset location is taken from the ``ROADSCRIBE_PVS_DIR`` environment
variable by the data-dependent tests; it must contain ``PVS 1`` ... ``PVS 9``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal

from ..schemas import RawTrip

__all__ = [
    "PVS_ENV",
    "ROUTES",
    "pvs_root",
    "trip_dir",
    "route_of",
    "vehicle_of",
    "read_logger",
    "read_labels",
    "load_pvs_trip",
    "surface_labels",
    "bump_mask",
    "xcorr_lag",
    "clock_offset_s",
]

PVS_ENV = "ROADSCRIBE_PVS_DIR"
#: Trips per route; each route is driven by all three vehicle-driver pairs.
ROUTES: dict[str, tuple[int, ...]] = {"A": (1, 4, 7), "B": (2, 5, 8), "C": (3, 6, 9)}
SIDES = ("left", "right")
SENSOR_COLS = {
    "acc": ["acc_x_{p}", "acc_y_{p}", "acc_z_{p}"],
    "gyro": ["gyro_x_{p}", "gyro_y_{p}", "gyro_z_{p}"],
}
GNSS_COLS = ["timestamp_gps", "latitude", "longitude", "speed"]
REFERENCE_COLS = ["acc_z_above_suspension", "acc_z_below_suspension"]


def pvs_root(path: str | os.PathLike | None = None) -> Path:
    p = path or os.environ.get(PVS_ENV)
    if not p:
        raise FileNotFoundError(f"set {PVS_ENV} to the folder holding 'PVS 1' ... 'PVS 9'")
    return Path(p)


def trip_dir(root: str | os.PathLike, trip: int) -> Path:
    return Path(root) / f"PVS {trip}"


def route_of(trip: int) -> str:
    return "ABC"[(trip - 1) % 3]


def vehicle_of(trip: int) -> int:
    return (trip - 1) // 3 + 1


def read_logger(root: str | os.PathLike, trip: int, side: Literal["left", "right"] = "left", usecols=None) -> pd.DataFrame:
    if side not in SIDES:
        raise ValueError("side must be 'left' or 'right'")
    return pd.read_csv(trip_dir(root, trip) / f"dataset_gps_mpu_{side}.csv", usecols=usecols)


def read_labels(root: str | os.PathLike, trip: int, usecols=None) -> pd.DataFrame:
    return pd.read_csv(trip_dir(root, trip) / "dataset_labels.csv", usecols=usecols)


def _sha256(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:16]


def load_pvs_trip(
    root: str | os.PathLike | None,
    trip: int,
    side: Literal["left", "right"] = "left",
    placement: str = "dashboard",
    with_labels: bool = True,
) -> RawTrip:
    """Load one logger of one PVS trip as a :class:`RawTrip` (gyro converted to rad/s).

    The two suspension channels of the same logger are kept in
    ``meta["reference"]`` as privileged references.
    """
    root = pvs_root(root)
    acc = [c.format(p=placement) for c in SENSOR_COLS["acc"]]
    gyr = [c.format(p=placement) for c in SENSOR_COLS["gyro"]]
    path = trip_dir(root, trip) / f"dataset_gps_mpu_{side}.csv"
    header = pd.read_csv(path, nrows=0).columns
    ref_cols = [c for c in REFERENCE_COLS if c in header]
    df = pd.read_csv(path, usecols=["timestamp", *acc, *gyr, *GNSS_COLS, *ref_cols])
    labels = None
    if with_labels:
        labels = read_labels(root, trip)
        n = min(len(df), len(labels))
        df, labels = df.iloc[:n].reset_index(drop=True), labels.iloc[:n].reset_index(drop=True)
    t = df["timestamp"].to_numpy(float)
    keep = np.concatenate([[True], np.diff(t) > 0])  # enforce a strictly increasing clock
    df = df[keep].reset_index(drop=True)
    if labels is not None:
        labels = labels[keep].reset_index(drop=True)
    t = df["timestamp"].to_numpy(float)
    fs = float(1.0 / np.median(np.diff(t)))
    gnss = (
        df[GNSS_COLS]
        .rename(columns={"timestamp_gps": "t", "latitude": "lat", "longitude": "lon", "speed": "speed_mps"})
        .drop_duplicates("t")
        .reset_index(drop=True)
    )
    return RawTrip(
        trip_id=f"pvs{trip}",
        source="pvs",
        route_id=route_of(trip),
        vehicle_id=str(vehicle_of(trip)),
        driver_id=str(vehicle_of(trip)),  # PVS pairs each vehicle with one driver
        t=t,
        f_b=df[acc].to_numpy(float),
        w_b=np.radians(df[gyr].to_numpy(float)),
        fs=fs,
        gnss=gnss,
        labels=labels,
        meta={
            "logger": side,
            "placement": placement,
            "mount": "fixed IMU module (not a smartphone)",
            "gyro_units_source": "deg/s",
            "reference": {c: df[c].to_numpy(float) for c in ref_cols},
            "provenance": {"file": str(path), "sha256_head": _sha256(path), "n_dropped_nonmonotonic": int((~keep).sum())},
        },
    )


def surface_labels(labels: pd.DataFrame) -> np.ndarray:
    """Per-sample surface: 'asphalt', 'cobble', 'dirt' or 'none'.

    Where several flags are set, dirt takes precedence over cobble over
    asphalt. This is the convention of ``pvs_facts.py``, under which PVS has
    54 surface transitions.
    """
    out = np.full(len(labels), "none", dtype=object)
    out[labels["asphalt_road"].to_numpy() == 1] = "asphalt"
    out[labels["cobblestone_road"].to_numpy() == 1] = "cobble"
    out[labels["dirt_road"].to_numpy() == 1] = "dirt"
    return out


def bump_mask(labels: pd.DataFrame) -> np.ndarray:
    return (labels["speed_bump_asphalt"].to_numpy() == 1) | (labels["speed_bump_cobblestone"].to_numpy() == 1)


def xcorr_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[int, float, float]:
    """Lag (samples) maximising the normalised cross-correlation of a and b.

    Returns (lag, peak correlation, correlation at lag 0). A positive lag
    means ``a`` lags ``b``.
    """
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    c = signal.correlate(a, b, mode="full", method="fft") / n
    lags = np.arange(-n + 1, n)
    k = np.abs(lags) <= maxlag
    i = int(np.argmax(c[k]))
    return int(lags[k][i]), float(c[k][i]), float(c[lags == 0][0])


def clock_offset_s(
    yaw_a: np.ndarray, yaw_b: np.ndarray, fs: float, maxlag_s: float = 30.0, band: tuple[float, float] = (0.05, 2.0)
) -> tuple[float, float]:
    """Clock offset between two loggers by yaw-rate cross-correlation (§8.1, §8.5).

    Both yaw rates are band-passed to ``band`` first. Returns (offset_s, peak r).
    """
    sos = signal.butter(2, [band[0] / (fs / 2), band[1] / (fs / 2)], "band", output="sos")
    n = min(len(yaw_a), len(yaw_b))
    a = signal.sosfiltfilt(sos, yaw_a[:n])
    b = signal.sosfiltfilt(sos, yaw_b[:n])
    lag, pk, _ = xcorr_lag(a, b, int(maxlag_s * fs))
    return lag / fs, pk
