r"""Distance-domain feature grid for segmentation (METHODOLOGY §4.2).

Every :math:`\Delta s=5` m grid point gets

.. math::
   \mathbf y_j=[\log\tilde a_{w,z},\ \log a_{w,z},\ R_{wb},\ R_{hb},\ \text{event rate per 100 m},\ \log a_{w,y}]_j,

each computed over a centred 25 m window and robust-standardised per trip
(the sparse event-rate column uses one event per window as its scale).
Speed is deliberately not a feature (check boundaries against a speed-only
PELT baseline, §4.5). Band energies are window means of squared band-passed
:math:`a^v_z`, which equal the band integrals of the PSD and work for 25 m
windows that are too short for a 5.12 s Welch segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import signal

from ..config import DistanceConfig, SpectralConfig, SpeedConfig
from ..schemas import Event, VehicleFrameTrip
from ..layer1.spectral import wheel_band

__all__ = ["FEATURE_NAMES", "GridFeatures", "robust_standardize", "build_grid"]

FEATURE_NAMES: tuple[str, ...] = (
    "log_aw_z_dec",
    "log_aw_z",
    "r_wb",
    "r_hb",
    "event_rate_100m",
    "log_aw_y",
)


@dataclass(frozen=True)
class GridFeatures:
    s: np.ndarray  # (n,) grid positions, m
    t: np.ndarray  # (n,) IMU time at each grid point
    Y: np.ndarray  # (n, 6) robust-standardised features
    raw: np.ndarray  # (n, 6) features before standardisation
    v: np.ndarray  # (n,) window-mean speed (diagnostic, not a feature)
    names: tuple[str, ...] = FEATURE_NAMES


def robust_standardize(X: np.ndarray, scale_floor: np.ndarray | float = 0.0) -> np.ndarray:
    """(x - median) / max(1.4826 MAD, scale_floor) per column.

    A column whose MAD is 0 (e.g. a sparse count) falls back to ``scale_floor``,
    then to its SD, then to 1.
    """
    X = np.asarray(X, dtype=float)
    med = np.median(X, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(X - med), axis=0), scale_floor)
    sd = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(sd > 1e-12, sd, 1.0))
    return (X - med) / scale


def _bp_power(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band[0], min(band[1], 0.45 * fs)
    sos = signal.butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], "band", output="sos")
    return signal.sosfiltfilt(sos, x) ** 2


def build_grid(
    vft: VehicleFrameTrip,
    aw: np.ndarray,
    events: Sequence[Event] = (),
    cfg: DistanceConfig | None = None,
    speed_cfg: SpeedConfig | None = None,
    spectral_cfg: SpectralConfig | None = None,
    f_wh: float | None = None,
) -> GridFeatures:
    """Build the 5 m feature grid from vehicle-frame data.

    ``aw`` is the (N, 3) ISO-weighted acceleration (W_d on x, y; W_k on z).
    The event rate counts impulse events not labelled ``artifact``. For the
    speed-decoupled feature the window speed is floored at ``speed_cfg.v_min``
    (the feature grid needs a value everywhere; HRNU-level values use the
    ``speed_low`` rule instead).
    """
    cfg = cfg or DistanceConfig()
    speed_cfg = speed_cfg or SpeedConfig()
    spectral_cfg = spectral_cfg or SpectralConfig()
    s = vft.s
    fs = vft.fs
    if s[-1] - s[0] < 2 * cfg.ds_m:
        raise ValueError("trip is too short for a distance grid")
    grid = np.arange(s[0], s[-1] + 1e-9, cfg.ds_m)
    half = cfg.window_m / 2.0
    lo = np.searchsorted(s, grid - half, "left")
    hi = np.searchsorted(s, grid + half, "right")
    hi = np.maximum(hi, lo + 1)
    hi = np.minimum(hi, len(s))
    lo = np.minimum(lo, hi - 1)
    cnt = (hi - lo).astype(float)

    def wmean(x: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(x)])
        return (c[hi] - c[lo]) / cnt

    eps = cfg.eps
    a_wz = np.sqrt(wmean(aw[:, 2] ** 2))
    lat = aw[:, 1]
    if cfg.lateral_highpass_hz is not None:
        sos = signal.butter(2, cfg.lateral_highpass_hz / (fs / 2.0), "high", output="sos")
        lat = signal.sosfiltfilt(sos, lat)
    a_wy = np.sqrt(wmean(lat**2))
    v = wmean(vft.v)
    a_wz_dec = a_wz * (speed_cfg.v_ref / np.maximum(v, speed_cfg.v_min)) ** speed_cfg.gamma
    az = vft.a_v[:, 2]
    e_body = wmean(_bp_power(az, fs, spectral_cfg.body_band)) + eps
    e_wheel = wmean(_bp_power(az, fs, wheel_band(f_wh, spectral_cfg))) + eps
    e_high = wmean(_bp_power(az, fs, spectral_cfg.high_band)) + eps
    ev_s = np.sort([e.s for e in events if e.kind == "impulse" and e.label != "artifact"])
    n_ev = np.searchsorted(ev_s, grid + half, "right") - np.searchsorted(ev_s, grid - half, "left")
    rate = n_ev * (100.0 / cfg.window_m)

    raw = np.column_stack(
        [
            np.log(a_wz_dec + eps),
            np.log(a_wz + eps),
            np.log(e_wheel / e_body),
            np.log(e_high / e_body),
            rate,
            np.log(a_wy + eps),
        ]
    )
    t_grid = vft.t[np.clip(np.searchsorted(s, grid, "left"), 0, len(s) - 1)]
    # The event rate is a sparse count whose MAD is usually 0: one event in the
    # window is taken as one robust SD, so a single bump does not dominate y.
    floor = np.zeros(raw.shape[1])
    floor[FEATURE_NAMES.index("event_rate_100m")] = 100.0 / cfg.window_m
    return GridFeatures(s=grid, t=t_grid, Y=robust_standardize(raw, floor), raw=raw, v=v)
