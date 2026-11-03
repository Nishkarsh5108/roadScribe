"""Synthetic trips with known ground truth for the Layer 1-3 tests."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation as Rot

from roadscribe.config import G0
from roadscribe.schemas import RawTrip, VehicleFrameTrip

# (duration s, a_lon m/s^2, yaw rate rad/s); speeds stay within ~4-15 m/s
SCHEDULE = [
    (10, 1.2, 0.0),
    (30, 0.0, 0.0),
    (8, 0.0, 0.15),
    (22, 0.0, 0.0),
    (6, -1.0, 0.0),
    (10, 0.0, -0.20),
    (8, 1.0, 0.0),
    (36, 0.0, 0.0),
    (6, -1.5, 0.0),
    (12, 0.0, 0.10),
    (9, 0.8, 0.0),
    (30, 0.0, 0.0),
    (7, 0.0, -0.18),
    (25, 0.0, 0.0),
]


@dataclass
class SynthTrip:
    raw: RawTrip
    R_vb: np.ndarray  # truth before any handling
    R_vb_after: np.ndarray | None  # truth after a handling event
    v: np.ndarray
    a_lon: np.ndarray
    w_z: np.ndarray
    f_v: np.ndarray  # vehicle-frame specific force
    meta: dict = field(default_factory=dict)


def _commands(t: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed a_lon and yaw-rate commands; the initial launch from rest is not repeated."""
    a = np.zeros_like(t)
    w = np.zeros_like(t)
    period = sum(d for d, _, _ in SCHEDULE)
    tt = t % period
    start = 0.0
    for d, al, wz in SCHEDULE:
        m = (tt >= start) & (tt < start + d)
        a[m], w[m] = al, wz
        start += d
    a[(t >= period) & (tt < SCHEDULE[0][0])] = 0.0
    sos = signal.butter(2, 0.5 / (fs / 2), "low", output="sos")
    return signal.sosfiltfilt(sos, a), signal.sosfiltfilt(sos, w)


def make_trip(
    R_vb: np.ndarray | None = None,
    duration: float = 600.0,
    fs: float = 100.0,
    seed: int = 0,
    vib_sd: float = 0.3,
    gyro: bool = True,
    handling_at: float | None = None,
    handling_rotvec_deg: tuple[float, float, float] = (0.0, 50.0, 30.0),
    handling_s: float = 1.0,
    v0: float = 0.5,
) -> SynthTrip:
    """A trip with a known mount R_vb (vehicle <- body)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fs)
    n = len(t)
    a_lon, w_z = _commands(t, fs)
    v = np.maximum(v0 + np.cumsum(a_lon) / fs, 0.0)
    a_lon = np.gradient(v, 1.0 / fs)

    # vehicle-frame specific force: longitudinal, centripetal (left positive), gravity
    sos_vib = signal.butter(2, [1.0 / (fs / 2), 20.0 / (fs / 2)], "band", output="sos")
    vib = signal.sosfiltfilt(sos_vib, rng.normal(0.0, 1.0, (n, 3)), axis=0)
    vib *= np.array([0.15, 0.15, 1.0]) * vib_sd / vib.std(axis=0)
    f_v = np.column_stack([a_lon, v * w_z, np.full(n, G0)]) + vib
    w_v = np.column_stack([np.zeros(n), np.zeros(n), w_z]) + rng.normal(0.0, 0.002, (n, 3))

    if R_vb is None:
        R_vb = Rot.from_euler("zyx", [90.0, 3.0, -2.0], degrees=True).as_matrix().T  # PVS-like
    R_bv = R_vb.T
    f_b = f_v @ R_bv.T
    w_b = w_v @ R_bv.T
    R_after = None
    if handling_at is not None:
        # the phone turns about a vehicle-frame axis k_v by theta(t) (raised cosine)
        rv = np.radians(np.asarray(handling_rotvec_deg, float))
        theta_f = np.linalg.norm(rv)
        k_v = rv / theta_f
        u = np.clip((t - handling_at) / handling_s, 0.0, 1.0)
        theta = theta_f * (1 - np.cos(np.pi * u)) / 2
        theta_dot = np.where((u > 0) & (u < 1), theta_f * np.pi / (2 * handling_s) * np.sin(np.pi * u), 0.0)
        R_vb_t = Rot.from_rotvec(theta[:, None] * k_v[None, :]).as_matrix() @ R_vb  # (n,3,3)
        R_bv_t = np.transpose(R_vb_t, (0, 2, 1))
        f_b = np.einsum("nij,nj->ni", R_bv_t, f_v)
        w_b = np.einsum("nij,nj->ni", R_bv_t, w_v + theta_dot[:, None] * k_v[None, :])
        R_after = R_vb_t[-1]

    # GNSS at 1 Hz: speed with noise; positions from the integrated heading
    tk = np.arange(0.0, duration, 1.0)
    idx = np.clip(np.round(tk * fs).astype(int), 0, n - 1)
    psi = np.cumsum(w_z) / fs  # heading, CCW from east
    east = np.cumsum(v * np.cos(psi)) / fs
    north = np.cumsum(v * np.sin(psi)) / fs
    lat0, lon0 = 52.0, 4.0
    gnss = pd.DataFrame(
        {
            "t": tk,
            "lat": lat0 + north[idx] / 111320.0,
            "lon": lon0 + east[idx] / (111320.0 * np.cos(np.radians(lat0))),
            "speed_mps": np.maximum(v[idx] + rng.normal(0.0, 0.05, len(tk)), 0.0),
        }
    )
    raw = RawTrip(
        trip_id=f"synth{seed}",
        source="phone",
        route_id="R",
        vehicle_id="V",
        driver_id="D",
        t=t,
        f_b=f_b,
        w_b=w_b if gyro else None,
        fs=fs,
        gnss=gnss,
        labels=None,
        meta={"synthetic": True},
    )
    return SynthTrip(raw, R_vb, R_after, v, a_lon, w_z, f_v)


def vehicle_frame(
    a_v: np.ndarray,
    v: np.ndarray,
    fs: float = 100.0,
    w_v: np.ndarray | None = None,
    a_lon: np.ndarray | None = None,
) -> VehicleFrameTrip:
    """Wrap synthetic vehicle-frame signals in a VehicleFrameTrip (single episode)."""
    n = len(v)
    t = np.arange(n) / fs
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) / fs)])
    return VehicleFrameTrip(
        trip_id="vft",
        t=t,
        fs=fs,
        a_v=a_v,
        w_v=w_v,
        v=v,
        a_lon_gnss=np.gradient(v, 1.0 / fs) if a_lon is None else a_lon,
        s=s,
        episode=np.zeros(n, dtype=int),
        R_vb=[np.eye(3)],
        yaw_deg=[0.0],
        yaw_ci_deg=[(-0.5, 0.5)],
        n_quasistatic=[n],
    )


def bump(fs: float, amp: float = 6.0, f0: float = 11.0, dur: float = 0.25) -> np.ndarray:
    """A short damped oscillation, the vertical signature of an impact."""
    tt = np.arange(0.0, dur, 1.0 / fs)
    return amp * np.exp(-tt / (dur / 4)) * np.sin(2 * np.pi * f0 * tt)


def make_hrnu(i: int, s0: float, s1: float, **kw):
    """An HRNU with plausible defaults for Layer 3 tests."""
    from roadscribe.layer1.iso2631 import comfort_class
    from roadscribe.schemas import HRNU

    a_v = kw.pop("a_v", 0.3)
    a_wz_dec = kw.pop("a_wz_dec", a_v * 0.9)
    base = dict(
        unit_id=f"U{i}",
        s0=s0,
        s1=s1,
        t0=s0 / 10.0,
        t1=s1 / 10.0,
        osm_way_ids=[100 + i],
        names=[f"Road {chr(64 + i)}"],
        osm_tags={"surface": "asphalt", "name": f"Road {chr(64 + i)}"},
        a_v=a_v,
        a_wz=a_v * 0.9,
        a_wz_dec=a_wz_dec,
        vdv_z=1.0,
        mtvv_z=a_v * 1.2,
        cf_z=4.0,
        kurt_z=3.5,
        shock=False,
        iso_felt=comfort_class(a_v),
        iso_road=comfort_class(a_wz_dec) if a_wz_dec is not None else None,
        r_wb=-1.0,
        r_hb=-2.0,
        v_mean=10.0,
        event_ids=[],
        boundary_left="trip_start" if i == 1 else "signal",
        map_signal_disagreement=None,
        conf={"map_match": 0.95, "orientation_yaw_ci_deg": 0.8, "channels": ["acc", "gyro", "gps", "map"]},
    )
    base.update(kw)
    return HRNU(**base)


def make_event(i: int, s: float, p: float = 0.9, label: str = "defect", mtvv: float = 1.5, maneuver: str = "cruise"):
    from roadscribe.schemas import Event

    return Event(f"E{i}", "impulse", label, s / 10, s / 10 + 0.2, s, 52.0, 4.0, 10.0, 5.0, mtvv, maneuver, p)
