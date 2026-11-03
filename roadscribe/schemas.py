"""Data contracts between the RoadScribe stages (METHODOLOGY §8.3).

SI units throughout: times in s, distances in m, accelerations in m/s^2,
angular rates in rad/s.

Deviations from the §8.3 listing, all additive:

* ``RawTrip`` is ``frozen=True, eq=False``: with ``eq=True`` a frozen
  dataclass compares and hashes its numpy fields, which raises.
* ``VehicleFrameTrip`` also carries ``t`` and ``fs``. The listing omits them,
  but every consumer of the vehicle frame (and ``vehicle_frame.parquet`` in
  §8.4) needs the clock.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

__all__ = [
    "RawTrip",
    "VehicleFrameTrip",
    "Event",
    "HRNU",
    "Fact",
    "Capsule",
    "Claim",
    "Narrative",
    "EventKind",
    "EventLabel",
    "Maneuver",
    "BoundaryType",
    "FactKind",
    "SelectionLevel",
    "to_jsonable",
]

EventKind = Literal["impulse", "avoidance"]
EventLabel = Literal["defect", "artifact", "ambiguous"]
Maneuver = Literal["cruise", "brake", "accel", "turn", "swerve", "stop_go"]
BoundaryType = Literal["trip_start", "map+signal", "signal", "map"]
FactKind = Literal["SEV", "NUM", "EVT", "ANCHOR", "CMP", "MISSING", "META"]
SelectionLevel = Literal["OMIT", "BRIEF", "FULL"]
Hedge = Literal["assertive", "likely", "possibly"]

MANEUVERS: tuple[str, ...] = ("cruise", "brake", "accel", "turn", "swerve", "stop_go")
EVENT_LABELS: tuple[str, ...] = ("defect", "artifact", "ambiguous")
FACT_KINDS: tuple[str, ...] = ("SEV", "NUM", "EVT", "ANCHOR", "CMP", "MISSING", "META")
BOUNDARY_TYPES: tuple[str, ...] = ("trip_start", "map+signal", "signal", "map")
SELECTION_LEVELS: tuple[str, ...] = ("OMIT", "BRIEF", "FULL")


def _check_choice(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {allowed}")


def _check_matrix(name: str, x: Any, n: int, cols: int) -> None:
    if not isinstance(x, np.ndarray) or x.shape != (n, cols):
        shape = getattr(x, "shape", None)
        raise ValueError(f"{name} must be an ndarray of shape ({n}, {cols}), got {shape}")


def _check_vector(name: str, x: Any, n: int) -> None:
    if not isinstance(x, np.ndarray) or x.shape != (n,):
        raise ValueError(f"{name} must be an ndarray of shape ({n},), got {getattr(x, 'shape', None)}")


@dataclass(frozen=True, eq=False)
class RawTrip:
    trip_id: str
    source: Literal["pvs", "phone"]
    route_id: str
    vehicle_id: str
    driver_id: str
    t: np.ndarray  # (N,) IMU clock, monotonic
    f_b: np.ndarray  # (N,3) specific force incl. gravity, body/sensor frame
    w_b: Optional[np.ndarray]  # (N,3) rad/s; None if no gyroscope
    fs: float
    gnss: pd.DataFrame  # t, lat, lon, speed_mps [, course_deg, hacc_m] at native rate
    labels: Optional[pd.DataFrame]  # PVS: surface, bump, quality per IMU sample
    meta: dict = field(default_factory=dict)  # logger side, mount, clock offsets, provenance, config hash

    GNSS_REQUIRED = ("t", "lat", "lon", "speed_mps")

    def __post_init__(self) -> None:
        _check_choice("source", self.source, ("pvs", "phone"))
        t = self.t
        if not isinstance(t, np.ndarray) or t.ndim != 1 or len(t) < 2:
            raise ValueError("t must be a 1-D ndarray with at least two samples")
        if not np.all(np.diff(t) > 0):
            raise ValueError("t must be strictly increasing (monotonic IMU clock)")
        _check_matrix("f_b", self.f_b, len(t), 3)
        if self.w_b is not None:
            _check_matrix("w_b", self.w_b, len(t), 3)
        if not (self.fs > 0 and math.isfinite(self.fs)):
            raise ValueError("fs must be positive and finite")
        missing = [c for c in self.GNSS_REQUIRED if c not in self.gnss.columns]
        if missing:
            raise ValueError(f"gnss is missing columns {missing}")
        if self.labels is not None and len(self.labels) != len(t):
            raise ValueError("labels must have one row per IMU sample")

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def has_gyro(self) -> bool:
        return self.w_b is not None


@dataclass(eq=False)
class VehicleFrameTrip:
    trip_id: str
    t: np.ndarray  # (N,) IMU clock (added; see module docstring)
    fs: float  # (added)
    a_v: np.ndarray  # (N,3) gravity-free, x fwd, y left, z up
    w_v: Optional[np.ndarray]
    v: np.ndarray  # (N,) speed, PCHIP from GNSS
    a_lon_gnss: np.ndarray
    s: np.ndarray  # along-track distance (arc length after map matching)
    episode: np.ndarray  # (N,) mount-episode index; -1 = discarded around a handling split
    R_vb: list[np.ndarray]  # one 3x3 per episode
    yaw_deg: list[float]
    yaw_ci_deg: list[tuple[float, float]]
    n_quasistatic: list[int]

    def __post_init__(self) -> None:
        n = len(self.t)
        _check_matrix("a_v", self.a_v, n, 3)
        if self.w_v is not None:
            _check_matrix("w_v", self.w_v, n, 3)
        for name in ("v", "a_lon_gnss", "s", "episode"):
            _check_vector(name, getattr(self, name), n)
        k = len(self.R_vb)
        if not (len(self.yaw_deg) == len(self.yaw_ci_deg) == len(self.n_quasistatic) == k):
            raise ValueError("per-episode lists must all have one entry per episode")
        if np.any(np.diff(self.s) < -1e-9):
            raise ValueError("s must be non-decreasing")

    @property
    def n(self) -> int:
        return len(self.t)


@dataclass
class Event:
    event_id: str
    kind: EventKind
    label: EventLabel
    t0: float
    t1: float
    s: float
    lat: float
    lon: float
    tkeo_peak: float
    band_ratio: float
    mtvv_z: float
    maneuver: Maneuver
    p_defect: float

    def __post_init__(self) -> None:
        _check_choice("kind", self.kind, ("impulse", "avoidance"))
        _check_choice("label", self.label, EVENT_LABELS)
        _check_choice("maneuver", self.maneuver, MANEUVERS)
        if not (0.0 <= self.p_defect <= 1.0):
            raise ValueError("p_defect must be in [0, 1]")
        if self.t1 < self.t0:
            raise ValueError("t1 must be >= t0")

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class HRNU:
    unit_id: str
    s0: float
    s1: float
    t0: float
    t1: float
    osm_way_ids: list[int]
    names: list[str]
    osm_tags: dict
    a_v: float
    a_wz: float
    a_wz_dec: Optional[float]  # None if speed_low
    vdv_z: float
    mtvv_z: float
    cf_z: float
    kurt_z: float
    shock: bool
    iso_felt: int
    iso_road: Optional[int]
    r_wb: float
    r_hb: float
    v_mean: float
    event_ids: list[str]
    boundary_left: BoundaryType
    map_signal_disagreement: Optional[float]
    conf: dict  # map_match, orientation, channels present

    def __post_init__(self) -> None:
        _check_choice("boundary_left", self.boundary_left, BOUNDARY_TYPES)
        if self.s1 < self.s0:
            raise ValueError("s1 must be >= s0")
        if not 0 <= self.iso_felt <= 5:
            raise ValueError("iso_felt must be a comfort class in 0..5")
        if self.iso_road is not None and not 0 <= self.iso_road <= 5:
            raise ValueError("iso_road must be a comfort class in 0..5")

    @property
    def length_m(self) -> float:
        return self.s1 - self.s0

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


@dataclass(frozen=True)
class Fact:
    fact_id: str
    kind: FactKind
    unit_id: Optional[str]
    key: str
    value: Any
    tol: Optional[float] = None
    conf: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        _check_choice("kind", self.kind, FACT_KINDS)
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError("conf must be in [0, 1]")


@dataclass
class Capsule:
    capsule_id: str
    trip_id: str
    persona: str
    budget: int
    text: str  # exactly what the SLM sees
    n_tokens: int
    tokenizer: str
    selection: dict  # unit_id -> OMIT/BRIEF/FULL
    ledger: list[Fact]

    def __post_init__(self) -> None:
        for uid, level in self.selection.items():
            _check_choice(f"selection[{uid}]", level, SELECTION_LEVELS)


@dataclass
class Claim:
    span: tuple[int, int]
    kind: str
    unit_id: Optional[str]
    key: str
    value: Any
    hedge: Hedge
    supp_ledger: Optional[bool] = None
    supp_gt: Optional[bool] = None


@dataclass
class Narrative:
    capsule_id: str
    condition: str
    model: str
    seed: int
    text: str
    plan: Optional[dict]
    claims: list[Claim]
    regenerations: int
    fallback_sentences: int
    latency_ms: dict
    energy_j: Optional[float]


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / numpy scalars and arrays into JSON-serialisable objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    return obj
