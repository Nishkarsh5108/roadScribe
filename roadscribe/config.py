"""Frozen configuration for every RoadScribe threshold (METHODOLOGY §8.2).

Every stage takes its parameters from one of the frozen dataclasses below.
An experiment is described by a single YAML file that overrides any subset of
the defaults; :meth:`RoadScribeConfig.config_hash` is saved with every output
so that any number in the paper can be traced back to the configuration that
produced it.

Defaults marked *[methodology]* are fixed by the methodology document.
Defaults marked *[placeholder]* must be tuned on training routes only
(leave-one-route-out) before they are used for reported results.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

__all__ = [
    "OrientationConfig",
    "ISOConfig",
    "SpeedConfig",
    "SpectralConfig",
    "EventConfig",
    "DistanceConfig",
    "MIPELTConfig",
    "ArbitrationConfig",
    "LedgerConfig",
    "KSTConfig",
    "RoadScribeConfig",
    "config_hash",
]

#: Standard gravity (m/s^2).
G0 = 9.80665


@dataclass(frozen=True)
class OrientationConfig:
    """Mount episodes, gated gravity, Rodrigues tilt and Procrustes yaw (§3.1)."""

    g: float = G0
    lp_hz: float = 0.5  # [methodology] low-pass for f-bar and omega-bar
    lp_order: int = 2
    eps_g: float = 0.3  # [methodology] | ||f|| - g | gate, m/s^2
    eps_omega_deg: float = 3.0  # [methodology] gyro gate, deg/s
    eps_a: float = 0.3  # [methodology] |a_lon^G| gate, m/s^2
    v_min: float = 3.0  # [methodology] speed gate for the quasi-static set, m/s
    v_min_yaw: float = 3.0  # [methodology] fixes used for the yaw fit, m/s
    min_quasistatic: int = 200  # samples required to trust an episode's gravity estimate
    min_fixes: int = 10  # GNSS fixes required for the yaw fit
    # Mount episodes (Step 0)
    handling_omega_deg: float = 30.0  # [methodology] Omega_h, deg/s
    handling_theta_deg: float = 10.0  # [methodology] theta_h, deg
    handling_guard_s: float = 5.0  # [methodology] discard +-5 s around a split
    handling_side_window_s: float = 120.0  # data on each side used to compare gravity
    # Geometric median (Weiszfeld / Vardi-Zhang)
    geomedian_tol: float = 1e-12
    geomedian_max_iter: int = 1000
    # Yaw confidence: moving-block bootstrap over fixes
    bootstrap_n: int = 200
    bootstrap_block_s: float = 10.0
    bootstrap_level: float = 0.95
    seed: int = 0


@dataclass(frozen=True)
class ISOConfig:
    """ISO 2631-1 weighting and evaluation rule (§3.3)."""

    method: Literal["fft", "iir"] = "fft"  # [methodology] offline FFT weighting is exact
    pad_s: float = 10.0  # [methodology] zero padding for FFT weighting
    mtvv_tau_s: float = 1.0  # [methodology] running-RMS integration time
    cf_threshold: float = 9.0  # [methodology]
    mtvv_ratio_threshold: float = 1.5  # [methodology]
    vdv_ratio_threshold: float = 1.75  # [methodology]
    k_x: float = 1.0  # comfort multiplying factors
    k_y: float = 1.0
    k_z: float = 1.0


@dataclass(frozen=True)
class SpeedConfig:
    """Speed decoupling (§3.2)."""

    gamma: float = 0.35  # [placeholder] PVS dashboard estimates were 0.32-0.36; fit per corpus
    v_ref: float = 10.0  # [methodology] m/s
    # Calibration on shared roads (identifiability study, 2026-11-12): log gain of this vehicle
    # relative to the fleet mean, estimated where several vehicles drove the same places.
    # 0.0 means uncalibrated (the original behaviour).
    vehicle_gain_log: float = 0.0
    v_min: float = 5.0  # [methodology] below this report raw a_w with speed_low


@dataclass(frozen=True)
class SpectralConfig:
    """Welch PSD and modal band ratios (§3.4)."""

    nperseg: int = 512  # [methodology] 5.12 s at 100 Hz
    nperseg_short: int = 256  # [methodology] for units shorter than short_unit_s
    short_unit_s: float = 15.0
    body_band: tuple[float, float] = (0.8, 2.5)  # [methodology]
    wheel_search: tuple[float, float] = (6.0, 20.0)  # [methodology]
    wheel_default: tuple[float, float] = (8.0, 16.0)  # [methodology]
    wheel_rel: tuple[float, float] = (0.75, 1.33)  # [methodology]
    high_band: tuple[float, float] = (16.0, 40.0)  # [methodology]


@dataclass(frozen=True)
class EventConfig:
    """TKEO impulse detection, maneuver context and decision rule (§3.5)."""

    bp_lo_hz: float = 4.0  # [methodology]
    bp_hi_hz: float = 25.0  # [methodology]
    bp_order: int = 4  # [methodology]
    envelope_s: float = 0.05  # [methodology] h_50ms
    threshold_window_s: float = 10.0  # [methodology] centred median/MAD window
    threshold_hop_s: float = 0.1  # exact median/MAD on this hop, interpolated between
    # [placeholder] choose on training routes. On stationary Gaussian background
    # kappa = 8 / 12 / 15 gives about 200 / 15-20 / 1-4 false runs per hour.
    kappa: float = 15.0
    merge_gap_s: float = 0.3  # [methodology]
    dmax_base_s: float = 0.5  # [methodology] D_max = 0.5 s + 3 m / v
    dmax_dist_m: float = 3.0  # [methodology]
    dmax_v_floor: float = 1.0  # avoids D_max -> inf when stationary, m/s
    rho_hi_band: tuple[float, float] = (8.0, 25.0)  # [methodology]
    rho_lo_band: tuple[float, float] = (0.5, 4.0)  # [methodology]
    rho_window_s: float = 2.0  # minimum analysis window for rho_e (0.5 Hz resolution)
    rho0: float = 1.0  # [placeholder]
    jx_threshold: float = 3.0  # [placeholder] J_x, m/s^3
    jerk_lp_hz: float = 5.0  # [methodology]
    brake_a: float = 1.0  # [placeholder] A_b, m/s^2 (PVS maneuver analysis used 1.0)
    accel_a: float = 1.0  # [placeholder] A_a, m/s^2
    turn_a: float = 1.5  # [placeholder] A_t, m/s^2
    turn_min_s: float = 1.0  # [methodology]
    swerve_a: float = 1.5  # [placeholder] A_s, m/s^2
    swerve_min_s: float = 1.0  # [methodology]
    swerve_max_s: float = 4.0  # [methodology]
    swerve_net_heading_deg: float = 5.0  # [methodology]
    yaw_lp_hz: float = 1.0
    stopgo_v: float = 2.0  # [methodology]
    stopgo_jerk: float = 2.0  # [placeholder] J_sg, m/s^3
    avoidance_window_s: float = 2.0  # [methodology]
    # Used only until a leave-one-route-out calibrator is fitted.
    prior_p_defect: float = 0.7  # [placeholder]
    prior_p_ambiguous: float = 0.5  # [placeholder]
    prior_p_artifact: float = 0.15  # [placeholder]
    avoidance_p: float = 0.5  # [placeholder] low confidence -> "possibly"


@dataclass(frozen=True)
class DistanceConfig:
    """Distance-domain feature grid (§4.2)."""

    ds_m: float = 5.0  # [methodology]
    window_m: float = 25.0  # [methodology]
    eps: float = 1e-6  # floor inside log()
    # None follows §4.2 (log a_w,y of the full W_d-weighted lateral channel). Below
    # ~1 Hz that channel carries cornering (§3.3), which then produces change
    # points at every turn; e.g. 1.0 removes it for an ablation.
    lateral_highpass_hz: Optional[float] = None


@dataclass(frozen=True)
class MIPELTConfig:
    """Map-informed PELT (§4.3)."""

    # [placeholder] beta_0 = c * d * ln(n). Grid features share 25 m windows on a
    # 5 m grid (about 5 correlated samples), so c is ~5x the iid BIC-like value.
    c: float = 10.0
    lam: float = 0.5  # [placeholder] lambda in [0, 1)
    sigma_b_m: float = 15.0  # [methodology]
    w_surface: float = 1.0  # [methodology]
    w_highway: float = 0.6  # [methodology]
    w_junction: float = 0.3  # [methodology]
    w_traffic_calming: float = 0.3  # [methodology]
    min_size: int = 1  # minimum segment length in grid samples
    pruning: Literal["corrected", "textbook", "none"] = "corrected"


@dataclass(frozen=True)
class ArbitrationConfig:
    """Snapping, boundary typing, merging (§4.4)."""

    snap_m: float = 25.0  # [methodology] delta_snap
    l_min_m: float = 50.0  # [methodology]
    hotelling_alpha: float = 0.01  # [methodology]
    hotelling_window_m: float = 100.0  # data on each side of a map-only candidate
    hotelling_thin: int = 5  # use every 5th grid point (features use 25 m windows)
    ridge: float = 1e-6
    dedupe_m: float = 1.0  # boundaries closer than this are the same boundary


@dataclass(frozen=True)
class LedgerConfig:
    """Claim-ledger tolerances (§5.1, §6.2)."""

    eps_rel: float = 0.10
    eps_abs_acc: float = 0.05  # m/s^2 (and m/s^1.75 for VDV)
    eps_abs_len: float = 10.0  # m
    eps_abs_speed: float = 0.5  # m/s
    eps_abs_time: float = 5.0  # s
    eps_abs_score: float = 0.05  # dimensionless scores
    event_tol_m: float = 25.0  # [methodology] delta_s for event matching (§6.4)
    cmp_margin: float = 0.2  # relative a_v margin before "rougher"/"smoother"


@dataclass(frozen=True)
class KSTConfig:
    """Kinetic-Spatial Tokenizer (§5.2, Algorithm 1)."""

    budget: int = 256  # B; ablate over {64, 128, 256, 512}
    eta_brief: float = 0.5  # [placeholder]
    eta_full: float = 1.0
    eta_star: float = 2.0  # [placeholder] must-mention threshold on p_e * sev(e)
    w1: float = 1.0  # [methodology] all weights 1 after z-scoring
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0
    salience_floor: float = 1.0  # shift so the least salient unit keeps a positive gain
    anchor_rmax_m: float = 200.0
    max_refits: int = 8


def _section(cls):
    return field(default_factory=cls)


@dataclass(frozen=True)
class RoadScribeConfig:
    """All configuration for one experiment."""

    orientation: OrientationConfig = _section(OrientationConfig)
    iso: ISOConfig = _section(ISOConfig)
    speed: SpeedConfig = _section(SpeedConfig)
    spectral: SpectralConfig = _section(SpectralConfig)
    events: EventConfig = _section(EventConfig)
    distance: DistanceConfig = _section(DistanceConfig)
    mipelt: MIPELTConfig = _section(MIPELTConfig)
    arbitration: ArbitrationConfig = _section(ArbitrationConfig)
    ledger: LedgerConfig = _section(LedgerConfig)
    kst: KSTConfig = _section(KSTConfig)

    # ------------------------------------------------------------ (de)serialization
    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RoadScribeConfig":
        """Build a config from a (possibly partial) nested mapping.

        Unknown keys raise ``KeyError`` so that a typo in an experiment YAML
        cannot silently fall back to a default.
        """
        return _from_dict(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoadScribeConfig":
        import yaml  # optional dependency

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True)

    def replace(self, **sections: Any) -> "RoadScribeConfig":
        """Return a copy with whole sections, or ``{"section": {"key": value}}`` overrides."""
        d = self.to_dict()
        for name, value in sections.items():
            if name not in d:
                raise KeyError(f"unknown config section {name!r}")
            if dataclasses.is_dataclass(value):
                d[name] = _to_plain(dataclasses.asdict(value))
            else:
                d[name] = {**d[name], **dict(value)}
        return RoadScribeConfig.from_dict(d)

    def config_hash(self) -> str:
        return config_hash(self)


def config_hash(cfg: Any, length: int = 16) -> str:
    """Stable short SHA-256 of a config dataclass (or any JSON-able mapping)."""
    payload = _to_plain(dataclasses.asdict(cfg)) if dataclasses.is_dataclass(cfg) else _to_plain(cfg)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _from_dict(tp, value)
        else:
            kwargs[name] = _coerce(tp, value)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise ValueError(f"{value!r} is not one of {typing.get_args(tp)}")
        return value
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
