r"""Impulsive events and maneuver gating (METHODOLOGY §3.5).

Signal chain:

* vertical impulse channel :math:`\tilde z=\mathrm{BP}_{[4,25]}(a^v_z)`
  (4th-order Butterworth, zero-phase);
* Teager-Kaiser energy :math:`\Psi[n]=\tilde z[n]^2-\tilde z[n-1]\tilde z[n+1]`;
* envelope :math:`E=h_{50\,ms}*\max(\Psi,0)`;
* adaptive threshold :math:`\theta=\mathrm{med}_W(E)+\kappa\cdot1.4826\,\mathrm{MAD}_W(E)`,
  :math:`W=10` s centred, evaluated exactly on a 0.1 s hop and interpolated;
* candidates: runs of :math:`E>\theta` merged across gaps < 0.3 s; runs longer
  than :math:`D_{max}=0.5\,s+3\,m/\bar v` are rough stretches for Layer 2.

Each candidate is labelled *artifact*, *defect* or *ambiguous* by the rule
of Eq. (gate). Braking and turning never veto a defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from ..config import EventConfig, ISOConfig
from ..schemas import Event, VehicleFrameTrip
from .iso2631 import running_rms, weight

__all__ = [
    "bandpass",
    "tkeo",
    "tkeo_envelope",
    "adaptive_threshold",
    "find_runs",
    "merge_runs",
    "max_event_duration",
    "ManeuverStates",
    "maneuver_states",
    "swerve_mask",
    "band_ratio",
    "label_event",
    "event_feature_vector",
    "DefectCalibrator",
    "Candidate",
    "EventDetectionResult",
    "detect_candidates",
    "detect_events",
]

MANEUVER_PRIORITY = ("stop_go", "accel", "swerve", "brake", "turn")
FEATURE_NAMES = ("log_peak", "duration_s", "log_rho", "jx_max", "is_accel", "is_stop_go", "is_brake", "is_turn")


# --------------------------------------------------------------------------- signal chain
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass; ``hi`` is clipped below Nyquist."""
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(order, [lo / (fs / 2.0), hi / (fs / 2.0)], "band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def tkeo(x: np.ndarray) -> np.ndarray:
    """Psi[n] = x[n]^2 - x[n-1] x[n+1]; the two end samples are 0.

    For x = A cos(Omega n + phi), Psi = A^2 sin^2(Omega) exactly (Kaiser 1990).
    """
    x = np.asarray(x, dtype=float)
    psi = np.zeros_like(x)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return psi


def tkeo_envelope(z: np.ndarray, fs: float, win_s: float = 0.05) -> np.ndarray:
    """E = h * max(Psi, 0) with a centred boxcar h of ``win_s`` seconds."""
    w = max(1, int(round(win_s * fs)))
    psi = np.maximum(tkeo(z), 0.0)
    return np.convolve(psi, np.ones(w) / w, mode="same")


def adaptive_threshold(
    E: np.ndarray, fs: float, window_s: float = 10.0, kappa: float = 15.0, hop_s: float = 0.1, chunk: int = 512
) -> np.ndarray:
    """theta[n] = med_W(E) + kappa * 1.4826 * MAD_W(E) over a centred window.

    Median and MAD are exact for windows centred on a ``hop_s`` grid and
    linearly interpolated between; the threshold is held constant within
    half a window of either end.
    """
    E = np.asarray(E, dtype=float)
    n = len(E)
    W = int(round(window_s * fs)) | 1  # odd, so it has a centre
    if n <= W:
        med = float(np.median(E))
        mad = float(np.median(np.abs(E - med)))
        return np.full(n, med + kappa * 1.4826 * mad)
    half = W // 2
    hop = max(1, int(round(hop_s * fs)))
    centres = np.arange(half, n - half, hop)
    if centres[-1] != n - half - 1:
        centres = np.append(centres, n - half - 1)
    win = sliding_window_view(E, W)
    med = np.empty(len(centres))
    mad = np.empty(len(centres))
    for k in range(0, len(centres), chunk):
        block = win[centres[k : k + chunk] - half]
        m = np.median(block, axis=1)
        med[k : k + chunk] = m
        mad[k : k + chunk] = np.median(np.abs(block - m[:, None]), axis=1)
    return np.interp(np.arange(n), centres, med + kappa * 1.4826 * mad)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as half-open (i0, i1) index pairs."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def merge_runs(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by fewer than ``max_gap`` samples."""
    out: list[list[int]] = []
    for a, b in runs:
        if out and a - out[-1][1] < max_gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def max_event_duration(v_mean: float, cfg: EventConfig | None = None) -> float:
    """D_max = 0.5 s + 3 m / v (speed floored at ``dmax_v_floor``)."""
    cfg = cfg or EventConfig()
    return cfg.dmax_base_s + cfg.dmax_dist_m / max(float(v_mean), cfg.dmax_v_floor)


# --------------------------------------------------------------------------- maneuvers
@dataclass(frozen=True)
class ManeuverStates:
    """Per-sample maneuver indicators (§3.5, Maneuver context)."""

    brake: np.ndarray
    accel: np.ndarray
    turn: np.ndarray
    swerve: np.ndarray
    stop_go: np.ndarray
    jx: np.ndarray  # d/dt LP_5Hz(a^v_x), m/s^3
    a_lat: np.ndarray  # v * omega^v_z, m/s^2 (0 without a gyroscope)

    def at(self, i0: int, i1: int) -> str:
        """Single maneuver label for samples [i0, i1): gate-relevant states first."""
        for name in MANEUVER_PRIORITY:
            if getattr(self, name)[i0:i1].any():
                return name
        return "cruise"


def _lp(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, fc / (fs / 2.0), "low", output="sos")
    return signal.sosfiltfilt(sos, x)


def swerve_mask(wz: np.ndarray, v: np.ndarray, fs: float, cfg: EventConfig | None = None) -> np.ndarray:
    """Yaw-rate sign reversal within 1-4 s, net heading change < 5 deg, peak |v w_z| > A_s.

    ``wz`` is the low-passed vehicle yaw rate. Two consecutive yaw-rate lobes
    of opposite sign form a candidate; each lobe must reach A_s/2 so that
    noise crossings around zero yaw rate do not count as reversals.
    """
    cfg = cfg or EventConfig()
    n = len(wz)
    out = np.zeros(n, dtype=bool)
    a_lat = np.abs(v * wz)
    zc = np.flatnonzero(np.diff(np.signbit(wz))) + 1
    bounds = np.concatenate([[0], zc, [n]])
    heading = np.concatenate([[0.0], np.cumsum(wz) / fs])  # heading[k] = integral up to sample k
    for k in range(len(bounds) - 2):
        z0, z1, z2 = bounds[k], bounds[k + 1], bounds[k + 2]
        dur = (z2 - z0) / fs
        if not (cfg.swerve_min_s <= dur <= cfg.swerve_max_s):
            continue
        p1, p2 = a_lat[z0:z1].max(), a_lat[z1:z2].max()
        if max(p1, p2) <= cfg.swerve_a or min(p1, p2) <= 0.5 * cfg.swerve_a:
            continue
        if abs(np.degrees(heading[z2] - heading[z0])) >= cfg.swerve_net_heading_deg:
            continue
        out[z0:z2] = True
    return out


def maneuver_states(
    v: np.ndarray,
    a_lon: np.ndarray,
    a_x: np.ndarray,
    w_z: np.ndarray | None,
    fs: float,
    cfg: EventConfig | None = None,
) -> ManeuverStates:
    """Longitudinal states from GNSS (a_lon), lateral ones from the gyroscope.

    Using GNSS for the longitudinal states avoids the road-grade tilt bias of
    the IMU x channel (§3.1, Step 1).
    """
    cfg = cfg or EventConfig()
    n = len(v)
    brake = a_lon < -cfg.brake_a
    accel = a_lon > cfg.accel_a
    jx = np.gradient(_lp(a_x, fs, min(cfg.jerk_lp_hz, 0.45 * fs)), 1.0 / fs)
    stop_go = (v < cfg.stopgo_v) & (np.abs(jx) > cfg.stopgo_jerk)
    if w_z is not None:
        wz = _lp(w_z, fs, cfg.yaw_lp_hz)
        a_lat = v * wz
        turn = np.zeros(n, dtype=bool)
        min_len = int(round(cfg.turn_min_s * fs))
        for a, b in find_runs(np.abs(a_lat) > cfg.turn_a):
            if b - a >= min_len:
                turn[a:b] = True
        swerve = swerve_mask(wz, v, fs, cfg)
    else:
        a_lat = np.zeros(n)
        turn = np.zeros(n, dtype=bool)
        swerve = np.zeros(n, dtype=bool)
    return ManeuverStates(brake=brake, accel=accel, turn=turn, swerve=swerve, stop_go=stop_go, jx=jx, a_lat=a_lat)


# --------------------------------------------------------------------------- features and rule
def band_ratio(az: np.ndarray, fs: float, i0: int, i1: int, cfg: EventConfig | None = None) -> float:
    """rho_e = E_[8,25](e) / E_[0.5,4](e) of a^v_z around the event.

    The analysis window is centred on the event and at least ``rho_window_s``
    long, because a 0.1-0.5 s event window cannot resolve the 0.5-4 Hz band.
    """
    cfg = cfg or EventConfig()
    n = len(az)
    L = max(i1 - i0, int(round(cfg.rho_window_s * fs)))
    c = (i0 + i1) // 2
    a = max(0, c - L // 2)
    b = min(n, a + L)
    a = max(0, b - L)
    seg = az[a:b]
    f, P = signal.periodogram(seg, fs=fs, window="hann", detrend="constant")
    hi_band = (cfg.rho_hi_band[0], min(cfg.rho_hi_band[1], fs / 2.0))
    e_hi = P[(f >= hi_band[0]) & (f <= hi_band[1])].sum()
    e_lo = P[(f >= cfg.rho_lo_band[0]) & (f <= cfg.rho_lo_band[1])].sum()
    return float(e_hi / e_lo) if e_lo > 0 else float("inf")


def label_event(
    maneuver: str, jx_max: float, rho: float, duration: float, d_max: float, cfg: EventConfig | None = None
) -> str:
    """Decision rule l(e) of §3.5.

    artifact  if m_e in {stop_go, accel} and max|j_x| > J_x and rho_e < rho_0;
    defect    if not artifact and rho_e >= rho_0 and d_e <= D_max;
    ambiguous otherwise.
    """
    cfg = cfg or EventConfig()
    if maneuver in ("stop_go", "accel") and jx_max > cfg.jx_threshold and rho < cfg.rho0:
        return "artifact"
    if rho >= cfg.rho0 and duration <= d_max:
        return "defect"
    return "ambiguous"


def event_feature_vector(peak: float, duration: float, rho: float, jx_max: float, maneuver: str) -> np.ndarray:
    """Features for the p_defect calibrator, in the order of ``FEATURE_NAMES``."""
    rho_c = rho if np.isfinite(rho) and rho > 0 else 1e6
    return np.array(
        [
            np.log(max(peak, 1e-12)),
            duration,
            np.log(rho_c),
            jx_max,
            float(maneuver == "accel"),
            float(maneuver == "stop_go"),
            float(maneuver == "brake"),
            float(maneuver == "turn"),
        ]
    )


class Calibrator(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class DefectCalibrator:
    """L2-regularised logistic regression (Newton-IRLS) mapping features to p_defect.

    Train it leave-one-route-out: positives are bump traversals, negatives are
    stop-go and acceleration episodes > 50 m from any labelled bump (§3.5).
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DefectCalibrator":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        A = self._design(X)
        beta = np.zeros(A.shape[1])
        reg = np.full(A.shape[1], self.l2)
        reg[0] = 0.0  # intercept is not penalised
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-A @ beta))
            g = A.T @ (p - y) + reg * beta
            Hm = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-12 * np.eye(len(beta))
            step = np.linalg.solve(Hm, g)
            beta -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("calibrator is not fitted")
        return 1.0 / (1.0 + np.exp(-self._design(np.atleast_2d(X)) @ self.coef_))


# --------------------------------------------------------------------------- detection
@dataclass(frozen=True)
class Candidate:
    i0: int
    i1: int  # exclusive
    i_peak: int
    peak: float
    duration: float
    v_mean: float
    d_max: float


@dataclass
class EventDetectionResult:
    events: list[Event]
    candidates: list[Candidate]
    rough_runs: list[tuple[float, float]]  # (t0, t1) of runs longer than D_max -> Layer 2
    envelope: np.ndarray
    threshold: np.ndarray
    states: ManeuverStates
    diagnostics: dict = field(default_factory=dict)


def detect_candidates(
    az: np.ndarray, v: np.ndarray, fs: float, cfg: EventConfig | None = None
) -> tuple[list[Candidate], list[tuple[int, int]], np.ndarray, np.ndarray]:
    """TKEO candidates (kept) and rough runs (rejected as longer than D_max)."""
    cfg = cfg or EventConfig()
    z = bandpass(az, fs, cfg.bp_lo_hz, cfg.bp_hi_hz, cfg.bp_order)
    E = tkeo_envelope(z, fs, cfg.envelope_s)
    theta = adaptive_threshold(E, fs, cfg.threshold_window_s, cfg.kappa, cfg.threshold_hop_s)
    runs = merge_runs(find_runs(E > theta), int(round(cfg.merge_gap_s * fs)))
    cands: list[Candidate] = []
    rough: list[tuple[int, int]] = []
    for a, b in runs:
        dur = (b - a) / fs
        vm = float(np.mean(v[a:b]))
        dmax = max_event_duration(vm, cfg)
        if dur > dmax:
            rough.append((a, b))
            continue
        ip = a + int(np.argmax(E[a:b]))
        cands.append(Candidate(a, b, ip, float(E[ip]), dur, vm, dmax))
    return cands, rough, E, theta


def _interp_or_nan(t: float, tg: np.ndarray | None, y: np.ndarray | None) -> float:
    if tg is None or y is None or len(tg) == 0:
        return float("nan")
    return float(np.interp(t, tg, y))


def detect_events(
    vft: VehicleFrameTrip,
    cfg: EventConfig | None = None,
    iso_cfg: ISOConfig | None = None,
    gnss: Optional[pd.DataFrame] = None,
    calibrator: Optional[Calibrator] = None,
    aw_z: Optional[np.ndarray] = None,
) -> EventDetectionResult:
    """Detect, label and score impulsive events and avoidance manoeuvres.

    ``gnss`` (columns t, lat, lon) supplies event coordinates; without it they
    are NaN. ``aw_z`` is the trip's W_k-weighted vertical acceleration (computed
    here if omitted) and gives each event's MTVV. Without a ``calibrator``,
    p_defect falls back to the per-label priors in :class:`EventConfig`.
    """
    cfg = cfg or EventConfig()
    iso_cfg = iso_cfg or ISOConfig()
    fs, t = vft.fs, vft.t
    az = vft.a_v[:, 2]
    if aw_z is None:
        aw_z = weight(az, fs, "Wk", iso_cfg.method, iso_cfg.pad_s)
    rr = running_rms(aw_z, fs, iso_cfg.mtvv_tau_s)
    w_mtvv = max(1, int(round(iso_cfg.mtvv_tau_s * fs)))

    def event_mtvv(i0: int, i1: int) -> float:
        lo = int(np.clip(i0 - w_mtvv + 1, 0, len(rr) - 1))
        hi = int(np.clip(i1 - 1, lo, len(rr) - 1))
        return float(rr[lo : hi + 1].max())

    states = maneuver_states(
        vft.v, vft.a_lon_gnss, vft.a_v[:, 0], vft.w_v[:, 2] if vft.w_v is not None else None, fs, cfg
    )
    cands, rough, E, theta = detect_candidates(az, vft.v, fs, cfg)

    tg = lat = lon = None
    if gnss is not None and len(gnss):
        g = gnss.sort_values("t").drop_duplicates("t")
        tg, lat, lon = g["t"].to_numpy(float), g["lat"].to_numpy(float), g["lon"].to_numpy(float)

    prior = {"defect": cfg.prior_p_defect, "ambiguous": cfg.prior_p_ambiguous, "artifact": cfg.prior_p_artifact}
    raw_events: list[dict] = []
    for c in cands:
        m = states.at(c.i0, c.i1)
        jx_max = float(np.max(np.abs(states.jx[c.i0 : c.i1])))
        rho = band_ratio(az, fs, c.i0, c.i1, cfg)
        lab = label_event(m, jx_max, rho, c.duration, c.d_max, cfg)
        if calibrator is not None:
            p = float(calibrator.predict_proba(event_feature_vector(c.peak, c.duration, rho, jx_max, m)[None, :])[0])
        else:
            p = prior[lab]
        tp = float(t[c.i_peak])
        raw_events.append(
            dict(
                kind="impulse",
                label=lab,
                t0=float(t[c.i0]),
                t1=float(t[c.i1 - 1]),
                s=float(vft.s[c.i_peak]),
                lat=_interp_or_nan(tp, tg, lat),
                lon=_interp_or_nan(tp, tg, lon),
                tkeo_peak=c.peak,
                band_ratio=rho,
                mtvv_z=event_mtvv(c.i0, c.i1),
                maneuver=m,
                p_defect=float(np.clip(p, 0.0, 1.0)),
            )
        )

    # avoidance: a swerve with no vertical candidate (kept or rough) within +-2 s
    vert_t = [(t[c.i0], t[c.i1 - 1]) for c in cands] + [(t[a], t[b - 1]) for a, b in rough]
    for a, b in find_runs(states.swerve):
        t0, t1 = float(t[a]), float(t[b - 1])
        w = cfg.avoidance_window_s
        if any(v1 >= t0 - w and v0 <= t1 + w for v0, v1 in vert_t):
            continue
        im = (a + b) // 2
        raw_events.append(
            dict(
                kind="avoidance",
                label="ambiguous",
                t0=t0,
                t1=t1,
                s=float(vft.s[im]),
                lat=_interp_or_nan(float(t[im]), tg, lat),
                lon=_interp_or_nan(float(t[im]), tg, lon),
                tkeo_peak=float(E[a:b].max()),
                band_ratio=float("nan"),
                mtvv_z=event_mtvv(a, b),
                maneuver="swerve",
                p_defect=cfg.avoidance_p,
            )
        )

    raw_events.sort(key=lambda e: (e["t0"], e["kind"]))
    events = [Event(event_id=f"E{k + 1}", **e) for k, e in enumerate(raw_events)]
    return EventDetectionResult(
        events=events,
        candidates=cands,
        rough_runs=[(float(t[a]), float(t[b - 1])) for a, b in rough],
        envelope=E,
        threshold=theta,
        states=states,
        diagnostics={"n_runs": len(cands) + len(rough), "kappa": cfg.kappa},
    )
