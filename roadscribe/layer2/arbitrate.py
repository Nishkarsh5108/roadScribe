r"""Map-signal arbitration and HRNU construction (METHODOLOGY §4.4).

* **(A1) Snap.** A change point within :math:`\delta_{snap}=25` m of an OSM
  transition moves to it: the map wins on location, the signal on existence.
* **(A2) Type.** ``map+signal`` (snapped), ``signal`` (no OSM transition
  nearby) or ``map`` (an OSM surface change without a change point, kept only
  if a two-sample Hotelling :math:`T^2` test on :math:`\mathbf y` across it
  rejects at :math:`\alpha=0.01`).
* **(A3) Merge** units shorter than :math:`\ell_{min}=50` m into the neighbour
  whose mean feature vector is closer in Mahalanobis distance.
* **(A4) Disagreement** :math:`\delta^{map}=1-P(\text{surface}=\text{OSM tag}\mid\mathbf y)`
  from a surface classifier; undefined (``None``) without a tag or classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from ..config import ArbitrationConfig, ISOConfig, MIPELTConfig, SpectralConfig, SpeedConfig
from ..schemas import HRNU, Event, VehicleFrameTrip
from ..layer1.iso2631 import axis_measures, comfort_class, rms, vector_sum
from ..layer1.spectral import modal_band_ratios
from ..layer1.speed import decoupled_intensity
from .mipelt import changepoint_positions

__all__ = [
    "MapTransition",
    "WaySpan",
    "Boundary",
    "transition_weights",
    "snap_changepoints",
    "hotelling_t2",
    "map_only_boundaries",
    "merge_short_units",
    "arbitrate",
    "build_hrnus",
]

TransitionKind = Literal["surface", "highway", "junction", "traffic_calming"]


@dataclass(frozen=True)
class MapTransition:
    """An OSM transition b in B at along-track position ``s`` (m)."""

    s: float
    kind: TransitionKind
    weight: Optional[float] = None  # None -> the default w_b for its kind
    ref: str = ""  # OSM node / way ids, for provenance


@dataclass(frozen=True)
class WaySpan:
    """A matched OSM way covering [s0, s1)."""

    s0: float
    s1: float
    way_id: int
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Boundary:
    s: float
    kind: Literal["map+signal", "signal", "map"]
    transition: Optional[MapTransition] = None
    s_signal: Optional[float] = None  # change-point position before snapping
    p_value: Optional[float] = None  # Hotelling p for map-only boundaries


def transition_weights(transitions: Sequence[MapTransition], cfg: MIPELTConfig | None = None) -> np.ndarray:
    """w_b: 1.0 surface change, 0.6 highway-class change, 0.3 junction / traffic_calming."""
    cfg = cfg or MIPELTConfig()
    default = {
        "surface": cfg.w_surface,
        "highway": cfg.w_highway,
        "junction": cfg.w_junction,
        "traffic_calming": cfg.w_traffic_calming,
    }
    return np.array([b.weight if b.weight is not None else default[b.kind] for b in transitions], dtype=float)


# --------------------------------------------------------------------------- A1
def snap_changepoints(
    cp_s: Sequence[float], transitions: Sequence[MapTransition], snap_m: float = 25.0
) -> list[Boundary]:
    """Snap each change point to the nearest transition within ``snap_m``.

    Two change points never snap to the same transition: the closer one wins
    and the other stays a ``signal`` boundary.
    """
    tr_s = np.array([b.s for b in transitions], dtype=float)
    out: list[Boundary] = []
    claimed: dict[int, tuple[float, int]] = {}
    for i, s in enumerate(cp_s):
        if len(tr_s):
            j = int(np.argmin(np.abs(tr_s - s)))
            d = abs(tr_s[j] - s)
            if d <= snap_m and (j not in claimed or d < claimed[j][0]):
                if j in claimed:
                    k = claimed[j][1]
                    out[k] = Boundary(out[k].s_signal, "signal", s_signal=out[k].s_signal)
                claimed[j] = (d, i)
                out.append(Boundary(float(tr_s[j]), "map+signal", transitions[j], s_signal=float(s)))
                continue
        out.append(Boundary(float(s), "signal", s_signal=float(s)))
    return sorted(out, key=lambda b: b.s)


# --------------------------------------------------------------------------- A2
def hotelling_t2(X1: np.ndarray, X2: np.ndarray, ridge: float = 1e-6) -> tuple[float, float]:
    """Two-sample Hotelling T^2 with pooled covariance; returns (T^2, p-value)."""
    X1 = np.atleast_2d(np.asarray(X1, float))
    X2 = np.atleast_2d(np.asarray(X2, float))
    n1, p = X1.shape
    n2 = X2.shape[0]
    if n1 + n2 - p - 1 <= 0 or n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    d = X1.mean(0) - X2.mean(0)
    S = ((n1 - 1) * np.cov(X1, rowvar=False) + (n2 - 1) * np.cov(X2, rowvar=False)) / (n1 + n2 - 2)
    S = np.atleast_2d(S) + ridge * np.eye(p)
    t2 = float(n1 * n2 / (n1 + n2) * d @ np.linalg.solve(S, d))
    f = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    return t2, float(stats.f.sf(f, p, n1 + n2 - p - 1))


def map_only_boundaries(
    boundaries: Sequence[Boundary],
    transitions: Sequence[MapTransition],
    s_grid: np.ndarray,
    Y: np.ndarray,
    cfg: ArbitrationConfig | None = None,
) -> list[Boundary]:
    """Surface-tag changes with no boundary within ``snap_m`` that pass Hotelling T^2.

    The test uses up to ``hotelling_window_m`` of grid points on each side,
    stopping at the neighbouring boundaries, thinned to every
    ``hotelling_thin``-th point because adjacent features share their 25 m
    windows (unthinned, the test would over-reject).
    """
    cfg = cfg or ArbitrationConfig()
    s_grid = np.asarray(s_grid, float)
    b_s = np.array(sorted(b.s for b in boundaries), dtype=float)
    out: list[Boundary] = []
    for tr in transitions:
        if tr.kind != "surface":
            continue
        if len(b_s) and np.min(np.abs(b_s - tr.s)) <= cfg.snap_m:
            continue
        left_lim = b_s[b_s < tr.s].max() if np.any(b_s < tr.s) else -np.inf
        right_lim = b_s[b_s > tr.s].min() if np.any(b_s > tr.s) else np.inf
        lo = max(tr.s - cfg.hotelling_window_m, left_lim)
        hi = min(tr.s + cfg.hotelling_window_m, right_lim)
        L = Y[(s_grid >= lo) & (s_grid < tr.s)][:: cfg.hotelling_thin]
        R = Y[(s_grid > tr.s) & (s_grid <= hi)][:: cfg.hotelling_thin]
        _, p = hotelling_t2(L, R, cfg.ridge)
        if np.isfinite(p) and p < cfg.hotelling_alpha:
            out.append(Boundary(float(tr.s), "map", tr, p_value=p))
    return out


# --------------------------------------------------------------------------- A3
def _unit_slices(bounds_s: np.ndarray, s_grid: np.ndarray) -> list[np.ndarray]:
    edges = np.concatenate([[-np.inf], bounds_s, [np.inf]])
    return [(s_grid >= edges[k]) & (s_grid < edges[k + 1]) for k in range(len(edges) - 1)]


def merge_short_units(
    boundaries: Sequence[Boundary],
    s_start: float,
    s_end: float,
    s_grid: np.ndarray,
    Y: np.ndarray,
    l_min: float = 50.0,
) -> list[Boundary]:
    """Repeatedly merge the shortest unit below ``l_min`` into its closer neighbour.

    Closeness is the Mahalanobis distance between unit mean feature vectors
    under the pooled within-unit (diagonal) variance.
    """
    bs = sorted(boundaries, key=lambda b: b.s)
    s_grid = np.asarray(s_grid, float)
    while bs:
        edges = np.array([s_start] + [b.s for b in bs] + [s_end])
        lengths = np.diff(edges)
        k = int(np.argmin(lengths))
        if lengths[k] >= l_min:
            break
        masks = _unit_slices(np.array([b.s for b in bs]), s_grid)
        means = [Y[m].mean(0) if m.any() else None for m in masks]
        resid = np.concatenate([Y[m] - Y[m].mean(0) for m in masks if m.sum() > 1] or [Y - Y.mean(0)])
        D = np.maximum(resid.var(0), 1e-6)

        def dist(i: int, j: int) -> float:
            if means[i] is None or means[j] is None:
                return np.inf
            d = means[i] - means[j]
            return float(np.sqrt(np.sum(d * d / D)))

        # unit k is bounded on the left by boundary k-1 and on the right by boundary k
        candidates = []
        if k > 0:
            candidates.append((dist(k, k - 1), k - 1))  # merge left: drop boundary k-1
        if k < len(bs):
            candidates.append((dist(k, k + 1), k))  # merge right: drop boundary k
        _, drop = min(candidates, key=lambda c: (c[0], c[1]))
        bs.pop(drop)
    return bs


def arbitrate(
    changepoints: Sequence[int],
    s_grid: np.ndarray,
    Y: np.ndarray,
    transitions: Sequence[MapTransition] = (),
    cfg: ArbitrationConfig | None = None,
    s_start: float | None = None,
    s_end: float | None = None,
) -> list[Boundary]:
    """A1-A3: change-point indices from MI-PELT -> typed, merged boundaries."""
    cfg = cfg or ArbitrationConfig()
    s_grid = np.asarray(s_grid, float)
    s_start = float(s_grid[0]) if s_start is None else s_start
    s_end = float(s_grid[-1]) if s_end is None else s_end
    pos = changepoint_positions(s_grid)
    cp_s = [float(pos[c]) for c in changepoints]
    bs = snap_changepoints(cp_s, transitions, cfg.snap_m)
    bs = bs + map_only_boundaries(bs, transitions, s_grid, Y, cfg)
    bs = sorted(bs, key=lambda b: b.s)
    dedup: list[Boundary] = []
    for b in bs:
        if not (s_start < b.s < s_end):
            continue
        if dedup and b.s - dedup[-1].s < cfg.dedupe_m:
            # keep the better-supported one: map+signal > signal > map
            rank = {"map+signal": 0, "signal": 1, "map": 2}
            if rank[b.kind] < rank[dedup[-1].kind]:
                dedup[-1] = b
            continue
        dedup.append(b)
    return merge_short_units(dedup, s_start, s_end, s_grid, Y, cfg.l_min_m)


# --------------------------------------------------------------------------- HRNUs
def _osm_for_unit(ways: Sequence[WaySpan], s0: float, s1: float) -> tuple[list[int], list[str], dict]:
    ov = [(min(w.s1, s1) - max(w.s0, s0), w) for w in ways if min(w.s1, s1) > max(w.s0, s0)]
    if not ov:
        return [], [], {}
    ov.sort(key=lambda o: o[1].s0)
    way_ids = []
    names = []
    for _, w in ov:
        if w.way_id not in way_ids:
            way_ids.append(int(w.way_id))
        name = w.tags.get("name")
        if name and name not in names:
            names.append(str(name))
    dominant = max(ov, key=lambda o: o[0])[1]
    return way_ids, names, dict(dominant.tags)


def build_hrnus(
    boundaries: Sequence[Boundary],
    vft: VehicleFrameTrip,
    aw: np.ndarray,
    events: Sequence[Event] = (),
    ways: Sequence[WaySpan] = (),
    *,
    grid_s: Optional[np.ndarray] = None,
    grid_Y: Optional[np.ndarray] = None,
    surface_proba: Optional[Callable[[np.ndarray], Mapping[str, float]]] = None,
    surface_alias: Optional[Callable[[str], str]] = None,
    map_match_conf: Optional[np.ndarray] = None,
    f_wh: float | None = None,
    iso_cfg: ISOConfig | None = None,
    speed_cfg: SpeedConfig | None = None,
    spectral_cfg: SpectralConfig | None = None,
) -> list[HRNU]:
    """Cut the trip at the boundaries and compute every HRNU field (§4.4, §8.3).

    ``aw`` is the trip's (N, 3) weighted acceleration; per-unit measures are
    computed on slices of it so that each unit does not get its own filter
    transient. ``surface_proba`` maps a unit's grid features (rows of
    ``grid_Y``) to class probabilities for A4; ``surface_alias`` maps OSM
    ``surface`` values to the classifier's class names. ``map_match_conf`` is
    a per-IMU-sample confidence (e.g. forward-backward marginals interpolated
    from the fixes).
    """
    iso_cfg = iso_cfg or ISOConfig()
    speed_cfg = speed_cfg or SpeedConfig()
    spectral_cfg = spectral_cfg or SpectralConfig()
    fs = vft.fs
    s = vft.s
    bs = sorted(boundaries, key=lambda b: b.s)
    edges = [float(s[0])] + [b.s for b in bs] + [float(s[-1])]
    kinds: list[str] = ["trip_start"] + [b.kind for b in bs]
    channels = ["acc"] + (["gyro"] if vft.w_v is not None else []) + ["gps"] + (["map"] if ways else [])
    ev_sorted = sorted(events, key=lambda e: e.s)
    units: list[HRNU] = []
    for k in range(len(edges) - 1):
        s0, s1 = edges[k], edges[k + 1]
        last = k == len(edges) - 2
        m = (s >= s0) & ((s <= s1) if last else (s < s1))
        idx = np.flatnonzero(m)
        if len(idx) < 2:
            raise ValueError(f"unit {k + 1} [{s0:.1f}, {s1:.1f}) m holds fewer than two IMU samples")
        sl = slice(int(idx[0]), int(idx[-1]) + 1)
        a_wx, a_wy = rms(aw[sl, 0]), rms(aw[sl, 1])
        mz = axis_measures(aw[sl, 2], fs, iso_cfg)
        a_v = vector_sum(a_wx, a_wy, mz.a_w, (iso_cfg.k_x, iso_cfg.k_y, iso_cfg.k_z))
        v_mean = float(np.mean(vft.v[sl]))
        a_dec = decoupled_intensity(mz.a_w, v_mean, speed_cfg)
        r_wb, r_hb = modal_band_ratios(vft.a_v[sl, 2], fs, f_wh, spectral_cfg)
        way_ids, names, tags = _osm_for_unit(ways, s0, s1)
        ev_ids = [e.event_id for e in ev_sorted if s0 <= e.s < s1 or (last and e.s == s1)]

        disagreement = None
        tag = tags.get("surface")
        if surface_proba is not None and tag and grid_s is not None and grid_Y is not None:
            gm = (grid_s >= s0) & (grid_s < s1)
            if gm.any():
                proba = surface_proba(grid_Y[gm])
                key = surface_alias(tag) if surface_alias else tag
                disagreement = float(1.0 - proba.get(key, 0.0))

        eps_in_unit = sorted(set(int(e) for e in vft.episode[sl] if e >= 0))
        ci_w = [vft.yaw_ci_deg[e][1] - vft.yaw_ci_deg[e][0] for e in eps_in_unit if e < len(vft.yaw_ci_deg)]
        conf = {
            "map_match": float(np.mean(map_match_conf[sl])) if map_match_conf is not None else None,
            "orientation_yaw_ci_deg": float(max(ci_w)) if ci_w else None,
            "channels": channels,
            "discarded_frac": float(np.mean(vft.episode[sl] < 0)),
        }
        units.append(
            HRNU(
                unit_id=f"U{k + 1}",
                s0=s0,
                s1=s1,
                t0=float(vft.t[idx[0]]),
                t1=float(vft.t[idx[-1]]),
                osm_way_ids=way_ids,
                names=names,
                osm_tags=tags,
                a_v=a_v,
                a_wz=mz.a_w,
                a_wz_dec=a_dec,
                vdv_z=mz.vdv,
                mtvv_z=mz.mtvv,
                cf_z=mz.cf,
                kurt_z=mz.kurtosis,
                shock=mz.shock,
                iso_felt=comfort_class(a_v),
                iso_road=comfort_class(a_dec) if a_dec is not None else None,
                r_wb=r_wb,
                r_hb=r_hb,
                v_mean=v_mean,
                event_ids=ev_ids,
                boundary_left=kinds[k],  # type: ignore[arg-type]
                map_signal_disagreement=disagreement,
                conf=conf,
            )
        )
    return units
