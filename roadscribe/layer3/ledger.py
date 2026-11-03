r"""Claim ledger: HRNUs and events -> typed facts (METHODOLOGY §5.1).

The ledger is the single source of truth for the run-time verifier and the
evaluation. Every fact is ``(fact_id, kind, unit_id, key, value, tol, conf)``
with kinds SEV, NUM, EVT, ANCHOR, CMP, MISSING and META. A MISSING fact
records that a value is unavailable *and why*, so that a narrator that states
it can be caught (the hallucinated-missing rate, §6.2).

Fact ids are assigned in a fixed order (META first, then units in route
order, then external MISSING facts), so the same inputs always give the same
ledger.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..config import LedgerConfig
from ..layer1.iso2631 import comfort_label
from ..schemas import HRNU, Event, Fact, to_jsonable

__all__ = [
    "TripMeta",
    "Anchor",
    "MissingField",
    "event_word",
    "nearest_anchor",
    "compile_ledger",
    "facts_by_unit",
    "write_jsonl",
    "read_jsonl",
    "format_fact",
    "format_ledger",
    "LedgerView",
]


@dataclass(frozen=True)
class TripMeta:
    trip_id: str
    persona: str = "commuter"
    channels: tuple[str, ...] = ("acc", "gyro", "gps", "map")
    duration_s: float = float("nan")
    length_m: float = float("nan")
    map_coverage: Optional[float] = None  # fraction of matched length carrying a surface tag
    orientation_yaw_ci_deg: Optional[float] = None
    history: str = "none"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Anchor:
    """A named place (POI, junction, landmark) at along-track position ``s``."""

    name: str
    s: float
    kind: str = "poi"


@dataclass(frozen=True)
class MissingField:
    """A field that is not available, e.g. ``MissingField("U6", "anchor", "gps_gap_48s")``."""

    unit_id: Optional[str]
    key: str
    reason: str


def event_word(e: Event) -> str:
    """Short type word used in capsules and EVT facts."""
    if e.kind == "avoidance":
        return "avoidance"
    return {"defect": "bump", "ambiguous": "jolt", "artifact": "artifact"}[e.label]


def nearest_anchor(u: HRNU, anchors: Sequence[Anchor], r_max: float) -> Optional[Anchor]:
    """Anchor closest to the unit interval (distance 0 inside it), within ``r_max``."""
    best, best_d = None, math.inf
    for a in anchors:
        d = 0.0 if u.s0 <= a.s < u.s1 else min(abs(a.s - u.s0), abs(a.s - u.s1))
        if d <= r_max and d < best_d:
            best, best_d = a, d
    return best


class _Builder:
    def __init__(self, cfg: LedgerConfig):
        self.cfg = cfg
        self.facts: list[Fact] = []

    def add(self, kind: str, unit_id: Optional[str], key: str, value: Any, tol=None, conf=1.0, note="") -> Fact:
        f = Fact(f"F{len(self.facts) + 1}", kind, unit_id, key, to_jsonable(value), tol, float(conf), note)  # type: ignore[arg-type]
        self.facts.append(f)
        return f

    def num(self, unit_id: Optional[str], key: str, value: float, eps_abs: float, note: str = "", conf: float = 1.0):
        tol = max(eps_abs, self.cfg.eps_rel * abs(value))
        return self.add("NUM", unit_id, key, float(value), tol=tol, conf=conf, note=note)

    def missing(self, unit_id: Optional[str], key: str, reason: str):
        return self.add("MISSING", unit_id, key, None, note=reason)


def _relation(x: float, ref: float, margin: float) -> str:
    if ref <= 0 or not np.isfinite(ref) or not np.isfinite(x):
        return "similar"
    r = x / ref
    if r > 1.0 + margin:
        return "rougher"
    if r < 1.0 / (1.0 + margin):
        return "smoother"
    return "similar"


def compile_ledger(
    units: Sequence[HRNU],
    events: Sequence[Event],
    trip: TripMeta,
    cfg: LedgerConfig | None = None,
    anchors: Sequence[Anchor] = (),
    missing: Iterable[MissingField] = (),
    history: Optional[Mapping[str, float]] = None,
    anchor_rmax_m: float = 200.0,
) -> list[Fact]:
    """Compile the claim ledger for one trip.

    ``history`` maps unit ids to the route's historical median a_v (aligned
    by the caller); it adds ``a_v_vs_history`` comparisons.
    """
    cfg = cfg or LedgerConfig()
    b = _Builder(cfg)
    ev_by_id = {e.event_id: e for e in events}

    # ---- META
    b.add("META", None, "channels", list(trip.channels))
    b.add("META", None, "persona", trip.persona)
    b.add("META", None, "n_units", len(units))
    if np.isfinite(trip.duration_s):
        b.num(None, "trip_duration_s", trip.duration_s, cfg.eps_abs_time)
    if np.isfinite(trip.length_m):
        b.num(None, "trip_length_m", trip.length_m, cfg.eps_abs_len)
    if trip.map_coverage is not None:
        b.add("META", None, "map_coverage", float(trip.map_coverage))
    else:
        b.missing(None, "map_coverage", "no_map")
    if trip.orientation_yaw_ci_deg is not None:
        b.add("META", None, "orientation_yaw_ci_deg", float(trip.orientation_yaw_ci_deg))
    if "gyro" not in trip.channels:
        b.missing(None, "gyro", "no_gyro")

    a_v_all = np.array([u.a_v for u in units], dtype=float)
    trip_median = float(np.median(a_v_all)) if len(a_v_all) else float("nan")

    for i, u in enumerate(units):
        uid = u.unit_id
        mm = u.conf.get("map_match") if isinstance(u.conf, Mapping) else None
        anchor_conf = float(mm) if mm is not None else 1.0
        # severity
        b.add("SEV", uid, "felt_class", int(u.iso_felt), note=comfort_label(u.iso_felt))
        b.num(uid, "a_v", u.a_v, cfg.eps_abs_acc, note="ISO 2631-1-weighted cabin acceleration, felt")
        if u.iso_road is not None and u.a_wz_dec is not None:
            b.add("SEV", uid, "road_class", int(u.iso_road), note=comfort_label(u.iso_road))
            b.num(uid, "a_wz_dec", u.a_wz_dec, cfg.eps_abs_acc, note="speed-decoupled vertical, road")
        else:
            b.missing(uid, "road_class", "speed_low")
        b.num(uid, "a_wz", u.a_wz, cfg.eps_abs_acc)
        if u.shock:
            b.add("SEV", uid, "shock", True, note="shock-dominated (ISO 2631-1 evaluation rule)")
            b.num(uid, "mtvv_z", u.mtvv_z, cfg.eps_abs_acc, note="peak 1-s exposure")
            b.num(uid, "vdv_z", u.vdv_z, cfg.eps_abs_acc, note="no ISO comfort scale; never map to comfort words")
        # geometry and speed
        b.num(uid, "length_m", u.length_m, cfg.eps_abs_len)
        b.num(uid, "duration_s", u.duration_s, cfg.eps_abs_time)
        b.num(uid, "v_mean_mps", u.v_mean, cfg.eps_abs_speed)
        # anchors
        b.add("ANCHOR", uid, "extent", {"s0": float(u.s0), "s1": float(u.s1)}, tol=cfg.event_tol_m, conf=anchor_conf)
        if u.names:
            for name in u.names:
                b.add("ANCHOR", uid, "name", name, conf=anchor_conf, note=f"{u.s0:.0f}-{u.s1:.0f} m")
        else:
            reason = "no_osm_name" if u.osm_way_ids else "no_map" if trip.map_coverage is None else "mapmatch_failed"
            b.missing(uid, "name", reason)
        poi = nearest_anchor(u, anchors, anchor_rmax_m)
        if poi is not None:
            b.add("ANCHOR", uid, "near", {"name": poi.name, "s": poi.s, "kind": poi.kind}, conf=anchor_conf)
        surface = u.osm_tags.get("surface") if u.osm_tags else None
        if surface:
            b.add("ANCHOR", uid, "osm_surface", surface, conf=anchor_conf)
        pred = u.conf.get("surface_pred") if isinstance(u.conf, Mapping) else None
        if pred:
            p_pred = u.conf.get("surface_pred_p", 1.0)
            b.add("SEV", uid, "surface_signal", pred, conf=float(p_pred), note="signal surface classifier")
        if u.map_signal_disagreement is not None:
            b.num(uid, "map_signal_disagreement", u.map_signal_disagreement, cfg.eps_abs_score)
        else:
            b.missing(uid, "map_signal_disagreement", "no_osm_surface_tag" if not surface else "no_surface_classifier")
        # events
        unit_events = [ev_by_id[e] for e in u.event_ids if e in ev_by_id]
        b.num(uid, "n_events", len(unit_events), 0.0)
        for e in unit_events:
            b.add(
                "EVT",
                uid,
                e.event_id,
                {
                    "type": event_word(e),
                    "kind": e.kind,
                    "label": e.label,
                    "s": e.s,
                    "t": e.t0,
                    "p_defect": e.p_defect,
                    "mtvv_z": e.mtvv_z,
                    "maneuver": e.maneuver,
                },
                tol=cfg.event_tol_m,
                conf=e.p_defect,
            )
        # comparisons
        if i > 0:
            prev = units[i - 1]
            b.add(
                "CMP",
                uid,
                "a_v_vs_prev",
                {"other": prev.unit_id, "relation": _relation(u.a_v, prev.a_v, cfg.cmp_margin), "ratio": u.a_v / prev.a_v if prev.a_v > 0 else None},
            )
        if len(units) > 1:
            b.add(
                "CMP",
                uid,
                "a_v_vs_trip_median",
                {"relation": _relation(u.a_v, trip_median, cfg.cmp_margin), "ratio": u.a_v / trip_median if trip_median > 0 else None},
            )
        if history is not None:
            if uid in history and history[uid] is not None:
                h = float(history[uid])
                b.add("CMP", uid, "a_v_vs_history", {"relation": _relation(u.a_v, h, cfg.cmp_margin), "ratio": u.a_v / h if h > 0 else None})
            else:
                b.missing(uid, "history", "no_history_for_unit")

    for m in missing:
        b.missing(m.unit_id, m.key, m.reason)
    return b.facts


def facts_by_unit(facts: Sequence[Fact]) -> dict[Optional[str], list[Fact]]:
    out: dict[Optional[str], list[Fact]] = {}
    for f in facts:
        out.setdefault(f.unit_id, []).append(f)
    return out


def write_jsonl(facts: Sequence[Fact], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(to_jsonable(f), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Fact]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Fact(**json.loads(line)))
    return out


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, Mapping):
        return " ".join(f"{k}={_short(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ",".join(_short(x) for x in v)
    return str(v)


def format_fact(f: Fact, compact: bool = False) -> str:
    """One line per fact, e.g. ``F13 U2 NUM a_v=1.62 tol=0.162``.

    ``compact=True`` is the prompt form: no tolerances, and events as
    ``F27 U1 EVT E2 bump s=734 p=0.92 brake``.
    """
    head = f"{f.fact_id} {f.unit_id or '-'} {f.kind} {f.key}"
    if f.kind == "MISSING":
        return f"{head} ({f.note})"
    if compact and f.kind == "EVT":
        v = f.value
        return f"{head} {v['type']} s={v['s']:.0f} p={f.conf:.2f} {v['maneuver']}"
    if compact and f.key == "extent":
        return f"{head}={f.value['s0']:.0f}-{f.value['s1']:.0f}m"
    body = f"{head}={_short(f.value)}"
    if f.tol is not None and not compact:
        body += f" tol={f.tol:.3g}"
    if f.conf < 1.0 and (not compact or f.kind != "ANCHOR"):
        body += f" conf={f.conf:.2f}"
    return body


def format_ledger(
    facts: Sequence[Fact], unit_ids: Optional[Iterable[Optional[str]]] = None, compact: bool = False
) -> str:
    """Fact index for prompts; ``unit_ids`` restricts it (``None`` = trip-level facts)."""
    keep = None if unit_ids is None else set(unit_ids)
    return "\n".join(format_fact(f, compact) for f in facts if keep is None or f.unit_id in keep)


class LedgerView:
    """Read-only index over a ledger, used by the planner, realizer and verifier."""

    def __init__(self, facts: Sequence[Fact]):
        self.facts = list(facts)
        self.by_id = {f.fact_id: f for f in self.facts}
        self.units: list[str] = []
        self._unit: dict[Optional[str], list[Fact]] = {}
        for f in self.facts:
            if f.unit_id is not None and f.unit_id not in self._unit:
                self.units.append(f.unit_id)
            self._unit.setdefault(f.unit_id, []).append(f)
        names: dict[str, list[str]] = {}
        for f in self.facts:
            if f.kind == "ANCHOR" and f.key == "name":
                names.setdefault(f.value, []).append(f.unit_id)
        self.name_units = names

    def facts_of(self, unit_id: Optional[str]) -> list[Fact]:
        return self._unit.get(unit_id, [])

    def get(self, unit_id: Optional[str], key: str, kind: Optional[str] = None) -> Optional[Fact]:
        for f in self.facts_of(unit_id):
            if f.key == key and (kind is None or f.kind == kind):
                return f
        return None

    def all(self, unit_id: Optional[str], key: str, kind: Optional[str] = None) -> list[Fact]:
        return [f for f in self.facts_of(unit_id) if f.key == key and (kind is None or f.kind == kind)]

    def value(self, unit_id: Optional[str], key: str, default: Any = None) -> Any:
        f = self.get(unit_id, key)
        return default if f is None or f.kind == "MISSING" else f.value

    def missing(self, unit_id: Optional[str]) -> dict[str, str]:
        return {f.key: f.note for f in self.facts_of(unit_id) if f.kind == "MISSING"}

    def events(self, unit_id: Optional[str]) -> list[Fact]:
        return [f for f in self.facts_of(unit_id) if f.kind == "EVT"]

    def names(self, unit_id: str) -> list[str]:
        return [f.value for f in self.all(unit_id, "name", "ANCHOR")]

    def extent(self, unit_id: str) -> tuple[float, float]:
        v = self.value(unit_id, "extent")
        return (float(v["s0"]), float(v["s1"]))

    def unit_of_event(self, event_id: str) -> Optional[str]:
        for f in self.facts:
            if f.kind == "EVT" and f.key == event_id:
                return f.unit_id
        return None
