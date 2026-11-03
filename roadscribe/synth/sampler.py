r"""Synthetic trips and capsules (METHODOLOGY §8.6, Step 1, ``synth/sampler.py``).

A trip is a Markov chain over surfaces (asphalt, cobblestone, unpaved). Each
unit gets a felt comfort class (ISO 2631-1 Annex C, 0-5) drawn from a
surface-dependent distribution, capped by a per-trip *regime* so that every
class is equally represented as a trip's worst stretch. Physical quantities
are made mutually consistent:

* a_v lies inside the class's cut-point interval;
* a_wz = a_v / sqrt(1 + h^2) with a horizontal share h;
* the speed-decoupled road intensity a_wz (v_ref / v)^gamma (None below 5 m/s);
* events arrive as Poisson processes per km whose rate grows with roughness;
  labels, confidences, maneuvers and MTVV follow the §3.5 evidence (drivers
  brake before bumps, powertrain artifacts come with pulling away);
* a unit is shock-dominated when a confident defect's MTVV exceeds 1.5 a_wz
  (the ISO MTVV criterion).

Degraded trips add missingness and contradiction patterns (no gyroscope,
GPS gap, map-match failure, OSM-signal contradiction, low-confidence events,
too slow to decouple, no map). History trips add a per-unit historical a_v
for cross-trip comparison. Every trip is then compiled with the real ledger
and KST code, so a synthetic capsule is exactly what Layer 3 would build.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import KSTConfig, LedgerConfig, SpeedConfig
from ..layer1.iso2631 import COMFORT_CUTPOINTS, comfort_class
from ..layer3.kst import KSTResult, RegexTokenizer, build_capsule, event_severity
from ..layer3.ledger import Anchor, MissingField, TripMeta, compile_ledger
from ..layer3.personas import PERSONAS
from ..schemas import HRNU, Event, Fact

__all__ = [
    "SURFACES",
    "ROAD_NAMES",
    "POI_NAMES",
    "SynthConfig",
    "SyntheticTrip",
    "sample_trip",
]

SURFACES = ("asphalt", "cobblestone", "unpaved")
_PRED = {"asphalt": "asphalt", "cobblestone": "cobble", "unpaved": "unpaved"}
_OSM_TAGS = {"asphalt": ("asphalt",), "cobblestone": ("sett", "cobblestone", "paving_stones"), "unpaved": ("unpaved", "dirt", "gravel", "compacted")}
_TRANSITION = {
    "asphalt": {"asphalt": 0.45, "cobblestone": 0.30, "unpaved": 0.25},
    "cobblestone": {"asphalt": 0.55, "cobblestone": 0.30, "unpaved": 0.15},
    "unpaved": {"asphalt": 0.55, "cobblestone": 0.15, "unpaved": 0.30},
}
_FELT_DIST = {
    "asphalt": (0.30, 0.30, 0.20, 0.12, 0.06, 0.02),
    "cobblestone": (0.02, 0.08, 0.20, 0.30, 0.25, 0.15),
    "unpaved": (0.02, 0.08, 0.20, 0.28, 0.25, 0.17),
}
_SPEED = {"asphalt": (7.0, 22.0), "cobblestone": (4.0, 13.0), "unpaved": (3.5, 12.0)}
_LENGTH = {"asphalt": (300, 2500), "cobblestone": (150, 900), "unpaved": (200, 1800)}
_DEFECT_RATE = {"asphalt": (0.5, 0.3), "cobblestone": (0.6, 0.3), "unpaved": (0.8, 0.4)}  # per km: base + slope * class

ROAD_NAMES = (
    "Mill Road", "Station Street", "Harbour Way", "Church Lane", "Elm Avenue", "Market Street", "Ridge Road",
    "Canal Road", "Orchard Lane", "Victoria Road", "Park Avenue", "Quarry Lane", "Bridge Street", "Hill Road",
    "North Ring", "Farm Track", "Riverside Drive", "School Lane", "Forest Road", "Kings Road", "Abbey Street",
    "Meadow Way", "Linden Avenue", "Brook Lane", "Castle Street", "Garden Row", "Willow Drive", "Factory Road",
    "Harvest Lane", "Signal Way", "Chapel Street", "Ferry Road", "Granary Lane", "Beacon Way", "Tannery Row",
    "Copper Street", "Lantern Lane", "Weavers Way", "Ashgrove Road", "Pottery Lane",
)
POI_NAMES = (
    "Fire Station", "Central Station", "Town Hall", "Market Hall", "City Library", "General Hospital",
    "Sports Ground", "Bus Depot", "Primary School", "Petrol Station", "Level Crossing", "Water Tower",
    "Old Mill", "Retail Park", "Swimming Pool", "Post Office",
)
DEGRADATIONS = ("no_gyro", "gps_gap", "mapmatch_failed", "osm_contradiction", "low_conf_events", "speed_low", "no_map")


@dataclass(frozen=True)
class SynthConfig:
    personas: tuple[str, ...] = tuple(PERSONAS)
    budgets: tuple[int, ...] = (64, 128, 256, 512)
    budget_weights: tuple[float, ...] = (0.10, 0.35, 0.35, 0.20)
    n_units: tuple[int, int] = (3, 8)
    p_degraded: float = 0.22
    p_history: float = 0.15
    max_events_per_unit: int = 3
    min_event_gap_m: float = 60.0
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    kst: KSTConfig = field(default_factory=KSTConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)


@dataclass
class SyntheticTrip:
    trip_id: str
    persona: str
    regime: int  # worst felt class of the trip
    task_mix: str  # "narrative" | "degraded" | "history"
    patterns: list[str]
    units: list[HRNU]
    events: list[Event]
    anchors: list[Anchor]
    missing: list[MissingField]
    history: Optional[dict[str, float]]
    meta: TripMeta
    ledger: list[Fact]
    kst: KSTResult
    must_mention_events: list[str]

    @property
    def capsule(self):
        return self.kst.capsule


def _choice(r: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return r.choices(keys, [weights[k] for k in keys])[0]


def _class_interval(k: int) -> tuple[float, float]:
    cuts = (0.10, *COMFORT_CUTPOINTS, 3.6)
    return cuts[k], cuts[k + 1]


def _sample_in_class(r: random.Random, k: int) -> float:
    lo, hi = _class_interval(k)
    lo, hi = lo * 1.01, hi * 0.99
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


def _event_p(r: random.Random, label: str, low_conf: bool) -> float:
    u = r.random()
    if label == "defect":
        p = r.uniform(0.9, 0.99) if u < 0.70 else r.uniform(0.7, 0.9) if u < 0.90 else r.uniform(0.4, 0.7) if u < 0.97 else r.uniform(0.2, 0.4)
    elif label == "ambiguous":
        p = r.uniform(0.7, 0.9) if u < 0.30 else r.uniform(0.4, 0.7) if u < 0.90 else r.uniform(0.2, 0.4)
    else:
        p = r.uniform(0.05, 0.3)
    if low_conf:
        p *= r.uniform(0.45, 0.8)
    return round(p, 2)


def sample_trip(r: random.Random, idx: int, cfg: SynthConfig | None = None, persona: Optional[str] = None,
                regime: Optional[int] = None, task_mix: Optional[str] = None) -> SyntheticTrip:
    """Sample one synthetic trip and compile its ledger and capsule."""
    cfg = cfg or SynthConfig()
    persona = persona or r.choice(cfg.personas)
    regime = r.randrange(6) if regime is None else regime
    if task_mix is None:
        u = r.random()
        task_mix = "degraded" if u < cfg.p_degraded else "history" if u < cfg.p_degraded + cfg.p_history else "narrative"
    patterns: list[str] = []
    if task_mix == "degraded":
        patterns = r.sample(DEGRADATIONS, k=1 if r.random() < 0.7 else 2)
    gyro = "no_gyro" not in patterns
    no_map = "no_map" in patterns

    # ---- surfaces and felt classes
    n = r.randint(*cfg.n_units)
    surf = [r.choices(SURFACES, (0.6, 0.2, 0.2))[0]]
    for _ in range(n - 1):
        surf.append(_choice(r, _TRANSITION[surf[-1]]))
    if regime <= 1:
        surf = [s if s == "asphalt" or r.random() < 0.2 else "asphalt" for s in surf]
    elif regime >= 4 and all(s == "asphalt" for s in surf):
        surf[r.randrange(n)] = r.choice(("cobblestone", "unpaved"))
    felt = [min(regime, r.choices(range(6), _FELT_DIST[s])[0]) for s in surf]
    worst = max(range(n), key=lambda i: (surf[i] != "asphalt" if regime >= 3 else surf[i] == "asphalt", felt[i]))
    felt[worst] = regime
    for i in range(1, n):  # adjacent units are distinct (a boundary exists for a reason)
        if surf[i] == surf[i - 1] and felt[i] == felt[i - 1]:
            felt[i] = max(0, felt[i] - 1) if felt[i] == regime else min(regime, felt[i] + 1)
            if felt[i] == felt[i - 1]:
                felt[i] = max(0, felt[i] - 1)

    # ---- names
    names: list[Optional[str]] = []
    pool = list(ROAD_NAMES)
    r.shuffle(pool)
    for i in range(n):
        if i > 0 and names[-1] and r.random() < 0.3:
            names.append(names[-1])  # surface changes mid-street
        else:
            names.append(pool.pop())
    speed_low_unit = r.randrange(n) if "speed_low" in patterns else None
    mm_failed = r.randrange(n) if "mapmatch_failed" in patterns else None
    gps_gap_unit = r.randrange(n) if "gps_gap" in patterns else None
    contra_unit = max(range(n), key=lambda i: felt[i]) if "osm_contradiction" in patterns else None

    units: list[HRNU] = []
    raw_events: list[dict] = []
    s0 = 0.0
    t0 = 0.0
    for i in range(n):
        sf = surf[i]
        length = float(round(r.uniform(*_LENGTH[sf]), -1))
        v = r.uniform(*_SPEED[sf])
        if i == speed_low_unit or (sf != "asphalt" and r.random() < 0.06):
            v = r.uniform(2.5, 4.8)
        v = round(v, 3)  # stored value; decide speed_low on exactly what the HRNU holds
        a_v = _sample_in_class(r, felt[i])
        h = r.uniform(0.25, 0.55)
        a_wz = a_v / math.sqrt(1 + h * h)
        a_dec = a_wz * (cfg.speed.v_ref / v) ** cfg.speed.gamma if v >= cfg.speed.v_min else None
        T = length / v
        uid = f"U{i + 1}"
        # events
        evs = []
        k = felt[i]
        rates = {
            "defect": _DEFECT_RATE[sf][0] + _DEFECT_RATE[sf][1] * k,
            "ambiguous": 0.35,
            "artifact": 0.25,
            "avoidance": 0.06 if gyro else 0.0,
        }
        for label, rate in rates.items():
            lam = rate * length / 1000.0
            m = 0
            # Poisson by inversion
            pk, cdf, x = math.exp(-lam), math.exp(-lam), r.random()
            while x > cdf and m < 10:
                m += 1
                pk *= lam / m
                cdf += pk
            for _ in range(m):
                evs.append(label)
        r.shuffle(evs)
        evs = evs[: cfg.max_events_per_unit]
        placed: list[float] = []
        for label in evs:
            for _try in range(20):
                s = round(r.uniform(s0 + 20, s0 + length - 20), 0)
                if all(abs(s - q) >= cfg.min_event_gap_m for q in placed):
                    break
            else:
                continue
            placed.append(s)
            kind = "avoidance" if label == "avoidance" else "impulse"
            lab = "ambiguous" if label == "avoidance" else label
            if kind == "avoidance":
                p = round(r.uniform(0.42, 0.85), 2)
                man = "swerve"
                mt = a_wz * r.uniform(1.0, 1.4)
            else:
                p = _event_p(r, lab, "low_conf_events" in patterns)
                man = {
                    "defect": lambda: r.choices(("cruise", "brake", "turn"), (0.55, 0.35, 0.10))[0],
                    "ambiguous": lambda: r.choices(("stop_go", "accel", "cruise"), (0.40, 0.35, 0.25))[0],
                    "artifact": lambda: r.choice(("stop_go", "accel")),
                }[lab]()
                mt = a_wz * {"defect": r.uniform(1.6, 4.0), "ambiguous": r.uniform(1.2, 2.2), "artifact": r.uniform(1.0, 1.6)}[lab]
            te = t0 + (s - s0) / v
            raw_events.append(dict(kind=kind, label=lab, t0=round(te, 2), t1=round(te + r.uniform(0.1, 0.4), 2), s=float(s),
                                   lat=52.0 + s / 111320.0, lon=4.0, tkeo_peak=round(r.uniform(2, 40), 2),
                                   band_ratio=round(r.uniform(0.3, 15), 2) if kind == "impulse" else float("nan"),
                                   mtvv_z=round(mt, 3), maneuver=man, p_defect=p, _unit=uid))
        unit_evs = [e for e in raw_events if e["_unit"] == uid]
        shock = any(e["label"] == "defect" and e["p_defect"] >= 0.7 and e["mtvv_z"] > 1.5 * a_wz for e in unit_evs)
        kappa = r.uniform(9.5, 25) if shock else r.uniform(3.0, 6.0)
        mtvv = max([a_wz * r.uniform(1.05, 1.35)] + [e["mtvv_z"] for e in unit_evs])
        # map
        tags: dict = {}
        way_ids = [1000 * idx + i + 1]
        nm = names[i]
        if not no_map and i != mm_failed:
            tags["highway"] = r.choice(("residential", "secondary", "tertiary", "unclassified", "service"))
            if r.random() < 0.8 or i == contra_unit:
                if i == contra_unit or r.random() < 0.08:
                    tags["surface"] = r.choice([t for s2 in SURFACES if s2 != sf for t in _OSM_TAGS[s2]])
                else:
                    tags["surface"] = r.choice(_OSM_TAGS[sf])
            if nm and r.random() < 0.96:
                tags["name"] = nm
            else:
                nm = None
        else:
            way_ids, nm = [], None
        pred = _PRED[sf] if r.random() < 0.9 else _PRED[r.choice([s2 for s2 in SURFACES if s2 != sf])]
        conf = {"map_match": round(r.uniform(0.85, 0.99), 2), "orientation_yaw_ci_deg": round(r.uniform(0.3, 2.5), 2),
                "channels": ["acc"] + (["gyro"] if gyro else []) + ["gps"] + ([] if no_map else ["map"]),
                "surface_pred": pred, "surface_pred_p": round(r.uniform(0.6, 0.97), 2)}
        if i == mm_failed:
            conf["map_match"] = round(r.uniform(0.2, 0.4), 2)
        if i == gps_gap_unit:
            conf["map_match"] = round(r.uniform(0.5, 0.7), 2)
        disagreement = None
        if "surface" in tags:
            tag_class = next(s2 for s2 in SURFACES if tags["surface"] in _OSM_TAGS[s2])
            disagreement = round(r.uniform(0.03, 0.35) if _PRED[tag_class] == pred else r.uniform(0.55, 0.95), 2)
        units.append(HRNU(
            unit_id=uid, s0=s0, s1=s0 + length, t0=round(t0, 1), t1=round(t0 + T, 1),
            osm_way_ids=way_ids, names=[nm] if nm else [], osm_tags=tags,
            a_v=round(a_v, 4), a_wz=round(a_wz, 4), a_wz_dec=round(a_dec, 4) if a_dec is not None else None,
            vdv_z=round(a_wz * T**0.25 * kappa**0.25, 3), mtvv_z=round(mtvv, 3),
            cf_z=round(r.uniform(9.5, 16) if shock else r.uniform(3, 7), 2), kurt_z=round(kappa, 2), shock=shock,
            iso_felt=comfort_class(round(a_v, 4)),
            iso_road=comfort_class(round(a_dec, 4)) if a_dec is not None else None,
            r_wb=round(r.uniform(-2.5, 1.0) + (0.8 if sf != "asphalt" else 0), 3), r_hb=round(r.uniform(-4, -1), 3),
            v_mean=round(v, 3), event_ids=[], boundary_left="trip_start" if i == 0 else r.choice(("map+signal", "signal", "map")),
            map_signal_disagreement=disagreement, conf=conf,
        ))
        s0 += length
        t0 += T

    raw_events.sort(key=lambda e: e["s"])
    events = []
    for k, e in enumerate(raw_events):
        uid = e.pop("_unit")
        ev = Event(event_id=f"E{k + 1}", **e)
        events.append(ev)
        next(u for u in units if u.unit_id == uid).event_ids.append(ev.event_id)

    missing: list[MissingField] = []
    if gps_gap_unit is not None:
        missing.append(MissingField(units[gps_gap_unit].unit_id, "anchor", f"gps_gap_{r.randint(12, 90)}s"))

    anchors = []
    if not no_map:
        for name in r.sample(POI_NAMES, k=r.randint(0, 3)):
            u = r.choice(units)
            anchors.append(Anchor(name, round(r.uniform(u.s0 + 10, u.s1 - 10), 0)))

    history = None
    if task_mix == "history":
        history = {u.unit_id: round(u.a_v * math.exp(r.gauss(0, 0.35)), 4) for u in units if r.random() < 0.85}

    length = sum(u.length_m for u in units)
    tagged = sum(u.length_m for u in units if u.osm_tags.get("surface"))
    meta = TripMeta(
        trip_id=f"syn{idx:05d}",
        persona=persona,
        channels=tuple(units[0].conf["channels"]),
        duration_s=round(sum(u.duration_s for u in units), 1),
        length_m=length,
        map_coverage=None if no_map else round(tagged / length, 2),
        orientation_yaw_ci_deg=round(r.uniform(0.3, 2.5), 2),
        history=f"{r.randint(3, 12)} drives" if history else "none",
    )
    ledger = compile_ledger(units, events, meta, cfg.ledger, anchors=anchors, missing=missing, history=history,
                            anchor_rmax_m=cfg.kst.anchor_rmax_m)
    hist_delta = {u: abs(math.log(next(x.a_v for x in units if x.unit_id == u) / h)) for u, h in history.items()} if history else None
    budget = r.choices(cfg.budgets, cfg.budget_weights)[0]
    kst = build_capsule(units, events, meta, budget, RegexTokenizer(), cfg.kst, ledger=ledger, anchors=anchors,
                        history_delta=hist_delta)
    must = [e.event_id for u in units if u.unit_id in kst.must_mention for e in events
            if e.event_id in u.event_ids and e.p_defect * event_severity(e) >= cfg.kst.eta_star]
    return SyntheticTrip(meta.trip_id, persona, regime, task_mix, patterns, units, events, anchors, missing, history,
                         meta, ledger, kst, must)
