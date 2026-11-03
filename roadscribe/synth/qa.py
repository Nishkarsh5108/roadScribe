"""Driver questions with ledger-grounded gold answers (METHODOLOGY §6.7).

Each item pairs a question in a driver's own words with a gold answer built
from the ledger and checked by the claim verifier, or with an abstention and
its reason when the drive cannot answer it (about 20 %).

These questions were written at design time, so they are suitable for
development and regression testing only. E1 (§2.2) needs questions elicited
from drivers who are not in the study, with a sealed half that nothing is
tuned on.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..eval.claims import verify_text
from ..layer3.hedging import MENTION_THRESHOLD
from ..layer3.ledger import LedgerView, format_ledger
from ..layer3.lexicon import SURFACE_WORDS
from ..layer3.personas import get_persona
from ..layer3.templates import _cap, _event_sentences, _felt_word, _road_word, fmt_kmh, ground, unit_ref
from .sampler import SyntheticTrip

__all__ = ["QAItem", "ANSWERABLE_TYPES", "UNANSWERABLE_TYPES", "candidate_questions", "qa_record"]

Q = {
    "roughest": {
        None: ["Which part of the drive was the roughest?", "Where was the worst bit of road today?",
               "Which stretch shook the car the most?", "Where should I expect the roughest ride on this route?"],
        "fleet_manager": ["Which stretch is hardest on the vans?", "Where was the worst vibration on this run?"],
        "car_sick_passenger": ["Which part of the ride was the most uncomfortable?", "Where was the ride worst for me?"],
        "sports_car": ["Where was the car most unsettled?"],
    },
    "smoothest": {
        None: ["Which stretch was the smoothest?", "Where was the ride nicest?", "Was any part of the drive actually smooth?"],
        "car_sick_passenger": ["Where could I relax a bit?"],
        "sports_car": ["Where did the car feel most settled?"],
    },
    "unit_felt": {None: ["How was {name}?", "How rough was {name} today?", "What was the ride like on {name}?",
                         "Was {name} bad?", "How did {name} feel?"]},
    "unit_events": {None: ["Did I hit anything on {name}?", "Were there any bumps on {name}?",
                           "Anything to watch out for on {name}?", "I felt a knock somewhere on {name}. What was it?"]},
    "speed_cause": {None: ["Was {name} rough because of the road or because I was going fast?",
                           "Is {name} actually a bad road, or was I just driving too quickly?",
                           "Would {name} have been fine if I'd slowed down?"]},
    "jolt_cause": {None: ["Was that jolt on {name} the road or me?", "Did I cause that jerk on {name}?",
                          "What was that lurch on {name}?"]},
    "map_check": {None: ["The map says {name} is {osm}. Is that right?", "Does {name} match what the map says?",
                         "Is {name} really {osm}?"]},
    "history": {None: ["Was {name} worse than usual?", "Has {name} got any better?", "Was {name} rougher than normal today?"]},
    "speed_on": {None: ["How fast was I going on {name}?", "What speed was I doing on {name}?"]},
    # unanswerable
    "out_of_scope": {None: []},
    "history_missing": {None: ["Was {name} worse than usual?", "Is {name} getting worse?"]},
    "name_missing": {None: ["What's the name of the road between {a} and {b} km?", "Which street was I on from {a} to {b} km?"]},
    "road_vs_speed_unknown": {None: ["Is {name} a bad road, or was it just me?", "Was {name} rough because of the road?"]},
}
OUT_OF_SCOPE = [
    ("Did that last pothole damage my tyres?", "tyre or vehicle damage is not measured"),
    ("How deep was the worst pothole on {name}?", "the size or depth of a defect is not measured"),
    ("What was the weather like on the drive?", "weather is not recorded"),
    ("Was there much traffic on {name}?", "traffic is not recorded"),
    ("How much fuel did I use?", "fuel use is not recorded"),
    ("Is {name} going to be resurfaced soon?", "maintenance plans are not in the map data"),
    ("Was I over the speed limit on {name}?", "speed limits are not part of this trip's data"),
    ("How loud was the cabin on {name}?", "cabin noise is not recorded"),
    ("What did I swerve around?", "what caused a steering manoeuvre cannot be identified from motion data"),
]
ANSWERABLE_TYPES = ("roughest", "smoothest", "unit_felt", "unit_events", "speed_cause", "jolt_cause", "map_check", "history", "speed_on")
UNANSWERABLE_TYPES = ("out_of_scope", "history_missing", "name_missing", "road_vs_speed_unknown")


@dataclass
class QAItem:
    qtype: str
    question: str
    answerable: bool
    answer: str
    fact_ids: list[str]
    abstain_reason: Optional[str] = None
    must_events: list[str] = field(default_factory=list)


def _frame(L: LedgerView, u: str, keys: tuple[str, ...], events: bool = False) -> tuple[dict, list[str]]:
    ids = [L.get(u, "extent").fact_id] + [f.fact_id for f in L.all(u, "name", "ANCHOR")]
    for k in keys:
        f = L.get(u, k)
        if f is not None:
            ids.append(f.fact_id)
    if events:
        ids += [f.fact_id for f in L.events(u) if f.conf >= MENTION_THRESHOLD and f.value["label"] != "artifact"]
    plan = {"intent": "trip_summary", "items": [{"ref": u, "facts": ids, "rel": "elaboration", "hedge": "none"}],
            "omit": [], "close": {"type": "none"}}
    return ground(plan, L.facts)[0], ids


def _lead(ref: str, sents: list[str]) -> str:
    """'On X, there was ...' followed by the remaining sentences."""
    first = sents[0][:1].lower() + sents[0][1:]
    return " ".join([f"On {ref}, {first}", *sents[1:]])


def _ask(r: random.Random, qtype: str, persona: str, **kw) -> str:
    bank = Q[qtype].get(persona, []) + Q[qtype][None]
    return r.choice(bank).format(**kw)


def candidate_questions(trip: SyntheticTrip, r: random.Random) -> list[QAItem]:
    """Every question type this trip can support, with its gold answer."""
    L = LedgerView(trip.ledger)
    P = get_persona(trip.persona)
    out: list[QAItem] = []
    unique = [u for u in L.units if L.names(u) and len(L.name_units[L.names(u)[0]]) == 1]
    a_v = {u: L.value(u, "a_v") for u in L.units}

    def name(u):
        return L.names(u)[0]

    if len(L.units) >= 2:
        u = max(L.units, key=lambda x: a_v[x])
        fr, ids = _frame(L, u, ("felt_class", "a_v"))
        out.append(QAItem("roughest", _ask(r, "roughest", trip.persona),
                          True, _cap(f"the roughest stretch was {unit_ref(fr, P)}, which felt {_felt_word(fr['felt'], r)}."), ids))
        u = min(L.units, key=lambda x: a_v[x])
        fr, ids = _frame(L, u, ("felt_class", "a_v"))
        out.append(QAItem("smoothest", _ask(r, "smoothest", trip.persona),
                          True, _cap(f"{unit_ref(fr, P)} was the smoothest stretch and felt {_felt_word(fr['felt'], r)}."), ids))
    for u in unique:
        nm = name(u)
        miss = L.missing(u)
        fr, ids = _frame(L, u, ("felt_class", "road_class", "shock"))
        ans = f"{unit_ref(fr, P)} felt {_felt_word(fr['felt'], r)}"
        if "road" in fr:
            ans += f", and the surface itself is {_road_word(fr['road'])}"
        ans = _cap(ans) + "."
        if fr.get("shock"):
            ans += " It was jarring in places."
        out.append(QAItem("unit_felt", _ask(r, "unit_felt", trip.persona, name=nm), True, ans, ids))

        fr, ids = _frame(L, u, (), events=True)
        evs = fr["events"]
        if evs:
            sents = _event_sentences(fr, P, r, report=False)
            ans = _lead(unit_ref(fr, P), sents)
            must = [e["id"] for e in evs]
        else:
            ans = f"I didn't pick up any bumps or jolts on {unit_ref(fr, P)}."
            must = []
        out.append(QAItem("unit_events", _ask(r, "unit_events", trip.persona, name=nm), True, ans, ids, must_events=must))

        fc, rc, v = L.value(u, "felt_class"), L.value(u, "road_class"), L.value(u, "v_mean_mps")
        if "road_class" in miss:
            fr, ids = _frame(L, u, ("felt_class",))
            reason = "you were moving too slowly there to separate the road from your speed"
            ans = f"I can't tell that from this drive: {reason}. {_cap(unit_ref(fr, P))} felt {_felt_word(fr['felt'], r)}, though."
            out.append(QAItem("road_vs_speed_unknown", _ask(r, "road_vs_speed_unknown", trip.persona, name=nm), False, ans, ids + [L.get(u, 'road_class').fact_id], reason))
        elif None not in (fc, rc, v) and fc - rc >= 1 and v >= 12.0:
            fr, ids = _frame(L, u, ("felt_class", "road_class", "v_mean_mps"))
            ans = (f"{unit_ref(fr, P)} felt {_felt_word(fr['felt'], r)} mostly because of speed: at about "
                   f"{fmt_kmh(fr['speed_mps'])} km/h you were quick for a surface that is {_road_word(fr['road'])}.")
            out.append(QAItem("speed_cause", _ask(r, "speed_cause", trip.persona, name=nm), True, _cap(ans), ids))
        elif None not in (fc, rc) and rc >= fc and rc >= 2:
            fr, ids = _frame(L, u, ("felt_class", "road_class"))
            ans = f"Mostly the road: {unit_ref(fr, P)} is {_road_word(fr['road'])}, and it felt {_felt_word(fr['felt'], r)}."
            out.append(QAItem("speed_cause", _ask(r, "speed_cause", trip.persona, name=nm), True, ans, ids))

        jolts = [f for f in L.events(u) if f.value["type"] == "jolt" and f.value["maneuver"] in ("accel", "stop_go") and f.conf >= MENTION_THRESHOLD]
        if jolts:
            fr, ids = _frame(L, u, ())
            fr["events"] = []
            fr2, _ = _frame(L, u, (), events=True)
            fr["events"] = [e for e in fr2["events"] if e["id"] in {j.key for j in jolts}]
            Pm = get_persona("commuter")  # maneuver-focused wording
            sents = _event_sentences(fr, Pm, r, report=False)
            ans = _lead(unit_ref(fr, P), sents)
            out.append(QAItem("jolt_cause", _ask(r, "jolt_cause", trip.persona, name=nm), True, ans,
                              ids + [j.fact_id for j in jolts], must_events=[j.key for j in jolts]))

        osm, sig = L.value(u, "osm_surface"), L.value(u, "surface_signal")
        if osm and sig:
            fr, ids = _frame(L, u, ("osm_surface", "surface_signal", "map_signal_disagreement"))
            osm_txt = osm.replace("_", " ")
            if (L.value(u, "map_signal_disagreement") or 0) >= 0.5:
                ans = f"Not quite: {unit_ref(fr, P)} rides like {SURFACE_WORDS[sig]}, although the map lists it as {osm_txt}."
            else:
                ans = f"It looks consistent. On {unit_ref(fr, P)} the map lists it as {osm_txt}, and it rides like {SURFACE_WORDS[sig]}."
            out.append(QAItem("map_check", _ask(r, "map_check", trip.persona, name=nm, osm=osm_txt), True, ans, ids))

        h = L.get(u, "a_v_vs_history")
        if h is not None and h.kind == "CMP":
            fr, ids = _frame(L, u, ("a_v_vs_history", "felt_class"))
            rel = h.value["relation"]
            tail = "about the same as usual" if rel == "similar" else f"{rel} than usual"
            ans = f"{_cap(unit_ref(fr, P))} was {tail} today, and it felt {_felt_word(fr['felt'], r)}."
            out.append(QAItem("history", _ask(r, "history", trip.persona, name=nm), True, ans, ids))
        elif trip.history is None:
            reason = "there are no earlier drives of this route to compare with"
            out.append(QAItem("history_missing", _ask(r, "history_missing", trip.persona, name=nm), False,
                              f"I can't tell that from this drive: {reason}.", [], reason))

        if v is not None:
            fr, ids = _frame(L, u, ("v_mean_mps",))
            out.append(QAItem("speed_on", _ask(r, "speed_on", trip.persona, name=nm), True,
                              f"You averaged about {fmt_kmh(fr['speed_mps'])} km/h on {unit_ref(fr, P)}.", ids))

    for u in L.units:
        miss = L.missing(u)
        if "name" in miss:
            s0, s1 = L.extent(u)
            why = {"mapmatch_failed": "the map match failed there", "no_osm_name": "the map has no street name there",
                   "no_map": "no map data was available for this drive"}[miss["name"]]
            out.append(QAItem("name_missing", _ask(r, "name_missing", trip.persona, a=f"{s0 / 1000:.1f}", b=f"{s1 / 1000:.1f}"),
                              False, f"I can't tell that from this drive: {why}.", [L.get(u, 'name').fact_id], why))

    has_swerve = any(f.value["kind"] == "avoidance" for f in L.facts if f.kind == "EVT")
    oos = [q for q in OUT_OF_SCOPE if ("{name}" not in q[0] or unique) and ("swerve" not in q[0] or has_swerve)]
    q, why = r.choice(oos)
    out.append(QAItem("out_of_scope", q.format(name=name(r.choice(unique)) if unique else ""), False,
                      f"I can't tell that from this drive: {why}.", [], why))
    return out


def qa_record(trip: SyntheticTrip, item: QAItem, qa_id: str, config_hash: str, seed: int) -> Optional[dict]:
    """Verify the gold answer; return the record, or ``None`` if it fails."""
    rep = verify_text(item.answer, trip.ledger, must_mention_events=item.must_events)
    if not rep.passed:
        return None
    P = get_persona(trip.persona)
    return {
        "qa_id": qa_id,
        "split": "dev",
        "persona": trip.persona,
        "e2_test_persona": P.e2_test_persona,
        "capsule_id": trip.capsule.capsule_id,
        "capsule": trip.capsule.text,
        "ledger": format_ledger(trip.ledger, compact=True),
        "question": item.question,
        "question_type": item.qtype,
        "answerable": item.answerable,
        "gold_answer": item.answer,
        "gold_fact_ids": item.fact_ids,
        "abstain_reason": item.abstain_reason,
        "verification": {"passed": rep.passed, "n_claims": rep.n_claims, "p_ledger": rep.p_ledger},
        "generator": {"source": "roadscribe.synth.qa v1 (synthetic, design-time questions)", "seed": seed, "config_hash": config_hash},
    }
