"""DPO negatives: minimal, controlled perturbations of verified narratives (METHODOLOGY §8.6, Step 2).

Each perturbation edits the fact frames of a verified output and re-realizes
them with the same seed. Every item keeps its own wording RNG, so the rejected
text differs from the chosen one only where the error was introduced. A pair is
kept only if the verifier rejects the perturbed text for the expected reason.

Types (the §8.6 list plus the three named in the request):

* ``number_drift``           a stated value drifted by 25-60 %
* ``anchor_swap``            two stretches' street names swapped
* ``fabricated_defect``      a bump or pothole invented on a stretch with no defect
* ``dropped_must_mention``   a must-mention event left out
* ``unhedged_low_confidence`` a p < 0.9 event stated with "definitely"
* ``missing_value_supplied`` a value given for a MISSING fact (road class, street name, history)
* ``misattributed_braking``  a bump or jolt attributed to braking that did not happen then
* ``severity_shift``         felt comfort class shifted by two or more
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..eval.claims import VerificationReport, verify_text
from ..layer3.ledger import LedgerView
from ..layer3.templates import realize
from .sampler import ROAD_NAMES, SyntheticTrip
from .teacher import TeacherOutput

__all__ = ["PERTURBATIONS", "EXPECTED_REASONS", "Perturbed", "perturb"]


@dataclass
class Perturbed:
    kind: str
    frames: list[dict]
    text: str
    report: VerificationReport
    edit: dict


def _units(frames):
    return [f for f in frames if f.get("ref", "").startswith("U") and "s0" in f]


def _number_drift(fr, L, trip, r):
    cands = [(f, k) for f in _units(fr) for k in ("a_v", "speed_mps", "length_m", "mtvv") if k in f]
    cands += [(f, k) for f in fr if f.get("ref") == "TRIP" for k in ("duration_min", "length_km") if k in f]
    if not cands:
        return None
    f, k = r.choice(cands)
    factor = r.uniform(1.25, 1.6) if r.random() < 0.5 else r.uniform(0.4, 0.75)
    old = f[k]
    f[k] = type(old)(round(old * factor)) if isinstance(old, int) else old * factor
    if f[k] == old:
        return None
    return {"ref": f["ref"], "field": k, "from": old, "to": f[k]}


def _anchor_swap(fr, L, trip, r):
    named = [f for f in _units(fr) if f.get("name")]
    pairs = [(a, b) for i, a in enumerate(named) for b in named[i + 1:]
             if a["name"] != b["name"] and abs(a.get("felt", 0) - b.get("felt", 0)) >= 2]
    if not pairs:
        return None
    a, b = r.choice(pairs)
    a["name"], b["name"] = b["name"], a["name"]
    a["name_ambiguous"], b["name_ambiguous"] = b["name_ambiguous"], a["name_ambiguous"]
    return {"refs": [a["ref"], b["ref"]], "names": [b["name"], a["name"]]}


def _fabricated_defect(fr, L, trip, r):
    cands = [f for f in _units(fr) if not any(e.value["label"] == "defect" and e.value["kind"] == "impulse" for e in L.events(f["ref"]))]
    if not cands:
        return None
    f = r.choice(cands)
    s = r.uniform(f["s0"] + 20, f["s1"] - 20)
    noun = r.choice(("bump", "pothole"))
    f["events"].append({"id": "FAKE", "type": "bump", "p": 0.95, "hedge": "assertive", "s": s, "maneuver": "cruise", "noun": noun})
    return {"ref": f["ref"], "invented": noun, "s": round(s)}


def _dropped_must_mention(fr, L, trip, r):
    must = set(trip.must_mention_events)
    cands = [(f, e) for f in _units(fr) for e in f["events"] if e["id"] in must]
    if not cands:
        return None
    f, e = r.choice(cands)
    f["events"] = [x for x in f["events"] if x["id"] != e["id"]]
    return {"ref": f["ref"], "dropped": e["id"]}


def _unhedged(fr, L, trip, r):
    cands = [(f, e) for f in _units(fr) for e in f["events"] if e["hedge"] in ("likely", "possibly")]
    if not cands:
        return None
    f, e = r.choice(cands)
    e["force_hedge"] = "assertive"
    e["force_word"] = r.choice(("definitely", "certainly", "clearly"))
    return {"ref": f["ref"], "event": e["id"], "p": e["p"], "word": e["force_word"]}


def _missing_supplied(fr, L, trip, r):
    opts = []
    for f in _units(fr):
        miss = L.missing(f["ref"])
        if "road_class" in miss and "road" not in f:
            opts.append(("road_class", f))
        if "name" in miss and not f.get("name"):
            opts.append(("name", f))
        if "history" in miss and "history" not in f:
            opts.append(("history", f))
    if not opts:
        return None
    key, f = r.choice(opts)
    if key == "road_class":
        f["road"] = r.randint(0, 5)
        val = f["road"]
    elif key == "name":
        known = set(L.name_units)
        f["name"] = r.choice([n for n in ROAD_NAMES if n not in known])
        f["name_ambiguous"] = True  # forces "Name (a-b km)", tying the name to this stretch
        val = f["name"]
    else:
        f["history"] = r.choice(("rougher", "smoother"))
        val = f["history"]
    return {"ref": f["ref"], "missing_key": key, "supplied": val}


def _misattributed_braking(fr, L, trip, r):
    cands = [(f, e) for f in _units(fr) for e in f["events"] if e["type"] in ("bump", "jolt") and e["maneuver"] != "brake"]
    if not cands:
        return None
    f, e = r.choice(cands)
    e["attrib"] = "brake"
    return {"ref": f["ref"], "event": e["id"], "true_maneuver": e["maneuver"]}


def _severity_shift(fr, L, trip, r):
    cands = [f for f in _units(fr) if "felt" in f]
    if not cands:
        return None
    f = r.choice(cands)
    old = f["felt"]
    opts = [k for k in range(6) if abs(k - old) >= 2]
    f["felt"] = r.choice(opts)
    f.pop("rank", None)
    return {"ref": f["ref"], "from": old, "to": f["felt"]}


PERTURBATIONS: dict[str, Callable] = {
    "number_drift": _number_drift,
    "anchor_swap": _anchor_swap,
    "fabricated_defect": _fabricated_defect,
    "dropped_must_mention": _dropped_must_mention,
    "unhedged_low_confidence": _unhedged,
    "missing_value_supplied": _missing_supplied,
    "misattributed_braking": _misattributed_braking,
    "severity_shift": _severity_shift,
}

#: a rejected narrative must fail with at least one of these reasons
EXPECTED_REASONS: dict[str, set[str]] = {
    "number_drift": {"number_mismatch"},
    "anchor_swap": {"severity_mismatch", "wrong_name", "fabricated_event", "event_count", "number_mismatch",
                    "comparison_mismatch", "wrong_anchor", "wrong_map_tag", "wrong_surface", "overclaimed_hedge",
                    "misattributed_maneuver", "wrong_position", "unsupported_cause", "hallucinated_missing"},
    "fabricated_defect": {"fabricated_event"},
    "dropped_must_mention": {"missed_must_mention"},
    "unhedged_low_confidence": {"overclaimed_hedge"},
    "missing_value_supplied": {"hallucinated_missing"},
    "misattributed_braking": {"misattributed_maneuver"},
    "severity_shift": {"severity_mismatch"},
}


def perturb(kind: str, trip: SyntheticTrip, chosen: TeacherOutput, r: random.Random) -> Optional[Perturbed]:
    """Apply one perturbation; ``None`` if it does not apply or the verifier misses it."""
    frames = copy.deepcopy(chosen.frames)
    edit = PERTURBATIONS[kind](frames, LedgerView(trip.ledger), trip, r)
    if edit is None:
        return None
    text = realize(frames, trip.persona, seed=chosen.seed, vary=True)
    if text == chosen.text:
        return None
    rep = verify_text(text, trip.ledger, omitted_units=chosen.plan["omit"], must_mention_events=trip.must_mention_events)
    if rep.passed or not (set(rep.reasons) & EXPECTED_REASONS[kind]):
        return None
    return Perturbed(kind, frames, text, rep, edit)
