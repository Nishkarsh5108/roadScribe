"""Template NLG pipeline: rule planner, grounding and realizer (conditions T / T+V).

This is the deterministic narrator of METHODOLOGY §2.1-2.4:

* :func:`rule_plan` does content selection per persona (which units and
  facts, relations, hedges) and emits a plan in the §5.3 schema;
* :func:`ground` expands plan items into fact frames whose values come only
  from the ledger;
* :func:`realize` writes text from the frames. With ``vary=True`` (T+V) it
  samples wording and optional content per item from a seeded RNG; each item
  uses its own sub-RNG, so editing one frame leaves the wording of the others
  unchanged (used to build minimal DPO perturbations).

All severity, hedge and event wording comes from :mod:`.lexicon` so that the
claim extractor in :mod:`roadscribe.eval.claims` can check every sentence
against the ledger.
"""
from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from ..schemas import Fact
from .hedging import MENTION_THRESHOLD, hedge_for
from .ledger import LedgerView
from .lexicon import FELT_WORDS, ROAD_WORDS, SURFACE_WORDS, number_word
from .personas import Persona, get_persona

__all__ = ["rule_plan", "ground", "realize", "fmt_len", "fmt_acc", "fmt_kmh", "unit_ref"]

V_FAST = 12.0  # m/s; above this the speed-cause explanation is offered


# --------------------------------------------------------------------------- formatting
def fmt_len(m: float) -> str:
    return f"{int(round(m, -1))} m" if m < 995 else f"{m / 1000:.1f} km"


def fmt_acc(x: float) -> str:
    return f"{x:.1f}" if x >= 0.995 else f"{x:.2f}"


def fmt_kmh(v_mps: float) -> str:
    return f"{int(round(v_mps * 3.6))}"


def _km(s: float) -> str:
    return f"{s / 1000:.1f}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def unit_ref(fr: dict, persona: Persona, with_len: bool = False) -> str:
    """How a unit is named in text: unique name, name + km range, or km range."""
    rng_txt = f"{_km(fr['s0'])}–{_km(fr['s1'])} km"
    name = fr.get("name")
    if name and not fr.get("name_ambiguous") and not persona.always_range:
        out = name
        if with_len and fr.get("length_m") is not None:
            out += f" ({fmt_len(fr['length_m'])})"
        return out
    if name:
        return f"{name} ({rng_txt})"
    return f"the stretch from {_km(fr['s0'])} to {_km(fr['s1'])} km"


# --------------------------------------------------------------------------- planning
def _mentionable(f: Fact) -> bool:
    return f.kind == "EVT" and f.conf >= MENTION_THRESHOLD and f.value.get("label") != "artifact"


def rule_plan(
    ledger: Sequence[Fact],
    selection: dict[str, str],
    persona: str,
    rng: Optional[random.Random] = None,
    must_mention_events: Sequence[str] = (),
) -> dict:
    """Content selection by persona rules. ``rng=None`` gives the deterministic T plan.

    Stretches holding a must-mention event are always described individually,
    even beyond the persona's ``max_units``.
    """
    P = get_persona(persona)
    L = LedgerView(ledger)
    r = rng or random.Random(0)
    coin = (lambda p: r.random() < p) if rng is not None else (lambda p: p >= 0.5)
    visible = [u for u in L.units if selection.get(u, "OMIT") != "OMIT"]
    omit = [u for u in L.units if u not in visible]
    must_units = {L.unit_of_event(e) for e in must_mention_events}
    if len(visible) > P.max_units:
        # keep must-mention stretches, then those with events and the roughest; the rest go to REST
        def score(u):
            evs = [f for f in L.events(u) if _mentionable(f)]
            return (u in must_units, len(evs) > 0, L.value(u, "felt_class", 0), len(evs), -L.units.index(u))
        ranked = sorted(visible, key=score, reverse=True)
        keep = set(ranked[: P.max_units]) | (must_units & set(visible))
        omit += [u for u in visible if u not in keep]
        visible = [u for u in visible if u in keep]
    items: list[dict] = []

    trip = [f.fact_id for f in L.facts_of(None) if f.kind == "MISSING" and f.key in ("gyro", "map_coverage")]
    if P.key != "car_sick_passenger":
        for key in ("trip_duration_s", "trip_length_m"):
            f = L.get(None, key, "NUM")
            if f is not None and (P.numbers or P.key == "commuter"):
                trip.append(f.fact_id)
    if trip:
        items.append({"ref": "TRIP", "facts": trip, "rel": "elaboration", "hedge": "none"})

    prev_planned: Optional[str] = None
    for u in visible:
        evts = [f for f in L.events(u) if _mentionable(f)]
        felt = L.get(u, "felt_class", "SEV")
        if P.key == "car_sick_passenger" and not evts and felt is not None and felt.value <= 1 and coin(0.5):
            omit.append(u)
            continue
        ids = [L.get(u, "extent").fact_id]
        ids += [f.fact_id for f in L.all(u, "name", "ANCHOR")]
        if felt is not None:
            ids.append(felt.fact_id)
        miss = {f.key: f for f in L.facts_of(u) if f.kind == "MISSING"}
        road = L.get(u, "road_class", "SEV")
        v = L.get(u, "v_mean_mps", "NUM")
        speed_cause = (
            road is not None and felt is not None and v is not None
            and felt.value - road.value >= 1 and v.value >= V_FAST
            and P.key != "fleet_manager" and coin(0.8)
        )
        rel, with_ = "elaboration", None
        if speed_cause:
            ids += [road.fact_id, v.fact_id]
            rel = "cause"
        elif road is not None and (P.road_focus or coin(0.2)):
            ids.append(road.fact_id)
        if "road_class" in miss and (P.key == "fleet_manager" or coin(0.7)):
            ids.append(miss["road_class"].fact_id)
        if P.numbers:
            for key, p in (("a_v", 1.0 if P.report_style else 0.4), ("v_mean_mps", 0.6), ("length_m", 1.0 if P.report_style else 0.3)):
                f = L.get(u, key, "NUM")
                if f is not None and f.fact_id not in ids and coin(p):
                    ids.append(f.fact_id)
        shock = L.get(u, "shock", "SEV")
        if shock is not None:
            ids.append(shock.fact_id)
            mt = L.get(u, "mtvv_z", "NUM")
            if P.numbers and mt is not None:
                ids.append(mt.fact_id)
        dis = L.get(u, "map_signal_disagreement", "NUM")
        osm, sig = L.get(u, "osm_surface", "ANCHOR"), L.get(u, "surface_signal", "SEV")
        if dis is not None and dis.value >= 0.5 and osm is not None and sig is not None:
            if coin({"commuter": 0.7, "car_sick_passenger": 0.3}.get(P.key, 1.0)):
                ids += [osm.fact_id, sig.fact_id, dis.fact_id]
        near = L.get(u, "near", "ANCHOR")
        if near is not None and coin(1.0 if P.key in ("commuter", "fleet_manager") else 0.5):
            ids.append(near.fact_id)
        for key in ("name", "anchor"):
            if key in miss and miss[key].note != "no_map":  # the trip-level no-map line covers it
                ids.append(miss[key].fact_id)
        hist = L.get(u, "a_v_vs_history", "CMP")
        if hist is not None and P.key != "car_sick_passenger" and (hist.value["relation"] != "similar" or coin(0.3)):
            ids.append(hist.fact_id)
            if rel == "elaboration":
                rel = "comparison_history"
        cmp_prev = L.get(u, "a_v_vs_prev", "CMP")
        if (
            rel == "elaboration" and cmp_prev is not None and prev_planned == cmp_prev.value["other"]
            and cmp_prev.value["relation"] != "similar" and coin(0.6)
        ):
            ids.append(cmp_prev.fact_id)
            rel, with_ = "contrast", prev_planned
        ids += [f.fact_id for f in evts]
        hedges = [hedge_for(f.conf) for f in evts]
        hedge = "possibly" if "possibly" in hedges else "likely" if "likely" in hedges else "none"
        item = {"ref": u, "facts": ids, "rel": rel, "hedge": hedge}
        if with_:
            item["with"] = with_
        items.append(item)
        prev_planned = u

    omit = [u for u in L.units if u in set(omit)]
    if omit:
        items.append({"ref": "REST", "facts": [], "rel": "elaboration", "hedge": "none"})

    planned = [it["ref"] for it in items if it["ref"] not in ("TRIP", "REST")]
    close: dict[str, Any] = {"type": "none"}
    if planned:
        by_felt = sorted(planned, key=lambda u: (-L.value(u, "felt_class", 0), L.units.index(u)))
        worst = by_felt[0]
        wf = L.value(worst, "felt_class", 0)
        if P.key == "commuter" and wf >= 3:
            close = {"type": "advice_speed", "ref": worst}
        elif P.key == "sports_car" and wf >= 3:
            close = {"type": "advice_pace", "ref": worst}
        elif P.key == "car_sick_passenger" and wf >= 2:
            close = {"type": "advice_gentle", "ref": worst}
        elif P.key == "fleet_manager":
            bad = [u for u in planned if L.get(u, "shock", "SEV") is not None or (L.value(u, "road_class") or 0) >= 4]
            if bad:
                close = {"type": "maintenance", "ref": bad[0]}
    return {"intent": "trip_summary", "persona": persona, "items": items, "omit": omit, "close": close}


# --------------------------------------------------------------------------- grounding
def _unit_frame(L: LedgerView, u: str) -> dict:
    s0, s1 = L.extent(u)
    names = L.names(u)
    name = names[0] if names else None
    return {"ref": u, "name": name, "name_ambiguous": bool(name and len(L.name_units.get(name, [])) > 1), "s0": s0, "s1": s1}


def ground(plan: dict, ledger: Sequence[Fact]) -> list[dict]:
    """Plan -> fact frames. Every value is copied from the ledger (§2.3, Ground)."""
    L = LedgerView(ledger)
    a_v = {u: L.value(u, "a_v") for u in L.units if L.value(u, "a_v") is not None}
    frames: list[dict] = []
    for it in plan["items"]:
        ref = it["ref"]
        facts = [L.by_id[i] for i in it.get("facts", [])]
        if ref == "TRIP":
            fr: dict[str, Any] = {"ref": "TRIP", "fact_ids": [f.fact_id for f in facts], "missing": []}
            for f in facts:
                if f.key == "trip_duration_s":
                    fr["duration_min"] = int(round(f.value / 60))
                elif f.key == "trip_length_m":
                    fr["length_km"] = round(f.value / 1000, 1)
                elif f.kind == "MISSING":
                    fr["missing"].append({"key": f.key, "reason": f.note})
            frames.append(fr)
            continue
        if ref == "REST":
            omitted = plan.get("omit", [])
            felt = [L.value(u, "felt_class") for u in omitted if L.value(u, "felt_class") is not None]
            n_ev = sum(1 for u in omitted for f in L.events(u) if _mentionable(f))
            length = sum(L.value(u, "length_m", 0.0) for u in omitted)
            frames.append({"ref": "REST", "units": list(omitted), "n": len(omitted), "length_m": length,
                           "felt_lo": min(felt) if felt else None, "felt_hi": max(felt) if felt else None,
                           "n_events": n_ev})
            continue
        fr = _unit_frame(L, ref)
        fr.update({"rel": it["rel"], "hedge": it["hedge"], "fact_ids": [f.fact_id for f in facts],
                   "events": [], "missing": [], "with": None})
        if not any(f.key == "name" and f.kind == "ANCHOR" for f in facts):
            fr["name"], fr["name_ambiguous"] = None, False
        for f in facts:
            k = f.key
            if f.kind == "MISSING":
                fr["missing"].append({"key": k, "reason": f.note})
            elif k == "felt_class":
                fr["felt"] = int(f.value)
            elif k == "road_class":
                fr["road"] = int(f.value)
            elif k == "a_v":
                fr["a_v"] = float(f.value)
            elif k == "v_mean_mps":
                fr["speed_mps"] = float(f.value)
            elif k == "length_m":
                fr["length_m"] = float(f.value)
            elif k == "shock":
                fr["shock"] = True
            elif k == "mtvv_z":
                fr["mtvv"] = float(f.value)
            elif k == "osm_surface":
                fr["osm"] = str(f.value)
            elif k == "surface_signal":
                fr["signal"] = str(f.value)
            elif k == "near":
                fr["near"] = f.value["name"]
            elif k == "a_v_vs_history":
                fr["history"] = f.value["relation"]
            elif k == "a_v_vs_prev" and it.get("with"):
                fr["with"] = {**_unit_frame(L, it["with"]), "relation": f.value["relation"]}
            elif f.kind == "EVT":
                v = f.value
                word = {"bump": "bump", "jolt": "jolt", "avoidance": "swerve"}.get(v["type"])
                if word:
                    fr["events"].append({"id": k, "type": word, "p": float(f.conf), "hedge": hedge_for(f.conf),
                                         "s": float(v["s"]), "maneuver": v["maneuver"]})
        fr["speed_cause"] = it["rel"] == "cause" and "road" in fr and "speed_mps" in fr
        if len(a_v) >= 3 and ref in a_v and "felt" in fr:
            if a_v[ref] >= max(a_v.values()):
                fr["rank"] = "roughest"
            elif a_v[ref] <= min(a_v.values()):
                fr["rank"] = "smoothest"
        frames.append(fr)
    close = plan.get("close") or {"type": "none"}
    if close.get("type", "none") != "none" and close.get("ref"):
        cf = _unit_frame(L, close["ref"])
        if not any(fr.get("ref") == close["ref"] and fr.get("name") for fr in frames):
            cf["name"], cf["name_ambiguous"] = None, False
        frames.append({"ref": "CLOSE", "type": close["type"], "unit": cf})
    return frames


# --------------------------------------------------------------------------- realization
def _pick(r: random.Random, options: Sequence[str]) -> str:
    return options[r.randrange(len(options))]


def _felt_word(k: int, r: random.Random) -> str:
    return _pick(r, FELT_WORDS[k])


def _road_word(k: int) -> str:
    return ROAD_WORDS[k][0]


def _positions(evs: list[dict]) -> str:
    xs = [_km(e["s"]) for e in evs]
    if len(xs) == 1:
        return f" around the {xs[0]} km mark"
    return " around the " + ", ".join(xs[:-1]) + f" and {xs[-1]} km marks"


_MAN_TEXT = {"brake": "as you braked", "accel": "pulling away", "stop_go": "stop-and-go driving", "turn": "in a turn"}


def _event_sentences(fr: dict, P: Persona, r: random.Random, report: bool) -> list[str]:
    """One clause per (type, hedge, attribution) group of events."""
    groups: dict[tuple, list[dict]] = {}
    for e in sorted(fr["events"], key=lambda e: e["s"]):
        attrib = None
        if e.get("attrib"):
            attrib = e["attrib"]
        elif e["type"] == "jolt" and e["maneuver"] in ("accel", "stop_go") and P.maneuver_focus:
            attrib = e["maneuver"]
        elif e["type"] == "bump" and e["maneuver"] == "brake" and P.maneuver_focus and r.random() < 0.5:
            attrib = "brake"
        key = (e["type"], e.get("force_hedge") or e["hedge"], attrib, e.get("noun"))
        groups.setdefault(key, []).append(e)
    out: list[str] = []
    show_pos = P.report_style or (P.numbers and r.random() < 0.7) or (not P.numbers and P.key != "car_sick_passenger" and r.random() < 0.4)
    for (typ, hedge, attrib, noun), evs in groups.items():
        n = len(evs)
        pos = _positions(evs) if show_pos else ""
        noun = noun or typ
        nouns = noun + ("s" if n > 1 else "")
        cnt = "a" if n == 1 else number_word(n)
        was = "was" if n == 1 else "were"
        if typ == "swerve":
            what = f"{cnt} {nouns}{pos}"
            if hedge == "assertive":
                fw = f"{evs[0]['force_word']} " if evs[0].get("force_word") else ""
                s = f"there {was} {fw}{what}, around an obstacle"
            elif hedge == "likely":
                s = f"there {was} probably {what}, likely around an obstacle"
            else:
                s = f"there may have been {what}, possibly around an obstacle"
        elif attrib in ("accel", "stop_go"):
            verb = {"assertive": "came from", "likely": "probably came from", "possibly": "may have come from"}[hedge]
            if hedge == "assertive" and evs[0].get("force_word"):
                verb = f"{evs[0]['force_word']} came from"
            s = f"{cnt} {nouns}{pos} {verb} {_MAN_TEXT[attrib]} rather than from the road"
        else:
            adj = _pick(r, ("sharp ", "hard ", "")) if typ == "bump" and hedge == "assertive" else ""
            if hedge == "assertive":
                hw = f"{evs[0]['force_word']} " if evs[0].get("force_word") else ""
                s = f"there {was} {hw}{cnt} {adj}{nouns}{pos}"
            elif hedge == "likely":
                s = _pick(r, (f"there {was} probably {cnt} {nouns}{pos}", f"there {was} likely {cnt} {nouns}{pos}"))
            else:
                s = _pick(r, (f"there may have been {cnt} {nouns}{pos}", f"there was possibly {cnt} {nouns}{pos}" if n == 1 else f"there were possibly {cnt} {nouns}{pos}"))
            if attrib == "brake" or attrib == "turn":
                s += f" {_MAN_TEXT[attrib]}"
        if out and not report and s.startswith("there "):
            w = s.split(" ", 2)  # "there was|were|may ..." -> "there was also ..."
            s = f"{w[0]} {w[1]} also {w[2]}"
        out.append(s if report else _cap(s) + ".")
    return out


def _missing_text(m: dict, report: bool) -> Optional[str]:
    key, why = m["key"], m["reason"]
    if key == "road_class":
        return "surface not rated (speed too low)" if report else "You were moving too slowly there to separate the road from your speed."
    if key == "name":
        if why == "mapmatch_failed":
            return "map match failed, no street name" if report else "The map match failed there, so I have no street name for it."
        return "no street name in the map" if report else "The map has no street name for it."
    if key == "anchor" and why.startswith("gps_gap_"):
        n = why[len("gps_gap_"):].rstrip("s")
        return f"{n}-second GPS gap" if report else f"A {n}-second GPS gap means its exact position is uncertain."
    if key == "gyro":
        return "no gyroscope, steering events not checked" if report else "Without a gyroscope, steering events could not be checked."
    if key == "map_coverage":
        return "no map data" if report else "No map data was available, so places are given as distances."
    return None


def _unit_prose(fr: dict, P: Persona, r: random.Random) -> list[str]:
    ref = unit_ref(fr, P, with_len=P.numbers and "length_m" in fr and r.random() < 0.6)
    sents: list[str] = []
    felt = fr.get("felt")
    fw = _felt_word(felt, r) if felt is not None else None
    speed = f" at about {fmt_kmh(fr['speed_mps'])} km/h" if "speed_mps" in fr else ""
    near = f" near {fr['near']}" if fr.get("near") else ""
    wth = fr.get("with")
    if fr.get("speed_cause") and fw:
        if P.numbers:
            s = f"{ref} felt {fw} mostly because of speed:{speed} you were quick for a surface that is {_road_word(fr['road'])}"
        else:
            s = f"{ref} felt {fw} mostly because of speed; the surface itself is {_road_word(fr['road'])}"
    elif wth and fw and wth["relation"] in ("rougher", "smoother"):
        s = f"{ref} was {wth['relation']} than {unit_ref(wth, P)} and felt {fw}"
    elif fr.get("rank") and fw and r.random() < 0.6:
        s = (f"the roughest stretch was {ref}, which felt {fw}" if fr["rank"] == "roughest"
             else f"{ref} was the smoothest stretch and felt {fw}")
    elif fw:
        if "road" in fr and P.road_focus:
            s = _pick(r, (f"on {ref} the surface is {_road_word(fr['road'])}, and{speed or ' overall'} it felt {fw}",
                          f"{ref} is {_road_word(fr['road'])} and felt {fw}{speed}"))
        else:
            s = _pick(r, (f"{ref} felt {fw}{speed}", f"the ride on {ref} was {fw}{speed}", f"on {ref} the ride was {fw}{speed}"))
            if "road" in fr:
                s += f", and the surface itself is {_road_word(fr['road'])}"
    else:
        s = f"on {ref} nothing stood out"
    if "a_v" in fr:
        s += f" ({fmt_acc(fr['a_v'])} m/s² weighted)"
    sents.append(_cap(s) + ".")
    if near:
        sents.append(f"It runs{near}.")
    if fr.get("shock"):
        if "mtvv" in fr:
            sents.append(f"It was shock-dominated, with a peak 1-s exposure of {fmt_acc(fr['mtvv'])} m/s².")
        else:
            sents.append("It was jarring in places.")
    if fr.get("osm") and fr.get("signal"):
        sents.append(f"It rides like {SURFACE_WORDS[fr['signal']]}, although the map lists it as {fr['osm'].replace('_', ' ')}.")
    if fr.get("history"):
        h = fr["history"]
        sents.append("That is about the same as usual on this route." if h == "similar" else f"That is {h} than usual on this route.")
    sents += _event_sentences(fr, P, r, report=False)
    for m in fr["missing"]:
        t = _missing_text(m, report=False)
        if t:
            sents.append(t)
    return sents


def _unit_report(fr: dict, P: Persona, r: random.Random) -> str:
    head = unit_ref(fr, P)
    if "length_m" in fr:
        head += f", {fmt_len(fr['length_m'])}"
    parts = []
    if fr.get("felt") is not None:
        p = f"ride {_felt_word(fr['felt'], r)}"
        if "a_v" in fr:
            p += f" ({fmt_acc(fr['a_v'])} m/s²)"
        if "speed_mps" in fr:
            p += f" at about {fmt_kmh(fr['speed_mps'])} km/h"
        parts.append(p)
    if "road" in fr:
        parts.append(f"surface {_road_word(fr['road'])}")
    if fr.get("shock"):
        parts.append(f"shock-dominated, peak 1-s exposure {fmt_acc(fr['mtvv'])} m/s²" if "mtvv" in fr else "shock-dominated")
    if fr.get("osm") and fr.get("signal"):
        parts.append(f"the map lists it as {fr['osm'].replace('_', ' ')}, but it rides like {SURFACE_WORDS[fr['signal']]}")
    if fr.get("near"):
        parts.append(f"near {fr['near']}")
    if fr.get("history"):
        h = fr["history"]
        parts.append("the same as usual" if h == "similar" else f"{h} than usual")
    parts += _event_sentences(fr, P, r, report=True)
    for m in fr["missing"]:
        t = _missing_text(m, report=True)
        if t:
            parts.append(t)
    return f"- {head}: " + "; ".join(parts) + "."


def _trip_text(fr: dict, P: Persona, r: random.Random) -> list[str]:
    out = []
    d, km = fr.get("duration_min"), fr.get("length_km")
    if d is not None and km is not None:
        if P.report_style:
            out.append(f"Trip report: {km:.1f} km drive, {d} minutes.")
        elif P.key == "sports_car":
            out.append(f"Your {km:.1f} km run took {d} minutes.")
        else:
            out.append(_pick(r, (f"Here is your {d}-minute, {km:.1f} km drive.", f"Your {km:.1f} km drive took {d} minutes.")))
    elif P.report_style:
        out.append("Trip report:")
    for m in fr["missing"]:
        t = _missing_text(m, report=False)
        if t:
            out.append(t)
    return out


def _rest_text(fr: dict, P: Persona, r: random.Random) -> str:
    n = fr["n"]
    if fr["felt_lo"] is None:
        return ""
    lo, hi = _felt_word(fr["felt_lo"], r), _felt_word(fr["felt_hi"], r)
    ev = "no bumps or jolts" if fr["n_events"] == 0 else f"{number_word(fr['n_events'])} lower-priority event" + ("s" if fr["n_events"] > 1 else "")
    length = f" ({fmt_len(fr['length_m'])})" if P.numbers else ""
    if P.report_style:
        felt = lo if lo == hi or fr["felt_lo"] == fr["felt_hi"] else f"{lo} to {hi}"
        what = "remaining stretch" if n == 1 else f"remaining {number_word(n)} stretches"
        return f"- The {what}{', ' + fmt_len(fr['length_m']) if P.numbers else ''}: ride {felt}; {ev}."
    if n == 1:
        return f"The other stretch{length} was {lo}, with {ev}."
    felt = f"were {lo}" if fr["felt_lo"] == fr["felt_hi"] else f"ranged from {lo} to {hi}"
    return f"The other {number_word(n)} stretches{length} {felt}, with {ev}."


def _close_text(fr: dict, P: Persona) -> str:
    ref = unit_ref(fr["unit"], P)
    t = fr["type"]
    if t == "advice_speed":
        return f"Easing off on {ref} next time should make for a gentler ride."
    if t == "advice_pace":
        return f"{_cap(ref)} is the place to back off; elsewhere the surface supports a brisk pace."
    if t == "advice_gentle":
        return f"Gentler braking and a slower pace on {ref} would help most."
    if t == "maintenance":
        return f"- Action: flag {ref} for a maintenance review."
    return ""


def realize(frames: Sequence[dict], persona: str, seed: int = 0, vary: bool = True) -> str:
    """Fact frames -> narrative. ``vary=False`` is the fixed-wording T condition."""
    P = get_persona(persona)
    lines: list[str] = []
    prose: list[str] = []
    for k, fr in enumerate(frames):
        r = random.Random(f"{seed}:{k}:{fr.get('ref')}") if vary else random.Random(0)
        ref = fr["ref"]
        if ref == "TRIP":
            (lines if P.report_style else prose).extend(_trip_text(fr, P, r))
        elif ref == "REST":
            t = _rest_text(fr, P, r)
            if t:
                (lines if P.report_style else prose).append(t)
        elif ref == "CLOSE":
            t = _close_text(fr, P)
            if t:
                (lines if P.report_style else prose).append(t)
        elif P.report_style:
            lines.append(_unit_report(fr, P, r))
        else:
            prose.extend(_unit_prose(fr, P, r))
    if P.report_style:
        return "\n".join(lines)
    return " ".join(prose)
