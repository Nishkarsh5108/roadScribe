"""Plan schema, validation and per-capsule GBNF grammar (METHODOLOGY §5.3).

A plan says which units and facts to mention, in what order, with which
rhetorical relation and hedge::

    {"intent": "trip_summary",
     "items": [{"ref": "TRIP", "facts": ["F4", "F5"], "rel": "elaboration", "hedge": "none"},
               {"ref": "U2", "facts": ["F18", "F17"], "rel": "elaboration", "hedge": "none"},
               {"ref": "U3", "facts": ["F22"], "rel": "contrast", "with": "U2", "hedge": "possibly"},
               {"ref": "REST", "facts": [], "rel": "elaboration", "hedge": "none"}],
     "omit": ["U4", "U5"],
     "close": {"type": "advice_speed", "ref": "U2"}}

``ref`` is a unit id, ``TRIP`` (trip-level facts) or ``REST`` (the omitted
units in aggregate). Each unit's ``facts`` may only be that unit's ledger ids.
The grammar enumerates one alternative per unit, which keeps the unit-fact
dependency, so plan validity is 100 % by construction under constrained
decoding (llama.cpp GBNF).
"""
from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence

from ..schemas import Fact

__all__ = ["RELATIONS", "HEDGES", "CLOSE_TYPES", "INTENTS", "canonical_json", "validate_plan", "plan_gbnf"]

RELATIONS = ("elaboration", "contrast", "cause", "comparison_history")
HEDGES = ("none", "likely", "possibly")
CLOSE_TYPES = ("none", "advice_speed", "advice_pace", "advice_gentle", "maintenance")
INTENTS = ("trip_summary",)


def canonical_json(plan: dict) -> str:
    """Compact JSON with the key order the grammar expects."""
    items = []
    for it in plan["items"]:
        d = {"ref": it["ref"], "facts": list(it.get("facts", [])), "rel": it["rel"]}
        if it.get("with") is not None:
            d["with"] = it["with"]
        d["hedge"] = it["hedge"]
        items.append(d)
    close = {"type": plan.get("close", {}).get("type", "none")}
    if plan.get("close", {}).get("ref") is not None:
        close["ref"] = plan["close"]["ref"]
    out = {"intent": plan["intent"], "items": items, "omit": list(plan.get("omit", [])), "close": close}
    return json.dumps(out, separators=(",", ":"))


def _facts_by_ref(ledger: Sequence[Fact]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"TRIP": set(), "REST": set()}
    for f in ledger:
        out.setdefault(f.unit_id or "TRIP", set()).add(f.fact_id)
    return out


def validate_plan(plan: dict, ledger: Sequence[Fact], visible_units: Optional[Iterable[str]] = None) -> list[str]:
    """Return a list of violations (empty = valid).

    ``visible_units`` are the units serialised in the capsule (BRIEF/FULL); a
    plan may only describe those individually.
    """
    errs: list[str] = []
    by_ref = _facts_by_ref(ledger)
    units = [r for r in by_ref if r not in ("TRIP", "REST")]
    visible = set(units if visible_units is None else visible_units)
    try:
        json.dumps(plan)
    except (TypeError, ValueError) as e:
        return [f"not JSON-serialisable: {e}"]
    if plan.get("intent") not in INTENTS:
        errs.append(f"bad intent {plan.get('intent')!r}")
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        return errs + ["items must be a non-empty list"]
    seen: set[str] = set()
    for k, it in enumerate(items):
        ref = it.get("ref")
        if ref not in by_ref:
            errs.append(f"item {k}: unknown ref {ref!r}")
            continue
        if ref not in ("TRIP", "REST") and ref not in visible:
            errs.append(f"item {k}: {ref} is omitted from the capsule")
        if ref in seen:
            errs.append(f"item {k}: {ref} planned twice")
        seen.add(ref)
        facts = it.get("facts", [])
        if ref == "REST" and facts:
            errs.append("REST items carry no facts")
        bad = [f for f in facts if f not in by_ref[ref]]
        if bad:
            errs.append(f"item {k}: facts {bad} do not belong to {ref}")
        if len(set(facts)) != len(facts):
            errs.append(f"item {k}: duplicate facts")
        if it.get("rel") not in RELATIONS:
            errs.append(f"item {k}: bad rel {it.get('rel')!r}")
        if it.get("hedge") not in HEDGES:
            errs.append(f"item {k}: bad hedge {it.get('hedge')!r}")
        w = it.get("with")
        if w is not None and (w not in visible or w == ref):
            errs.append(f"item {k}: bad 'with' {w!r}")
    omit = plan.get("omit", [])
    if not set(omit) <= set(units):
        errs.append(f"omit has unknown units {sorted(set(omit) - set(units))}")
    if set(omit) & seen:
        errs.append(f"units both planned and omitted: {sorted(set(omit) & seen)}")
    if "REST" in seen and not omit:
        errs.append("REST item without omitted units")
    close = plan.get("close", {"type": "none"})
    if close.get("type") not in CLOSE_TYPES:
        errs.append(f"bad close type {close.get('type')!r}")
    if close.get("ref") is not None and close["ref"] not in visible:
        errs.append(f"close ref {close['ref']!r} is not a visible unit")
    return errs


def _alt(xs: Iterable[str]) -> str:
    return " | ".join(f'"\\"{x}\\""' for x in xs)


def plan_gbnf(ledger: Sequence[Fact], visible_units: Sequence[str]) -> str:
    """GBNF grammar for compact plans over this capsule's ledger ids.

    Keys appear in the order of :func:`canonical_json` without whitespace:
    intent, items (ref, facts, rel, optional with, hedge), omit, close
    (type, optional ref).
    """
    by_ref = _facts_by_ref(ledger)
    rules = [
        'root ::= "{\\"intent\\":\\"trip_summary\\",\\"items\\":[" item ("," item)* "],\\"omit\\":[" omitlist? "],\\"close\\":" close "}"',
        f"rel ::= {_alt(RELATIONS)}",
        f"hedge ::= {_alt(HEDGES)}",
        'tail ::= ",\\"rel\\":" rel (",\\"with\\":" vunit)? ",\\"hedge\\":" hedge "}"',
    ]
    item_alts = []
    for ref in ["TRIP", *visible_units]:
        name = "f" + ref.lower()
        ids = sorted(by_ref.get(ref, ()), key=lambda x: int(x[1:]))
        if ids:
            rules.append(f"{name} ::= {_alt(ids)}")
            item_alts.append(f'"{{\\"ref\\":\\"{ref}\\",\\"facts\\":[" ({name} ("," {name})*)? "]" tail')
        else:
            item_alts.append(f'"{{\\"ref\\":\\"{ref}\\",\\"facts\\":[]" tail')
    item_alts.append('"{\\"ref\\":\\"REST\\",\\"facts\\":[]" tail')
    rules.append("item ::= " + " | ".join(item_alts))
    units = [r for r in by_ref if r not in ("TRIP", "REST")]
    unit_alt = _alt(units) if units else '"\\"-\\""'
    ref_alt = _alt(visible_units) if visible_units else '"null"'
    rules.append("unit ::= " + unit_alt)
    rules.append("vunit ::= " + ref_alt)
    rules.append('omitlist ::= unit ("," unit)*')
    rules.append('close ::= "{\\"type\\":" (' + _alt(CLOSE_TYPES) + ') (",\\"ref\\":" (' + ref_alt + '))? "}"')
    return "\n".join(rules) + "\n"
