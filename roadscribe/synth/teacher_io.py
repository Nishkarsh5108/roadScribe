"""Interactive teacher: trip sheets out, verified narratives and answers in.

A frontier model (the "teacher" of METHODOLOGY §8.6) reads a compact *sheet*
for each synthetic trip and writes a narrative, or a driver question and
answer, in its own words. Claim-bearing words (comfort classes, road
condition, hedges, event mentions, positions, numbers) must come from the
fixed lexicon so that :func:`roadscribe.eval.claims.verify_text` can check
every claim; everything else (selection, ordering, aggregation, explanation,
advice, tone) is free.

``ingest`` verifies each narrative, derives the plan from the verified
claims (stretches in order of first mention, the ledger facts that support
each claim, relations and hedges), validates the plan against the capsule's
grammar, grounds it into fact frames and writes training records. Outputs
that fail are reported, never repaired automatically.

    python -m roadscribe.synth.teacher_io sheet --kind sft --start 0 --n 15
    python -m roadscribe.synth.teacher_io sheet --kind qa --start 0 --n 15
    python -m roadscribe.synth.teacher_io ingest
    python -m roadscribe.synth.teacher_io exemplars

Raw teacher text lives in ``data/processed/teacher/{sft,qa}_*.txt`` as
blocks headed ``=== <id>``; QA blocks hold a ``Q:`` and an ``A:`` line.
``teacher/first_pass.json`` records how many narratives of each raw file
passed before any revision, and ``teacher/exemplars.txt`` lists the frozen
few-shot exemplars (§8.6 Step 0) that ``exemplars`` proposed. ``ingest``
also writes ``TEACHER_CARD.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ..eval.claims import VerificationReport, verify_text
from ..layer3.hedging import MENTION_THRESHOLD, hedge_for
from ..layer3.ledger import LedgerView, format_ledger
from ..layer3.lexicon import SURFACE_WORDS
from ..layer3.plan_grammar import canonical_json, validate_plan
from ..layer3.templates import ground
from .sampler import SynthConfig, SyntheticTrip, sample_trip
from .teacher import visible_units

__all__ = ["TEACHER_PERSONAS", "teacher_trip", "sheet", "derive_plan", "ingest", "propose_exemplars", "main"]

TEACHER_PERSONAS = ("commuter", "sports_car")  # never the E2 test personas
TEACHER_ID = "frontier-llm, interactive session, lexicon-constrained free composition v1"
DATA = Path("data/processed")
RAW = DATA / "teacher"
_MIX = ["narrative"] * 8 + ["degraded"] * 7 + ["history"] * 5  # 40 / 35 / 25 %

_ACK = {
    "road_class": re.compile(r"too slow|too slowly|speed (?:was )?too low|separate the road from (?:your |the )?speed", re.I),
    "name": re.compile(r"street name|map match|no name", re.I),
    "history": re.compile(r"no earlier drive|first recorded drive|no history|nothing to compare", re.I),
    "gyro": re.compile(r"gyroscope|\bgyro\b", re.I),
    "map_coverage": re.compile(r"no map data|without map data|no map\b", re.I),
}
_ADVICE = re.compile(r"\b(ease off|easing off|slow down|slower|back off|take it easy|gentler|brisk pace|lift off)\b", re.I)
# one few-shot exemplar per cell: both personas, all three task mixes
EXEMPLAR_CELLS = (("commuter", "narrative"), ("sports_car", "degraded"), ("commuter", "history"))


def teacher_trip(kind: str, i: int) -> SyntheticTrip:
    """Trip ``i`` of the teacher streams (disjoint from the SFT/DPO/QA build streams)."""
    cfg = SynthConfig(personas=TEACHER_PERSONAS)
    persona = TEACHER_PERSONAS[i % len(TEACHER_PERSONAS)]
    regime = (i // len(TEACHER_PERSONAS)) % 6
    seed, base = ("teacher", 400000) if kind == "sft" else ("teacher-qa", 450000)
    return sample_trip(random.Random(f"{seed}:{i}"), base + i, cfg, persona=persona, regime=regime,
                       task_mix=_MIX[i % len(_MIX)])


def _rid(kind: str, i: int) -> str:
    return f"syn-{'t' if kind == 'sft' else 'q'}{i:04d}"


def _parse_id(rid: str) -> tuple[str, int]:
    m = re.fullmatch(r"syn-([tq])(\d{4})", rid)
    if not m:
        raise ValueError(f"bad id {rid!r}")
    return ("sft" if m.group(1) == "t" else "qa"), int(m.group(2))


# --------------------------------------------------------------------------- sheets
def _unit_line(L: LedgerView, u: str, trip: SyntheticTrip) -> list[str]:
    s0, s1 = L.extent(u)
    names = L.names(u)
    ambig = bool(names and len(L.name_units.get(names[0], [])) > 1)
    ref = f'"{names[0]}"' + (" AMBIG->use range" if ambig else "") if names else "(unnamed: use range)"
    v = L.value(u, "v_mean_mps")
    parts = [f"{u} {s0 / 1000:.1f}-{s1 / 1000:.1f}km {ref} len {L.value(u, 'length_m'):.0f}m",
             f"felt {L.value(u, 'felt_class')} (a_v {L.value(u, 'a_v'):.2f})"]
    rc = L.value(u, "road_class")
    parts.append(f"road {rc} ({L.value(u, 'a_wz_dec'):.2f})" if rc is not None else "road MISSING")
    if v is not None:
        parts.append(f"{v * 3.6:.0f} km/h")
    if L.get(u, "shock", "SEV") is not None:
        mt = L.value(u, "mtvv_z")
        parts.append(f"SHOCK mtvv {mt:.2f}" if mt is not None else "SHOCK")
    osm, sig, dis = L.value(u, "osm_surface"), L.value(u, "surface_signal"), L.value(u, "map_signal_disagreement")
    if osm or sig:
        parts.append(f"osm {osm or '-'} / rides like {SURFACE_WORDS.get(sig, sig) if sig else '-'}" + (f" dis {dis:.2f}" if dis is not None else ""))
    near = L.all(u, "near", "ANCHOR")
    if near:
        parts.append("near " + ", ".join(f.value["name"] for f in near))
    prev = L.value(u, "a_v_vs_prev")
    if prev and prev["relation"] != "similar":
        parts.append(f"{prev['relation']} than {prev['other']}")
    hist = L.get(u, "a_v_vs_history")
    if hist is not None:
        parts.append(f"history {hist.value['relation']}" if hist.kind == "CMP" else "history MISSING")
    miss = {k: v_ for k, v_ in L.missing(u).items() if k not in ("map_signal_disagreement", "history")}
    if miss:
        parts.append("MISSING " + ", ".join(f"{k}({v_})" for k, v_ in miss.items()))
    lines = [" | ".join(parts)]
    evs = []
    for f in L.events(u):
        ev = f.value
        if ev["label"] == "artifact":
            evs.append(f"{f.key} artifact p{f.conf:.2f} {ev['maneuver']} (driving, not a road event)")
            continue
        h = hedge_for(f.conf)
        tag = h if h else "p<.4 DO NOT MENTION"
        word = "swerve" if ev["kind"] == "avoidance" else ev["type"]
        evs.append(f"{f.key} {word} p{f.conf:.2f} [{tag}] @{ev['s'] / 1000:.3f}km {ev['maneuver']}"
                   + (" MUST" if f.key in trip.must_mention_events else ""))
    if evs:
        lines.append("    " + " ; ".join(evs))
    return lines


def sheet(kind: str, i: int) -> str:
    """Compact view of one teacher trip (what the teacher reads)."""
    t = teacher_trip(kind, i)
    L = LedgerView(t.ledger)
    vis = visible_units(t) if kind == "sft" else list(L.units)
    trip_miss = L.missing(None)
    head = (f"=== {_rid(kind, i)} | {t.persona} | B={t.capsule.budget} | {t.task_mix}"
            + (f" {t.patterns}" if t.patterns else "")
            + f" | {L.value(None, 'trip_duration_s') / 60:.0f} min, {L.value(None, 'trip_length_m') / 1000:.1f} km"
            + (" | MISSING " + ", ".join(f"{k}({v})" for k, v in trip_miss.items()) if trip_miss else ""))
    out = [head]
    for u in vis:
        out += _unit_line(L, u, t)
    om = [u for u in L.units if u not in vis]
    if om:
        felt = [L.value(u, "felt_class") for u in om]
        n_ev = sum(1 for u in om for f in L.events(u) if f.conf >= MENTION_THRESHOLD and f.value["label"] != "artifact")
        out.append(f"REST {len(om)} units ({', '.join(om)}), {sum(L.value(u, 'length_m') for u in om):.0f} m, "
                   f"felt {min(felt)}-{max(felt)}, {n_ev} mentionable events")
    return "\n".join(out)


# --------------------------------------------------------------------------- plan derivation
def derive_plan(trip: SyntheticTrip, text: str, rep: VerificationReport, visible: list[str]) -> dict:
    """Plan implied by a verified narrative: stretches in order of first mention and the
    ledger facts behind each supported claim."""
    L = LedgerView(trip.ledger)
    first: dict[str, int] = {}
    facts: dict[str, list[str]] = defaultdict(list)
    flags: dict[str, set] = defaultdict(set)
    with_: dict[str, str] = {}
    hedges: dict[str, set] = defaultdict(set)

    def touch(ref: str, pos: int) -> None:
        first.setdefault(ref, pos)

    def add(ref: str, f) -> None:
        if f is not None and f.fact_id not in facts[ref]:
            facts[ref].append(f.fact_id)

    for c in sorted(rep.claims, key=lambda c: c.span):
        if not c.supported:
            continue
        u = c.unit_id
        if c.kind == "REST":
            touch("REST", c.span[0])
            continue
        if u is None:
            if c.key in ("trip_duration_s", "trip_length_m"):
                touch("TRIP", c.span[0])
                add("TRIP", L.get(None, c.key))
            continue
        touch(u, c.span[0])
        k = c.key
        if c.kind == "NAME":
            add(u, next((f for f in L.all(u, "name", "ANCHOR") if f.value == c.value), None))
        elif c.kind == "RANGE":
            add(u, L.get(u, "extent"))
        elif c.kind == "SEV":
            add(u, L.get(u, k))
        elif c.kind == "NUM":
            add(u, L.get(u, k))
        elif c.kind == "CMP" and k == "speed_cause":
            for kk in ("felt_class", "road_class", "v_mean_mps"):
                add(u, L.get(u, kk))
            flags[u].add("cause")
        elif c.kind == "CMP" and k == "superlative":
            add(u, L.get(u, "a_v"))
        elif c.kind == "CMP" and k == "a_v_vs_unit":
            add(u, L.get(u, "a_v"))
            prev = L.get(u, "a_v_vs_prev", "CMP")
            if prev is not None and prev.value["other"] == c.value["other"]:
                add(u, prev)
            if c.value["other"]:
                with_[u] = c.value["other"]
            flags[u].add("contrast")
        elif c.kind == "CMP" and k == "a_v_vs_history":
            add(u, L.get(u, k))
            flags[u].add("history")
        elif c.kind == "ANCHOR" and k == "near":
            add(u, next((f for f in L.all(u, "near", "ANCHOR") if f.value["name"] == c.value), None))
        elif c.kind == "ANCHOR":
            add(u, L.get(u, k))
        elif c.kind == "MISSING_ACK":
            add(u, L.get(u, "anchor") if L.get(u, "anchor") is not None else L.get(None, "anchor"))
        elif c.kind == "EVT" and k == "no_events":
            add(u, L.get(u, "n_events"))
        elif c.kind == "EVT":
            add(u, next((f for f in L.events(u) if f.key == c.value["event"]), None))
            hedges[u].add(c.hedge)
    # acknowledgements of missing evidence (sentences, not claims)
    for a, b, su in rep.sentence_units:
        sent = text[a:b]
        ref = su if su not in (None, "TRIP", "REST") else "TRIP"
        owner = None if ref == "TRIP" else ref
        for key, rx in _ACK.items():
            f = L.get(owner, key)
            if f is not None and f.kind == "MISSING" and rx.search(sent):
                touch(ref, a)
                add(ref, f)
    close = {"type": "none"}
    for a, b, su in rep.sentence_units:
        if su in visible and _ADVICE.search(text[a:b]):
            close = {"type": "advice_pace" if trip.persona == "sports_car" else "advice_speed", "ref": su}
    items = []
    for ref in sorted(first, key=first.get):
        if ref in ("TRIP", "REST"):
            items.append({"ref": ref, "facts": [] if ref == "REST" else facts[ref], "rel": "elaboration", "hedge": "none"})
            continue
        ids = facts[ref]
        ext = L.get(ref, "extent")
        if ext.fact_id not in ids:
            ids.insert(0, ext.fact_id)
        rel = ("cause" if "cause" in flags[ref] else "contrast" if "contrast" in flags[ref]
               else "comparison_history" if "history" in flags[ref] else "elaboration")
        h = hedges[ref]
        item = {"ref": ref, "facts": ids, "rel": rel,
                "hedge": "possibly" if "possibly" in h else "likely" if "likely" in h else "none"}
        if rel == "contrast" and with_.get(ref) in visible and with_[ref] != ref:
            item["with"] = with_[ref]
        items.append(item)
    planned = {it["ref"] for it in items}
    omit = [u for u in L.units if u not in planned]
    return {"intent": "trip_summary", "persona": trip.persona, "items": items, "omit": omit, "close": close}


# --------------------------------------------------------------------------- ingest
def _blocks(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    cur: Optional[str] = None
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^=== (syn-[tq]\d{4})\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def _split(rid: str) -> str:
    return "dev" if int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.1 else "train"


def _failure(rid: str, text: str, rep: VerificationReport, errs: list[str]) -> str:
    lines = [f"FAIL {rid}: {rep.reasons} {errs}"]
    for c in rep.claims:
        if not c.supported:
            lines.append(f"    {c.kind} {c.unit_id} {c.key} -> {c.reason}: {text[c.span[0]:c.span[1]]!r}")
    if rep.must_mention_missed:
        lines.append(f"    missed must-mention {rep.must_mention_missed}")
    return "\n".join(lines)


def _id_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.split("#")[0].strip()]


def _stats(xs: list[float]) -> str:
    if not xs:
        return "-"
    s = sorted(xs)
    return (f"mean {sum(s) / len(s):.1f}, median {s[len(s) // 2]:.0f}, "
            f"p95 {s[min(len(s) - 1, int(0.95 * len(s)))]:.0f}, max {s[-1]:.0f}")


def _card(sft: list[dict], qa: list[dict], ex: list[dict], first_pass: dict) -> str:
    n_fp = sum(v["first_pass"] for v in first_pass.values())
    n_all = sum(v["n"] for v in first_pass.values())
    fp_rows = "\n".join(f"| `{k}` | {v['first_pass']} / {v['n']} | {v.get('note', '')} |" for k, v in sorted(first_pass.items()))
    degr = Counter(p for r in sft for p in r["degradations"])
    abst = sum(not r["answerable"] for r in qa)
    return f"""# RoadScribe frontier-teacher data

Written by `python -m roadscribe.synth.teacher_io ingest` from the raw teacher files in
`data/processed/teacher/`. Edit the raw files and re-run `ingest`; do not edit this card.

| File | Records | Use |
|---|---|---|
| `roadscribe_teacher_sft.jsonl` | {len(sft)} | capsule → plan → narrative written by a frontier-model teacher |
| `roadscribe_teacher_qa.jsonl` | {len(qa)} | driver question → teacher answer or abstention |
| `roadscribe_teacher_exemplars.jsonl` | {len(ex)} | frozen few-shot exemplars for §8.6 Step 0 |

Records have the fields of `roadscribe_sft.jsonl` plus `"source": "teacher"`; `generator.teacher` is
`{TEACHER_ID}`.

## How the data were made

1. **Trips.** `teacher_trip(kind, i)` draws trip *i* with the same sampler and ledger code as the template
   data, on its own seed stream (`teacher:i`, `teacher-qa:i`), and only for the development personas
   {" and ".join(TEACHER_PERSONAS)}. The task mix cycles 40 % narrative, 35 % degraded and 25 % history.
2. **Sheets.** `teacher_io sheet` prints each capsule as a trip sheet. For every visible stretch it gives the
   name (or the range, when the name is ambiguous or missing), length, felt and road class with a_v and
   a_wz, speed, shock flag and MTVV, map tag and signal surface, nearby places, the comparison with the
   previous stretch, history, MISSING fields, and every event with its hedge band, position, maneuver
   and must-mention flag. Events below p = 0.4 and maneuver artifacts are marked as not to be mentioned. A
   REST line summarises the stretches the budget left out.
3. **Teacher.** A frontier LLM, working in an interactive session, wrote one narrative per sheet. It chose
   sentence structure, order, emphasis, aggregation, comparisons and closing advice freely. For every
   claim it used the closed lexicon of `roadscribe.layer3.lexicon` (felt and road classes, hedges,
   maneuver phrases, event nouns and counts, positions), so the verifier can check every claim.
4. **Verification and plans.** `ingest` verifies each narrative against the full claim ledger, with the
   must-mention events and the REST units. It derives the plan implied by the supported claims: stretches
   in order of first mention, the ledger facts behind each claim, acknowledgements of MISSING fields and
   the closing advice. It then re-verifies, validates the plan against the plan grammar and grounds it
   into frames. A narrative that fails is revised by the teacher and checked again; the checks are never
   relaxed to let a narrative through.

**First-pass acceptance:** {n_fp} of {n_all} outputs, narratives and answers ({100 * n_fp / max(n_all, 1):.1f} %), passed
before any revision. After revision all pass. Before the first batch, trial narratives exposed two verifier gaps
(coordinated subjects, and claims in compound sentences). Both were fixed in the verifier, with
regression tests, before any narrative was counted.

| Raw file | First pass | Note |
|---|---|---|
{fp_rows}

## Read before training or evaluating

1. **Terms of use.** These narratives are model outputs. Check the model provider's terms on using outputs
   to train another model before training on them or releasing them (§8.6).
2. **Spot check.** §8.6 asks for 200 outputs to be spot-checked by hand. That has not been done yet.
3. **E2-safe.** Only development personas appear (`e2_test_persona` is false everywhere), so E2's held-out
   personas stay unseen.
4. **Not C-PGR.** C-PGR, the cloud-model upper reference, must be run at evaluation time on test capsules
   under the same prompt and protocol as the SLM. These training narratives cannot stand in for it. If the
   same model family later serves as C-PGR or as a judge, say so, because a judge can prefer text written
   in its own style.
5. **Lexicon-constrained.** Sentence structure is free but the claim vocabulary is closed. This is not the
   unconstrained free text that §6.1's LLM claim extractor is meant to handle.
6. **Baselines.** The SLM is trained on this data and on the template data; T+Intent sees neither. When
   comparing them, report the teacher data as part of the SLM's training budget.
7. **QA questions.** Teacher QA questions are written by the designer, not elicited from drivers. Keep them
   out of E1, as with `roadscribe_qa_bank.jsonl`.
8. **Exemplars.** The exemplars come from the train split. Freeze them, with the prompt, before any sealed
   set is touched (§8.6 leakage controls).

## Composition

| | |
|---|---|
| SFT personas | {dict(Counter(r["persona"] for r in sft))} |
| SFT task mix | {dict(Counter(r["task_mix"] for r in sft))} |
| SFT degradations | {dict(degr)} |
| SFT split | {dict(Counter(r["split"] for r in sft))} |
| Budgets | {dict(sorted(Counter(r["budget"] for r in sft).items()))} |
| Worst felt class per trip | {dict(sorted(Counter(r["regime_max_felt_class"] for r in sft).items()))} |
| Narrative words | {_stats([len(r["narrative"].split()) for r in sft])} |
| Claims per narrative | {_stats([r["verification"]["n_claims"] for r in sft])} |
| Event mentions per narrative | {_stats([r["verification"]["event_mentions"] for r in sft])} |
| QA answerable / abstain | {len(qa) - abst} / {abst} |
| Exemplars | {", ".join(f"`{r['id']}` ({r['persona']}, {r['task_mix']})" for r in ex) or "-"} |
"""


def ingest(raw: Path = RAW, out: Path = DATA, quiet: bool = False) -> dict:
    sft_rows, qa_rows, fails = [], [], []
    for path in sorted(raw.glob("sft_*.txt")):
        for rid, text in _blocks(path).items():
            kind, i = _parse_id(rid)
            t = teacher_trip(kind, i)
            vis = visible_units(t)
            rep = verify_text(text, t.ledger, must_mention_events=t.must_mention_events,
                              omitted_units=[u for u in LedgerView(t.ledger).units if u not in vis])
            plan = derive_plan(t, text, rep, vis)
            rep = verify_text(text, t.ledger, must_mention_events=t.must_mention_events, omitted_units=plan["omit"])
            errs = validate_plan(plan, t.ledger, vis) if rep.passed else []
            if not rep.passed or errs:
                fails.append(_failure(rid, text, rep, errs))
                continue
            sft_rows.append({
                "id": rid, "split": _split(rid), "source": "teacher", "persona": t.persona, "e2_test_persona": False,
                "intent": "trip_summary", "task_mix": t.task_mix, "degradations": t.patterns,
                "regime_max_felt_class": t.regime, "budget": t.capsule.budget, "budget_effective": t.kst.budget_effective,
                "tokenizer": t.capsule.tokenizer, "capsule_id": t.capsule.capsule_id, "capsule": t.capsule.text,
                "ledger": format_ledger(t.ledger, [None, *vis], compact=True),
                "must_mention_events": t.must_mention_events, "plan": canonical_json(plan),
                "frames": ground(plan, t.ledger), "narrative": text,
                "verification": {"passed": True, "plan_valid": True, "n_claims": rep.n_claims, "p_ledger": rep.p_ledger,
                                 "hallucinated_missing": 0, "event_mentions": rep.event_mentions, "must_mention_missed": []},
                "generator": {"teacher": TEACHER_ID, "trip": f"teacher:{i}"},
            })
    for path in sorted(raw.glob("qa_*.txt")):
        for rid, block in _blocks(path).items():
            kind, i = _parse_id(rid)
            t = teacher_trip(kind, i)
            q = re.search(r"^Q:\s*(.+)$", block, re.M)
            a = re.search(r"^A:\s*(.+(?:\n(?!Q:).+)*)", block, re.M)
            if not q or not a:
                fails.append(f"FAIL {rid}: needs a Q: and an A: line")
                continue
            ans = a.group(1).strip()
            rep = verify_text(ans, t.ledger)
            if not rep.passed:
                fails.append(_failure(rid, ans, rep, []))
                continue
            abst = ans.startswith("I can't tell")
            fids = sorted({fid for it in derive_plan(t, ans, rep, list(LedgerView(t.ledger).units))["items"] for fid in it["facts"]},
                          key=lambda x: int(x[1:]))
            qa_rows.append({
                "qa_id": rid, "split": "dev", "source": "teacher", "persona": t.persona, "e2_test_persona": False,
                "capsule_id": t.capsule.capsule_id, "capsule": t.capsule.text, "ledger": format_ledger(t.ledger, compact=True),
                "question": q.group(1).strip(), "question_type": "free", "answerable": not abst, "gold_answer": ans,
                "gold_fact_ids": fids, "abstain_reason": ans.split(":", 1)[1].strip().rstrip(".") if abst and ":" in ans else None,
                "verification": {"passed": True, "n_claims": rep.n_claims, "p_ledger": rep.p_ledger},
                "generator": {"teacher": TEACHER_ID, "trip": f"teacher-qa:{i}"},
            })
    by_id = {r["id"]: r for r in sft_rows}
    ex_rows = []
    for rid in _id_list(raw / "exemplars.txt"):
        if rid not in by_id or by_id[rid]["split"] != "train":
            fails.append(f"FAIL exemplar {rid}: not a verified train record")
        else:
            ex_rows.append(by_id[rid])
    for name, rows in (("roadscribe_teacher_sft.jsonl", sft_rows), ("roadscribe_teacher_qa.jsonl", qa_rows),
                       ("roadscribe_teacher_exemplars.jsonl", ex_rows)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows if name.endswith("exemplars.jsonl") else sorted(rows, key=lambda r: r.get("id") or r["qa_id"]):
                fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    first_pass = json.loads((raw / "first_pass.json").read_text(encoding="utf-8")) if (raw / "first_pass.json").exists() else {}
    (out / "TEACHER_CARD.md").write_text(_card(sft_rows, qa_rows, ex_rows, first_pass), encoding="utf-8")
    summary = {"sft": len(sft_rows), "qa": len(qa_rows), "failed": len(fails),
               "sft_mix": dict(Counter(r["task_mix"] for r in sft_rows)),
               "sft_persona": dict(Counter(r["persona"] for r in sft_rows)),
               "qa_answerable": sum(r["answerable"] for r in qa_rows), "exemplars": [r["id"] for r in ex_rows]}
    if not quiet:
        for f in fails:
            print(f)
        print(json.dumps(summary))
    return summary


def propose_exemplars(rows: list[dict], target_words: int = 130, max_words: int = 200) -> list[dict]:
    """One train-split narrative of at most ``max_words`` words per :data:`EXEMPLAR_CELLS`
    cell (short enough for a KV-cached few-shot prompt): the one that shows the most
    kinds of claim, relation and hedge, then the one closest to ``target_words``."""
    out = []
    for persona, mix in EXEMPLAR_CELLS:
        best = None
        for r in rows:
            if (r["split"] != "train" or r["persona"] != persona or r["task_mix"] != mix
                    or len(r["narrative"].split()) > max_words):
                continue
            kind, i = _parse_id(r["id"])
            t = teacher_trip(kind, i)
            plan = json.loads(r["plan"])
            rep = verify_text(r["narrative"], t.ledger, must_mention_events=t.must_mention_events, omitted_units=plan["omit"])
            feats = ({c.kind for c in rep.claims} | {f"hedge:{c.hedge}" for c in rep.claims if c.kind == "EVT"}
                     | {f"rel:{it['rel']}" for it in plan["items"]} | {f"close:{plan['close']['type']}"})
            words = len(r["narrative"].split())
            key = (len(feats), -abs(words - target_words), r["id"])
            if best is None or key > best["key"]:
                best = {"key": key, "id": r["id"], "persona": persona, "task_mix": mix, "words": words, "features": sorted(feats)}
        if best is not None:
            best.pop("key")
            out.append(best)
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--kind", choices=("sft", "qa"), default="sft")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--n", type=int, default=10)
    sub.add_parser("ingest")
    sub.add_parser("exemplars")
    args = ap.parse_args(argv)
    logging.getLogger("roadscribe.layer3.kst").setLevel(logging.WARNING)
    if args.cmd == "sheet":
        print("\n".join(sheet(args.kind, i) for i in range(args.start, args.start + args.n)))
    elif args.cmd == "exemplars":
        with open(DATA / "roadscribe_teacher_sft.jsonl", encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh]
        for e in propose_exemplars(rows):
            print(json.dumps(e))
    else:
        ingest()


if __name__ == "__main__":  # pragma: no cover
    main()
