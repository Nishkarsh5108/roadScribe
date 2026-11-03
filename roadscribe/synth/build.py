"""Build the synthetic distillation, DPO and QA datasets.

    python -m roadscribe.synth.build --out data/processed

Writes ``roadscribe_sft.jsonl``, ``roadscribe_dpo.jsonl``,
``roadscribe_qa_bank.jsonl``, ``manifest.json`` and ``DATASET_CARD.md``.
Every record is reproducible from (seed, stream, index): trips are sampled
from ``random.Random(f"{seed}:{stream}:{index}")``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

from .. import __version__
from ..config import RoadScribeConfig, config_hash
from ..layer3.personas import PERSONAS, get_persona
from ..layer3.plan_grammar import canonical_json
from .perturb import PERTURBATIONS, perturb
from .qa import ANSWERABLE_TYPES, UNANSWERABLE_TYPES, candidate_questions, qa_record
from .sampler import SynthConfig, sample_trip
from .teacher import TEACHER_ID, prompt_ledger, sft_record, teach

__all__ = ["build_sft", "build_dpo", "build_qa", "main"]

log = logging.getLogger("roadscribe.synth.build")


def _quota(total: int, keys) -> dict[str, int]:
    keys = list(keys)
    base, extra = divmod(total, len(keys))
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(keys)}


def _split(record_id: str, dev_frac: float) -> str:
    h = int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_frac else "train"


def build_sft(n: int, seed: int, personas, cfg: SynthConfig, chash: str, dev_frac: float = 0.05):
    records, stats = [], Counter()
    i = 0
    while len(records) < n:
        persona = personas[len(records) % len(personas)]
        regime = (len(records) // len(personas)) % 6
        trip = sample_trip(random.Random(f"{seed}:sft:{i}"), i, cfg, persona=persona, regime=regime)
        out = teach(trip, seed=i)
        stats["attempted"] += 1
        if out.ok:
            rid = f"sft-{len(records):05d}"
            records.append(sft_record(trip, out, rid, _split(rid, dev_frac), chash))
        else:
            stats["rejected_by_filter"] += 1
        i += 1
    return records, stats


def build_dpo(n: int, seed: int, personas, cfg: SynthConfig, chash: str):
    need = _quota(n, PERTURBATIONS)
    records, stats = [], Counter()
    j = 0
    while sum(need.values()) > 0:
        persona = personas[j % len(personas)]
        trip = sample_trip(random.Random(f"{seed}:dpo:{j}"), 100000 + j, cfg, persona=persona)
        ch = teach(trip, seed=100000 + j)
        j += 1
        if not ch.ok:
            continue
        r = random.Random(f"{seed}:dpo-perturb:{j}")
        for kind in sorted((k for k in need if need[k] > 0), key=lambda k: (-need[k], k)):
            stats[f"attempt:{kind}"] += 1
            p = perturb(kind, trip, ch, r)
            if p is None:
                continue
            need[kind] -= 1
            pid = f"dpo-{len(records):05d}"
            rej = p.report.summary()
            records.append({
                "pair_id": pid,
                "split": "train",
                "perturbation": kind,
                "persona": trip.persona,
                "e2_test_persona": get_persona(trip.persona).e2_test_persona,
                "task_mix": trip.task_mix,
                "capsule_id": trip.capsule.capsule_id,
                "capsule": trip.capsule.text,
                "ledger": prompt_ledger(trip),
                "plan": canonical_json(ch.plan),
                "frames": ch.frames,
                "chosen": ch.text,
                "rejected": p.text,
                "edit": p.edit,
                "chosen_verification": {"passed": True, "n_claims": ch.report.n_claims, "p_ledger": ch.report.p_ledger},
                "rejected_verification": {"passed": False, "reasons": rej["reasons"], "p_ledger": rej["p_ledger"],
                                          "hallucinated_missing": rej["hallucinated_missing"],
                                          "must_mention_missed": rej["must_mention_missed"],
                                          "unsupported": rej["unsupported"]},
                "generator": {"teacher": TEACHER_ID, "seed": ch.seed, "config_hash": chash},
            })
            break  # one pair per trip keeps the prompts diverse
    stats["trips"] = j
    return records, stats


def build_qa(n: int, seed: int, personas, cfg: SynthConfig, chash: str, frac_unanswerable: float = 0.2):
    n_un = round(n * frac_unanswerable)
    need = {**_quota(n - n_un, ANSWERABLE_TYPES), **_quota(n_un, UNANSWERABLE_TYPES)}
    records, stats = [], Counter()
    k = 0
    while sum(need.values()) > 0 and k < 50 * n:
        persona = personas[k % len(personas)]
        trip = sample_trip(random.Random(f"{seed}:qa:{k}"), 200000 + k, cfg, persona=persona)
        r = random.Random(f"{seed}:qa-q:{k}")
        k += 1
        cands = [c for c in candidate_questions(trip, r) if need.get(c.qtype, 0) > 0]
        if not cands:
            continue
        cands.sort(key=lambda c: (-need[c.qtype], c.qtype))
        best = [c for c in cands if c.qtype == cands[0].qtype]
        item = r.choice(best)
        rec = qa_record(trip, item, f"qa-{len(records):04d}", chash, 200000 + k - 1)
        if rec is None:
            stats["rejected_by_filter"] += 1
            continue
        need[item.qtype] -= 1
        records.append(rec)
    stats["trips"] = k
    if sum(need.values()) > 0:
        log.warning("QA quotas not met: %s", {q: v for q, v in need.items() if v})
    return records, stats


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n")


def _pct(c: Counter, total: int) -> dict:
    return {k: f"{v} ({100 * v / total:.1f} %)" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _stats(sft, sft_s, dpo, dpo_s, qa, qa_s) -> dict:
    words = [len(r["narrative"].split()) for r in sft]
    felt = Counter()
    for r in sft:
        for fr in r["frames"]:
            if "felt" in fr:
                felt[fr["felt"]] += 1
    return {
        "sft": {
            "n": len(sft),
            "teacher_filter": {"attempted": sft_s["attempted"], "rejected": sft_s["rejected_by_filter"],
                               "acceptance": round(1 - sft_s["rejected_by_filter"] / max(sft_s["attempted"], 1), 4)},
            "split": dict(Counter(r["split"] for r in sft)),
            "persona": dict(Counter(r["persona"] for r in sft)),
            "e2_test_persona_records": sum(r["e2_test_persona"] for r in sft),
            "task_mix": dict(Counter(r["task_mix"] for r in sft)),
            "degradations": dict(Counter(d for r in sft for d in r["degradations"])),
            "regime_max_felt_class": dict(sorted(Counter(r["regime_max_felt_class"] for r in sft).items())),
            "felt_class_of_described_units": dict(sorted(felt.items())),
            "budget": dict(sorted(Counter(r["budget"] for r in sft).items())),
            "budget_raised": sum(r["budget_effective"] > r["budget"] for r in sft),
            "narrative_words": {"mean": round(statistics.mean(words), 1), "median": statistics.median(words),
                                "p95": sorted(words)[int(0.95 * len(words)) - 1], "max": max(words)},
            "claims_per_narrative": round(statistics.mean(r["verification"]["n_claims"] for r in sft), 1),
        },
        "dpo": {"n": len(dpo), "perturbation": dict(Counter(r["perturbation"] for r in dpo)),
                "persona": dict(Counter(r["persona"] for r in dpo)), "trips_sampled": dpo_s["trips"],
                "attempts": {k.split(":", 1)[1]: v for k, v in dpo_s.items() if k.startswith("attempt:")}},
        "qa": {"n": len(qa), "answerable": sum(r["answerable"] for r in qa),
               "question_type": dict(Counter(r["question_type"] for r in qa)),
               "persona": dict(Counter(r["persona"] for r in qa)), "trips_sampled": qa_s["trips"]},
    }


def _card(manifest: dict) -> str:
    s = manifest["stats"]
    e2 = ", ".join(sorted(k for k, p in PERSONAS.items() if p.e2_test_persona))
    return f"""# RoadScribe synthetic datasets

Generated {manifest['date']} with `python -m roadscribe.synth.build` (roadscribe {manifest['version']},
seed {manifest['seed']}, config hash `{manifest['config_hash']}`). Every record can be regenerated from the
seed; `manifest.json` holds the full statistics.

| File | Records | Use |
|---|---|---|
| `roadscribe_sft.jsonl` | {s['sft']['n']} | capsule → plan → grounded narrative (PLAN and REALIZE tasks) |
| `roadscribe_dpo.jsonl` | {s['dpo']['n']} | verified narrative (chosen) vs controlled error (rejected) |
| `roadscribe_qa_bank.jsonl` | {s['qa']['n']} | driver question → ledger-grounded answer or abstention |

`roadscribe.synth.formats.to_chat(record, task)` turns a record into chat messages for the
`plan`, `realize`, `dpo` or `qa` task.

## How the data were made

* **Capsules.** `roadscribe.synth.sampler` simulates trips (Markov chain over asphalt, cobblestone and
  unpaved surfaces; ISO 2631-1 felt classes 0–5 balanced by a per-trip regime; Poisson events; physically
  consistent a_v, a_wz, speed decoupling, MTVV and shock flags), then compiles them with the package's real
  ledger and KST code. Token budgets B ∈ {{64, 128, 256, 512}} use the approximate regex tokenizer, not a
  model tokenizer; {s['sft']['budget_raised']} SFT capsules needed a raised budget because their
  must-mention stretches alone did not fit.
* **Teacher.** A programmatic teacher: the rule planner and the T+V template realizer in
  `roadscribe.layer3.templates` (`{TEACHER_ID}`). No language model wrote these narratives.
* **Verification filter.** Every plan validates against its capsule's ledger ids, and every claim in every
  narrative and gold answer is checked by `roadscribe.eval.claims.verify_text`: numbers within tolerance,
  severity words band-consistent, events matched to ledger events with allowed hedges, nothing stated for a
  MISSING field, every must-mention event mentioned. Teacher acceptance: {s['sft']['teacher_filter']['acceptance']:.2%}.
* **DPO negatives.** `roadscribe.synth.perturb` edits one fact frame of a verified output and re-realizes it
  with the same seed, so chosen and rejected differ only at the error. A pair is kept only if the verifier
  rejects the perturbed text for the intended reason (listed in `rejected_verification`).
* **QA.** Question wordings written at design time with gold answers built from the ledger and verified;
  {s['qa']['n'] - s['qa']['answerable']} of {s['qa']['n']} are unanswerable and the gold answer abstains with a reason.

## Read before training or evaluating

1. **E2 leakage.** METHODOLOGY §2.2 holds out {e2} as *test* personas for E2, and §8.6 forbids test
   personas in training material. {s['sft']['e2_test_persona_records']} of {s['sft']['n']} SFT records use them;
   they carry `"e2_test_persona": true`. Drop them before training if E2 is to stay valid:
   `[r for r in rows if not r["e2_test_persona"]]`.
2. **Template teacher.** Fine-tuning on these narratives teaches the SLM to imitate the T+V template system,
   which is the paper's baseline. It can still teach the plan format, grounding and hedging, but it cannot
   show that the SLM covers what templates miss (E1–E5). §8.6 plans a frontier-model teacher for that; its free
   text needs the validated LLM claim extractor of §6.1, because this verifier only reads the template lexicon.
3. **All synthetic.** §8.6 plans 30 % real HRNUs from training routes; none are included here.
4. **QA is not E1.** These questions were written by the system designer. Use the bank for development and
   regression tests only; E1 needs questions elicited from drivers, with a sealed half.
5. **Closed-vocabulary verifier.** The extractor recognises the template lexicon. It is not the validated LLM
   claim extractor that §6.1 requires for free text.

## Record fields

**SFT** `id, split (train/dev by hash, 5 % dev), persona, e2_test_persona, intent, task_mix
(narrative/degraded/history), degradations, regime_max_felt_class, budget, budget_effective, tokenizer,
capsule_id, capsule, ledger (fact index of trip-level and visible units), must_mention_events, plan (canonical
JSON, the grammar's key order), frames, narrative, verification, generator`.

**DPO** `pair_id, split, perturbation, persona, e2_test_persona, task_mix, capsule_id, capsule, ledger, plan,
frames (the realization prompt), chosen, rejected, edit, chosen_verification, rejected_verification, generator`.

**QA** `qa_id, split (dev), persona, e2_test_persona, capsule_id, capsule, ledger (full fact index), question,
question_type, answerable, gold_answer, gold_fact_ids, abstain_reason, verification, generator`.

## Composition

| | |
|---|---|
| SFT personas | {s['sft']['persona']} |
| SFT task mix | {s['sft']['task_mix']} |
| SFT degradations | {s['sft']['degradations']} |
| Worst felt class per trip | {s['sft']['regime_max_felt_class']} |
| Felt class of described stretches | {s['sft']['felt_class_of_described_units']} |
| Budgets | {s['sft']['budget']} |
| Narrative words | {s['sft']['narrative_words']} |
| Claims per narrative | {s['sft']['claims_per_narrative']} |
| DPO perturbations | {s['dpo']['perturbation']} |
| DPO personas | {s['dpo']['persona']} |
| QA question types | {s['qa']['question_type']} |
| QA personas | {s['qa']['persona']} |
"""


def main(argv: Optional[list[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-sft", type=int, default=5000)
    ap.add_argument("--n-dpo", type=int, default=1500)
    ap.add_argument("--n-qa", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS))
    ap.add_argument("--exclude-e2-test-personas", action="store_true",
                    help="drop the personas METHODOLOGY §2.2 holds out for E2")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("roadscribe.layer3.kst").setLevel(logging.WARNING)
    personas = [p for p in args.personas if not (args.exclude_e2_test_personas and get_persona(p).e2_test_persona)]
    cfg = SynthConfig(personas=tuple(personas))
    chash = config_hash({"roadscribe": RoadScribeConfig().to_dict(), "synth": json.loads(json.dumps(
        {k: v for k, v in cfg.__dict__.items() if k not in ("speed", "kst", "ledger")}))})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log.info("SFT: %d examples", args.n_sft)
    sft, sft_s = build_sft(args.n_sft, args.seed, personas, cfg, chash)
    log.info("DPO: %d pairs", args.n_dpo)
    dpo, dpo_s = build_dpo(args.n_dpo, args.seed, personas, cfg, chash)
    log.info("QA: %d items", args.n_qa)
    qa, qa_s = build_qa(args.n_qa, args.seed, personas, cfg, chash)

    _write_jsonl(out / "roadscribe_sft.jsonl", sft)
    _write_jsonl(out / "roadscribe_dpo.jsonl", dpo)
    _write_jsonl(out / "roadscribe_qa_bank.jsonl", qa)
    manifest = {"version": __version__, "date": date.today().isoformat(), "seed": args.seed, "config_hash": chash,
                "personas": personas, "teacher": TEACHER_ID, "stats": _stats(sft, sft_s, dpo, dpo_s, qa, qa_s)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "DATASET_CARD.md").write_text(_card(manifest), encoding="utf-8")
    log.info("wrote %s", ", ".join(p.name for p in sorted(out.iterdir())))
    return manifest


if __name__ == "__main__":  # pragma: no cover
    main()
