# RoadScribe synthetic datasets

Generated 2026-11-03 with `python -m roadscribe.synth.build` (roadscribe 0.1.0,
seed 20261103, config hash `33f98a43dbccfb0c`). Every record can be regenerated from the
seed; `manifest.json` holds the full statistics.

| File | Records | Use |
|---|---|---|
| `roadscribe_sft.jsonl` | 5000 | capsule → plan → grounded narrative (PLAN and REALIZE tasks) |
| `roadscribe_dpo.jsonl` | 1500 | verified narrative (chosen) vs controlled error (rejected) |
| `roadscribe_qa_bank.jsonl` | 300 | driver question → ledger-grounded answer or abstention |

`roadscribe.synth.formats.to_chat(record, task)` turns a record into chat messages for the
`plan`, `realize`, `dpo` or `qa` task.

## How the data were made

* **Capsules.** `roadscribe.synth.sampler` simulates trips (Markov chain over asphalt, cobblestone and
  unpaved surfaces; ISO 2631-1 felt classes 0–5 balanced by a per-trip regime; Poisson events; physically
  consistent a_v, a_wz, speed decoupling, MTVV and shock flags), then compiles them with the package's real
  ledger and KST code. Token budgets B ∈ {64, 128, 256, 512} use the approximate regex tokenizer, not a
  model tokenizer; 1100 SFT capsules needed a raised budget because their
  must-mention stretches alone did not fit.
* **Teacher.** A programmatic teacher: the rule planner and the T+V template realizer in
  `roadscribe.layer3.templates` (`roadscribe.synth.teacher/rule-planner+template-realizer v1`). No language model wrote these narratives.
* **Verification filter.** Every plan validates against its capsule's ledger ids, and every claim in every
  narrative and gold answer is checked by `roadscribe.eval.claims.verify_text`: numbers within tolerance,
  severity words band-consistent, events matched to ledger events with allowed hedges, nothing stated for a
  MISSING field, every must-mention event mentioned. Teacher acceptance: 100.00%.
* **DPO negatives.** `roadscribe.synth.perturb` edits one fact frame of a verified output and re-realizes it
  with the same seed, so chosen and rejected differ only at the error. A pair is kept only if the verifier
  rejects the perturbed text for the intended reason (listed in `rejected_verification`).
* **QA.** Question wordings written at design time with gold answers built from the ledger and verified;
  60 of 300 are unanswerable and the gold answer abstains with a reason.

## Read before training or evaluating

1. **E2 leakage.** METHODOLOGY §2.2 holds out car_sick_passenger, fleet_manager as *test* personas for E2, and §8.6 forbids test
   personas in training material. 2500 of 5000 SFT records use them;
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
| SFT personas | {'commuter': 1250, 'sports_car': 1250, 'car_sick_passenger': 1250, 'fleet_manager': 1250} |
| SFT task mix | {'narrative': 3151, 'history': 759, 'degraded': 1090} |
| SFT degradations | {'osm_contradiction': 188, 'low_conf_events': 188, 'mapmatch_failed': 203, 'no_gyro': 206, 'speed_low': 200, 'gps_gap': 209, 'no_map': 206} |
| Worst felt class per trip | {0: 836, 1: 836, 2: 832, 3: 832, 4: 832, 5: 832} |
| Felt class of described stretches | {0: 4323, 1: 4655, 2: 4501, 3: 2881, 4: 1547, 5: 842} |
| Budgets | {64: 529, 128: 1712, 256: 1792, 512: 967} |
| Narrative words | {'mean': 157.3, 'median': 150.0, 'p95': 274, 'max': 406} |
| Claims per narrative | 34.1 |
| DPO perturbations | {'dropped_must_mention': 188, 'number_drift': 188, 'anchor_swap': 188, 'fabricated_defect': 188, 'misattributed_braking': 187, 'missing_value_supplied': 187, 'severity_shift': 187, 'unhedged_low_confidence': 187} |
| DPO personas | {'commuter': 376, 'sports_car': 376, 'car_sick_passenger': 373, 'fleet_manager': 375} |
| QA question types | {'roughest': 27, 'jolt_cause': 27, 'smoothest': 27, 'speed_cause': 27, 'unit_events': 27, 'unit_felt': 27, 'map_check': 26, 'speed_on': 26, 'history': 26, 'out_of_scope': 15, 'name_missing': 15, 'history_missing': 15, 'road_vs_speed_unknown': 15} |
| QA personas | {'commuter': 75, 'sports_car': 75, 'car_sick_passenger': 75, 'fleet_manager': 75} |
