# RoadScribe frontier-teacher data

Written by `python -m roadscribe.synth.teacher_io ingest` from the raw teacher files in
`data/processed/teacher/`. Edit the raw files and re-run `ingest`; do not edit this card.

| File | Records | Use |
|---|---|---|
| `roadscribe_teacher_sft.jsonl` | 172 | capsule → plan → narrative written by a frontier-model teacher |
| `roadscribe_teacher_qa.jsonl` | 60 | driver question → teacher answer or abstention |
| `roadscribe_teacher_exemplars.jsonl` | 3 | frozen few-shot exemplars for §8.6 Step 0 |

Records have the fields of `roadscribe_sft.jsonl` plus `"source": "teacher"`; `generator.teacher` is
`frontier-llm, interactive session, lexicon-constrained free composition v1`.

## How the data were made

1. **Trips.** `teacher_trip(kind, i)` draws trip *i* with the same sampler and ledger code as the template
   data, on its own seed stream (`teacher:i`, `teacher-qa:i`), and only for the development personas
   commuter and sports_car. The task mix cycles 40 % narrative, 35 % degraded and 25 % history.
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

**First-pass acceptance:** 228 of 232 outputs, narratives and answers (98.3 %), passed
before any revision. After revision all pass. Before the first batch, trial narratives exposed two verifier gaps
(coordinated subjects, and claims in compound sentences). Both were fixed in the verifier, with
regression tests, before any narrative was counted.

| Raw file | First pass | Note |
|---|---|---|
| `qa_000.txt` | 20 / 20 | 4 of 20 answers abstain |
| `qa_001.txt` | 20 / 20 | 4 of 20 answers abstain |
| `qa_002.txt` | 20 / 20 | 4 of 20 answers abstain |
| `sft_000.txt` | 10 / 12 | two event positions rounded more than 50 m from the event |
| `sft_001.txt` | 15 / 16 | one verifier gap: a pronoun after a sentence naming two stretches (verifier fixed, test added) |
| `sft_002.txt` | 16 / 16 |  |
| `sft_003.txt` | 16 / 16 |  |
| `sft_004.txt` | 16 / 16 | while writing, a REST count of seven exposed a verifier crash; fixed before ingest, test added |
| `sft_005.txt` | 16 / 16 |  |
| `sft_006.txt` | 16 / 16 |  |
| `sft_007.txt` | 16 / 16 |  |
| `sft_008.txt` | 16 / 16 |  |
| `sft_009.txt` | 16 / 16 |  |
| `sft_010.txt` | 15 / 16 | one felt word ('rough') in a trip-level sentence |

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
| SFT personas | {'commuter': 86, 'sports_car': 86} |
| SFT task mix | {'narrative': 72, 'degraded': 60, 'history': 40} |
| SFT degradations | {'no_map': 15, 'gps_gap': 10, 'speed_low': 13, 'osm_contradiction': 10, 'mapmatch_failed': 10, 'no_gyro': 8, 'low_conf_events': 9} |
| SFT split | {'train': 157, 'dev': 15} |
| Budgets | {64: 26, 128: 56, 256: 68, 512: 22} |
| Worst felt class per trip | {0: 30, 1: 30, 2: 28, 3: 28, 4: 28, 5: 28} |
| Narrative words | mean 203.0, median 196, p95 348, max 436 |
| Claims per narrative | mean 38.3, median 36, p95 69, max 91 |
| Event mentions per narrative | mean 5.6, median 5, p95 12, max 16 |
| QA answerable / abstain | 48 / 12 |
| Exemplars | `syn-t0004` (commuter, narrative), `syn-t0009` (sports_car, degraded), `syn-t0018` (commuter, history) |
