# Pre-registration P5: does the narrator pass verification when its inputs contain what its targets state?

*Written 2026-11-12 and committed before the training run. The commit time is authoritative.*

## Background

The regenerated S-PGR run passed verification in 0 of 108 narratives, with a raw claim precision of 48%. The diagnosis was that the deployed training asked the model for values its input did not contain:
- **plan fact IDs:** 0 of 16 in the prompt;
- **event positions:** 7.7% present;
- **speeds:** 0.7% present;
- **accelerations:** 5% present.

The repository's own design, `synth/formats.py`, avoids this:
- **PLAN** sees the capsule *and* the fact index, and writes a plan over fact IDs.
- **REALIZE** sees grounded frames, which contain the values, and writes the text.

## Design

- **Model:** the same base model, Qwen2.5-3B-Instruct with QLoRA (r 16, α 32, lr 2e-4, 1 epoch). The loss is computed on assistant tokens only.
- **Training data:** a seeded sample from `roadscribe_sft.jsonl`, train split, personas commuter and sports_car:
  - 1,200 PLAN examples;
  - 1,200 different records as REALIZE examples;
  - the 196 frontier-teacher records as REALIZE examples.
  - Examples longer than 2,048 tokens are dropped.
- **PVS inputs:** capsules rebuilt per persona, with the correct channel list (no `map`, because PVS has no map coverage), plus the fact index of the visible units. Both the original and the P4-calibrated ledgers are used.
- **Generation:** a plan for each (trip, persona) and 3 seeds, T = 0.3 and top-p 0.9 as before. A plan that passes `validate_plan` is grounded deterministically into frames, and the model then realizes the frames into text. Anything that fails falls back to the template narrator.
- **Evaluation** uses the same verifier and acceptance rule as before.

## Hypotheses

| ID | Prediction | Previous run |
|---|---|---|
| P5-H1 | Plan validity (`validate_plan`) ≥ 50% | 1–2 of 54 |
| P5-H2 | Acceptance ≥ 30% of narratives | 0 of 54 |
| P5-H3 | Raw claim precision of the model-realized narratives ≥ 0.90 | 0.48 |
| P5-H4 | Wrong-position share of event claims below 40% | 470 of 1,161 |

**Exploratory:** whether road-claim contradictions across vehicles in accepted narratives fall with the calibrated ledgers, as they did for the template.

**Caveats known in advance:**
- One epoch on 2,596 examples.
- The training capsules are synthetic, so a shift to real trips remains.
- If the hypotheses fail, the cause could be either the identifiability fix being insufficient or too little training. Both readings will be reported.
