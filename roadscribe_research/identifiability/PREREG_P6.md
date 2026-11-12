# Pre-registration P6: controlled test of input identifiability (same targets, different inputs)

*Written 2026-11-13 and committed before any training run. The commit time is authoritative.*

## Why

P5 changed several things at once: training format, training set, and the PVS inputs. Its improvement therefore cannot be attributed to input identifiability (simulated review, Reviewer A #2). P6 holds everything fixed except **what the input contains**.

## Design

**Shared across all arms**
- **Records:** the same 1,200 training records, a seeded sample of the `roadscribe_sft.jsonl` train split with personas commuter and sports_car.
- **Target:** the same narrative for every arm, the record's `narrative`.
- **Model and training:** Qwen2.5-3B-Instruct with QLoRA r 16, α 32, lr 2e-4, 1 epoch, loss on the target tokens only, maximum length 2,048 tokens. Sequences over the limit are dropped **in all arms if they are over the limit in any arm**, so all arms keep identical record sets.
- **System prompt:** one neutral prompt for all arms.

**Arms (input only)**

| Arm | Input |
|---|---|
| A, capsule | persona card + capsule |
| B, capsule + frames | persona card + capsule + the record's grounded frames, which hold every value the target states |
| C, capsule + frames without positions | as B, with event positions (`s`) removed from the frames; everything else unchanged |

**PVS evaluation.** 9 trips × 2 personas × 3 seeds = 54 narratives per arm, with the original ledgers and capsules rebuilt without the spurious map channel (the P5 inputs). For arms B and C the frames are the grounded frames of the rule planner's plan over the visible units. Sampling is T = 0.3, top-p 0.9, at most 768 new tokens. The verifier and the acceptance rule are unchanged.

## Measures

- Claim precision against the ledger.
- Unsupported claims by reason, with the share of event claims at a wrong position as the key measure.
- Hallucinated-missing claims.
- Acceptance (whole narrative passes).
- Claims per narrative.

## Hypotheses

| ID | Prediction |
|---|---|
| P6-H1 | Precision B > A, by at least 0.15 |
| P6-H2 | Wrong-position share: C ≥ 2 × B, and A ≥ 2 × B |
| P6-H3 | Unsupported claims for reasons *other* than position: C < 0.5 × A |
| P6-H4 | Precision B > C |

If these hold, errors appear where values are missing from the input and disappear where they are present, with targets held fixed. That is the causal claim the paper makes.

**Uncertainty:** cluster bootstrap over drives (9 clusters, 10,000 resamples) for precision differences.

**Also run (not part of the hypotheses):** prompted Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct without fine-tuning, given arm B's input.

**Known limits:**
- One epoch on at most 1,200 examples per arm.
- Synthetic training data.
- The frames used at PVS time come from the rule planner, so arms B and C do not choose content.
