# Identifiability of sensor-language claims: overnight results (2026-11-12)

This page is for reading over coffee. The full chronological record, with every decision (D1–D17), is `WORKLOG.md`.

The work is on the local branch `night/2026-11-12-identifiability`: 12 commits, plus the one that fixed this sentence. **Nothing was pushed.**

## What was done

1. **Re-checked my earlier claims (C1–C11).**
   - The fabrication findings hold. No Kaggle notebook version ever ran QA or E2 inference.
   - One of my notes was imprecise: PELT failed in 136/300 cases for the fair variant (300/300 applies only to `verify_math.py`'s version). The paper's number is the right one.
   - The identifiability lemma is now confirmed numerically: channel differences are identified from shared roads, but the common channel needs speed diversity at the same place (30–100× error otherwise).
2. **Phase 0, integrity fix.**
   - Rewrote `scripts/run_full_evaluation.py`: E1/E2 report "not run", P_G is removed, grounding is scored per drive against a chance level.
   - Added a regression test for the cross-trip pooling bug.
   - Corrected the paper's results section, abstract, contributions list and one related-work sentence. It still compiles: 24 pp, no overfull boxes.
   - All 232 tests pass.
3. **Three pre-registered pilots on PVS** (P1–P3), each hypothesis committed before its data was read, plus three exploratory analyses (X1–X3), labelled as such.

## What was found

| Claim tested | Result |
|---|---|
| P1: RoadScribe's road claims are vehicle-invariant | **No.** κ_road 0.72. Vehicle identity explains **18.2%** of the road index (placebo null 0.06%). All three cars agree on the road class at only 4.5% of places |
| P1: speed elasticity from one traversal is usable | **No.** Pooled −0.64 vs within-place +0.86. The sign flip is mostly surface composition; within asphalt the cross-place estimate is still about 80% too small (0.16 vs 0.96). RoadScribe's γ = 0.35 is not trustworthy |
| P2: SCR channel calibration fixes vehicle dependence | **Partly, and not for the reason I proposed.** A plain per-vehicle gain learned on *other* roads removes the vehicle share as well as SCR does, and gets the best held-out-vehicle accuracy (97.3%). SCR's small rank gain (+0.06) comes mostly from using a spatial-frequency index. SCR keeps more speed dependence |
| P3: SCR predicts how felt vibration changes for an unseen car | **No.** Calibrated γ + gains beats it: RMSE 0.158 vs 0.196, comfort class 76% vs 69%. Both beat naive transfer by a wide margin (0.47) |
| X1: the problem is the same for every sensor | **No.** Vehicle share: dashboard 18%, above suspension 3%, below suspension 12%. The phone-like mount is the worst case |
| X2: RoadScribe's narratives contradict each other about the same road | **Illustrated, not measured.** About 80% of co-located road-surface claim pairs disagree, but this rests on only 24 claims |

**Hypotheses that failed as stated:** P1-H1 (felt agreement was higher than I predicted), P1-H5, P2-H4 and both P3 hypotheses. **My physics prediction for X1 was also wrong.**

## What it means

- **The identifiability problem is real and large** for the realistic phone-like mount, and it shows up in RoadScribe's own text.
- **The remedies that work are simple:** calibrate each vehicle on *shared* roads, estimate speed effects *within* places, and describe roads in spatial frequency.
- **My SCR spectral factorisation did not earn its place on PVS.** It is demoted to optional.
- **What a paper can still claim:** the claim-level measurement and the identifiability criterion. A new estimator is not a candidate. Road/vehicle separation from phones is prior art (Botshekan et al. 2020).

## Decisions that are yours

1. **The branch:** review it, then push or merge. I did not push.
2. **Commit trailers:** I followed your earlier wish for no AI trailers in this repo (WORKLOG D2). Say so if you want them.
3. **The late-November DTU deadline:** an honest short RoadScribe paper (PHASES Track R), or skip it.
4. **Whether to pursue this direction.** My recommendation is yes, as a measurement paper. The next steps, in order:
   1. A literature check on cross-place speed-elasticity bias, and on claim-level identifiability in sensor narration.
   2. One more domain with a measured rate (Ottawa bearings).
   3. Phase 4: two or three LLM narrators on PVS, scored with per-claim units.

## Files

| File | What it is |
|---|---|
| `WORKLOG.md` | Chronological log: every decision, action, result and correction |
| `PHASES.md` | The plan, gate outcomes, Phase 3 data sources |
| `PREREG_P1.md`, `PREREG_P2.md`, `PREREG_P3.md` | Pre-registrations (commit times are authoritative) |
| `CLAIM_TAXONOMY.md` | Invariant, covariant and non-identifiable road claims |
| `p1_cross_vehicle.py`, `p2_channel_calibration.py`, `p3_counterfactual.py` | Pre-registered analyses |
| `x1_placements.py`, `x2_narrative_contradictions.py`, `x3_robustness.py`, `p2_exploratory_decomposition.py` | Exploratory analyses |
| `fig_speed_elasticity.py` → `results/p1_speed_elasticity_flip.png` | Figure: pooled vs within-place elasticity |
| `results/*.json` | All numbers quoted here. The stretch CSVs are gitignored; the scripts regenerate them |

Code added to the package: `roadscribe/paired/` (route alignment, stretch features, agreement statistics), `roadscribe/scr/` (Whittle SCR fit), `roadscribe/eval/metrics.py::anomaly_grounding_by_recording`. Tests: `tests/test_paired.py`, `tests/test_scr.py`, and new cases in `tests/test_metrics.py`.

**Reproduce**, from the repo root, with `ROADSCRIBE_PVS_DIR` set:

```
python scripts/run_full_evaluation.py
python roadscribe_research/identifiability/p1_cross_vehicle.py
python roadscribe_research/identifiability/p2_channel_calibration.py
python roadscribe_research/identifiability/p3_counterfactual.py
python roadscribe_research/identifiability/x1_placements.py
python roadscribe_research/identifiability/x2_narrative_contradictions.py
python roadscribe_research/identifiability/x3_robustness.py
```

P1 and P2 take about 10 minutes each, because of the bootstraps.
