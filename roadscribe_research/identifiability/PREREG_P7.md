# Pre-registration P7: noise floor, data efficiency, held-out correctness and speed-floor sensitivity

*Written 2026-11-13, about 00:43, and committed before any of these analyses were run. The commit time is authoritative.*

## Why

The simulated review (memory `roadscribe-icdsi-review-2026-11-13`) raised four questions about the cross-vehicle study:
- **No noise floor.** Is κ = 0.96 after calibration good or merely expected?
- **Calibration data.** How much shared road does calibration need?
- **Agreement or correctness?** Agreement between vehicles is not correctness, and the held-out-vehicle accuracy still trains on the test route.
- **Speed floor.** The 5 m/s filter excludes stretches, possibly the rough ones.

P7 answers each question with an analysis fixed here.

## Data

- **Stretches:** the P1 stretch tables (100 m stretches, dashboard IMU) and the P1/P4 code paths.
- **New in P7:** the right-side logger of every trip. Its dashboard module is matched to the left logger's reference track, using its own GNSS fixes and the same forward-window matcher.
- **Places:** the P1 primary set (310 places, all three vehicles at v ≥ 5 m/s), unless stated otherwise.

## Analyses and hypotheses

| ID | Analysis | Prediction |
|---|---|---|
| P7-H1 | **Noise floor.** Left vs right dashboard module, same vehicle, same pass. Quadratic-weighted κ of the road class over all (trip, place) pairs where both modules are valid. | κ_LR ≥ 0.90 |
| P7-H2 | **Calibration reaches the floor.** P4's calibrated cross-vehicle κ (0.958) compared with κ_LR. | \|κ_cal − κ_LR\| ≤ 0.05 |
| P7-H3 | **Replication on the other logger.** P1's vehicle share of log a_dec and P4's leave-one-route-out calibration, recomputed on the right logger. | Vehicle share ≥ 10%, and κ_cal ≥ 0.90 |
| P7-H4 | **Data efficiency.** Leave one route out. γ̂ and the gains are fitted on K km of contiguous 1 km blocks drawn from the other two routes, for K ∈ {0.5, 1, 2, 4, 8, all}, with 200 draws per K. κ_cal is computed over all 310 places. | Median κ_cal ≥ 0.90 at K = 2 km |
| P7-H5 | **Held-out correctness.** For each (route, vehicle) cell, everything is fitted without that cell's route: calibration on the other two routes, cut-points for the quality label (good / regular / bad) on the other two routes and the other two vehicles. Prediction on the held-out cell. Pooled accuracy and balanced accuracy for log a_v, log a_dec and log a_cal. | Accuracy of a_cal > accuracy of a_dec, with the 95% CI of the difference above 0 |
| P7-H6 | **Speed floor.** P1's vehicle share and P4's κ_cal with a floor of 3 m/s instead of 5. Report the number of places at each floor and the number of places all three vehicles covered at any speed. | Vehicle share ≥ 10%, and κ_cal ≥ 0.90 |

**Uncertainty.**
- Block bootstrap over contiguous 10-place (1 km) blocks within each route, 2,000 resamples, as in P1. It also gives CIs for:
  - κ_cal and κ_uncal themselves (P4 bootstrapped only their difference);
  - the felt-value vehicle share;
  - γ̂ (fitted on all three routes).
- For H5 and P4's leave-one-vehicle-out accuracies, predictions are held fixed and the correctness indicators are resampled by block.

**Ablation (no hypothesis).** Which part of the calibration does the work, measured as κ over the 310 places, leave one route out:
- the gain only, with γ = 0.35 (RoadScribe's default);
- the elasticity only, with γ̂ and no gains;
- both.

## Known limits

- Left and right modules see different wheel paths, and the quality labels are per side. κ_LR is therefore a floor for *same-vehicle, same-pass* agreement, which includes left–right road differences. It is not pure sensor noise.
- Three vehicles, one pass each. Vehicle and driver remain confounded.
