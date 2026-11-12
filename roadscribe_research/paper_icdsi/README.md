# ICDSI-2027 paper (Springer llncs)

## Build

Run from this folder:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where every number comes from

Result files are in `../identifiability/results/` unless the path says otherwise.

| Paper location | Number(s) | Source |
|---|---|---|
| §3, simulation | channel errors 0.010–0.023 vs 0.67–1.09; differences ≤ 0.025 | `tests/test_scr.py` and the seed sweep in `../identifiability/WORKLOG.md` |
| §4 | template passes the verifier on every PVS drive | `data/evaluation_summary.json` (`scripts/run_full_evaluation.py`); `p5_summary.json` (`template_precision`) |
| §5.1 data | route lengths, alignment 1.5–3.8 m, 97.9–99.3%, 310 of 361 places, κ_GT 0.93 | `p1_results.json`; 361 from `p7_results.json` (`H6_speed_floor`) |
| Table 1 | commits, times, outcomes | `git log` of `../identifiability/PREREG_P*.md`; verdicts in `p1_results.json`, `p2_results.json`, `p3_results.json`, `p4_results.json`, `p5_summary.json`, `p6_results.json`, `p7_results.json` |
| §5.1 γ = 0.35 date | 2026-11-03 | `git log -S"gamma: float = 0.35" -- roadscribe/config.py` (commit `aec2308`) |
| §5.1 bootstrap | 2,000 resamples; 12/10/10 blocks | `p7_results.json` (`missing_cis.bootstrap`) |
| Table 2, rows 1–2 | κ, CIs, agreement, vehicle share and CIs | `p1_results.json` (`primary_100m`); all-agree CI from `p7_results.json` (`missing_cis`) |
| Table 2, row 3 | κ 0.96 [0.93, 0.97], 64.8% [58.7, 70.9], 24%, < 0.1% | `p4_results.json` (`calibrated`, `ci95`); CIs from `p7_results.json` (`missing_cis`) |
| Table 2, row 4; §5.2 floor | 0.98 [0.97, 0.98], 7.8%, never ≥ 2, side 0.03% | `p7_results.json` (`H1_noise_floor`) |
| §5.2 placebo, interaction | 0.06%, 0.27%, 2.1% | `x3_robustness.json`, `p1_results.json` |
| §5.2 replication | 17.8%, κ 0.72 | `p7_results.json` (`H3_replication_right`) |
| §5.2 placement | 2.9% / 11.8% | `x1_placements.json` (exploratory) |
| §5.3, Fig. 2 | −0.64 / +0.86; 1.50 [1.26, 1.64]; surface slopes; +0.96; 0.51 | `p1_results.json`, `x3_robustness.json`, `fig_speed_elasticity.py` |
| §5.4 calibration | γ̂ 0.87–0.88, gains, κ 0.72 → 0.96, [0.20, 0.33], slope −0.04 | `p4_results.json` |
| §5.4 gap closed 92%; right logger 0.72 → 0.95 | | `p7_results.json` (`H1`, `H3`, `missing_cis`) |
| §5.4 ablation | 0.96 (gain only), slope 0.51; 0.86 (elasticity only) | `p7_results.json` (`ablation`) |
| §5.4 shared road | 0.95 [0.83, 0.98] at 0.5 km; 0.96 [0.91, 0.99] at 1 km | `p7_results.json` (`H4_data_efficiency`) |
| §5.4 held-out accuracy | 98.3 [97.2, 99.2], 90.5 [86.4, 94.5], 86.5, 76.2; balanced 0.82 / 0.66 / 0.60; +7.7 [4.4, 11.4] | `p7_results.json` (`H5_correctness.route_and_vehicle_out`) |
| §5.4 speed floor | 27 of 38; 15.4%; 0.75 → 0.95 | `p7_results.json` (`H6_speed_floor`) |
| §5.4 spectral | RMSE 0.196 vs 0.158, [0.024, 0.051] | `p3_results.json`, `p2_results.json` |
| §5.4 narratives; Table 3 | 23% vs 80%; 28 and 24 claims; 70 of 310; example rows | `p4_results.json`, `x2_narrative_contradictions.json`; `x12_worked_example.json` |
| §5.5 events | 337, 52.8%, 6.8%; 17%, 34 of 60; 72% / 8% | `x11_event_reproducibility.json`, `x10_event_precision.json` (exploratory) |
| §6, Table 4 | T, T+V 100%; 300-token row; 1,024 tokens: 0 of 54, 46.2, 48.5% (1,208 of 2,493) | `data/evaluation_summary.json`; `spgr_regen_summary.json`; `x5_slm_failures.json` (claim counts) |
| §6.1 failure reasons | 470, 261, 132, 163, 47, 41; median 244 words | `x5_slm_failures.json`; `data/models/kaggle_regen/spgr_regen_eval.json` |
| Table 5, §6.1 audit | 0 of 7,793; 0 of 2,621; 1.7%; 21% lenient | `x8_target_identifiability.json` |
| §6.2, Table 6 | arms A–C and prompted rows; H1–H4 with CIs; per-type contrasts | `p6_results.json` (`scripts/eval_p6.py`; rows printed by `../identifiability/p6_table.py`) |
| §6.2 sentence filter 33/41/52%; §6.3 prompted remainders 2.5–6; descriptors 7–10 | | `x9_sentence_filter.json`, `x13_vocabulary_coverage.json` (exploratory) |
| Table 5 error column | 86 / 98 / 80 / 1.2 / 0% | `x14_per_type_errors.json` |
| §6.3 P5 | 52%, 40% → 18%, 27 → 1, 69.4% | `p5_summary.json`, `spgr_regen_summary.json`, `x5_slm_failures.json` |
| §6.3 model plan + template | 28 of 54, 100%, 77% | `x6_slm_plan_template_realize.json` |
| §6.3 sentence-level fallback | 36%, 12.0 of 22.4; 63%, 20.2 of 23.6 | `x9_sentence_filter.json` (exploratory) |
| §6.3 grounding | 84% / 98% | `data/evaluation_summary.json` |
| §7 | γ̂ 0.86 [0.68, 1.02]; κ 0.95 with γ = 0.5 | `p7_results.json` (`missing_cis`, `ablation`) |

Paths that begin with `scripts/`, `tests/` or `data/` are relative to the repository root.

The pre-registrations are `../identifiability/PREREG_P1.md` to `PREREG_P7.md`; their commit times are the authoritative timestamps. The decision log is `../identifiability/WORKLOG.md`.

The verifier was corrected on 2026-11-13 (WORKLOG D26): summaries of "the other N stretches" had been checked against an empty set of units. Every number above comes from a rerun with the corrected verifier.
