# ICDSI-2027 paper: two-column IEEE version (9 pages)

**Title:** *Same Road, Three Stories: Why Verified Sensor Narratives Can Still Be Wrong*.
The single-column Springer LNCS version (15 pages) is preserved at tag `icdsi-llncs-15pp`.

## Build

Run from this folder:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where every number comes from

Result files are in `../identifiability/results/` unless the path says otherwise.

Sections: II related work, III identifiability, IV testbed, V world level, VI input level, VII discussion.

All language-model numbers use the verifier as corrected on 2026-11-13:
- D26: summaries of "the other N stretches" are checked against the units the narrative does not describe;
- D29: event positions may have two decimals.

| Paper location | Number(s) | Source |
|---|---|---|
| Abstract, §I, Table III | three wordings at route A stretch 107 | `x12_worked_example.json` |
| §III, simulation | 0.010–0.023 vs 0.67–1.09 | `tests/test_scr.py` and the seed sweep in `../identifiability/WORKLOG.md` |
| §IV, Table IV | template passes on every drive (P_L 100%) | `data/evaluation_summary.json` (`scripts/run_full_evaluation.py`) |
| §V-A | alignment 1.5–3.8 m, 97.9–99.3%, 310 of 361, κ_GT 0.93 | `p1_results.json`, `p7_results.json` (`H6_speed_floor`) |
| Table I | commits, times, outcomes | `git log` of `../identifiability/PREREG_P*.md`; verdicts in `p1`–`p4` results, `p5_summary.json`, `p6_results.json`, `p7_results.json` |
| §V-A, γ = 0.35 date | 2026-11-03 | `git log -S"gamma: float = 0.35" -- roadscribe/config.py` (`aec2308`) |
| Table II | κ, agreement, shares, CIs; floor row | `p1_results.json`, `p4_results.json`, `p7_results.json` (`missing_cis`, `H1_noise_floor`) |
| §V-B | placebo 0.06% / 0.27%; residual 2.1%; right logger 17.8%, 0.72; mounts 2.9% / 11.8% | `x3_robustness.json`, `p1_results.json`, `p7_results.json` (`H3`), `x1_placements.json` |
| §V-C, Fig. 2 | −0.64 / +0.86; 1.50 [1.26, 1.64]; surface slopes; +0.96; 0.51 | `p1_results.json`, `x3_robustness.json`, `fig_speed_elasticity.py` |
| §V-D, estimates and agreement | γ̂ 0.87–0.88; gains; 0.72 → 0.96, [0.20, 0.33]; 92%; right logger 0.95; slope −0.04 | `p4_results.json`, `p7_results.json` |
| §V-D, ablation; shared road | 0.96 with gain only (slope 0.51), 0.86 elasticity only; 0.95 [0.83, 0.98] at 0.5 km, 0.96 [0.91, 0.99] at 1 km | `p7_results.json` (`ablation`, `H4_data_efficiency`) |
| §V-D, accuracy table (361 places) | 94.6 [92.0, 97.0], 87.3 [82.7, 91.8], 81.0, 66%; balanced 0.87 [0.79, 0.90], 0.69 [0.65, 0.72], 0.59; bad passes 50 vs 24 of 69 | `x16_accuracy_speed_floor.json` (`"0.0"`) |
| §V-D, 5 m/s floor | 98.3 vs 90.5; balanced 0.82 vs 0.66; 11 of 38; 15.4%; 0.75 → 0.95 | `p7_results.json` (`H5`, `H6`) |
| §V-D, spectral; narratives | RMSE 0.196 vs 0.158; 70 of 310; 23% vs 80%; 28 and 24 claims | `p3_results.json`; `x12_worked_example.json`; `p4_results.json`, `x2_narrative_contradictions.json` |
| §V-E, events | 337; 52.8%; 6.8%; 17% | `x11_event_reproducibility.json`, `x10_event_precision.json` |
| Table IV, §VI-A | 300-token row; 0 of 54, 46.2, 48.5% (2,493 claims); 470 / 261 / 163 / 47 / 41; median 244 words | `data/evaluation_summary.json`, `spgr_regen_summary.json`, `x5_slm_failures.json`, `data/models/kaggle_regen/spgr_regen_eval.json` |
| Table IV, rebuilt-capsule row; §VI-A P9 sentence | P9 | `p9_results.json` (`p9_eval.py`) |
| Table V, §VI-A | presence 0 / 0 / 1.7 / 98.2 / 100%; errors 86 / 98 / 80 / 1.2 / 0%; lenient 21% | `x8_target_identifiability.json`, `x14_per_type_errors.json` |
| §VI-B, Table VI | arms A–C, prompted 3B / 7B; H1–H4 with CIs; per-type errors | `p6_results.json` (`scripts/eval_p6.py`); rows printed by `../identifiability/p6_table.py` |
| §VI-B, greedy and second epoch | 76.4%; 75.3%, 44%, loss 0.09 | `p8_results.json`; loss from `data/models/p8/epoch2/*.log` |
| §VI-B, sentence filter; descriptors | 33 / 41 / 52%; 7–10 | `x9_sentence_filter.json`, `x13_vocabulary_coverage.json` |
| §VI-C | 52%; 86 → 41%; 27 → 1; 48.5 → 69.4%; 104; 28 of 54; 77%; 36% / 63%; 2.5–6 | `p5_summary.json`, `x14_per_type_errors.json`, `x6_slm_plan_template_realize.json`, `x9_sentence_filter.json`, `x13_vocabulary_coverage.json` |
| §VII | γ̂ 0.86 [0.68, 1.02]; κ 0.95 with γ = 0.5 | `p7_results.json` (`missing_cis`, `ablation`) |

Paths that begin with `scripts/`, `tests/` or `data/` are relative to the repository root.
The pre-specifications are `../identifiability/PREREG_P1.md` to `PREREG_P7.md`; P8 and P9 were declared in `WORKLOG.md` before they ran.
