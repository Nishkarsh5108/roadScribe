# ICDSI-2027 paper (Springer llncs)

## Build

Run from this folder:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where every number comes from

| Paper location | Number(s) | Source |
|---|---|---|
| §4 (identifiability), synthetic | channel errors 0.010–0.023 vs 0.67–1.09; differences ≤ 0.025 | `tests/test_scr.py` (and the seed sweep logged in `../identifiability/WORKLOG.md`) |
| §5.1 data | route lengths, alignment 1.5–3.8 m, 97.9–99.3%, 310 places, κ_GT 0.93 | `../identifiability/results/p1_results.json` (alignment, primary_100m) |
| §5.2 grounding | 84% / 98% / 7.5%; 35 of 60 | `scripts/run_full_evaluation.py` → `data/evaluation_summary.json`; `../verification/recheck_eval_2026_11_11.py` |
| §5.2 S-PGR | acceptance, claims, P_L, H1 | `scripts/eval_spgr_regen.py` → `data/models/kaggle_regen/spgr_regen_eval.json` |
| Table 2, §5.3 | κ, agreement, vehicle share, CIs, placebo | `p1_results.json`, `x3_robustness.json` |
| §5.3 placement | 2.9% / 11.8% / 18.2% | `x1_placements.json` (exploratory) |
| §5.4, Fig. 2 | −0.64 / +0.86; surface split; 0.51 | `p1_results.json`, `x3_robustness.json`, `fig_speed_elasticity.py` |
| §5.5 calibration | γ̂, gains, κ 0.96, 64.8%, slope −0.04, 98.5%, 23% vs 80% | `p4_results.json`, `x2_narrative_contradictions.json` |
| §5.5 spectral | +0.014; RMSE 0.196 vs 0.158 | `p2_exploratory_decomposition.json`, `p3_results.json` |

Paths in the table are relative to this folder, except those that begin with `scripts/`, `tests/` or `data/`, which are relative to the repository root.

Pre-registrations for the cross-vehicle study are in `../identifiability/PREREG_P1.md` to `PREREG_P4.md`. Their commit times are the authoritative timestamps.
