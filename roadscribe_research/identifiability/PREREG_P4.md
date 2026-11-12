# Pre-registration P4: do road claims agree across vehicles once each vehicle is calibrated on shared roads?

*Written 2026-11-12 and committed before the analysis ran. The commit time is authoritative.*

## Question

P1 showed that RoadScribe's road claims carry an 18% vehicle share. They also keep a within-place speed slope of 0.51, because γ = 0.35 is too small. Calibration is *identifiable* when several vehicles drive the same places. P4 tests whether calibrating each vehicle on **other** roads makes the road claims about a held-out road agree across vehicles, both in the road index and in the narrator's text.

## Method (leave one route out)

For each held-out route r:

1. **Calibrate** on the other two routes using P1's stretch tables (100 m; primary set). Fit
   log a_wz ~ location FE + vehicle FE + log v̄
   with the within estimator. Take γ̂_r as the slope, and the vehicle gains ĝ_{r,v} as the vehicle effects centred to mean 0.
2. **Calibrated road value** on route r: a_cal = a_wz · (10 / v̄)^γ̂_r · exp(−ĝ_{r,v}), with class `comfort_class(a_cal)`. In the package this is `SpeedConfig(gamma=γ̂_r, vehicle_gain_log=ĝ_{r,v})`; the defaults reproduce the old behaviour.
3. **Stretch level:** compare the calibrated road class with P1's uncalibrated one, on the same 310 places.
4. **Narrative level:**
   - Run the full RoadScribe pipeline per trip with the calibrated `SpeedConfig`.
   - Have the template narrator write both personas.
   - Measure cross-vehicle disagreement of road-surface claims exactly as in X2: co-located 100 m stretch pairs, plus the number of distinct claims.
   - Save the calibrated capsules for the SLM run (`data/processed/pvs_capsules_cal/`).

## Hypotheses

| ID | Prediction |
|---|---|
| P4-H1 | κ_road(calibrated) > 0.718 (P1's uncalibrated value), **and** the share of places where all three vehicles agree exceeds P1's 4.5% |
| P4-H2 | Vehicle share of log a_cal (location FE + vehicle FE + log v̄) < 5% |
| P4-H3 | \|within-place slope of log a_cal on log v̄\| < 0.2. P1's uncalibrated value was 0.51 |
| P4-H4 | In the template narratives, the share of co-located road-claim stretch pairs that disagree is below the uncalibrated 79.7% |

**Uncertainty:** block bootstrap CIs as in P1 for κ and vehicle share (2,000 resamples), plus the CI of the κ difference.

**Caveat known in advance:** H4 rests on few claims (24 when uncalibrated). It is reported with its claim count and is not over-interpreted.

**Why the result could fail:** comfort-class cut-points are fixed. Calibration also moves values across cut-points, so κ could fall even when the continuous agreement improves. Spearman agreement is reported too.
