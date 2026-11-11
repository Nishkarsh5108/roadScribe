# Pre-registration P1: do RoadScribe's road claims hold across vehicles?

*Written 2026-11-12 and committed at 00:21:48 IST (commit `6a135d8`), before any vibration data was read for this analysis. The header first said "about 00:50"; that was a guess, corrected from the commit time.*

**Already seen before writing this:**
- A GNSS-only feasibility check. The tracks of the three vehicles overlap: median gap 2.5–4.7 m, 95th percentile at most 8 m. Each route is driven in the same direction, path lengths agree within 0.3%, and all three routes are loops.
- Labels and vibration statistics were not examined for this question.
- Known from earlier sessions: PVS drivers slow to 3.6–5.1 m/s before speed bumps (2026-11-01).

## Question

RoadScribe makes two kinds of claim about each stretch of road:

| Claim | Quantity | What it describes |
|---|---|---|
| **Felt** | ISO a_v (W_k/W_d-weighted vector sum), mapped to a comfort class 0–5 | The ride in *this* car at *this* speed. Should differ between vehicles. |
| **Road** | a_wz · (v_ref / v̄)^γ, with γ = 0.35 and v_ref = 10 m/s, mapped to the same classes (`iso_road`) | The road. Presented as a property of the road, so it **should agree** across vehicles at the same place. |

P1 measures whether road claims agree across the three vehicle–driver pairs at matched locations, and how much of their variation belongs to the vehicle rather than the place.

## Data and processing (fixed in advance)

- **Trips:** PVS trips 1–9, left logger, dashboard IMU as the primary sensor. Routes A = {1, 4, 7}, B = {2, 5, 8}, C = {3, 6, 9}. Vehicle = 1 + (trip − 1) // 3.
- **Pipeline:** RoadScribe's own code is used unchanged: `estimate_orientation`, `to_vehicle_frame`, `weight_vehicle_axes`, with the config defaults.
- **Route coordinate θ:**
  - The vehicle-1 trip's GNSS track, densified to 1 m, is the reference polyline for its route.
  - Each trip's fixes are matched in time order. Each fix goes to the nearest reference point within a forward window of −30 m to +300 m from the previous match, starting at 0.
  - Samples get θ by linear interpolation in time.
  - Samples whose fix lies more than 15 m from the reference are invalid.
- **Stretches:** 100 m bins of θ (primary); 50 m (secondary). A stretch counts for a trip if at least 70% of its length is covered by valid samples.
- **Per (trip, stretch):**
  - mean speed v̄, from GNSS speed interpolated to samples;
  - RMS of the weighted axes a_wx, a_wy, a_wz;
  - felt a_v = √(a_wx² + a_wy² + a_wz²) and felt class `comfort_class(a_v)`;
  - road value a_dec = a_wz · (10 / v̄)^0.35 and road class `comfort_class(a_dec)`;
  - secondary index R2: log of the vertical (unweighted) acceleration energy in the temporal band [0.1 v̄, 1.0 v̄] Hz, i.e. the spatial band 0.1–1 cycles/m;
  - ground truth: majority left-side quality (good = 0, regular = 1, bad = 2), majority surface, and whether a speed bump occurs.
- **Primary analysis set:** stretches valid in all three trips of a route with v̄ ≥ 5 m/s in all three. 5 m/s is RoadScribe's own floor for road claims. **Secondary set:** the same set with bump stretches removed.

## Hypotheses and predictions

| ID | Hypothesis | Test | Prediction |
|---|---|---|---|
| H1 | Felt claims differ across vehicles; this is legitimate covariance | Mean pairwise quadratic-weighted κ of felt class | κ_felt < 0.6 |
| **H2 (main)** | Road claims are not vehicle-invariant | (a) Mean pairwise weighted κ of road class, **and** (b) vehicle share of variance in log a_dec in an OLS with location fixed effects, vehicle fixed effects and log v̄ | (a) κ_road < 0.8 **and** (b) vehicle share ≥ 5% |
| H3 | Speed is endogenous | (a) Spearman correlation between location roughness (location mean of log a_wz) and location mean speed; (b) γ̂_within (location and vehicle FE) minus γ̂_pooled (no FE), from log a_wz on log v̄ | (a) ρ < −0.2; (b) difference > 0.1 |
| H4 (sanity) | Ground-truth quality labels agree across trips at the same stretch | Mean pairwise weighted κ of GT quality | κ_GT ≥ 0.6. If not, the GT-based results are flagged as uninterpretable. |
| H5 | Speed decoupling makes road claims transfer across vehicles better than felt claims | Leave-one-vehicle-out accuracy predicting GT quality (0/1/2) from a 1-D index, with cut-points fitted on the other two vehicles | accuracy(a_dec) − accuracy(a_v) ≥ 5 points |

**Kill criterion** (from PHASES.md): if κ_road ≥ 0.8 and the vehicle share is below 5%, roads do not show the identifiability problem, and roads stop being the lead domain.

## Reporting

- **Uncertainty:** 95% CIs by block bootstrap. Blocks are 10 consecutive stretches (1 km) within a route, with 2,000 resamples.
- **Also reported:** the share of matched pairs whose road class differs by ≥ 1 and ≥ 2 classes; the full variance decomposition (location, vehicle, speed, residual); per-route values; the secondary sensors (above and below suspension) as exploratory.
- **Failed hypotheses** are reported as failed. Analyses not listed here are labelled exploratory.

**Known limitations:**
- Vehicle and driver are confounded, because each vehicle has one driver.
- There is one traversal per vehicle and route, so measurement noise cannot be separated from the location × vehicle interaction.
- The left-wheel track depends on the driver's lane position.
