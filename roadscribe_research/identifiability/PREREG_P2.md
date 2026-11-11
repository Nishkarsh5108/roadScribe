# Pre-registration P2: does channel calibration make road claims vehicle-invariant?

*Written 2026-11-12, about 01:05 IST, while P1 was still running. P1's results had not been seen. Committed before P2 runs.*

## Question

Pilot P1 asks whether RoadScribe's road claims agree across vehicles. P2 asks whether a vehicle-channel calibration fixes the disagreement. The calibration is estimated on *other* roads by the SCR fit `log S = A_vehicle(f) + B_segment(f/ρ) − log ρ`. P2 also asks whether the frequency-dependent part of that calibration adds anything beyond a plain per-vehicle gain. If it does not, the SCR factorisation is unnecessary for this purpose.

## Design: leave-one-route-out

For each held-out route r:

1. **Calibration**, on the other two routes with all three vehicles:
   - Fit SCR with channel = vehicle, segments = 100 m stretches of those routes, and smoothness edges between consecutive stretches.
   - Penalties: λ_A = λ_B = 10, λ_θ = 1. These are the package defaults, fixed now.
   - Also fit the gain-only calibration: the vehicle fixed effects of log a_dec ~ location FE + vehicle FE + log v̄ on the same two routes.
2. **Evaluation** on route r, using P1's primary stretch set (100 m, valid for all three vehicles, v̄ ≥ 5 m/s in all three).

**Frames:**
- unweighted gravity-free vertical acceleration (dashboard, vehicle frame);
- 256-sample frames, hop 128, K = 3 DPSS tapers (NW = 2), f from 0.5 to 20 Hz;
- frame speed = mean v over the frame, which must be at least 5 m/s;
- frame θ = θ at the frame centre, which must be valid.

**Indices per (trip, stretch):**

| Index | Definition |
|---|---|
| `a_dec` | RoadScribe's road value: a_wz · (10/v̄)^0.35 (from P1) |
| `a_dec_gain` | `a_dec` divided by the calibrated vehicle gain |
| `R3_uncal` | Intercept c₀ of an OLS fit of r = log I + log ρ on (log ν − log 0.3), over all frame–frequency cells in the stretch with ν = f/ρ in [0.05, 2] cycles/m |
| `R3_cal` | The same, with r = log I − Â_vehicle(f) + log ρ |

## Metrics

All metrics pool the rows of the three held-out routes.

- (a) **Vehicle share** of the variance: type-II share in index ~ location FE + vehicle FE + log v̄. Lower is better.
- (b) **Cross-vehicle agreement:** mean pairwise Spearman across vehicles at matched locations. Higher is better.
- (c) **Leave-one-vehicle-out accuracy** for the ground-truth quality label (good/regular/bad), plain and balanced, with cut-points fitted on the other two vehicles.
- (d) **Within-location slope of the index on log v̄.** Values near 0 mean the speed effect has been removed.

## Hypotheses

| ID | Prediction |
|---|---|
| P2-H1 | Vehicle share of R3_cal < vehicle share of a_dec, **and** < that of R3_uncal |
| **P2-H2 (gate)** | Spearman(R3_cal) > Spearman(a_dec) |
| P2-H3 | R3_cal beats a_dec_gain by ≥ 2 points of vehicle share, **or** by ≥ 0.05 in Spearman. If not, a plain gain calibration does as well and the spectral factorisation adds nothing here |
| P2-H4 | \|within-location slope\| of R3_cal < that of log a_dec |

**Gate (PHASES.md, Phase 2):** if P2-H2 fails, the instrument is not good enough to measure identifiability on PVS, and the plan must change before anything is built on it.

## Uncertainty and reporting

- **Uncertainty:** block bootstrap as in P1 (blocks of 10 consecutive stretches within a route). Differences between indices are computed inside each draw.
- **Exploratory** (labelled as such): penalty sensitivity (all λ × 0.1 and × 10); the per-route results.

**Known limitations:**
- Each route has one traversal per vehicle, so on the held-out route the speed diversity at a location is only the three vehicles' speeds.
- Tyre non-uniformity is a vehicle-specific source indexed by distance (about 1/(π·D) ≈ 0.5 cycles/m). The model attributes it to the road term, and it may leak into the index.
