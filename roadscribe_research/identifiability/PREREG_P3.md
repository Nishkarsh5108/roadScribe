# Pre-registration P3: does the SCR model predict covariant (felt) claims for an unseen vehicle?

*Written 2026-11-12 and committed before P1 or P2 results were seen. The commit time in git is authoritative.*

## Question

Felt comfort is *supposed* to change with vehicle and speed. A useful model of the nuisances should predict *how* it changes. For each place on a held-out route, can vehicle B's felt vertical vibration be predicted from two things?
- the place's road spectrum, estimated from the **other two vehicles** at their own speeds;
- B's channel, estimated on **other routes**.

And is that prediction better than the strongest scalar alternative, a calibrated speed elasticity γ plus per-vehicle gains?

## Design: leave one route out

For the held-out route r and each location in P1's primary set (100 m, valid for all three vehicles, v̄ ≥ 5 m/s), take each target vehicle B in turn.

**Inputs**
- The channels Â_1, Â_2, Â_3 come from an SCR fit on the other two routes. Settings are identical to P2 (λ_A = λ_B = 10, λ_θ = 1).
- Frames are identical to P2: dashboard, unweighted vertical, K = 3, f from 0.5 to 20 Hz, ρ ≥ 5 m/s.

**SCR prediction**
1. Pool the other two vehicles' frames at the location.
2. Fit the place's road term as a power law in spatial frequency ν = f/ρ, by OLS of r = log I − Â_X(f) + log ρ on (log ν − log 0.3). Use all cells.
3. For each of B's frames at the location, predict log I = Â_B(f) + B̂(f/ρ_B) − log ρ_B.
4. Convert to S by removing the χ² log bias, ψ(3) − log 3.
5. Compute the predicted band-limited weighted RMS: a_wz² = mean over frames of Σ_f |W_k(f)|² S Δf.

**Measured value:** the same band-limited weighted RMS, computed from B's own periodograms.

**Baselines** (prediction = mean over the other two vehicles X):

| Baseline | Prediction for each X |
|---|---|
| γ + gains | log a_wz,X + γ̂·(log ρ̄_B − log ρ̄_X) + (ĝ_B − ĝ_X), with γ̂ and ĝ from the within-location OLS of log a_wz on location FE, vehicle FE and log v̄, fitted on the other two routes |
| Naive | log a_wz,X (no adjustment) |

## Metrics

| Metric | Direction |
|---|---|
| RMSE of the log a_wz prediction, pooled over all held-out (location, vehicle) pairs | lower is better |
| Exact agreement of comfort class between predicted and measured a_wz (ISO cut-points) | higher is better |
| Weighted κ of comfort class | higher is better |

## Hypotheses

- **P3-H1:** RMSE(SCR) < RMSE(γ + gains).
- **P3-H2:** exact comfort-class agreement, SCR > γ + gains.
- **Sanity check:** both beat the naive baseline. If they do not, speed and vehicle differences are too small on PVS for this test to say anything.

**Uncertainty:** block bootstrap of the RMSE difference (blocks of 10 consecutive stretches within a route, 2,000 resamples).

**Limitation:** the road term is a power law in ν per place, which is a strong simplification. A failure of P3-H1 could therefore mean either that the model is wrong or that this road parametrisation is too crude. Both readings will be reported.
