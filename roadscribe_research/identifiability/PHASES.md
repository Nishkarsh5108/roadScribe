# Research plan: identifiability of sensor-language claims

*Drafted 2026-11-12. This plan replaces the SCR-ET "paradigm" framing. Background is in `../NOVELTY_AUDIT_2026-11-11.md`, including its Section II addendum.*

## The question

**Can the data a sensor-language system receives actually determine the claims it makes?**

Some quantities can't be recovered from the input. Speed, vehicle (the channel), mount, gain and clock vary between recordings, and some claims depend on which of them produced the signal. Two examples:

- From one traversal by one car, "the road is rough" and "the car transmits a lot" can produce the same spectrum.
- A sensor placed differently changes "how vigorous was this activity".

A claim about a quantity the data can't determine is a guess. Verifiers that check claims against a record (a ledger, a judge, a reference caption) can't catch it, because the record carries the same confound.

## What would be new

The mathematics is known (spectral disentangling; Bedell et al. 2019; drive-by multi-speed separation; blind-identification diversity) and is cited, not claimed. The contribution would be:

1. **A criterion.** A claim can be judged faithful only if the quantity it is about is identifiable, given the nuisances that vary (rate, channel, mount, gain, clock).
2. **A measurement.** Naturally paired recordings, each holding the content fixed while one nuisance changes, with claim-level metrics:
   - flips of claims that should stay invariant;
   - violations of the covariance law for claims that should change;
   - the rate of claims about non-identifiable quantities.
3. **A fix.** Generation gated by identifiability. When the factorised model's Fisher information or profile likelihood does not pin the quantity down, the system states the ambiguity and says what data would resolve it.

The machinery (convex Whittle SCR, evidence tree, recomputing verifier) is the instrument and the baseline, not the novelty.

## Status after the night of 2026-11-12

Details are in `WORKLOG.md`.

| Phase | Status | Gate outcome |
|---|---|---|
| 0: integrity | **Done** (`092ae35`) | Fabricated numbers replaced by measured ones; the paper compiles |
| 1: pilot P1 | **Done** | Gate **passed** for the phone-like dashboard sensor: road κ 0.72, vehicle share 18.2%. **Not** passed for the above-suspension IMU (κ 0.86, share 2.9%; exploratory X1). The speed elasticity flips sign across places (−0.64 vs +0.86 within) at every placement |
| 2: SCR v0, pilot P2 | **Done, with caveats** | The gate (rank agreement better than RoadScribe's index) passed: +0.06 [0.02, 0.12]. But a gain-only calibration removes the vehicle main effect equally well; most of the rank gain comes from a spatial-frequency index, not channel removal; SCR keeps more speed dependence (P2-H4 failed); and SCR predicts felt vibration *worse* than calibrated γ + gains (P3 failed) |
| Narrative claims (exploratory X2) | **Done** | In RoadScribe's own template narratives, co-located road-surface claims for different cars disagree in about 80% of stretch pairs (pairs are not independent; see the log) |

**Decisions that follow**
- **The identifiability question stands, and its evidence is simple.** It needs no spectral model: a vehicle signature in the "road" claims, and a sign-flipped speed effect estimated across places.
- **The default instrument for Phase 4 becomes:**
  - per-vehicle gains learned on shared roads;
  - a within-place speed elasticity;
  - a spatial-frequency road index.
- **SCR's spectral factorisation is optional future work.** It must first fix its bias and its per-place road parametrisation, and beat the scalar model on P3.

**Before any claim of novelty**
- Run a proper literature check on endogenous speed (drivers slowing on rough roads) in crowdsourced roughness work. One search found nothing, which is not proof.
- Botshekan et al. (2020) and others already separate road and vehicle from phones, so only the *claim-level* framing and the endogeneity finding are candidates for novelty.

## Phases and gates

### Phase 0: integrity and an honest baseline (night of 2026-11-12)

- Remove the fabricated numbers: E1, E2, P_G, the bootstrap and McNemar claims, and the 28-driver claim.
- Fix the anomaly-grounding scoring: score per trip, add a random-placement control, and report point-only matches.
- Regenerate `data/evaluation_summary.json` from code. Add regression tests that fail if pooling across trips comes back.

**Gate:** none. This must happen whatever is decided next.

### Phase 1: pilot P1, cross-vehicle claim agreement on PVS (night of 2026-11-12)

Pre-registered in `PREREG_P1.md`, committed before the analysis runs.

- **Data:** for each route, the 3 vehicles are aligned on a common GNSS route coordinate, and the route is cut into fixed stretches.
- **Felt claims:** ISO a_v and comfort band. These are *expected* to differ across vehicles.
- **Road claims:** RoadScribe's speed-decoupled index and a distance-domain spectral index. These *should* agree.
- **Analysis:** agreement (κ, Spearman), a variance decomposition over location, vehicle and speed, a check for speed endogeneity, and agreement with the video-annotated quality labels.

**Kill gate (roads):** if the road claims already agree across vehicles (κ ≥ 0.8) and vehicle identity explains less than 5% of the variance in log energy once location and speed are accounted for, then roads do not show the problem. In that case, either drop roads as the lead domain or drop the direction.

### Phase 2: SCR instrument v0 (started night of 2026-11-12, finishes in week 1)

- **Estimator:** convex Whittle fit `log S = A_vehicle(f) + B_segment(f/ρ) − log ρ`, with B-spline bases and smoothness penalties.
- **Synthetic recovery tests:** cover 1, 2 and many speeds, to show the identifiability cases in C6.
- **Within-vehicle test:** same road and same time, with dashboard, above-suspension and below-suspension channels.
- **Cross-vehicle test:** the vehicle channel is calibrated on other routes (leave-one-route-out). Compare the road index's cross-vehicle agreement and its accuracy on the quality labels against the baselines.

**Gate:** SCR's road term must beat the γ-normalised index on cross-vehicle agreement. If it does not, the instrument is not good enough to measure identifiability, and the plan must be fixed before anything is built on it.

### Phase 3: paired-nuisance benchmark (weeks 2–5)

Collect naturally paired recordings across domains:

| Dataset | What is held fixed | What varies |
|---|---|---|
| PVS | Locations | 3 vehicles × speeds |
| Penn HaTT | Textures | Speeds and forces |
| Ottawa bearings | Faults | Speed profiles |
| RealWorld HAR | Activities | 7 placements |
| CWRU (optional) | Faults | Loads |

For each domain, build a taxonomy of claims: invariant, covariant (with its transformation law), and non-identifiable from a single recording. Then build question templates whose gold answers come from ground-truth labels, never from a system's own ledger.

**Gate:** at least 3 domains with clean pairs and a usable taxonomy.

### Phase 4: measuring current systems (weeks 4–7)

**Systems:**
- ChatTS (open weights, 4-bit)
- a SensorLLM-style encoder plus projector
- Time-LLM
- MOMENT or Chronos plus an adapter
- plots fed to Qwen2.5-VL-3B
- a code agent working on the raw signal
- the RoadScribe template

**Measurements:**
- invariant-claim flip rate;
- covariance-law violations;
- rate of non-identifiable claims;
- whether larger models fix any of these.

**Gate:** this is the paper's headline. If current systems rarely make such claims, there is no paper. Stop, or pivot to a short benchmark note.

### Phase 5: identifiability-gated generation (weeks 6–9)

- Gate each claim on the factorised model's Fisher information or profile likelihood.
- Wording: say what is ambiguous and what would resolve it.
- Training: DPO, reusing the existing perturbation generator.
- Report the trade-off between gating rate and informativeness.

### Phase 6: write-up (weeks 9–12)

A finding paper with benchmark, criterion and fix, aimed at NeurIPS (main track or Datasets & Benchmarks) or IMWUT. The human study is optional and only for IMWUT.

#### Phase 3 data sources (checked 2026-11-12)

| Dataset | Nuisance | What it offers | Source | Caveat |
|---|---|---|---|---|
| PVS | Vehicle, speed | 3 vehicles × the same 3 routes. Labels: speed bumps, surface type, and per-side road quality (good/regular/bad) | Local copy | One traversal per vehicle and route |
| Ottawa time-varying-speed bearings (Huang & Baddour, Data in Brief 2018) | Rate | v2 has 60 recordings: 10 s at 200 kHz, an accelerometer, and an encoder speed channel. Faults: healthy, inner race, outer race, ball, combined. Speed profiles: increasing, decreasing, up-down, down-up | https://data.mendeley.com/datasets/v43hmbwxpm, DOI 10.17632/v43hmbwxpm.2 | The rate is measured, so this is the cleanest SCR test |
| Penn Haptic Texture Toolkit (Culbertson et al. 2014) | Sliding speed, force | 100 textures. Two 10 s tool recordings per texture: 3-axis acceleration, force and speed, with natural speed variation | https://grasp.upenn.edu/projects/the-penn-haptic-texture-toolkit | Non-commercial research licence; one tool, so only the common channel is identifiable |
| RealWorld HAR (Sztyler & Stuckenschmidt, PerCom 2016) | Sensor placement | Accelerometer, gyroscope and others at several body positions, 8 activities | https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/ | "7 positions, 15 subjects" is from memory; an aggregator lists different numbers. **Verify before use** |

### Track R: a short, honest RoadScribe paper (optional)

Only if the user wants something for the late-November deadline. It would contain:
- the PVS sensing findings;
- the verifier stress test on synthetic corruptions;
- the SLM result as it stands, which is negative;
- Phase 0's honest numbers.

Do not let this track consume Phases 1–2.
