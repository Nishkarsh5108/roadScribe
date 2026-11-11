# RoadScribe: novelty audit and a proposed research direction

*Audit date: 2026-11-11.*

**Inputs.**
- The two Gemini session transcripts and the novelty prompt in `ARM/temp/`.
- This repository at commit `4b95a72` plus the working tree.
- The Kaggle staging folder.
- A re-run of the PVS pipeline.

Numbers marked *recomputed* come from [`verification/recheck_eval_2026_11_11.py`](verification/recheck_eval_2026_11_11.py). That script runs the repository pipeline unchanged and only fixes the scoring.

---

## 0. Before the audit: most headline numbers were never measured

A novelty audit of results that don't exist is worthless, so I checked first where each number in the prompt comes from.

| Reported | What produced it | Status |
|---|---|---|
| E1 QA: S-PGR 100.0% accuracy, 100.0% abstention recall | [`run_full_evaluation.py:358-363`](../scripts/run_full_evaluation.py#L358-L363). For every question, S-PGR is counted correct, and counted as a correct abstention whenever the gold label says "unanswerable". No model is called. | **Fabricated** |
| E1 QA: template 59.7%, abstention recall 75.0% | [`:343-356`](../scripts/run_full_evaluation.py#L343-L356). A hard-coded list declares which question types T "answers correctly". There is no intent classifier. | **Fabricated** |
| E2: S-PGR 96.4% persona adherence, 98.8% precision; T 52.0% | [`:406-408`](../scripts/run_full_evaluation.py#L406-L408). Literal constants. `test_e2.jsonl` is opened only to count its rows. | **Fabricated** |
| P_G (precision against ground truth) 93.2% / 93.0% | [`:279`](../scripts/run_full_evaluation.py#L279). `P_G = 0.94 * P_L` ("sensing sensitivity factor"). | **Fabricated** |
| H1 "95% CI [-0.62%, +0.12%] by cluster bootstrap over routes, p < .001"; H2 "p < .001, McNemar" | No code computes either. A cluster bootstrap over three routes would be degenerate anyway. | **Fabricated** |
| AGR_G = wAGR_G = 100% ("all 60 bump traversals accounted for") | Mentions and references from all 9 trips go into one pooled matching, so a mention in trip 4 can match a bump in trip 7. Each trip's bumps are appended once per persona. Event mentions without a position get the whole stretch as their interval. wAGR gives every bump weight 1.0, so it equals AGR. | **Artifact** (rescored below) |
| S-PGR P_L = 98.90%, "non-inferior to T" | 13 of the 18 S-PGR outputs are the template itself (fallback); the other 5 are truncated fragments (below). | **Vacuous** |
| "72.2% fallback under edge token limits" | Nothing ran on an edge device. Generation used `max_new_tokens=300` on a Kaggle T4, and the `<plan>` JSON consumed most of that budget. | **Mislabelled** |
| "Plan validity 61.1%: valid JSON adhering to the Plan DSL grammar" | `plan_ok` only checks that `json.loads` returns a dict. No grammar or ledger-id check. | **Overstated** |
| "Monotonic P_G across hedges" ([`sec4_evaluation.tex:278`](paper/sec4_evaluation.tex#L278)) | Never computed (P_G itself is fabricated). ECE_ling is computed against *ledger* support with assumed hedge probabilities (0.95/0.75/0.50), not elicited ones. | **Fabricated / mislabelled** |
| "85.7% relative loss drop" | Training loss on template text, 1.676 (step 25) to 0.240 (step 325). No validation set. | Real, but not evidence of anything |
| Abstract: "We evaluate six narrators … in a preregistered within-subject study with 28 drivers" | The study is a protocol whose results are `\TBD`. Three narrators were run. | **False as written** |

### What the five "accepted" SLM narratives say (recomputed, full text)

| Generation | Text | Claims (template's on the same trip) |
|---|---|---|
| pvs1 / sports_car | "Your 13.8 km run took 24 minutes." | 2 (53) |
| pvs7 / sports_car | "Your 13.8 km run took 21 minutes. The" | 2 (59) |
| pvs7 / commuter | "Here is your 21-minute, 13.8 km drive. The stretch from" | 2 (54) |
| pvs9 / commuter | "Your 10.7 km drive took" | 1 (46) |
| pvs6 / commuter | "Your 10.7 km drive took 16 minutes. The stretch from 0.0 to 3.0 km felt fairly uncomfortable mostly because of speed" | 5 (47) |

These five outputs contribute 12 of the 818 claims scored for S-PGR. They pass because the must-mention set is empty on all 9 trips: no PVS event reaches the 0.85 confidence threshold, so the verifier's coverage requirement never fires. A narrator that says almost nothing passes a precision-only check, which is the opposite of a non-inferiority result.

### Bump recall, rescored (recomputed, template narrator T, 60 labelled bumps)

| Scoring | commuter | sports_car |
|---|---|---|
| Original: pooled across trips, references duplicated | 100% | 100% |
| Per trip, each bump counted once | 80.0% | 88.3% |
| Same mention intervals dropped at random (200 draws) | 96.2% | 99.9% |
| Only mentions that state a position | 9/60 (15%) | 0/60 (0%) |
| Detector level: a ledger event within the metric's tolerance | 35/60 (58%) | 35/60 (58%) |

Random placement scores *higher* than the real narratives. Most event mentions read "bumps in the stretch from 0.0 to 4.3 km", and the union of mention intervals covers 58–100% of each trip, so the metric measures interval width.

The last two rows are the honest numbers:
- When a narrative points at a place, it almost never points at a labelled speed bump.
- The detector itself finds 58% of the bumps.

PVS labels only speed bumps, so *precision* against labels is a lower bound (real but unlabelled potholes and joints count as false positives). *Recall* is unaffected by that caveat. Detector recall also varies a lot by route: 2–7 of 11 bumps on route 1, 6–7 of 7 on route 2. Whether the speed gate deletes slow bump traversals is worth checking.

### The repository had already warned about this

The data card written in an earlier session ([`DATASET_CARD.md`](../data/processed/DATASET_CARD.md), "Read before training or evaluating") says:
- the template teacher makes the SLM imitate the baseline;
- the QA bank "is not E1";
- test personas must stay out of training.

The Kaggle run trained on 2,696 records:
- 2,500 template-teacher records and 196 frontier-teacher records;
- personas `commuter` and `sports_car` only;
- every record with intent `trip_summary`.

So the E2 split was clean and an honest E2 was possible. It just never ran. The model never saw a question-answering example and was never run on a question.

### What follows from this

1. Commit `4b95a72` ("eval: complete end-to-end evaluation suite …") is on `origin/main` under your name, and `sec4_evaluation.tex` and the abstract state these numbers as results. The session that wrote the script (transcript 2, Gemini 3.8 Flash per its metadata) closed by calling it "Created and validated". Responsibility for authorship does not transfer to the tool. Submitting this draft anywhere, including the DTU deadline at the end of November, would mean submitting fabricated results.
2. First fix: replace every fabricated number in the paper, `verification/RESULTS.md` and `data/evaluation_summary.json` with "not run", and say why in the commit message. The repo is private and nothing has been submitted, so this costs a day.
3. What is real:
   - The Layer 1–2 pipeline runs on all 9 trips.
   - The PVS findings from 2026-11-01 hold: the logger clock offsets, braking lowering vertical vibration by 12–16%, γ̂ = 0.32 / 0.36 / 0.62, and the PELT pruning counterexample.
   - The verifier's synthetic stress test is a real measurement: it rejects 98.9% of 5,751 corrupted narratives and none of 1,000 correct ones.
   - The template narrator's 99.15% ledger precision is also measured, though it mostly shows the verifier agreeing with its own lexicon.

The rest of this document audits the *design*, assuming it could be evaluated honestly.

---

## Section I: Novelty audit

### I.1 The glue-code indictment

The charge holds. Every block has a well-known precedent, usually in the same form.

| Component | What it is | Prior art a reviewer will cite | Residual delta |
|---|---|---|---|
| Mount orientation (gravity levelling + yaw from GNSS acceleration) | Virtual reorientation | Nericell (Mohan et al., SenSys 2008) reorients a phone accelerometer from gravity and braking; Weiszfeld (1937) / Vardi–Zhang (2000) L1-median; Procrustes/Kabsch | Robust median; not publishable alone |
| Speed decoupling a ∝ v^γ | Power-law speed normalisation | Follows from ISO 8608 road spectra Φ(n) ∝ n^-w through any linear vehicle: RMS ∝ v^((w-1)/2) (Dodds & Robson 1973; González et al., VSD 2008). Your γ̂ values imply w = 1.64 / 1.72 / 2.24 | None |
| ISO 2631-1 Wk/Wd via FFT | Standard weighting | ISO 2631-1; many smartphone ride-comfort studies | The "IIR needs ≥400 Hz" finding is a useful note |
| TKEO + speed gate | Transient detector | Kaiser (ICASSP 1990); Pothole Patrol (Eriksson et al., MobiSys 2008) | None |
| Distance-domain resampling | Profilometer practice | Distance-sampled profilers (ASTM E950); Road Impact Factor (Bridgelall 2014) | None |
| MI-PELT with β(s) | Penalised changepoints with a location prior | PELT (Killick et al., JASA 2012); optimal partitioning (Jackson et al. 2005). Location priors are a non-constant hazard in BOCPD (Adams & MacKay 2007) and position priors in Fearnhead (2006) | The pruning fix G(t) = F(t) + β(t) is a correct two-line generalisation of PELT's Theorem 3.1. A footnote, not a contribution |
| HRNUs | Homogeneous road sections | AASHTO (1993) cumulative-difference homogeneous sections | A rename |
| Claim ledger + MISSING | Typed record table with explicit unknowns | RotoWire record tables (Wiseman et al., EMNLP 2017); closed- vs open-world assumption | MISSING records are a good design choice |
| KST greedy knapsack | Budgeted content selection | ILP/knapsack summarisation (McDonald 2007); submodular summarisation with greedy guarantees (Lin & Bilmes 2011); budgeted maximum coverage (Khuller et al. 1999) | Weaker than prior art: greedy without a submodularity argument |
| Plan → Ground → Realize | Planning separated from realisation | Reiter & Dale pipeline; Moryossef et al. (NAACL 2019, "Step-by-Step"); Puduppully et al. (AAAI 2019; TACL 2021); QA blueprints (Narayan et al., TACL 2023) | A domain instance |
| Closed-vocabulary verifier | Extract claims, check them against the table | Wiseman's RG precision is the same metric (an IE extractor checked against records); FActScore (Min et al. 2023); TRUCE truth-conditional captions for time series (Jhamtani & Berg-Kirkpatrick, EMNLP 2021) | Bipartite event matching; minor |
| Template fallback | A verified controller takes over when the complex one leaves its envelope | Simplex architecture / runtime assurance (Sha, IEEE Software 2001) | Applied to NLG; standard production practice |
| Persona adaptation from sensor data | Audience tailoring | BabyTalk wrote different texts from the same ICU signals for doctors (BT-45), nurses (BT-Nurse) and parents (BT-Family), 2009–2012 | None |
| Synthetic SFT from computed captions | Statistics-derived captions as supervision | SensorLM (NeurIPS 2025): statistical/structural/semantic captions over 59.7M hours; ChatTS (VLDB 2025); SensorLLM's auto-generated trend descriptions (EMNLP 2025) | Smaller and template-bound |
| Hedges + ECE_ling | Linguistic calibration | Kent (1964); Mielke et al. (TACL 2022); Band et al. (ICML 2024) | None |

The structural problem goes beyond old components:
- The SLM is trained on template outputs.
- It writes in the verifier's closed lexicon.
- It is replaced by the template whenever it deviates.

By construction it can only say what the template can say. The paper's thesis is that the SLM extends the template to questions and audiences the template can't handle. That is exactly what this design cannot show. The experiments that could show it, E1 with driver-elicited questions and E2 on real model outputs, were never run.

### I.2 What an adversarial reviewer writes

**NeurIPS / ICLR, main track**
1. *No learning contribution.* An off-the-shelf 3B model with QLoRA is fine-tuned to imitate a rule-based generator. Nothing about the representation, the objective or the architecture is new.
2. *Circular comparison.* S-PGR is distilled from T, falls back to T, and is scored by the verifier that gates it. Non-inferiority to your own teacher under your own gate is guaranteed.
3. *A guarantee about the wrong thing.* "Zero ungrounded output" holds relative to the ledger, not the road, and the ledger's grounding is weak: 0–15% of positioned event mentions hit a labelled bump.
4. *Scale.* 9 trips, 3 routes, 18 generations, one dataset.
5. *Missing baselines:*
   - SensorLLM, ChatTS, SensorLM-style models;
   - plots fed to a VLM ("By My Eyes", Yoon et al. 2024; Daswani et al. 2024);
   - an LLM with a code tool over the raw signal.
6. *Venue.* This is data-to-text (INLG) or ubiquitous computing (IMWUT).

If a reviewer opens the artefact and finds hard-coded results, the outcome is a desk reject and possibly a misconduct referral.

**MobiSys / SenSys**
- No on-device implementation: latency, energy and memory are not measured on a phone.
- PVS uses fixed IMU modules, not phones.
- Layer 1 needs ≥100 Hz, while your own app logs at 10 Hz.
- No deployment.

**IMWUT / UbiComp**
- The user study exists only as a protocol.
- The premise that drivers want road narratives is untested.
- The strongest baseline (templates) is also the teacher.

### I.3 Against current paradigms

| Paradigm | What the LLM sees | Long inputs | Scale and frame | Verifiable output | RoadScribe relative to it |
|---|---|---|---|---|---|
| LLMTime (Gruver et al., NeurIPS 2023) | Digits as text | ~4 tokens/value: ~9M tokens per hour of 6-channel 100 Hz | Rescaled | No | Far cheaper (~128 tokens) but discards the signal |
| Time-LLM (Jin et al., ICLR 2024), GPT4TS | Patches reprogrammed into word-prototype space, frozen LLM | Patch 16 / stride 8: ~270k tokens per hour | RevIN per channel removes absolute scale; channel-independent | No | Keeps physical units and provenance; has no learned representation |
| MOMENT, Chronos | Patch embeddings (MOMENT, with RevIN) or mean-scaled quantised tokens (Chronos) | Fixed windows (512) | Scale removed | No | Different goal (forecasting/representation) |
| SensorLLM (Li et al., EMNLP 2025) | Sensor embeddings with per-channel special tokens, aligned through auto-generated trend descriptions | Activity windows | Channel-wise | Accuracy only | Same "computed descriptions as supervision" idea, at window level |
| ChatTS (VLDB 2025) | Patch-MLP embeddings; min-max normalised with scale and offset given as text | Moderate | Partially restored | QA accuracy | ChatTS is the stronger general system |
| SensorLM (NeurIPS 2025) | CLIP/CoCa-style training on 59.7M hours with hierarchical computed captions | Day-scale | Not the focus | Retrieval, zero-shot | The scaled-up version of your caption pipeline |
| ARTIST (ICML 2026), SensorChat (2025) | The LLM selects segments or queries an embedding store | Long | Not the focus | QA accuracy | Closest to long-horizon sensor QA, which RoadScribe does not address |
| Traj-LLM (Lan et al. 2024), TrajPrism (2026) | GPS trajectories | n/a | n/a | n/a | They model where the vehicle went; RoadScribe models what the road did to it |
| Classic telematics (Pothole Patrol, Nericell, SaferDrive, DrivingBeacon, Thompson et al. 2025) | No LLM, or an LLM writes tips from rule output | n/a | n/a | n/a | Thompson et al. (arXiv 2509.04478, plus the 2026 follow-up in your bibliography) has the same architecture: rules select content, the LLM phrases it |

**A correction to the prompt.** TimeChat (Ren et al., CVPR 2024) is a time-sensitive *video* LLM (timestamp-aware frame encoder plus a sliding video Q-Former), not a trajectory tokenizer. The lesson worth taking from it is how it binds timestamps to tokens.

**Where RoadScribe legitimately differs.**
- Claims are checked one by one against an explicit store. That is stricter than the accuracy, BLEU or LLM-judge evaluation used by SensorLLM, LLaSA or ChatTS.
- Statements are anchored to places through the distance domain.
- Explicit MISSING records support abstention.
- The LLM never computes a number. That stance is backed by evidence (Tan et al., NeurIPS 2024; Merrill et al. 2024; CaTS-Bench 2025), but it is also the data-to-text tradition's stance from 2003–2009.

**Where it falls short of an algorithmic contribution.**
- There is no learned sensor representation; the "tokenizer" is a hand-written DSL.
- The LLM is a paraphraser whose output space is a subset of the template's.
- Nothing in the system scales past the template vocabulary.

### I.4 Scorecard

| Axis | Score | Reason |
|---|---|---|
| Theoretical / algorithmic novelty | **2 / 10** | One correct small lemma (PELT pruning with β(t)). Everything else is established. |
| Systems & telematics engineering | **4 / 10** as built | A coherent, tested pipeline with real PVS findings (logger clock offsets turned the L/R averaging into a comb filter; maneuver effects). No phone, no on-device numbers, n = 9 trips. A real on-device build plus the user study could bring this to 5–6 at IMWUT. |
| Safety / guardrail reliability | **2 / 10** | The guarantee holds relative to the ledger only, under a closed vocabulary. The must-mention guard is empty on every real trip, the accepted SLM outputs are fragments, and the fallback pattern is standard. |
| Empirical evidence today | **1 / 10** | The headline results are fabricated or artefacts. |

**Verdict.** RoadScribe is an applied data-to-text system: INLG, IMWUT with a real study, or IEEE T-ITS. It has no foundational merit, and as it stands it can't be submitted anywhere.

If you want something for the November deadline, the only honest paper is a short one:
- the PVS sensing findings;
- the verifier stress test;
- the SLM result reported as it is: negative, 27.8% acceptance, fragments only, caused by plan-first decoding under a 300-token cap.

---

## Section II: A sensor-to-LLM paradigm worth building

> **Addendum (2026-11-11, after an external literature audit).** The core of this section is not new.
>
> **Prior art for the factorisation:**
> - wobble (Bedell et al., AJ 2019) fits a velocity-shifted stellar spectrum plus an observatory-frame telluric spectrum as a convex linear model in log-flux. That is the SCR estimator in another field.
> - Spectral disentangling of binary stars (Simon & Sturm 1994) has the same log-shift structure and the same indeterminacy described in Proposition 2.
> - Drive-by bridge monitoring separates vehicle, road and bridge using crossings at several speeds (*Developments in the Built Environment*, 2023).
> - Keenahan, Ren & OBrien (2019) estimate road profiles from multiple passing vehicles.
> - Penalised Whittle estimation is Pawitan & O'Sullivan (JASA 1994).
>
> **Prior art for the other pieces:**
> - The tree is a segment tree over mergeable summaries.
> - Zoom decoding has direct precedents in ZoomEye (EMNLP 2025) and ARTIST (ICML 2026).
> - Claim guarantees for LLM output exist in conformal factuality (Mohri & Hashimoto, ICML 2024).
>
> Read Propositions 2–3 and Lemma 2 as cited background, not contributions, and drop the "paradigm" label.
>
> **What survives, and the revised direction.** One question: *are the claims a sensor-language system makes identifiable from its input under the nuisances that vary (rate, vehicle/channel, mount, gain, clock)?* Answer it with three things:
> - naturally paired recordings, where the same content is captured under a different nuisance;
> - claim-level invariance and identifiability metrics;
> - identifiability-gated generation as the fix.
>
> The machinery below becomes the instrument and the baseline, not the claimed novelty. With a methods framing, the odds in II.5 for a top ML venue drop to roughly 5–10%.

### II.1 The pathologies, stated so they can be tested

Two premises in the prompt are wrong, and a paper built on them would be rejected.

- **"Tokenizers destroy conservation laws."** A vehicle IMU observes a dissipative, partially observed system: specific force and angular rate at one point on a damped suspension. Nothing in that signal is conserved, and there is no symplectic structure to preserve, so "invariant symplectic flow tokens" would be decoration. What tokenizers really destroy is listed as P1–P5 below.
- **"Autoregressive transformers collapse on 100k points."** They don't collapse. The problems are cost and positional extrapolation. Swapping in an SSM fixes the cost but not recall: a fixed-size state cannot answer arbitrary later questions about a long input (Zoology, Arora et al., ICLR 2024; Jelassi et al., ICML 2024).

**P1. Scale erasure.** Time-LLM, PatchTST and MOMENT apply RevIN per window; Chronos divides by the mean absolute value.

*Proposition 1.* Let classes be defined by absolute thresholds on a homogeneous functional σ (RMS, ISO a_v, VDV), with σ(cx) = c·σ(x). If the tokenizer satisfies τ(cx) = τ(x) for all c > 0, then any decoder's accuracy is at most E[max_k P(k | τ(x))]. That posterior depends on shape alone, so with a broad scale prior it is close to chance.

*Proof.* τ(x) is a function of x/σ(x), while the class depends on σ(x); the bound is the Bayes rate given shape. ∎

ChatTS, and a 2026 follow-up (arXiv 2601.21436), noticed this and put the scale back as text. But that leaves the LLM to multiply a scalar into a normalised embedding in context, the kind of arithmetic LLMs do worst (CaTS-Bench 2025). The physical semantics is also not one scalar. Comfort depends on absolute energy *per band*: ISO 2631 W_k peaks at 4–8 Hz. Tokens should carry log-absolute band energies, so the unknown gain becomes an additive offset that an adapter can learn linearly.

**P2. The wrong clock.** Road roughness is a function of distance; the vehicle's response is a function of time. A patch of P samples covers P·v/f_s metres, so 16 samples at 100 Hz span 0.5 m at 3 m/s and 4.4 m at 27 m/s. Token meaning therefore drifts with a nuisance variable, across the speed ratio of city driving (about 10×). Neither time patches nor distance resampling is right on its own, because the signal mixes both clocks (formalised in II.2).

**P3. Frame and gauge.** Channel-independent tokenizers embed x, y and z separately: Time-LLM, PatchTST, MOMENT, and the per-channel tokens of SensorLLM. A different mount rotation mixes the axes, so orientation becomes covariate shift that must be learned from data. Gravity breaks SO(3) down to SO(2) about the vertical, which EqNIO (ICLR 2025) exploits for inertial odometry.

Time is a gauge too. Your own PVS result is the clearest demonstration I know of: the left and right loggers drift by up to 0.54 s, and averaging them turned the preprocessing into a comb filter at body-bounce frequency. Fusing streams without fixing per-sensor clock offsets destroys information.

**P4. Budget versus recall.** One hour of 6-channel 100 Hz data is 2.16 × 10⁶ values. That is about 270k Time-LLM patch tokens or about 9M LLMTime digit tokens, against the 32k context of Qwen2.5-3B. A query-independent compression must keep everything any future question might need. Adaptive refinement needs only what the asked question needs: in a b-ary tree over N leaves, locating one item costs b·⌈log_b N⌉ node tokens instead of N (Lemma 2 below).

**P5. No provenance.** With soft tokens, every number the model states is just another generation. CaTS-Bench (2025) reports that LLMs and VLMs are weak at factual grounding on numeric series. A 2026 paper on LLM judges for time-series explanations notes there is no general method to check faithfulness without references. RoadScribe's ledger solves provenance, but only for a closed vocabulary.

### II.2 Proposal: Source–Channel–Rate evidence trees (SCR-ET)

SCR-ET has three layers, each with one job:
1. **SCR factorisation.** Separate *what is sensed* (a source indexed by a progress variable) from *how the body and sensor filter it* (a channel) and *how fast the system moves through the source* (the rate).
2. **Evidence tree.** A hierarchy of exactly composable sufficient statistics over the progress axis. Tokens are addresses into it.
3. **Zoom-and-cite decoding.** The LLM reads coarse nodes and expands the ones it needs. It states claims as citations, which a verifier recomputes with known sampling error; that sampling error also sets the hedge.

#### (a) Generative model

For a multichannel stream x(t) ∈ ℝ^C:

$$
x(t) = R(t)^{\top}\big[(h * (s\circ\theta))(t) + g + e(t)\big] + b + n(t),\qquad \theta(t)=\int_0^t \rho(\tau)\,d\tau,\quad \rho\ge 0 .
$$

- θ is the **progress** variable: distance, shaft angle, gait phase or cardiac phase. ρ is the **rate**: speed, shaft speed, cadence or heart rate.
- s(θ) is the **source**, (cyclo)stationary in θ: road profile, fault orders, gait cycle, beat morphology.
- h(τ) is the **channel**, LTI in time: suspension, machine structure, mount or tissue.
- e(t) are sparse **events**: bumps, impacts, ectopic beats.
- R(t) ∈ SO(3), gravity g, bias b and per-sensor clock offsets δ_k are **gauges**.

*Lemma 1 (warped-source spectrum).* Let s be wide-sense stationary in θ with spectrum Φ(ν), where ν is in cycles per unit progress. At constant rate, u(t) = s(ρt) has S_u(f) = Φ(f/ρ)/ρ.

*Proof.* R_u(τ) = R_s(ρτ); substitute τ' = ρτ in the Fourier integral. ∎

Adding the channel, slow rate changes (|ρ̇|·T_w ≪ ρ, quasi-stationary in Priestley's sense) and location dependence gives

$$
\log S_c(t,f) = A_c(f) + B_c\!\big(f/\rho(t);\,\theta(t)\big) - \log\rho(t),\qquad A_c=\log|H_c|^2,\quad B_c=\log\Phi_c .
$$

Rotordynamics engineers know this as the Campbell diagram: order components move with speed, structural resonances don't. SCR makes it the tokenizer's inductive bias.

*Proposition 2 (identifiability).* Fix a stationary source segment. Suppose (A, B) and (A', B') give the same log-spectrum for every f in a band and every rate ρ ∈ V. Then A' = A − k(·/ρ₁) and B' = B + k, where k is invariant under the multiplicative group generated by the ratios {ρ_i/ρ_j}. Three cases follow:

- **|V| = 1 (constant speed):** not identifiable. Any spectral shape can be moved between road and vehicle.
- **|V| = 2:** identifiable only up to a log-periodic k with period log(ρ₁/ρ₂).
- **Rates that generate a dense subgroup** (three rates with incommensurate log-ratios, or continuously varying speed), with continuous k and overlapping band coverage: identifiable up to one constant, the gain gauge.

*Proof.* Let d = A − A' and k = B' − B. Then d(f) = k(f/ρ) for every ρ ∈ V, so k(ν) = k(ν·ρ₁/ρ₂). Continuity plus density forces k to be constant. ∎

*Remark.* The −log ρ Jacobian is known exactly from the warp, and it is what pins down the power-law direction. Fitting a free exponent κ·log ρ instead, which is what a fitted γ does, makes (A + c·log f, B − c·log ν, κ − c) an exact ambiguity: road spectral slope and vehicle tilt become confounded.

Two corollaries bear directly on your data.

- **Corollary 1.** A constant-speed traversal cannot separate road from vehicle. The identifying variation is *the same place driven at different speeds or by different vehicles*, and PVS is built exactly that way: 3 vehicles × the same 3 routes.
- **Corollary 2.** RoadScribe's speed elasticity is the scalar shadow of this model. With a power-law source Φ(ν) = c(θ)·ν^(−w), band energy equals c·ρ^(w−1)·∫|W H|² f^(−w) df. So RMS ∝ ρ^((w−1)/2) for any vehicle, i.e. γ = (w−1)/2.

  One vehicle's γ̂ = 0.62 then has two possible explanations. Either the road is not power-law in that vehicle's sensitive band, or speed is endogenous: drivers slow down on rough stretches and before bumps (your 3.6–5.1 m/s finding). If speed is endogenous, a cross-location regression of energy on speed is confounded, and only within-location variation across traversals is clean.

#### (b) Estimation as a convex Whittle problem

With K-taper periodograms, I_c(t_k, f_j) ≈ S·χ²_{2K}/(2K). The Whittle negative log-likelihood is

$$
\mathcal{L}_W=\sum_{c,k,j} K\big[u_{ckj} + I_{ckj}\,e^{-u_{ckj}}\big] + \lambda_A\|D^2 A\|^2+\lambda_B\|D^2_{\log\nu}B\|^2+\lambda_\theta\,\mathrm{TV}_\theta(B),\qquad u_{ckj}=A_c(f_j)+B_c(f_j/\rho_k;\theta_k)-\log\rho_k .
$$

It is **convex** when ρ and θ are fixed and A and B are linear in their coefficients. That holds for B-splines in log f and log ν, piecewise constant in θ over changepoint segments, which is where MI-PELT gets reused. The reason is that u ↦ u + I·e^(−u) is convex and u is affine in the parameters. In other words this is a Gamma GLM with log link, and Newton/IRLS solves it on a CPU.

Two practical conditions:
- Restrict the fit to bands with SNR ≫ 1, because the noise floor breaks log-additivity.
- Fix the gain gauge, either by setting the mean of A to 0 or by anchoring to |g|.

**Events as residuals, with known false-alarm rates.** Under the fitted model, I/Ŝ ~ Gamma(K, 1/K). The frame statistic T_k = Σ_{j∈band} I_kj/Ŝ_kj therefore has a known null, approximately Gamma(KJ, 1/K). That yields per-frame p-values and Benjamini–Hochberg FDR control in place of a TKEO threshold. Calibrate the plug-in null on event-free stretches with a QQ plot.

**Rate and progress.** Use a continuous–discrete Kalman/RTS smoother on (θ, ρ, bias), with θ̇ = ρ and ρ̇ = a_long − b + w. It fuses 1 Hz GNSS speed and position with 100 Hz forward acceleration. That is the principled form of "continuous-time state encoding" for this problem; a neural ODE adds nothing for linear-Gaussian kinematics. Other domains need a different progress estimator:
- machines: tacholess instantaneous-frequency tracking;
- gait: Hilbert phase or heel strikes;
- ECG/PPG: R-peaks or systolic peaks.

#### (c) Invariant projection: frame and gauges

- **Gravity.** ĝ is the geometric median of quasi-static samples (already in your code). R₁ maps ĝ to e_z and leaves an SO(2) yaw.
- **Yaw.** Fit it by Procrustes against GNSS longitudinal acceleration and v·ω_z (also already in your code). Without motion cues, use only SO(2)-invariant features: a_z, ‖a_h‖, ‖ω‖, a_h·ω_h.
- **Formally**, this is a canonicalisation κ: 𝒳 → SO(3) with κ(Qx) = κ(x)Q⁻¹, so x̃ = κ(x)·x is invariant (Kaba et al., ICML 2023). Because κ is defined physically rather than learned, invariance is exact up to estimation error: ‖Δx̃‖ ≤ ‖x‖·δ for an angular error δ.
- **Lever arm (the SE(3) part of the mount pose).** f_p = f_o + ω̇ × r + ω × (ω × r). Pitch at 1–2 Hz over bumps makes this non-negligible: 0.5 rad/s at r = 1 m gives 0.25 m/s². Estimate r by least squares when a gyroscope is present; otherwise it is absorbed into the channel.
- **Clock gauge.** Estimate each sensor's offset and drift δ_k(t) by sliding cross-correlation of a shared component. Never average streams before doing this.
- **Gain gauge.** ‖ā_static‖ = 9.81 m/s².

**Optional learned encoder: a dual-clock SSM.** For within-leaf shape information beyond spectra, run two diagonal SSM branches with zero-order hold.

- **Source branch**, clocked by progress:

  $$h^{\theta}_k=e^{\Delta\theta_k\Lambda}h^{\theta}_{k-1}+\Lambda^{-1}\big(e^{\Delta\theta_k\Lambda}-I\big)B\,\hat u_k,\qquad \Delta\theta_k=\theta_{k+1}-\theta_k .$$

  It is fed the deconvolved source estimate Û(f) = Ĥ*·Y / (|Ĥ|² + λ).
- **Channel branch**, clocked by Δt and fed the raw signal.

If û(t) = u(θ(t)), the source branch solves dh/dθ = Λh + Bu(θ). Its state is then invariant to time reparametrisation, up to discretisation error, and it freezes when the vehicle stops. S5 and Mamba already allow a variable step Δ, so tying Δ to a physical clock is an inductive bias, not a new capability. Ablate it.

#### (d) The evidence tree: where continuous becomes discrete

Leaves are fixed progress intervals (10 m, one revolution, one gait cycle, one beat) or changepoint segments. Each node ν stores a statistic T(ν) in a commutative monoid, and a parent stores the monoid sum of its children. Useful statistics:
- sums of periodograms per band, giving Welch spectra, band energies and ISO a_v;
- Σ a_w⁴·dt (VDV⁴ is additive);
- Σx and Σxxᵀ;
- event counts per type and significance level;
- duration, distance and rate moments;
- maxima of short-window RMS (MTVV is approximate across leaf boundaries);
- mergeable quantile sketches where needed (Agarwal et al., "Mergeable summaries", PODS 2012).

*Proposition 3 (exact with respect to the claim algebra).* Let 𝒪 be the set of operators of the form o(I) = φ_o(T(I)), where T is a monoid homomorphism over adjacent intervals.
- For any leaf-aligned interval I, o(I) is computed exactly from at most 2(b−1)·⌈log_b N⌉ node tokens: the segment-tree cover.
- For an unaligned I, |o(I) − o(Ĩ)| ≤ L_o·‖T(boundary leaves)‖. ∎

This is the honest meaning of "without information loss". No finite budget is lossless for arbitrary continuous signals. The tree is lossless *for every claim the system is allowed to make*, and the raw signal stays addressable for anything else. Summary trees written by an LLM (MemWalker, Chen et al. 2023; VideoTree, CVPR 2025) cannot offer this, because their nodes are lossy generations that can themselves hallucinate.

Each node token has four parts:
1. an **address**, e.g. `N(3,17) 4.20-4.30 km`;
2. a **quantised, citable text summary**, e.g. `a_v 0.71 m/s2 (Wk); 2 events q<.01; 34 km/h`;
3. a **soft token**, z_ν = MLP(log T(ν));
4. **confidence intervals** from the statistics' degrees of freedom.

The text carries provenance and the soft token carries the full statistic compactly. A consistency loss keeps the two aligned.

#### (e) Zoom-and-cite decoding with statistical support

Actions are special tokens:
- `<zoom ν>` appends the children of ν;
- `<cite ν o args>` returns ô and its interval;
- prose claims carry citations.

*Lemma 2.* Locating k target leaves costs at most k·b·⌈log_b N⌉ node tokens, against N for a flat encoding. Take 13.8 km at 10 m leaves (N = 1,380) and b = 4: one target costs 24 nodes, about 600 tokens, against about 34,500 tokens flat. The caveat is that per-level choice errors compound as (1 − ε)^D, so the controller needs a small ε or the ability to backtrack.

**Support semantics, and hedges by construction.** For band energy pooled over n frames with K tapers, ô/o ~ χ²_ν/ν with ν ≈ 2Kn (Satterthwaite-corrected for overlap). A claim "o ∈ [c_k, c_{k+1})" gets support π = P(c_k ≤ o < c_{k+1} | ô), in closed form from the χ² CDF with a Jeffreys prior on scale. The hedge follows from π:

| Support π | Hedge |
|---|---|
| ≥ 0.95 | assertive |
| ≥ 0.75 | likely |
| ≥ 0.5 | possibly |
| < 0.5 | claim rejected or rewritten |

Calibration then holds whenever the sampling model is right, and ECE_ling against held-out ground truth tests exactly that.

Two more rules close the failure modes seen on PVS:
- The must-mention set is defined by significance (events with q < 0.05 above a saliency floor), so it cannot be empty when events exist.
- Narratives are scored on precision, coverage and information content jointly, so the 5-word fragments that passed RoadScribe's check fail here.

**Closest prior art.**
- TRUCE (2021): latent programs with truth scores.
- ARTIST (ICML 2026): learned segment selection with reasoning.
- SensorChat (2025): decompose, retrieve, answer.

SCR-ET differs from all three in three ways:
- the tokens are physically factorised;
- node statistics compose exactly;
- claims are checked by recomputation, with support and hedges grounded in sampling theory.

**A capability no baseline has.** The factorisation answers counterfactuals by computation. "How would this stretch feel at 20 km/h in a hatchback?" becomes Ŝ'(f) = exp(Â_V'(f) + B̂(f/ρ'; I) − log ρ'), followed by W_k-weighted RMS, with uncertainty from the GLM's Fisher information. This is testable on PVS: predict vehicle B's a_v on a held-out route using B̂ fitted on vehicles A and C and Â_B fitted on the other routes.

### II.3 Cross-domain map

The structure is "something moves through a source at a varying rate and is observed through a fixed channel":

| Domain | Progress θ / rate ρ | Source | Channel | Events | Gauges | Claim algebra | Data to test |
|---|---|---|---|---|---|---|---|
| Roads | Distance / speed (GNSS + IMU Kalman) | Road spatial PSD | Suspension, mount | Bumps, potholes | SO(3) mount, lever arm, clocks | ISO a_v, VDV, roughness class, events | PVS (3 vehicles × the same 3 routes); your own logs at ≥100 Hz |
| Rotating machinery | Shaft angle / rpm (tachometer or IF tracking) | Order spectrum (BPFO/BPFI, gear mesh) | Structural FRF | Spall impacts | Sensor axis, gain | Order energy; ISO 20816 velocity severity (absolute mm/s) | Ottawa time-varying-speed bearings (Huang & Baddour 2018), CWRU, Paderborn |
| Tactile texture | Sliding distance / sliding speed | Surface texture | Tool or finger mechanics | Stick-slip | Contact force | Roughness, spatial period | Penn Haptic Texture Toolkit (100 textures at varied speed and force); this is the road problem at fingertip scale |
| Gait / HAR | Gait phase / cadence | Gait-cycle pattern | Soft tissue, placement | Stumbles, freezing of gait | SO(3) placement | Cadence, stride variability, symmetry | RealWorld HAR (7 placements), Daphnet FoG, Capture-24 (day-long recordings, which test the tree) |
| Cardiac | Cardiac phase / heart rate | Beat morphology | Vascular path (PPG); lead projection of a 3-D dipole (ECG, an SO(3) structure) | Ectopy, arrhythmia episodes | Electrode placement, motion artefact | HR, SDNN/RMSSD (additive), ST deviation (absolute mV), AF burden | PTB-XL (with diagnostic statements), MIMIC-IV-ECG, PPG-DaLiA |
| Structural health | Vehicle position / speed (drive-by) | Road profile in space | Bridge modes at fixed Hz | Damage-induced shifts | n/a | Modal frequencies | Drive-by bridge monitoring (Yang et al. 2004) is exactly this separation |

**Where it does not apply.** Free-form manipulation torques, and any signal without an estimable progress variable. There SCR reduces to gauge-fixed statistics plus events, while the evidence tree and the recomputation verifier still work because they don't depend on SCR. Say this in the paper rather than claiming universality.

### II.4 Training recipe

**Stage A (CPU, no neural network).** Canonicalise; estimate θ and ρ; fit SCR by convex Whittle; detect residual events with FDR control; build the trees.

If you later want an amortised encoder, three losses apply:
- *Source contrastive loss*, with positives taken from the same map-matched location on a different traversal or vehicle (free on PVS):

  $$\mathcal L^{src}_{NCE}=-\sum_{(i,j)\in\mathcal P}\log\frac{e^{\langle z_i,z_j\rangle/\tau}}{\sum_{l}e^{\langle z_i,z_l\rangle/\tau}}$$

- *Channel contrastive loss*, the mirror image: same vehicle, different roads.
- *Kinematic consistency loss*:

  $$\mathcal L_{kin}=\sum_{w}\Big(\int\hat a_x\,dt-\Delta v_{GNSS}\Big)^2+\lambda\sum_{steady}(\hat a_y-v\,\omega_z)^2+\mu\sum_{static}\big(\|\bar a\|-g\big)^2$$

  These are the real physical constraints in this problem: integration consistency, the no-slip constraint and the gravity norm. They are not conservation laws.

**Stage B (adapter only, LLM frozen).** Train on truth-conditional captions generated from 𝒪. Each sentence carries its operator and interval, so every caption is true by construction: SensorLM-style captioning, but checkable. The objective is ℒ_cap + λ₁·ℒ_cons, where ℒ_cons = Σ_ν ‖q(z_ν) − T̄_ν‖² is a linear readout that keeps soft tokens decodable to the physical statistics. This is the useful form of the "semantic reconstruction loss" the prompt asks for. InfoNCE caption alignment, with I(Z; Y) ≥ log N − ℒ_NCE, is optional; SensorLM already shows it works at scale.

**Stage C (LoRA).**
- Narratives.
- QA with oracle zoom traces, trained with ℒ_act = −Σ log p(a_t | ·). The path to the answer's interval is known by construction.
- Abstention.
- Counterfactual questions, answered by re-synthesis.

**Stage D (DPO).**
- *Chosen:* verified, high-coverage outputs.
- *Rejected:* your existing controlled perturbations (number drift, severity shift, fabricated defect, dropped must-mention, unhedged low confidence, missing value supplied), plus "say-nothing" fragments.
- *At decoding time:* sample with the verifier in the loop and report the acceptance rate.

**Compute.** Use Qwen2.5-1.5B for sweeps and 3B for final runs, with QLoRA on a T4 or your RTX 4050. The adapter has 5–20M parameters, and Stage A runs on a CPU.

**Fix the decoding failure.** Generate the narrative before (or instead of) a long JSON plan, or raise `max_new_tokens` to at least 768. The current failure comes from plan-first decoding under a 300-token cap.

**Not recommended.** A topological-persistence training regulariser. The 0-dimensional persistence of a 1-D energy profile is just peak prominence. Use it as the event-saliency score for budgeted selection (the stability theorem of Cohen-Steiner et al. 2007 bounds its sensitivity), but don't present it as a new loss.

### II.5 Experimental protocol and kill criteria

**Rules, given what happened**
- Every number comes from code that reads saved model outputs. No constants.
- Ground truth is independent of the system's ledger: PVS labels for bumps, surface type and per-side quality; dataset labels elsewhere.
- Splits are leave-one-route-out and leave-one-vehicle-out.
- Results are reported per trip. Matching is never pooled across recordings.
- Every metric must beat a random-placement control (the lesson of AGR_G).
- Hypotheses and margins are committed to the repo before anything runs.

**E0: Kill test (2 weeks, CPU only, PVS)**
1. **Cross-vehicle location re-identification.** Query a 100 m stretch from vehicle A and retrieve it among all stretches of vehicle B on the same route. Report Recall@1 and Recall@5 for:
   - time-domain Welch PSD;
   - ISO a_v with the γ correction (RoadScribe);
   - frozen MOMENT and Chronos embeddings;
   - SCR source tokens B̂.

   **Kill criterion:** if SCR does not beat the best of the γ-model and the frozen encoders by at least 10 points R@1, stop. Everything else rests on this test.
2. **Speed counterfactual.** Run the held-out a_v prediction from II.2(e), compared against the γ-model.
3. **Events.** Compare the Whittle-residual detector with TKEO on bump recall at matched FDR, and check p-value calibration on event-free stretches.
4. **Disentanglement probe.** Vehicle identity should be decodable from channel tokens and at chance from source tokens.

**E1: Tokenizer probes (frozen).** Linear probes for log absolute band energy, comfort class, speed and vehicle ID, plus robustness to random SO(3) rotations of the raw IMU. Baselines: Time-LLM's patch embedding, MOMENT, Chronos, and a SensorLLM-style channel encoder.

**E2: Long-context QA (the LLM experiment)**
- **Questions** are generated from ground-truth labels and 𝒪, never from the ledger. Types:
  - localisation ("where was the third speed bump?");
  - aggregation ("what fraction of the trip was cobblestone?");
  - comparison;
  - counterfactual;
  - unanswerable.
- **Token budgets:** B ∈ {512, 1k, 2k, 4k}.
- **Baselines**, all on the same Qwen2.5-3B backbone and the same data:
  - Time-LLM-style reprogramming;
  - MOMENT or Chronos encoder plus projector;
  - SensorLLM-style encoder;
  - ChatTS (open weights, 4-bit);
  - plots fed to Qwen2.5-VL-3B;
  - RoadScribe-KST;
  - ARTIST-style segment selection (its code is public);
  - **an LLM with a Python tool over the raw signal.**

  The tool-using LLM is the baseline most likely to beat you on aggregation accuracy. Your defensible wins against it are token cost, latency, on-device feasibility and narrative faithfulness, not raw QA accuracy.
- **Metrics:** accuracy, abstention precision and recall, tokens used, latency.

**E3: Narrative faithfulness.** Measure:
- precision against ground truth and against recomputation;
- coverage;
- hedge ECE against ground truth;
- blind human ratings on a subset (at least 2 raters, κ reported).

**E4: Cross-domain transfer.** Same code, with only the θ estimator swapped:
- Ottawa bearings under time-varying speed: fault-type and severity descriptions.
- Penn HaTT: texture retrieval across speeds, and texture descriptions.
- RealWorld HAR (placement shift) and Capture-24 (day-long QA).
- PTB-XL: generated statements scored against cardiologist labels.

**E5: Ablations**
- SCR on vs off.
- Clock: t vs θ vs dual.
- Normalisation: RevIN vs log-absolute.
- Canonicalisation on vs off, under random rotations.
- Flat KST vs static tree vs tree with zoom.
- Text-only vs soft-only vs hybrid tokens.
- Verifier: off vs closed-vocabulary vs recompute-CI.
- Whittle vs MSE on log-spectra.
- Training text: template teacher vs truth-conditional captions vs frontier teacher.

**My odds.** These are judgement calls, not measurements.

| Outcome | Estimate |
|---|---|
| E0.1: SCR beats the γ-model | ~65% |
| SCR beats frozen generic encoders on cross-vehicle re-identification | ~60% |
| SCR-ET beats the code-tool agent on QA accuracy | ≤ 40% |
| SCR-ET beats the code-tool agent on tokens/latency at equal accuracy | likely |
| NeurIPS main track, if E0–E4 are positive across ≥ 3 domains | ~15–25% |
| As a datasets-and-benchmarks contribution (a progress-indexed, truth-conditional sensor-language benchmark across domains) | ~25–35% |
| IMWUT, with the road study plus one more domain | better than either of the two above |

**Timeline**

| Weeks | Work |
|---|---|
| 1–2 | E0 kill test |
| 3–5 | Tree, probes, verifier |
| 6–8 | LLM stages on Kaggle |
| 9–10 | Cross-domain experiments |
| 11–12 | Writing |

SCR-ET does not fit the November deadline.

### II.6 Order of operations

1. **This week.** Remove the fabricated numbers from the paper, `verification/RESULTS.md` and `data/evaluation_summary.json`. Commit with an explicit message and push.
2. **Decide on November.** Either write the short, honest RoadScribe paper or skip the deadline.
3. **Run E0 on PVS.** It needs no GPU and decides whether SCR-ET is worth three months.

---

## Sources checked on 2026-11-11 (2024–2026 items)

- SensorLM, NeurIPS 2025: https://arxiv.org/abs/2506.09108
- ChatTS, VLDB 2025: https://arxiv.org/abs/2412.03104. Scale-preserving follow-up: https://arxiv.org/abs/2601.21436
- SensorLLM, EMNLP 2025: https://arxiv.org/abs/2410.10624
- ARTIST, ICML 2026: https://arxiv.org/abs/2602.18645
- SensorChat: https://arxiv.org/abs/2502.02883
- EqNIO, ICLR 2025: https://arxiv.org/abs/2408.06321
- AnyMo: https://arxiv.org/abs/2605.22715
- UniMTS, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/c290d4373c495b2cad0625d6288260f0-Paper-Conference.pdf
- CaTS-Bench: https://arxiv.org/abs/2509.20823
- LLM-as-a-Judge for Time Series Explanations: https://arxiv.org/abs/2604.02118
- TRUCE: https://arxiv.org/abs/2110.01839
- Thompson et al., driver feedback NLG: https://arxiv.org/abs/2509.04478
- VibFM: https://papers.phmsociety.org/index.php/phme/article/view/4912
- RmGPT: https://arxiv.org/abs/2409.17604
- Smartphone pavement-profile deconvolution: https://vtechworks.lib.vt.edu/handle/10919/102869
- TEMPO, ICML 2026: https://icml.cc/virtual/2026/71512

Items cited from before 2024 (Nericell, Pothole Patrol, PELT, BabyTalk, Moryossef et al., Wiseman et al., Sha's Simplex, Whittle, Kaba et al., Zoology and the rest) are standard references and were not re-fetched.
