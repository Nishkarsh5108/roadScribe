# RoadScribe: methodology, the template objection, and implementation plan

Prepared 2026-11-01 for an ACM IMWUT (primary) or MobiSys submission.

**Status tags used throughout**

- **[verified]**: measured in this workspace on 2026-11-01. Scripts are in [verification/](verification/) and raw outputs in [verification/RESULTS.md](verification/RESULTS.md).
- **[audit]**: from your September 2026 IRI audit; not re-run today.
- **[proposed]**: a design that still needs validation.

Paper-ready LaTeX for Sections 3 and 4 is in [paper/sec3_system.tex](paper/sec3_system.tex) and [paper/sec4_evaluation.tex](paper/sec4_evaluation.tex). The LaTeX is written in paper voice. This file is the working document behind it.

## Contents

0. [Verdict](#0-verdict)
1. [Prior work you must position against](#1-prior-work-you-must-position-against)
2. [Answering "why not a template engine?"](#2-answering-why-not-a-template-engine)
3. [Layer 1: formalization](#3-layer-1-formalization)
4. [Layer 2: formalization](#4-layer-2-formalization)
5. [Layer 3: tokenizer, capsule, grounded generation](#5-layer-3-tokenizer-capsule-grounded-generation)
6. [Metrics](#6-metrics)
7. [User study (N = 28)](#7-user-study-n--28)
8. [Implementation plan for this workspace](#8-implementation-plan-for-this-workspace)
9. [Schedule, gates, kill criteria](#9-schedule-gates-kill-criteria)
10. [Appendix A: verified numbers](#appendix-a-verified-numbers)

---

## 0. Verdict

The pivot can work at IMWUT, but not as the blueprint states it. Three changes come before any experiment.

1. **Change the thesis.** "A 3B SLM beats a template engine" will lose. On a fixed-schema trip summary, a competent template is faithful by construction and costs nothing to run. A defensible thesis is narrower: *a grounded on-device SLM extends a deterministic narrator to outputs that cannot be enumerated at design time (questions users actually ask, audiences nobody wrote templates for, evidence states nobody anticipated), and the fidelity it costs can be measured and bounded.* The template engine stays inside RoadScribe as the fallback, and the paper measures what the SLM adds over it.
2. **Collect phone data of your own.** PVS can validate Layers 1–2. It cannot support three claims the paper needs:
   - phone mounting and handling (PVS uses fixed IMU modules);
   - repeated drives of one route by one driver (needed for habituation);
   - road conditions where your participants live.
3. **Repair the data preparation first.** `01_prepare_pvs_data.ipynb` averages two loggers whose clocks disagree by up to 0.54 s (§8.1).

### 0.1 Blueprint elements that will not survive review

| # | Blueprint element | What breaks (evidence) | Fix |
|---|---|---|---|
| B1 | "Smartphone" IMU, validated on PVS | PVS files are `dataset_gps_mpu_{left,right}.csv`. They hold IMUs at the dashboard, above the suspension and below the suspension, each with magnetometer and die-temperature channels. These are fixed IMU modules; nobody bolts a phone below a suspension. The notebook's text "smartphones were mounted" is wrong (confirm the hardware in the dataset paper before writing it). PVS never exercises orientation tracking: the mount is fixed, and the estimator simply recovers it **[verified]**. | Validate Layers 1–2 on PVS with synthetic rotations. Validate orientation, holder effects and handling on your own phone data (§8.5). |
| B2 | "ISO 2631-1 comfort filter bank" with a "switching boundary" between RMS and VDV | (a) The ISO comfort weighting applies at the seat–occupant interface. Annex C comfort reactions are informative and were written for public transport. A dashboard or holder is not the seat. (b) The standard has no switch. The weighted RMS is always reported; MTVV and VDV are *added* when the crest factor exceeds 9 or a ratio criterion trips. | Call the quantity "ISO 2631-1-weighted cabin acceleration". Implement the evaluation rule of §3.3. Validate the proxy with a second phone fixed at the seat rail. |
| B3 | "Texture ratios (… macro chatter)" | At ≤ 50 Hz and 5–25 m/s, wavelengths λ = v/f are ≥ 0.1 m. Pavement macrotexture is 0.5–50 mm (ISO 13473-1). A pavement engineer on the committee will flag this. | Rename them modal band ratios and give the λ = v/f mapping (§3.4). |
| B4 | TKEO + jerk "maneuver–defect decoupling" as a core contribution | On PVS asphalt, braking *lowers* dashboard vertical vibration by 12–16 % and turning has no effect. Only acceleration adds a dashboard-only high-band component, +36 % relative to the wheel sensor **[verified]**. Drivers slow to 3.6–5.1 m/s at bumps **[verified]**, so a braking veto would delete true bumps. The wheelbase double-hit cue I tested failed: ρ(lag, 1/v) ≈ 0, n = 60 **[verified]**. | Keep the gate as a false-positive suppressor for stop-and-go and acceleration artifacts. Ablate it and claim only what the ablation shows (§3.5). |
| B5 | "~90-token capsule" | One fully specified unit costs about 30–40 tokens in a compact line format and more in JSON (estimate; measure with the target tokenizer). 90 tokens hold about 2–3 units. A 20-minute PVS trip produces about 10–20. | Treat the budget B as a variable (64–512 tokens) and choose it from the fidelity–latency curve (§5.2). |
| B6 | Shannon entropy and "habituation over repeated commutes" | Entropy rises when you add random synonyms; a reviewer will say exactly that. PVS has no repeated commutes: its 9 trips are 3 routes × 3 vehicle–driver pairs **[verified]**. | Use Novelty–Change Alignment (§6.6), a repeated-route corpus you collect, and a field phase (§7.8). |
| B7 | No precedent cited | Data-to-text driver feedback from phone telematics already exists: SaferDrive (2018), DrivingBeacon (2022), and Thompson et al. (INLG 2026), which keeps content selection in rules and lets an LLM only paraphrase. | Position against it (§1). Its open gaps become your contribution. |

### 0.2 What the paper can claim

- **C1 Sensing.** A phone pipeline that turns IMU and GNSS into map-anchored, uncertainty-annotated ride facts: ISO 2631-1-weighted cabin vibration, impulsive events, and homogeneous road narrative units (HRNUs). It is validated against labelled events and against the dataset's reference sensors.
- **C2 Segmentation.** Map-informed PELT, a change-point detector whose penalty drops at OpenStreetMap transitions. Its pruning rule is corrected: the proof takes three lines, and the textbook rule returns sub-optimal partitions under such penalties **[verified]**.
- **C3 Generation.** Plan–Ground–Realize. An on-device SLM chooses and orders content by reference to a claim ledger and never emits a value. A verifier and a template fallback bound its errors.
- **C4 Measurement.** RoadScribe-Bench, with metrics that separate language errors from sensing errors: claim precision against the ledger and against ground truth, ρ_KSF, and anomaly grounding precision and recall.
- **C5 People.** A counterbalanced within-subject study (N = 28) and a two-week field phase. They measure usefulness, trust, error detection and habituation against strong template baselines.

### 0.3 Venue

IMWUT rewards the user study and the deployment. MobiSys expects the whole pipeline, including map matching, running on the phone, with energy measurements. With one laptop GPU and Kaggle, IMWUT is the realistic primary target.

IMWUT reviews on a quarterly cycle (historically Feb 1, May 1, Aug 1 and Nov 1; verify on the current call). §9 maps the work to Feb 1 (offline benchmark plus lab study) or May 1 (plus the field phase).

---

## 1. Prior work you must position against

| Work | What it does | What it leaves open |
|---|---|---|
| SaferDrive (Braun et al., 2018) | Rule-based NLG feedback on driving behaviour from insurance telematics | Behaviour, not road or ride; no LLM |
| DrivingBeacon (NLG4Health 2022; the UK system in Thompson et al., INLG 2026) | Post-trip reports from phone telematics with road and place context. Rules select content; gpt-3.5-turbo only paraphrases and "does not select incidents or infer new facts". 30 drivers, 12 weeks | No vibration or surface content; cloud LLM; no factuality evaluation; no template-vs-LLM comparison |
| Safe Drive Africa (the Nigeria system in Thompson et al., INLG 2026) | Harsh-event detection from phone accelerometer, gyro and GPS. LLM tips grounded in a retrieved legal knowledge base (GPT-4-turbo, then a GPT-4o-mini revision). 54 drivers, 802 trips, about 25 weeks | The same gaps. The paper states it "does not independently evaluate generated-text factuality" |
| LLaSA (arXiv 2406.14498), Penetrative AI (HotMobile 2024), By My Eyes (EMNLP 2024) | LLMs reasoning over raw IMU or sensor data | Human activity; no map anchoring; no claim-level fidelity |
| Nericell (SenSys 2008), Pothole Patrol (MobiSys 2008) | Phone and vehicle accelerometer road monitoring; virtual reorientation | No language layer. They are the orientation and detection baselines reviewers know |
| Less Is More (arXiv 2604.24636, 2026) | Engineering failures of on-device SLMs in a shipped app (output-format violations, context degradation) | Motivates grammar-constrained plans and cached prompt prefixes |

Thompson et al., "One Feedback System Does Not Fit All: Localising Data-to-Text Driver Coaching for the United Kingdom and Nigeria" (Thompson, Baig, Reiter, Yi; arXiv 2609.14687, accepted to INLG 2026), is the paper most likely to be in a reviewer's head. Read it in full. It leaves three gaps that RoadScribe can fill:

1. It does not evaluate the factuality of generated text.
2. It has no controlled template-versus-LLM comparison.
3. It has no vibration or road-surface content, and all its LLM use is in the cloud.

**Search caveat.** On 2026-11-01 I ran six web searches and found no system that narrates road or ride conditions from phone vibration with map grounding. That is a spot check, not a literature review. Before writing "first", search the ACM DL (IMWUT, MobiSys, SenSys, AutoUI) and the ACL Anthology (INLG) for 2024–2026.

---

## 2. Answering "why not a template engine?"

### 2.1 Concede what is true

A reviewer is right on three points, and the paper should say so before they do.

- **Templates win on the narrow task.** For a single-trip summary over a fixed schema, a template is faithful by construction, runs in microseconds, and is auditable. van Deemter, Krahmer and Theune (Computational Linguistics, 2005) showed that "template versus real NLG" is a false opposition: a template system can be linguistically rich.
- **The closest deployed systems keep content selection symbolic on purpose.** Thompson et al. state the principle as "structured components decide what can be said, while neural components may adapt how it is said".
- **An f-string baseline is a strawman, and the paper would be rejected for it.** The template baseline (T) must be a real NLG pipeline with:
  - content-selection rules;
  - aggregation ("both stretches were rough");
  - referring expressions;
  - lexical choice by ISO class;
  - comparison against the user's history;
  - an explicit fallback rule for every missing field.

  Budget about a week for it, and report its rule count.

### 2.2 Where templates break, and the experiment that shows each

The SLM is needed only where the output space cannot be enumerated when the system is built. Per HRNU, the capsule carries roughly felt class (6) × road class (6) × shock flag (2) × event pattern (≈4) × map–signal agreement (2) × confidence band (3) × missing-channel pattern (2³). That is about 13,800 joint states, before persona, trip history or a question enters.

Most states need no special wording. The ones that do are *interactions*, where the meaning is not the sum of the slots. A worked example:

```
U7   1.2 km   "Road C"
felt   a_v  = 1.62 m/s^2  -> class 4 ("very uncomfortable")
road   ã_w  = 0.70 m/s^2  -> class 2 ("fairly")   speed-decoupled, mean speed 16 m/s
OSM surface = asphalt;  signal classifier: cobblestone p = 0.81   -> disagreement
E12  impulse, label = ambiguous (stop-go context, p_defect = 0.55)
```

A good narrative says: *"Road C felt very rough mostly because of speed: at about 58 km/h you were fast for a surface that rides like cobblestone, although the map lists it as asphalt. One jolt near the junction may have come from pulling away rather than from the road."*

Each clause combines two or three facts. A template system needs a rule for every interaction that matters, plus a way to combine those rules. Compositional template grammars can do much of this, so the paper must **measure** where the coverage gap is rather than assert it. Five experiments do that:

| ID | Failure mode | Held out at design time | Fair template baseline | Primary metric | Falsified if (pre-register) |
|---|---|---|---|---|---|
| E1 | Questions users actually ask | Sealed half of an elicited question corpus | T plus an intent classifier over intents built from the dev half; abstains outside its intents | Correct-answer rate against the ledger (blind human judges); unsupported-claim rate; abstention precision | S-PGR's correct-answer rate is less than 10 pp above T+Intent, or its unsupported-claim rate is more than 2 pp worse |
| E2 | Audiences nobody wrote templates for | Test personas: municipal road engineer, motion-sensitive passenger, fleet manager | Template of the nearest seen persona | Persona-fit rating by matched raters; overlap with persona-specific gold plans; P_L | Nearest-persona templates come within 0.5 rating points |
| E3 | Evidence states nobody anticipated | Perturbation suite: no gyroscope, 60 s GNSS gap, map-match failure, OSM–signal contradiction, low-confidence events | T with an explicit fallback rule for each missing field | Hallucinated-missing rate (claims about MISSING facts); acknowledgement rate; well-formedness; P_L | S-PGR does not acknowledge missing evidence more often at an equal or lower hallucinated-missing rate |
| E4 | Change across repeated drives | none (repeated-route corpus) | T and T+V with the same history features in slots | Novelty–Change Alignment (§6.6); lexical-only variation LV | Not a win/lose test. It checks whether variation is informative, and the SLM may not win it |
| E5 | The fidelity cost and the rule budget | none | T grown rule by rule on the dev split | Frontier plot of P_L (x) against task coverage (y); T's coverage against its rule count | Not a falsification test. This is the paper's central figure |

**E1 elicitation.** This should start before you build anything. Ask 10–15 drivers who are not in the main study (a short online form is enough): "After a drive, what would you want to ask about the road or the ride?" Collect 200–300 questions, deduplicate them, and split them randomly 50/50 into dev and sealed test. Template intents, SLM few-shot examples and any fine-tuning data may use only the dev half.

**Pre-registration.** Put the falsification thresholds into a dated OSF pre-registration before running E1–E3 on the sealed sets. If you do not register publicly, say so in the supplement (your standing rule on disclosures).

### 2.3 The design that makes the defense honest: Plan–Ground–Realize (PGR)

```
capsule κ + ledger L ──► [SLM: PLAN, grammar-constrained to ids that exist in L]
                                  │ plan = (unit/fact ids, order, relations, hedges)
                                  ▼
                           [GROUND: ids → values, names, units from L]   (deterministic)
                                  │ fact frames
                                  ▼
                           [SLM: REALIZE text from fact frames]
                                  │
                                  ▼
                       [VERIFY claims vs L] ──fail twice──► [template realization of that plan item]
                                  │ pass
                                  ▼
                              narrative
```

1. **Plan.** The SLM emits a JSON plan under a grammar generated from this capsule's ledger. The plan says which units and facts to mention, in what order, with which rhetorical relation (elaboration, contrast, cause, comparison with history), and with which hedge. The grammar lists only the ledger's ids, so the model cannot reference a unit or event that does not exist.
2. **Ground.** Deterministic code expands each plan item into a fact frame with values, names and units taken from the ledger.
3. **Realize.** The SLM writes text from the fact frames. Ablate two variants:
   - (a) placeholders for numbers and names, filled after generation, which guarantees numeric fidelity;
   - (b) free text, checked by the verifier.
4. **Verify.** Claims are extracted from the text and checked against the ledger (§6.1). A failing sentence is regenerated up to twice, then replaced by the template realization of the same plan item. Report the fallback rate. If it is high, the SLM adds little, and the paper says so.

This is not "a template with extra steps". The SLM does content selection, ordering, aggregation and audience adaptation, which is where templates run out (E1–E3). It cannot do the one thing that makes LLM output untrustworthy here, which is to introduce a value.

### 2.4 Conditions

| Code | Condition | Role |
|---|---|---|
| T | Template NLG pipeline (deterministic) | Primary baseline; also the fallback |
| T+V | T with lexical and ordering variation sampled per run | Controls for "variety alone" |
| T→P | T, then an SLM paraphrase (the DrivingBeacon design) | Separates surface fluency from content planning |
| S-free | SLM given the capsule; free text; no plan, no verifier | Shows why the guardrails are needed |
| S-PGR | Full system, on-device | Proposed |
| C-PGR | Same protocol with a frontier cloud model | Upper reference; quantifies the privacy and latency trade-off |

The offline benchmark runs all six conditions. The user study runs four (T, T+V, T→P and S-PGR), so that N = 28 fits a Williams design exactly (§7).

### 2.5 Shannon entropy and habituation

Shannon entropy and distinct-n measure variety, and variety is cheap: sample synonyms and T+V scores high. Report them descriptively only. What matters is whether variation is *informative*: the narrative should change when the road or the drive changed and stay put when nothing did. §6.6 defines Novelty–Change Alignment for this.

Habituation is a behavioural effect that unfolds over days. Measure it in the field phase (§7.8) with engagement logs and short daily ratings, not with a text statistic.

### 2.6 Paragraph for the paper

> RoadScribe contains a template realizer, and every sentence that fails verification falls back to it. Templates suffice when the situations, questions and audiences are fixed when the system is built: on single-trip summaries over a fixed schema the template baseline matches the full system on claim precision ([TBD]) and on rated usefulness ([TBD]). The language model earns its cost on what cannot be enumerated in advance. On held-out user questions it answers [TBD] pp more correctly (E1); for unseen audiences it is rated [TBD] points closer to the intended reader (E2); on degraded inputs it acknowledges missing evidence in [TBD] % of cases versus [TBD] % (E3). It does so at a claim-precision cost of [TBD] pp (E5).

---

## 3. Layer 1: formalization

### 3.0 Notation and frames

| Symbol | Meaning |
|---|---|
| $n$, $f_s$, $T_s = 1/f_s$ | IMU sample index, sampling rate, sampling period |
| $\mathbf f^b[n]$ | accelerometer specific force in the phone/sensor body frame $b$ (includes gravity: at rest it equals $g$ along "up") |
| $\boldsymbol\omega^b[n]$ | gyroscope angular rate (rad/s) |
| $v$ frame | vehicle frame, ISO 8855: $x$ forward, $y$ left, $z$ up |
| $\mathbf R_{vb} \in SO(3)$ | rotation from body to vehicle frame |
| $\mathbf a^v[n] = \mathbf R_{vb}\mathbf f^b[n] - g\,\mathbf e_z$ | gravity-free acceleration in the vehicle frame |
| $v_k, t_k$ | GNSS speed at fix $k$ (1 Hz) |
| $v(t)$, $a^G_{\text{lon}}(t)$ | speed interpolated to IMU time (PCHIP) and its derivative |
| $s(t)$ | along-track distance: $\int v\,dt$ before map matching, arc length after |

**PVS facts [verified].** In PVS, forward is $+y_s$: corr(acc_y, GNSS dv/dt) is +0.41 to +0.69 while corr(acc_x, dv/dt) is about 0. The z axis points up (acc_z ≈ +9.8 at rest, and gyro_z correlates negatively with the clockwise GNSS course rate). So $\mathbf R_{vb} \approx \mathbf R_z(90^\circ)$. Do not hard-code this; use it as a regression test for the estimator.

### 3.1 Orientation

**Step 0: mount episodes.** A phone in a holder is static relative to the car except when someone handles it. Mark a candidate handling event where the gyroscope reports rotation that the vehicle does not perform:

$$\lVert\bar{\boldsymbol\omega}^b[n]\rVert - \lvert\dot\psi^G(t_n)\rvert > \Omega_h \quad(\Omega_h \approx 30^\circ/\text{s}),$$

where $\dot\psi^G$ is the GNSS course rate. Confirm a split when the quasi-static gravity directions on the two sides differ by more than $\theta_h \approx 10^\circ$. Discard ±5 s around each split, and estimate orientation per episode $e$.

**Step 1: gated gravity.** Let $\bar{\mathbf f}^b$ and $\bar{\boldsymbol\omega}^b$ be low-passed at 0.5 Hz. The quasi-static set of episode $e$ is

$$\mathcal Q_e=\Bigl\{n:\ \bigl\lvert\lVert\bar{\mathbf f}^b[n]\rVert-g\bigr\rvert<\epsilon_g,\ \lVert\bar{\boldsymbol\omega}^b[n]\rVert<\epsilon_\omega,\ \lvert a^G_{\text{lon}}(t_n)\rvert<\epsilon_a,\ v(t_n)>v_{\min}\Bigr\}.$$

From it, compute the up direction:

$$\hat{\mathbf g}_e=\operatorname{GeoMedian}\{\bar{\mathbf f}^b[n]: n\in\mathcal Q_e\},\qquad \mathbf u_e=\hat{\mathbf g}_e/\lVert\hat{\mathbf g}_e\rVert .$$

- **Thresholds on PVS:** $\epsilon_g = 0.3$ m/s², $\epsilon_\omega = 3^\circ$/s, $\epsilon_a = 0.3$ m/s², $v_{\min} = 3$ m/s. These leave 30k–51k quasi-static samples per trip **[verified]**.
- **Why gate:** an ungated low-pass absorbs sustained acceleration and tilts "up" by $\arctan(a/g)$, which is 17° at 3 m/s².
- **Why a geometric median:** it is robust and rotation-equivariant. A componentwise median gave up to 0.63° of equivariance error **[verified]**.
- **Caveat: road grade.** The route-average grade biases the tilt; a 5 % grade is 2.9°. The effect on the vertical channel is at most $1-\cos 2.9^\circ = 0.13\,\%$. On the longitudinal channel it can reach $g\sin\theta \approx 0.5$ m/s², so longitudinal maneuver thresholds use GNSS dv/dt, not IMU $a_x$.

**Step 2: tilt (Rodrigues).** Compute the rotation axis and angle that carry $\mathbf u_e$ onto $\mathbf e_z$:

$$\mathbf k=\frac{\mathbf u_e\times\mathbf e_z}{\lVert\mathbf u_e\times\mathbf e_z\rVert},\quad \theta=\operatorname{atan2}\bigl(\lVert\mathbf u_e\times\mathbf e_z\rVert,\ \mathbf u_e^\top\mathbf e_z\bigr),\quad \mathbf R_1=\mathbf I+\sin\theta\,[\mathbf k]_\times+(1-\cos\theta)[\mathbf k]_\times^2 .$$

Two degenerate cases need special handling: use $\mathbf R_1=\mathbf I$ if $\mathbf u_e\approx\mathbf e_z$, and $\mathbf R_1=\operatorname{diag}(1,-1,-1)$ if $\mathbf u_e\approx-\mathbf e_z$. The leveled yaw rate $\omega^\ell_z=(\mathbf R_1\bar{\boldsymbol\omega}^b)_z$ does **not** depend on the unknown heading offset. That is what makes Step 3 possible.

**Step 3: yaw by 2-D Procrustes against GNSS.** At each fix with $v_k>3$ m/s, take the leveled horizontal specific force $\mathbf h_k=(\bar f^\ell_x,\bar f^\ell_y)(t_k)$ and the reference $\mathbf r_k=\bigl(a^G_{\text{lon}}(t_k),\ v_k\,\omega^\ell_z(t_k)\bigr)$, i.e. longitudinal and centripetal acceleration, positive to the left. Then:

$$\hat\psi=\arg\min_\psi\sum_k\lVert\mathbf R_2(\psi)\mathbf h_k-\mathbf r_k\rVert^2,\qquad \mathbf R_2(\psi)=\begin{bmatrix}\cos\psi&\sin\psi\\-\sin\psi&\cos\psi\end{bmatrix}.$$

The closed form, with $\mathbf M=\sum_k\mathbf h_k\mathbf r_k^\top$, is

$$\hat\psi=\operatorname{atan2}\bigl(M_{21}-M_{12},\ M_{11}+M_{22}\bigr)=\operatorname{atan2}\Bigl(\sum_k h_{y,k}a_{\text{lon},k}-h_{x,k}a_{\text{lat},k},\ \sum_k h_{x,k}a_{\text{lon},k}+h_{y,k}a_{\text{lat},k}\Bigr).$$

It follows from maximizing $\operatorname{tr}(\mathbf R_2\mathbf M)=\cos\psi\,(M_{11}+M_{22})+\sin\psi\,(M_{21}-M_{12})$. Report the excitation $\sum_k\lVert\mathbf r_k\rVert^2$ and a bootstrap CI of $\hat\psi$ over fixes; these form the orientation confidence that goes into the capsule.

**Step 4.** Combine the rotations and transform the signals:

$$\mathbf R_{vb}=\mathbf R_z(\hat\psi)\mathbf R_1,\qquad \mathbf a^v=\mathbf R_{vb}\mathbf f^b-g\mathbf e_z,\qquad \boldsymbol\omega^v=\mathbf R_{vb}\boldsymbol\omega^b .$$

**PVS validation [verified].**

- **Known mount recovered.** $\hat\psi$ is 87.9–92.5° for vehicles 1 and 3 (expected 90°).
- **Vehicle 2 has a real mount offset.** It gives 103.1–105.5° consistently across its three routes, about a 14° mount yaw.
- **Equivariant to phone orientation.** Under 20 random 3-D rotations per trip, the estimate rotated with the data to within 0.09–0.28° (median) and 0.63° (max).

You may claim "recovers a rigid mount to within about 2° trip to trip and is equivariant to arbitrary phone orientation". Claims about handling or holders need your own data.

**Online variant [proposed].** Use a complementary filter for real-time use: $\hat{\mathbf g}^-[n]=\exp(-[\boldsymbol\omega^b]_\times T_s)\hat{\mathbf g}[n-1]$, then $\hat{\mathbf g}[n]=\hat{\mathbf g}^-[n]+\alpha\,\mathbf 1[n\in\mathcal Q](\bar{\mathbf f}^b[n]-\hat{\mathbf g}^-[n])$. Narratives are generated post-trip, so the batch version above is the primary one.

### 3.2 Speed decoupling

Vibration grows with speed. For a road whose elevation PSD follows ISO 8608, $G_d(n)=G_d(n_0)(n/n_0)^{-w}$, the temporal input PSD at speed $v$ scales as $v^{w-1}f^{-w}$. For a fixed vehicle response, the RMS therefore scales as $v^{(w-1)/2}$, which gives $\gamma=0.5$ for $w=2$. Estimate the exponent from data rather than assuming it, with this window-level model ($j$ indexes homogeneous segments):

$$\log a_{w,i}=\alpha_{j(i)}+\gamma\log v_i+\varepsilon_i .$$

**PVS asphalt, with trip fixed effects [verified]:** $\hat\gamma$ = 0.32 (dashboard, 0.5–4 Hz), 0.36 (dashboard, 4–25 Hz) and 0.62 (below the suspension, 4–25 Hz).

The decoupled intensity is

$$\tilde a_w=a_w\,(v_{\text{ref}}/\bar v)^{\hat\gamma},\qquad v_{\text{ref}}=10\ \text{m/s},$$

used only for $\bar v\ge 5$ m/s. Below that, report raw $a_w$ with a `speed_low` flag.

- **Do not reuse the earlier $(22.22/v)^2$ "kinetic norm".** It encoded speed (Spearman 0.995) **[audit]**.
- **Felt versus road.** Raw $a_w$ is how the ride *felt*; $\tilde a_w$ is how the *road* is. The narrative keeps them apart.
- **Counterfactual.** $a_w(v')\approx a_w(v'/\bar v)^{\hat\gamma}$. For example, slowing from 15 to 10 m/s at $\gamma=0.35$ cuts $a_w$ by 13 %. Mark it in the ledger as a model extrapolation.

### 3.3 ISO 2631-1 weighting and measures

**Transfer functions** ($p=j2\pi f$, $\omega_i=2\pi f_i$, $Q_1=1/\sqrt2$):

$$H_h=\frac{p^2}{p^2+\omega_1p/Q_1+\omega_1^2},\qquad H_l=\frac{\omega_2^2}{p^2+\omega_2p/Q_1+\omega_2^2},$$

$$H_t=\frac{(\omega_4^2/\omega_3)(p+\omega_3)}{p^2+\omega_4p/Q_4+\omega_4^2},\qquad H_s=\frac{p^2+\omega_5p/Q_5+\omega_5^2}{p^2+\omega_6p/Q_6+\omega_6^2},\qquad W=H_hH_lH_tH_s .$$

| Weighting (ISO 2631-1:1997 Annex A) | f1 | f2 | f3 | f4 | Q4 | f5 | Q5 | f6 | Q6 |
|---|---|---|---|---|---|---|---|---|---|
| W_k (vertical, seat) | 0.4 | 100 | 12.5 | 12.5 | 0.63 | 2.37 | 0.91 | 3.35 | 0.91 |
| W_d (horizontal, seat) | 0.4 | 100 | 2.0 | 2.0 | 0.63 | – | – | – | – |

**Unit-test values**, computed from these formulas **[verified]** (|W| × 1000). Compare them with the table in your copy of the standard; W_k at 1 Hz = 0.482 matches.

| f (Hz) | 0.5 | 1 | 2 | 4 | 5 | 8 | 12.5 | 16 | 20 | 31.5 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W_k | 418 | 482 | 531 | 967 | 1039 | 1036 | 902 | 769 | 636 | 405 | 314 |
| W_d | 853 | 1011 | 890 | 512 | 409 | 253 | 161 | 125 | 100 | 63 | 49 |

**Discrete realization.**

1. **Offline (recommended; narratives are post-trip).** Apply $W$ in the frequency domain with at least 10 s of zero padding: $a_w=\mathcal F^{-1}\{\mathcal F\{a\}\cdot W(j2\pi f_k)\}$. This is exact in magnitude *and* phase below $f_s/2$ **[verified]**: 0.0000 dB error on test tones, and the VDV and peak of an impulsive test bump match a 2 kHz IIR reference (1.3842 vs 1.3842; 3.7220 vs 3.7215).
2. **Streaming.** Use a per-section bilinear transform with pre-warping, $p\to K_i\frac{1-z^{-1}}{1+z^{-1}}$ with $K_i=\omega_{c,i}/\tan(\omega_{c,i}T_s/2)$. Here $\omega_c$ is the section corner frequency; for $H_s$, use the geometric mean of $f_5$ and $f_6$. For a section $(b_2p^2+b_1p+b_0)/(a_2p^2+a_1p+a_0)$ the coefficients are $B_0=b_2K^2+b_1K+b_0$, $B_1=2(b_0-b_2K^2)$, $B_2=b_2K^2-b_1K+b_0$, with $A_0$–$A_2$ formed the same way and all divided by $A_0$. Measured error against the analog response **[verified]**:
   - At $f_s=100$ Hz: ≤ 0.72 dB ($W_k$) and ≤ 1.22 dB ($W_d$) up to 20 Hz, but 7.3–7.7 dB at 40 Hz. Impulsive peaks are 3.8 % low, which biases the crest factor.
   - At $f_s=400$ Hz: ≤ 0.23 dB up to 40 Hz. For streaming, log at ≥ 400 Hz; otherwise use the offline path.
   - $H_l$ (100 Hz) cannot be realized at $f_s\le 200$ Hz. Dropping it changes $W$ by less than 0.11 dB below 40 Hz.
3. **Bandwidth disclosure.** The standard's comfort range is 0.5–80 Hz. At $f_s=100$ Hz the analysis band ends near 40 Hz. Say so.

**Measures per HRNU** (duration $T$, axis $j$):

$$a_{w,j}=\Bigl(\tfrac1T\int_0^T a_{w,j}^2\,dt\Bigr)^{1/2},\qquad a_v=\bigl(a_{w,x}^2+a_{w,y}^2+a_{w,z}^2\bigr)^{1/2}\ \ (\text{comfort: }k_x=k_y=k_z=1),$$

$$a_{w,j}(t_0)=\Bigl(\tfrac1\tau\int_{t_0-\tau}^{t_0}a_{w,j}^2\,dt\Bigr)^{1/2},\ \tau=1\ \text{s},\qquad \text{MTVV}_j=\max_{t_0}a_{w,j}(t_0),$$

$$\text{VDV}_j=\Bigl(\int_0^T a_{w,j}^4\,dt\Bigr)^{1/4}\ [\text{m s}^{-1.75}],\qquad \text{CF}_j=\max_t\lvert a_{w,j}(t)\rvert/a_{w,j}.$$

**Evaluation rule (ISO 2631-1 §6.2–6.3).** $a_w$ is always reported. MTVV and VDV are added, and RoadScribe flags the unit as shock-dominated, when

$$S_j=\mathbf 1\Bigl[\text{CF}_j>9\ \lor\ \text{MTVV}_j/a_{w,j}>1.5\ \lor\ \text{VDV}_j/(a_{w,j}T^{1/4})>1.75\Bigr].$$

**Useful identity.** For a zero-mean stationary signal, $\text{VDV}/(a_wT^{1/4})=\kappa^{1/4}$, where $\kappa$ is the kurtosis. So the 1.75 criterion is $\kappa>9.38$, and Gaussian vibration gives 1.32. The VDV criterion is a kurtosis test; put that sentence in the paper.

**Comfort words (Annex C, informative).** The bands are:

- < 0.315 m/s²: not uncomfortable
- 0.315–0.63: a little uncomfortable
- 0.5–1.0: fairly uncomfortable
- 0.8–1.6: uncomfortable
- 1.25–2.5: very uncomfortable
- > 2: extremely uncomfortable

The bands overlap. For a single label, use geometric-mean cut-points θ = (0.315, 0.561, 0.894, 1.414, 2.236) m/s² **[verified]**. For evaluation, accept any band that contains the measured value (§6.3).

**How the shock flag changes the narrative.**

- The comfort category always comes from $a_v$.
- When $S_z=1$, the narrative also reports the jolts as events, with MTVV labelled "peak 1-s exposure".
- ISO has no comfort scale for VDV, so never map VDV to comfort words.

**Horizontal channels.** Below about 1 Hz, $a_{w,x}$ and $a_{w,y}$ carry braking and cornering, which are driver inputs. Report them in a "driving" channel, not as road roughness.

### 3.4 Modal band ratios (replaces "texture ratios")

**PSD.** Compute the Welch PSD of $a^v_z$ per HRNU with a Hann window, $L=512$ (5.12 s, Δf = 0.195 Hz) and 50 % overlap. For units shorter than about 15 s, use $L=256$ or a multitaper estimate.

**Band energies.** Integrate the PSD over three bands, $E_B=\int_B P_{zz}(f)\,df$:

- $B_{\text{body}}=[0.8,2.5]$ Hz;
- $B_{\text{wheel}}=\hat f_{wh}\cdot[0.75,1.33]$, where $\hat f_{wh}=\arg\max_{[6,20]\,\text{Hz}}P_{zz}$ on rough units of that vehicle (default [8, 16] Hz);
- $B_{\text{high}}=[16,40]$ Hz.

**Ratios.**

$$R_{wb}=\log(E_{\text{wheel}}/E_{\text{body}}),\qquad R_{hb}=\log(E_{\text{high}}/E_{\text{body}}).$$

**Wavelengths ($\lambda=v/f$).** At 10 m/s, body ↔ 4–12.5 m, wheel ↔ 0.6–1.25 m, high ↔ 0.25–0.6 m: unevenness and megatexture, not macrotexture.

These ratios describe how the vehicle's modes are excited, so they are vehicle-specific. Use them within a vehicle and for segmentation, not as cross-vehicle road properties.

### 3.5 Impulsive events and maneuver gating

**Signal chain.**

- **Vertical impulse channel:** $\tilde z=\text{BP}_{[4,25]\,\text{Hz}}(a^v_z)$ (4th-order Butterworth, zero-phase offline).
- **TKEO:** $\Psi[n]=\tilde z[n]^2-\tilde z[n-1]\tilde z[n+1]$. For $\tilde z=A\cos(\Omega n+\phi)$, $\Psi=A^2\sin^2\Omega$, so it weights both amplitude and frequency (Kaiser 1990).
- **Envelope:** $E[n]=(h_{50\,\text{ms}}*\max(\Psi,0))[n]$.
- **Adaptive threshold:** $\theta[n]=\operatorname{med}_W(E)+\kappa\cdot1.4826\cdot\operatorname{MAD}_W(E)$, $W=10$ s centred. Cobblestone raises the background, and a fixed threshold would fire continuously there.
- **Candidates:** maximal runs with $E>\theta$, merged across gaps < 0.3 s. Reject runs longer than $D_{\max}=0.5\ \text{s}+3\ \text{m}/\bar v$; long runs are rough stretches and go to Layer 2.

**Maneuver context.** Longitudinal states come from GNSS and lateral states from the gyroscope, which avoids tilt bias:

- **brake:** $a^G_{\text{lon}}<-A_b$
- **accel:** $a^G_{\text{lon}}>A_a$
- **turn:** $\lvert v\,\omega^v_z\rvert>A_t$ for ≥ 1 s
- **swerve:** a yaw-rate sign reversal within 1–4 s, with net heading change below 5° and peak $\lvert v\omega^v_z\rvert>A_s$
- **stop-go:** $v<2$ m/s and $\lvert j_x\rvert>J_{sg}$, where $j_x=\frac{d}{dt}\text{LP}_{5\,\text{Hz}}(a^v_x)$

**Event features.** For each candidate $e$: peak $E$, duration $d_e$, the band ratio $\rho_e=E_{[8,25]}(e)/E_{[0.5,4]}(e)$ of $a^v_z$ in the event window, the coincident $\lvert j_x\rvert$, and the maneuver state $m_e$.

**Decision rule.**

$$\ell(e)=\begin{cases}\textsf{artifact}&m_e\in\{\textsf{stop-go},\textsf{accel}\}\ \land\ \max_{n\in e}\lvert j_x[n]\rvert>J_x\ \land\ \rho_e<\rho_0\\ \textsf{defect}&\lnot\textsf{artifact}\ \land\ \rho_e\ge\rho_0\ \land\ d_e\le D_{\max}\\ \textsf{ambiguous}&\text{otherwise}\end{cases}$$

- **Braking and turning never veto a defect.** On PVS they do not raise vertical vibration, and drivers brake *before* bumps.
- **Avoidance events.** A swerve with no vertical candidate within ±2 s becomes a separate low-confidence "avoidance" event (a possible obstacle the car steered around), and the narrative must hedge it.
- **Calibration.** Map the features to $p_{\text{defect}}$ with logistic regression trained leave-one-route-out. Positives are PVS bump traversals; negatives are stop-go and acceleration episodes more than 50 m from any labelled bump. Choose $\kappa,\rho_0,J_x$ on training routes only.

**Evidence behind the rule [verified]** (PVS, asphalt only, no bump within ±2 s, 5–20 m/s, 2-s windows, left logger; values are log-RMS differences from cruise, controlling for log speed and trip; cluster-robust SEs by trip):

| Maneuver (n windows) | Dashboard 0.5–4 Hz | Dashboard 4–25 Hz | Below suspension 4–25 Hz | Dashboard minus wheel, 4–25 Hz |
|---|---|---|---|---|
| brake (75) | −0.18 (p < .001) | −0.13 (p = .025) | −0.16 (p = .058) | +0.03 (n.s.) |
| accel (23) | −0.18 (p = .070) | −0.06 (n.s.) | −0.37 (p = .003) | **+0.31 (p < .001)** |
| turn (92) | +0.07 (n.s.) | −0.04 (n.s.) | +0.06 (n.s.) | −0.10 (p < .001) |

Cruise windows: 225. The dashboard-only excess under acceleration matches powertrain vibration. That is the artifact the gate should target.

**What to drop.** I tested the wheelbase double-hit cue (front then rear axle, lag = L/v). Spearman ρ(lag, 1/v) was between −0.12 and 0.00 across both channels and both lag estimators, n = 60 bump traversals, all p ≥ 0.36 **[verified]**. At 3.6–5.1 m/s, bumps excite body modes rather than two clean impulses. Treat any later attempt to use the cue as exploratory.

**Ablation to report.** False alarms per km at fixed bump recall, with and without the gate, broken down by maneuver class.

**Speed shortcut.** On PVS, speed alone separates surfaces with AUC 0.977, and RMS($a_z$)/√v reaches 0.997 **[audit]**. Every Layer-1 classifier must be compared with both baselines.

---

## 4. Layer 2: formalization

### 4.1 Map matching

Use the hidden Markov model of Newson and Krumm (2009). The states are candidate projections $x_{k,i}$ of fix $z_k$ onto OSM edges within $r=50$ m.

- **Emission:**
  $$p(z_k\mid x_{k,i})=\frac{1}{\sqrt{2\pi}\sigma_z}\exp\Bigl(-\frac{\lVert z_k-x_{k,i}\rVert_{gc}^2}{2\sigma_z^2}\Bigr),\qquad \sigma_z=1.4826\operatorname{med}_k\lVert z_k-x_k\rVert_{gc}.$$
- **Transition:**
  $$p(x_{k,i}\to x_{k+1,j})=\frac1\beta\exp\Bigl(-\frac{\bigl\lvert\lVert z_k-z_{k+1}\rVert_{gc}-\lVert x_{k,i}-x_{k+1,j}\rVert_{\text{route}}\bigr\rvert}{\beta}\Bigr),\qquad \beta=\operatorname{med}(d_k)/\ln2.$$
- **Decoding:** Viterbi. Forward–backward marginals give the per-fix map-match confidence $c^{mm}_k$.
- **Outputs:** the edge sequence, OSM way ids and tags, and the arc length $s(t)$.
- **Tools:** `leuvenmapmatching`, or your own Viterbi over an `osmnx` graph. Pure Python is fine for about 100 km. OSRM and Valhalla `match` need a server.
- **OSM content to pull:**
  - way tags `highway`, `surface`, `smoothness`, `name`, `bridge`, `tunnel`, `maxspeed`;
  - nodes `traffic_calming=*`, `railway=level_crossing`, `highway=traffic_signals`;
  - POIs for anchors.

**First task: measure OSM coverage on the PVS routes.** Compute two numbers:

- the fraction of matched length that carries a `surface` tag;
- how many of the 20 physical bumps carry `traffic_calming`.

If coverage is low, the map prior is weak, λ in MI-PELT will tune to about 0, and the paper must say so.

### 4.2 Distance-domain features

Resample at Δs = 5 m along $s$. Each grid point gets a feature vector
$$\mathbf y_j=[\log\tilde a_{w,z},\ \log a_{w,z},\ R_{wb},\ R_{hb},\ \text{event rate per 100 m},\ \log a_{w,y}]_j,$$
each computed over a centred 25 m window and robust-standardized per trip. Speed is deliberately excluded. Check that the resulting change points do not track speed, using the speed-only baseline in §4.5.

### 4.3 Map-informed PELT (MI-PELT)

**Objective.**

$$\min_{m,\ \tau_{1:m}}\ \sum_{i=1}^{m+1}C\bigl(\mathbf y_{\tau_{i-1}+1:\tau_i}\bigr)+\sum_{i=1}^{m}\beta(\tau_i),\qquad \tau_0=0,\ \tau_{m+1}=n .$$

**Cost.** A Gaussian mean shift with shared diagonal covariance $\mathbf D$, computed in O(1) from cumulative sums:

$$C(\mathbf y_{a:b})=\sum_{j=a}^{b}\lVert\mathbf y_j-\bar{\mathbf y}_{a:b}\rVert^2_{\mathbf D^{-1}}.$$

**Location-dependent penalty.**

$$\beta(\tau)=\beta_0\bigl(1-\lambda\,\pi(\tau)\bigr),\qquad \pi(\tau)=\max_{b\in\mathcal B}\ w_b\exp\Bigl(-\frac{(s_\tau-s_b)^2}{2\sigma_b^2}\Bigr),\qquad \lambda\in[0,1).$$

- **Map-transition weights $w_b$:** 1.0 for a surface-tag change, 0.6 for a highway-class change, 0.3 for a junction or `traffic_calming` node. Use $\sigma_b=15$ m.
- **Base penalty:** $\beta_0=c\,d\ln n$. Tune $(c,\lambda)$ leave-one-route-out, and show a CROPS path (Haynes et al., 2017) for sensitivity.
- **Special case:** λ = 0 recovers PELT exactly.

**Recursion.** Start from $F(0)=0$ and define $G(t)=F(t)+\beta(t)\mathbf 1[t>0]$. Then

$$F(s)=\min_{t\in\mathcal R_s,\ s-t\ge\ell_{\min}}\bigl\{G(t)+C(\mathbf y_{t+1:s})\bigr\},$$

$$\mathcal R_{s+1}=\bigl\{t\in\mathcal R_s:\ G(t)+C(\mathbf y_{t+1:s})<G(s)\bigr\}\cup\{s\}\quad(\text{with the usual }\ell_{\min}\text{ delay before }s\text{ enters}).$$

**Lemma (pruning under location-dependent penalties).** Assume $C(\mathbf y_{t+1:s})+C(\mathbf y_{s+1:T})\le C(\mathbf y_{t+1:T})$ for all $t<s<T$, which holds for the mean-shift cost. If $G(t)+C(\mathbf y_{t+1:s})\ge G(s)$, then $t$ is never the optimal last change point for any $T>s$.

*Proof.* For any $T>s$:
$$G(t)+C(\mathbf y_{t+1:T})\ \ge\ G(t)+C(\mathbf y_{t+1:s})+C(\mathbf y_{s+1:T})\ \ge\ G(s)+C(\mathbf y_{s+1:T}).\qquad\blacksquare$$

The textbook PELT condition $F(t)+C\ge F(s)$ equals this one only when β is constant. When $\beta(t)<\beta(s)$, as at map transitions, the textbook rule prunes valid candidates. In a randomized check, it returned a sub-optimal partition in **300 of 300** series. The corrected rule matched exhaustive optimal partitioning in **300 of 300** **[verified]**.

`ruptures` does not support position-dependent penalties, so implement MI-PELT directly; it is about 40 lines with cumulative sums. Complexity is O(n) expected under the usual PELT conditions and O(n²) in the worst case.

### 4.4 Arbitration and HRNUs

- **(A1) Snap.** If $\lvert s(\tau)-s(b)\rvert\le\delta_{\text{snap}}=25$ m for a relevant OSM transition $b$, move τ to $b$. The map wins on location; the signal wins on existence.
- **(A2) Type each boundary.**
  - `map+signal`: snapped.
  - `signal-only`: no OSM transition nearby. This is interesting content, e.g. the surface changes mid-street.
  - `map-only`: an OSM surface change without a PELT boundary. Keep it only if a two-sample test on $\mathbf y$ across it rejects (Hotelling T², α = 0.01). Otherwise every OSM way split would become a narrative boundary.
- **(A3) Merge** units shorter than $\ell_{\min}=50$ m into the neighbour with the smaller Mahalanobis distance.
- **(A4) Map–signal disagreement.** $\delta^{\text{map}}_m=1-P(\text{surface}=\text{OSM tag}\mid\mathbf y)$, from a surface classifier trained leave-one-route-out and compared against speed-only. It is undefined where there is no tag.

An **HRNU** is an interval $[s^0_m,s^1_m)$ carrying its metrics, events, anchors, boundary types and confidences (schema in §8.3).

### 4.5 Evaluating segmentation on PVS

- **Ground-truth boundaries:** surface-label transitions, 54 in all (4–9 per trip) **[verified]**.
- **Metrics:** boundary F1 at tolerance δ ∈ {25, 50} m, Hausdorff distance, and the number-of-segments error.
- **Comparators:**
  - constant-β PELT;
  - MI-PELT;
  - OSM-only boundaries;
  - fixed 100 m windows;
  - **PELT on speed alone.** Surface is confounded with speed on PVS. If speed-only PELT matches, the segmentation claim collapses.
- **Splits:** leave-one-route-out. With 3 routes, report per-route results descriptively and make inferential claims on your own corpus.

---

## 5. Layer 3: tokenizer, capsule, grounded generation

### 5.1 Claim ledger

The ledger is the single source of truth for the verifier and for the evaluation. Each fact is a typed tuple `(fact_id, kind, unit_id, key, value, tolerance, confidence)`.

| Kind | Example |
|---|---|
| SEV | felt class 4 of U2, with its band and $a_v$ |
| NUM | $a_v$, VDV, length, mean speed, count, with tolerance |
| EVT | event id, type, $s$, $p_{\text{defect}}$, MTVV |
| ANCHOR | name and the $s$-range it covers |
| CMP | U2 rougher than U1; today rougher than the route's median |
| MISSING | field not available, with a reason (`gps_gap_48s`, `no_gyro`, `mapmatch_failed`) |
| META | channels present, map coverage, orientation confidence |

### 5.2 Kinetic–Spatial Tokenizer (KST)

For each unit, compute a salience score:

$$\sigma_m=w_1k_m+w_2\sum_{e\in\mathcal E_m}p_e\,\text{sev}(e)+w_3\,\Delta_{\mathcal H}(u_m)+w_4\,\delta^{\text{map}}_m .$$

The weights are fixed *a priori*: all 1 after z-scoring, with an ablation. Do not learn them from 9 trips.

Each unit has three serializations, ℓ ∈ {OMIT, BRIEF, FULL}, with token cost $c_{m,\ell}$ (measured with the target tokenizer) and gain $g_{m,\ell}=\sigma_m\eta_\ell$, where $\eta_{\text{OMIT}}=0$. The must-mention set $\mathcal M$ contains the units holding an event with $p_e\,\text{sev}(e)\ge\eta^\star$. Selection is an exact multiple-choice knapsack:

$$V_m(b)=\max_{\ell\in\Lambda_m}\bigl\{V_{m-1}(b-c_{m,\ell})+g_{m,\ell}\bigr\},\quad V_0(b)=0,\quad \Lambda_m=\begin{cases}\{\text{BRIEF},\text{FULL}\}&m\in\mathcal M\\\{\text{OMIT},\text{BRIEF},\text{FULL}\}&\text{otherwise}\end{cases}$$

It runs in O(3MB). Serialize the chosen units in route order, and summarize the omitted ones in one REST line. The LaTeX `algorithmic` version is Algorithm 1 in [paper/sec3_system.tex](paper/sec3_system.tex).

**Example capsule** (compact line format; illustrative names):

```
TRIP pvs4 | 22.1 min | 13.8 km | persona=commuter | ch=acc,gyro,gps,map | hist=none
U1 0.0-1.9km Road A | asphalt (osm asphalt) | felt 1 | road 1 | ev - | mm .97
U2 1.9-2.6km Road B | cobble (osm -) | felt 4 SHOCK | road 3 | bump x2 p.93,.81 | near School 1 @2.3km | mm .95
U3 2.6-4.0km Road B | dirt (osm unpaved) | felt 3 | road 3 | jolt x1 p.55 stop-go | mm .91
REST 5 units, 7.4 km, asphalt, felt 0-1, no events
MISSING U6.anchor (gps gap 48 s)
```

**Token budget.** My character-count estimate is about 30–40 tokens per FULL line and 12–15 per BRIEF line. Measure with the target model's tokenizer. Make $B\in\{64,128,256,512\}$ an ablation axis and pick $B$ from the curve of P_L, ρ_KSF and AGR against prefill latency. Prefill of a few hundred tokens is cheap on a phone relative to decode; the real cost of a long capsule is weaker attention to numbers in small models. Also ablate the serialization format: JSON vs line format vs fact sentences, and numbers vs class words vs both.

### 5.3 Plan–Ground–Realize protocol

**Plan schema** (illustrative):

```json
{"intent": "trip_summary",
 "items": [{"ref": "U2", "facts": ["F18", "F17"], "rel": "elaboration", "hedge": "none"},
           {"ref": "U3", "facts": ["F22", "F23"], "rel": "contrast", "with": "U2", "hedge": "possibly"}],
 "omit": ["U4", "U5"],
 "close": {"type": "advice_speed", "ref": "U2"}}
```

**Grammar.** Generate a GBNF grammar per capsule (llama.cpp supports GBNF and JSON-schema grammars). `ref` may take only the ledger's unit ids, and each unit's `facts` only that unit's fact ids. The grammar enumerates one alternative per unit, which keeps the dependency between a unit and its facts. Plan validity is then 100 % by construction, which directly addresses the format-violation failures that "Less Is More" reports.

**Realization prompt.** The prompt holds the fact frames, the persona card and 2–3 worked examples. The static prefix (system text and examples) is KV-cached, so its prefill is paid once.

**Verifier.** It applies the claim extractor of §6.1 against the ledger. The rejection reasons are logged and become the DPO negatives (§8.6).

### 5.4 Hedging policy

| Confidence $p$ | Required wording |
|---|---|
| ≥ 0.9 | assertive (no hedge) |
| 0.7–0.9 | "likely", "probably" |
| 0.4–0.7 | "possibly", "may have" |
| < 0.4 | not mentioned unless the user asks |

The hedge is set in the plan and checked by the verifier. "Definitely" on a 0.6 event is a verification failure. Calibration is evaluated in §6.5.

---

## 6. Metrics

### 6.1 Claim extraction

A narrative $N$ is decomposed into atomic claims $\mathcal X(N)$ in the spirit of FActScore (Min et al., 2023) and the RotoWire information-extraction metrics (Wiseman et al., 2017). Claim types: SEV (felt or road), NUM, EVT, ANCHOR, CMP, HEDGE.

- **Template outputs:** extraction is exact, because the slots are logged.
- **Free text:** use an LLM extractor with a fixed prompt. Validate it against two human annotators on at least 300 claims stratified by condition, and require Cohen's κ ≥ 0.8 on support labels.
- **Run the same extractor on every condition.** Using slot logs for T but an LLM extractor for S would create a measurement asymmetry in T's favour.

### 6.2 Claim precision against two references

$$P_R(N)=\frac{1}{\lvert\mathcal X(N)\rvert}\sum_{c\in\mathcal X(N)}\text{supp}_R(c),\qquad R\in\{L\ (\text{ledger}),\ G\ (\text{ground truth})\}.$$

Support rules by claim type:

- **NUM:** within $\max(\epsilon_{\text{abs}},\epsilon_{\text{rel}}\lvert x\rvert)$ of the ledger value, for the same unit.
- **SEV:** band consistency (below).
- **EVT:** matching (§6.4).
- **ANCHOR:** the name exists and is attached to that unit.
- **CMP:** the direction holds with a minimum margin; otherwise the correct claim is "similar".

**Hallucination.** The rate is $1-P_L$. Report separately the claims about MISSING facts (the hallucinated-missing rate), which should be 0.

### 6.3 Kinetic–Semantic Fidelity: ρ_KSF, with band consistency and overstatement

Let $\mathcal S$ be all SEV claims of a condition, pooled over the corpus. Each claim $c$ has a measured intensity $x_c$ ($a_v$ for felt claims, $\tilde a_{w,z}$ for road claims) and an expressed class $\hat k_c\in\{0,\dots,5\}$ from a fixed severity lexicon. Let $R_c$ and $Q_c$ be the mid-ranks of $x_c$ and $\hat k_c$. Then

$$\rho_{\text{KSF}}=\frac{\sum_{c\in\mathcal S}(R_c-\bar R)(Q_c-\bar Q)}{\sqrt{\sum_{c\in\mathcal S}(R_c-\bar R)^2\ \sum_{c\in\mathcal S}(Q_c-\bar Q)^2}},$$

$$\text{BC}=\frac1{\lvert\mathcal S\rvert}\sum_{c\in\mathcal S}\mathbf 1\bigl[x_c\in\mathcal B(\hat k_c)\bigr],\qquad \text{OI}=\frac1{\lvert\mathcal S\rvert}\sum_{c\in\mathcal S}\bigl(\hat k_c-k^\star(x_c)\bigr),\qquad k^\star(x)=\sum_{j=1}^{5}\mathbf 1[x\ge\theta_j],$$

where $\mathcal B(k)$ is the ISO band of class $k$ (overlaps allowed) and θ are the cut-points of §3.3.

**Why three numbers.** ρ is invariant to monotone distortion: a narrator that calls every unit one class too rough still gets ρ = 1. BC catches absolute miscalibration, and OI gives its direction. Do not collapse them into a composite; a weighted sum hides which failure happened and invites a question about the weights.

**Computation rules.**

- **Pool over the corpus.** A single narrative with two severity claims has ρ = ±1 trivially.
- **Robustness.** Report Kendall's τ_b as a check.
- **CIs.** Use a cluster bootstrap over drives. PVS has only 3 route clusters, so PVS numbers are descriptive.

**Lexicon.** Fix it *a priori* (the ISO Annex C labels plus synonyms). Validate it with a short rating task: participants place about 30 phrases on the 6-point scale; take the median as each phrase's class and report agreement.

### 6.4 Anomaly grounding precision and recall

**Narrated mentions** $\hat{\mathcal E}$. A count claim ("two bumps on Road B") expands to two mentions. Vague quantifiers expand to the lower bound of a fixed lexicon ("a few" → 2, "several" → 3). Each mention has a type and a resolved location interval $I(\hat e)$.

**Reference events** $\mathcal E^R$ have a type and a position $s(e)$.

**Matching.** Add an edge between $\hat e$ and $e$ when the types are compatible and $\operatorname{dist}(s(e),I(\hat e))\le\delta_s$. Let $M^\star$ be the maximum bipartite matching (Hopcroft–Karp). Then

$$\text{AGP}_R=\frac{\lvert M^\star\rvert}{\lvert\hat{\mathcal E}\rvert},\qquad \text{AGR}_R=\frac{\lvert M^\star\rvert}{\lvert\mathcal E^R\rvert},\qquad \text{wAGR}_R=\frac{\sum_{e\in M^\star}w_e}{\sum_{e\in\mathcal E^R}w_e},\ \ w_e=\text{MTVV}_e .$$

**Two references.**

- **$R=L$ (ledger)** isolates the language layer. Use all ledger events for precision and the must-mention set for recall.
- **$R=G$ (ground truth)** measures the system end to end:
  - on PVS, the 60 bump traversals;
  - on your corpus, the video-annotated events.

**Tolerance.** δ_s = 25 m. The label file's clock is uncertain by up to ±0.5 s on some trips, which is about ±5 m at 10 m/s, plus map-match error. Report sensitivity at 10 and 50 m.

**Error attribution.** This is the table reviewers want. For each ground-truth event, ask: detected by Layer 1? kept in the capsule? mentioned in the narrative? For each narrated event, ask: is it in the ledger? is that ledger event true? This splits errors into perception misses, selection omissions, verbalization errors (hallucinations) and perception false positives.

### 6.5 Linguistic calibration

For event claims with hedge class $h\in\{\text{assertive},\text{likely},\text{possibly}\}$, let $P_G(h)$ be their precision against ground truth. Then

$$\text{ECE}_{\text{ling}}=\sum_h\frac{n_h}{n}\bigl\lvert P_G(h)-p_h\bigr\rvert ,$$

with nominal probabilities $p_h$ fixed beforehand. Elicit them in a 2-minute survey ("If a summary says a bump *likely* happened, how likely is it?"); Kent's (1964) words of estimative probability are the classical motivation. Also report whether $P_G(\text{assertive})\ge P_G(\text{likely})\ge P_G(\text{possibly})$. Related work: Mielke et al. (TACL 2022) on linguistic calibration.

### 6.6 Novelty–Change Alignment (NCA)

For consecutive drives $(d,d')$ of the same route by the same driver:

$$\Delta_{\text{kin}}(d,d')=\operatorname{mean}_u\bigl\lvert\log a_{v,u}(d)-\log a_{v,u}(d')\bigr\rvert+\bigl(1-J_\delta(\mathcal E_d,\mathcal E_{d'})\bigr),$$

$$\Delta_{\text{claim}}(d,d')=1-\frac{\lvert\tilde{\mathcal X}_d\cap\tilde{\mathcal X}_{d'}\rvert}{\lvert\tilde{\mathcal X}_d\cup\tilde{\mathcal X}_{d'}\rvert},\qquad \text{NCA}=\operatorname{Spearman}\bigl(\Delta_{\text{kin}},\Delta_{\text{claim}}\bigr)\ \text{over pairs}.$$

- **Unit alignment:** units are aligned by 100 m along-track bins.
- **Event overlap:** $J_\delta$ is the Jaccard index of the two event sets under spatial tolerance.
- **Canonical claims:** $\tilde{\mathcal X}$ are claims reduced to (type, 100 m bin, value class).
- **Lexical-only variation:** $\text{LV}=1-\text{BLEU-4}$ between the two narratives, with numbers and names masked. It measures wording change independent of content.

**Prediction.** T+V has high LV and NCA no better than T. A narrator that selects content by what changed has higher NCA. Whether that narrator is S-PGR or a history-aware T is an empirical question.

### 6.7 Question answering

- **Answerable rate:** the ledger contains the answer. This is a property of the question set.
- **Correct-answer rate:** judged against the ledger by blind raters.
- **Unsupported-claim rate** in answers.
- **Abstention precision and recall** for "I can't tell from this drive" on unanswerable questions.

### 6.8 System cost

Measure on at least two phones, one of them mid-range:

- time to first token;
- prefill and decode tokens/s;
- end-to-end time per narrative (p50/p95);
- peak RSS;
- energy per narrative (J);
- battery %/h for 100–400 Hz logging plus GNSS;
- throttling over 10 back-to-back generations;
- plan validity, verifier rejection rate and fallback rate.

---

## 7. User study (N = 28)

### 7.1 Design

**Factor.** One within-subject factor, the narrator, with four levels: T, T+V, T→P, S-PGR. S-free and C-PGR are evaluated offline only.

**Order.** A Williams 4 × 4 design balances first-order carryover. The rows are [verified: each ordered pair occurs once, and every column contains every condition]:

| Sequence | 1st | 2nd | 3rd | 4th | Participants |
|---|---|---|---|---|---|
| A | T | T+V | S-PGR | T→P | 7 |
| B | T+V | T→P | T | S-PGR | 7 |
| C | T→P | S-PGR | T+V | T | 7 |
| D | S-PGR | T | T→P | T+V | 7 |

With 28 = 7 × 4 the design is exactly balanced. Three conditions would need 6 sequences, which 28 does not divide.

**Stimuli.** Eight video excerpts of 2–3 minutes each.

- **Sources:** PVS (all 9 trips include a right-side environment video **[verified]**) and your own corpus.
- **Content:** each excerpt holds 2–4 HRNUs and 1–3 events.
- **Allocation:** every participant sees all 8 excerpts, 2 per condition. A second Latin square rotates the excerpt-to-condition assignment, so each excerpt appears in each condition equally often. This lets the analysis treat excerpts as random effects (Judd, Westfall & Kenny, 2012).

**Error injection.** In 2 of each participant's 8 narratives, one fact is altered: a wrong count, a wrong place or a wrong severity. Across participants, each condition receives the same number of altered narratives. This measures whether fluent text makes errors harder to notice. Analyse accuracy ratings on unaltered trials only, and detection on altered trials. Debrief at the end.

### 7.2 Participants and ethics

**Who.** Licensed drivers aged 18 or over with at least one year of regular driving. Mix daily commuters and occasional drivers. Record vehicle type and phone-in-car habits. Exclude anyone involved in the project.

**Ethics.**

- Obtain approval from your institution's ethics committee before recruiting, and take informed consent.
- GNSS traces identify people (de Montjoye et al., 2013). Store them encrypted, and publish only traces truncated by 500 m at each end, or aggregates.
- Never present narratives while a participant is driving (§7.8).

### 7.3 Procedure (about 75 minutes)

1. Consent, demographics, driving experience, and a one-line self-description used for the persona item (5 min).
2. Practice trial with an excerpt and narrative excluded from analysis (5 min).
3. Four blocks, one per condition in Williams order. For each of the block's two excerpts: watch the video, read the narrative, answer the per-narrative items, then the error question. At the end of each block, complete the trust checklist (about 48 min total).
4. QA task (10 min).
   - Two new excerpts.
   - The participant types three questions per excerpt.
   - Answers come from T+Intent or S-PGR, one system per excerpt, counterbalanced.
   - The participant rates each answer for helpfulness and apparent correctness.
5. Rank the four narrators, then a 10-minute semi-structured interview.

### 7.4 Measures and item wording

All items use a 7-point scale (1 = strongly disagree, 7 = strongly agree).

| Code | Item |
|---|---|
| ACC1 | "The summary accurately described the road conditions in the clip." |
| ACC2 | "The places the summary mentioned matched where things happened in the clip." |
| SPEC | "The summary was specific enough that I could find the places it mentioned again." |
| USE | "This summary would help me decide how to drive this route next time." |
| CLAR | "The summary was easy to understand on the first reading." |
| CERT | "When the summary was unsure about something, it said so in a way that felt right." |
| NOVEL | "The summary told me something about the drive that I had not noticed myself." |
| ANNOY (reversed) | "I would find getting a summary like this after every drive annoying." |
| FIT | "This summary was written for someone like me." |
| ERR | "Did anything in this summary seem wrong? If so, what?" (yes/no plus free text) |

**Additional instruments.**

- **Attention checks:** two instructed-response items, e.g. "Select 'somewhat disagree' for this statement."
- **Trust (per block):** the 12-item Checklist for Trust between People and Automation (Jian, Bisantz & Drury, 2000), 7-point.
- **Composite:** Accuracy = mean(ACC1, ACC2). Report its reliability (α or ω).
- **QA answers:** "This answer was helpful" and "As far as I can tell, this answer was correct" (7-point).

### 7.5 Hypotheses (confirmatory, fixed before data collection)

- **H1 (offline):** fidelity is non-inferior: $P_L(\text{S-PGR})-P_L(\text{T})>-0.02$.
- **H2 (offline):** S-PGR's correct-answer rate exceeds T+Intent's on the sealed question set.
- **H3 (study):** USE(S-PGR) > USE(T+V).
- **H4 (study):** perceived accuracy is non-inferior: Accuracy(S-PGR) − Accuracy(T) > −0.5 scale points.

Everything else is exploratory and must be labelled so in the paper: ANNOY, NOVEL, FIT, CERT, trust, error detection, rankings and habituation.

### 7.6 Analysis

- **Ratings (H3 and exploratory items).** Fit a cumulative link mixed model (R `ordinal::clmm`):
  $$\operatorname{logit}P(Y_{ij}\le k)=\theta_k-\bigl(\beta_{\text{cond}(ij)}+\beta_{\text{period}(ij)}+u_i+w_j\bigr),\quad u_i\sim\mathcal N(0,\sigma_u^2),\ w_j\sim\mathcal N(0,\sigma_w^2),$$
  with participant $i$ and excerpt $j$. Test H3 as a single contrast, with Holm correction across H2 and H3. Check carryover by adding a previous-condition term, which the Williams design makes estimable.
- **Robustness for ratings.** Repeat with an ART ANOVA (Wobbrock et al., 2011), and with Friedman plus Wilcoxon signed-rank tests on per-participant means.
- **H4.** Compute per-participant mean Accuracy per condition (unaltered trials). H4 holds if the lower bound of the 90 % CI of the paired difference exceeds −0.5 (a one-sided test at α = .05). Wilcoxon serves as a robustness check.
- **H1.** Cluster bootstrap over drives (10,000 resamples) of the P_L difference. The one-sided 95 % lower bound must exceed −0.02.
- **H2.** GLMM `correct ~ condition + (1|question) + (1|excerpt)`, or an exact McNemar test on paired outcomes.
- **Error detection.** GLMM `detected ~ condition + (1|participant) + (1|excerpt)` on altered trials.
- **Reliability of human coding.** Krippendorff's α (ordinal) for severity coding and Cohen's κ for claim support, each with a target of at least 0.8.
- **Interviews.** Reflexive thematic analysis (Braun & Clarke, 2006).

### 7.7 Power [verified]

- **Superiority tests.** N = 28 within subjects gives 80 % power for dz ≥ 0.55 at α = .05 (two-sided), and for dz ≥ 0.65 at α = .05/3 (Holm worst case). Detecting dz = 0.5 would need N ≈ 34.
- **H4 (non-inferiority).** N = 28 gives 82.5 % power if the 0.5-point margin equals 0.5 SD of the paired differences, i.e. if that SD is at most 1 scale point. If the SD is 1.25 points, you need N = 40. Estimate the SD in a pilot (n = 4–6) before fixing N. If it is larger, increase N or justify a wider margin before the main study.
- **Simulation.** Two excerpts per condition raise per-participant reliability. Running a simulation-based power analysis from pilot variance components (`simr`) is the honest version of this section.

### 7.8 Field phase (exploratory; n ≈ 10–12 of the 28; 14 days)

- **Delivery.** The app logs drives and delivers narratives only after the trip has ended and the car has been stationary for at least 2 minutes. Never deliver while driving; the distraction risk would sink the paper and endanger people.
- **Schedule.** Days alternate between T+V and S-PGR in a randomized balanced schedule, 7 days each.
- **Daily data.** A 3-item EMA (USE, ANNOY, NOVEL) plus logs: open rate, time to open, dwell time, scroll depth and questions asked.
- **Model.** `rating ~ cond * day + (1 + day | participant)`. Habituation is the cond × day interaction. With n ≈ 10 it is descriptive, and the paper should say so.

---

## 8. Implementation plan for this workspace

### 8.1 Fix the data first (all items verified today)

1. **Stop averaging left and right loggers.** The two sides are separate loggers with separate clocks.
   - **Size of the offsets.** Yaw-rate cross-correlation puts the left–right offset at +0.54 s on trip 5, +0.22 s on trip 6 and −0.09 s on trip 1. Trip 4 drifts from +0.10 to −0.05 s within the trip.
   - **Effect on the signal.** At zero lag, vertical-acceleration correlation drops to −0.09 (trip 5) and −0.23 (trip 6), against 0.56 and 0.72 at the best lag. Averaging $x(t)$ with $x(t-\Delta)$ is a comb filter with response $\lvert\cos(\pi f\Delta)\rvert$; for Δ = 0.54 s the first null is at 0.93 Hz, right on body bounce.
   - **Same-side sensors are fine.** Dashboard, above-suspension and below-suspension on one side share a logger and agree to ≤ 0.01 s.
   - **What to use.** The left logger, with its dashboard IMU as the phone proxy and its suspension IMUs as privileged references. If you need both sides, align per 60-s window by yaw-rate cross-correlation first.
2. **Fix the notebook's description.** The sensors are not smartphones (B1).
3. **Do not use `dataset_gps_mpu_mid.csv`** in the raw folders. It is a 10 Hz decimation of the left/right average, so it inherits the clock problem and aliases everything above 5 Hz.
4. **Ground truth.** Use the surface labels (54 transitions) and the bump labels (60 traversals of 20 physical bumps: 11, 7 and 2 per route). The per-sample good/regular/bad labels flip 66–272 times per trip; use them only aggregated over ≥ 50 m.
5. **Splits.** The routes are {1,4,7}, {2,5,8} and {3,6,9} (13.8, 11.6 and 10.7 km), each driven by all three vehicles. Use **leave-one-route-out**. For vehicle generalization, test on (route r, vehicle k) after training on the 4 trips with route ≠ r and vehicle ≠ k. Random window splits leak; this is the same failure as your ICONAT paper's same-track validation and test.
6. **Collection app.** RoadQualityMonitor logs at 10 Hz. That cannot feed Layer 1, so it is not your collection app.

### 8.2 Package layout

```
roadscribe/
  config.py           frozen dataclasses for every threshold; one YAML per experiment; config hash saved with outputs
  schemas.py          data contracts (§8.3)
  io/
    pvs.py            ONE logger (left) + labels; units; provenance; clock-offset check by yaw xcorr
    phone.py          Sensor Logger / phyphox CSV -> RawTrip; tap-sync of two phones
    store.py          Parquet/JSON I/O under data/roadscribe/{trip_id}/
  layer1/
    gnss.py           PCHIP speed, a_lon, course rate, outlier rejection
    orientation.py    mount episodes, gated gravity (geometric median), Rodrigues, Procrustes yaw, CI
    iso2631.py        analog W_k/W_d; FFT weighting; pre-warped SOS; a_w, a_v, MTVV, VDV, CF, kurtosis; shock flag; comfort class
    spectral.py       Welch PSD, wheel-hop peak, modal band ratios
    speed.py          elasticity fit (mixed model), decoupled intensity, counterfactual
    events.py         TKEO, adaptive threshold, maneuver states, decision rule, p_defect calibration
  layer2/
    osm.py            osmnx extract cached per bbox; tags, traffic_calming, POIs; coverage report
    mapmatch.py       HMM Viterbi + forward-backward confidence
    distance.py       5 m grid, feature matrix
    mipelt.py         MI-PELT: cumsum costs, corrected pruning, l_min
    arbitrate.py      snapping, boundary typing, merging, map-signal disagreement -> HRNUs
  layer3/
    ledger.py         HRNU/events -> Facts (incl. MISSING, META)
    kst.py            salience, serializations, knapsack DP, capsule, token counts
    templates.py      T and T+V realizers; also the per-sentence fallback
    plan_grammar.py   per-capsule GBNF from ledger ids
    slm.py            backends: llama-cpp-python (dev), transformers (training), remote (teacher, C-PGR)
    pgr.py            plan -> ground -> realize -> verify -> fallback
    hedging.py        confidence -> hedge band; checks
  eval/
    claims.py         extraction (slot logs, rules, LLM extractor) + human-validation export
    metrics.py        P_L, P_G, rho_KSF, BC, OI, AGP/AGR/wAGR, ECE_ling, NCA, LV, QA metrics
    stats.py          cluster bootstrap, non-inferiority, Holm; CSV export for R (clmm, glmer)
    bench.py          E1-E5 runners
  synth/
    sampler.py        synthetic capsules: Markov chain over surfaces, Poisson events, missingness patterns
    teacher.py        teacher generation + verification filter
    perturb.py        DPO negatives; E3 perturbation suite
tests/                one test per verified number in Appendix A (filters, PELT, orientation, offsets)
```

### 8.3 Data contracts

```python
# roadscribe/schemas.py  (SI units throughout; times in s; distances in m)
from dataclasses import dataclass, field
from typing import Literal, Optional, Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RawTrip:
    trip_id: str
    source: Literal["pvs", "phone"]
    route_id: str; vehicle_id: str; driver_id: str
    t: np.ndarray                  # (N,) IMU clock, monotonic
    f_b: np.ndarray                # (N,3) specific force incl. gravity, body/sensor frame
    w_b: Optional[np.ndarray]      # (N,3) rad/s; None if no gyroscope
    fs: float
    gnss: pd.DataFrame             # t, lat, lon, speed_mps [, course_deg, hacc_m] at native rate
    labels: Optional[pd.DataFrame] # PVS: surface, bump, quality per IMU sample
    meta: dict = field(default_factory=dict)  # logger side, mount, clock offsets, provenance, config hash

@dataclass
class VehicleFrameTrip:
    trip_id: str
    a_v: np.ndarray                # (N,3) gravity-free, x fwd, y left, z up
    w_v: Optional[np.ndarray]
    v: np.ndarray                  # (N,) speed, PCHIP from GNSS
    a_lon_gnss: np.ndarray
    s: np.ndarray                  # along-track distance (arc length after map matching)
    episode: np.ndarray            # (N,) mount-episode index
    R_vb: list[np.ndarray]         # one 3x3 per episode
    yaw_deg: list[float]; yaw_ci_deg: list[tuple[float, float]]; n_quasistatic: list[int]

@dataclass
class Event:
    event_id: str
    kind: Literal["impulse", "avoidance"]
    label: Literal["defect", "artifact", "ambiguous"]
    t0: float; t1: float; s: float; lat: float; lon: float
    tkeo_peak: float; band_ratio: float; mtvv_z: float
    maneuver: Literal["cruise", "brake", "accel", "turn", "swerve", "stop_go"]
    p_defect: float

@dataclass
class HRNU:
    unit_id: str
    s0: float; s1: float; t0: float; t1: float
    osm_way_ids: list[int]; names: list[str]; osm_tags: dict
    a_v: float; a_wz: float; a_wz_dec: Optional[float]           # None if speed_low
    vdv_z: float; mtvv_z: float; cf_z: float; kurt_z: float; shock: bool
    iso_felt: int; iso_road: Optional[int]
    r_wb: float; r_hb: float; v_mean: float
    event_ids: list[str]
    boundary_left: Literal["trip_start", "map+signal", "signal", "map"]
    map_signal_disagreement: Optional[float]
    conf: dict                      # map_match, orientation, channels present

@dataclass(frozen=True)
class Fact:
    fact_id: str
    kind: Literal["SEV", "NUM", "EVT", "ANCHOR", "CMP", "MISSING", "META"]
    unit_id: Optional[str]; key: str; value: Any
    tol: Optional[float] = None; conf: float = 1.0; note: str = ""

@dataclass
class Capsule:
    capsule_id: str; trip_id: str; persona: str; budget: int
    text: str                       # exactly what the SLM sees
    n_tokens: int; tokenizer: str
    selection: dict                 # unit_id -> OMIT/BRIEF/FULL
    ledger: list[Fact]

@dataclass
class Claim:
    span: tuple[int, int]; kind: str; unit_id: Optional[str]; key: str; value: Any
    hedge: Literal["assertive", "likely", "possibly"]
    supp_ledger: Optional[bool] = None; supp_gt: Optional[bool] = None

@dataclass
class Narrative:
    capsule_id: str; condition: str; model: str; seed: int
    text: str; plan: Optional[dict]
    claims: list[Claim]
    regenerations: int; fallback_sentences: int
    latency_ms: dict; energy_j: Optional[float]
```

### 8.4 Storage and data flow

```
data/roadscribe/
  raw_index.csv                         trip_id, source, route_id, vehicle_id, driver_id, path, logger, clock_offset_s
  {trip_id}/vehicle_frame.parquet       t, a_x, a_y, a_z, w_x, w_y, w_z, v, a_lon_gnss, s, episode
  {trip_id}/orientation.json            per-episode R_vb, yaw, CI, tilt, n_quasistatic
  {trip_id}/l1_grid.parquet             5 m grid features
  {trip_id}/events.parquet
  {trip_id}/mapmatch.parquet            fix -> edge, s, confidence
  {trip_id}/hrnu.json
  {trip_id}/ledger.jsonl
  {trip_id}/capsules/{persona}_{B}.txt
  {trip_id}/narratives/{condition}.jsonl   text, plan, claims, verification, latency
```

**Flow:** CSV → `RawTrip` → `VehicleFrameTrip` → (`l1_grid`, `Event`) → map match → MI-PELT → `HRNU` → `Fact` ledger → `Capsule` → `Narrative` → claims → metrics. Every stage writes its config hash, so any number in the paper traces to a config.

### 8.5 Your own data collection (needed, cheap)

**Equipment.** Two Android phones (ideally the same model), an ordinary dashboard or windshield holder, double-sided tape or a rigid bracket, and a dashcam or the holder phone's camera.

- **P1 (realistic):** in the holder. Logs accelerometer, gyroscope, GNSS and, if possible, video.
- **P2 (reference):** fixed rigidly at the seat rail or on the floor beside the driver's seat, never touched. The seat rail is the seat's floor input, not the seat surface; a true ISO seat measurement needs a seat-pad accelerometer. Describe P2 as "floor-referenced".

**Logging.** Use the maximum rate. Above 200 Hz, Android 12+ requires the `HIGH_SAMPLING_RATE_SENSORS` permission. Use Sensor Logger or phyphox, and verify the achieved rate on your phone by logging 60 s at rest and checking the timestamp histogram.

**Sync.** Tap the phones together three times at the start and end of each drive. Refine with yaw-rate cross-correlation (the method that found the PVS offsets). Aim for a residual offset below 10 ms.

**Drives.**

1. **Repeated route.** One 10–15 minute route with mixed surfaces and known speed breakers, driven at least 10 times on different days by the same driver. It feeds NCA, the habituation stimuli, and the test–retest reliability of Layer 1 (ICC of per-segment $a_v$ across days).
2. **Variety.** 10–20 drives on other routes, ideally with 2 or more vehicles. These add route clusters for statistics.
3. **Handling.** In 3 drives, pick up and replace P1 while stopped. In 2 drives, remount it elsewhere. This tests orientation re-estimation.

**Annotation.** From the video, mark speed breakers, potholes, rough patches, surface changes and rail crossings with timestamps. Two annotators code at least 20 % of the material (report κ). Annotate before anyone sees the narratives.

**What it buys.**

- orientation and handling under real conditions;
- the holder transfer function $H(f)=P_{21}(f)/P_{11}(f)$ from P2 to P1, with coherence; holder resonance shows up as a peak;
- proxy validity for the ISO measures: Bland–Altman of segment $a_w$, P1 against P2;
- repeated drives for NCA;
- more independent routes for inference.

### 8.6 Getting the SLM to work: few-shot first, SFT if needed, DPO only if still needed

**Step 0: few-shot PGR (about one week).**

- **Runtime:** `llama-cpp-python` with Q4_K_M GGUF models.
- **Models:** 2–3 models of 4B parameters or fewer. Llama-3.2-3B-Instruct (Sept 2024) is the familiar reference but will look dated to 2027 reviewers, so add a current one (e.g. Qwen3-4B or Gemma 3n). Check each license; licenses differ even within one model family.
- **Prompt:** the system spec plus 2–3 worked examples (plan and realization), KV-cached.
- **Measure on the dev split:** plan validity, P_L, fallback rate, persona fit, and QA correctness.
- **Decision:** stop here if P_L ≥ 0.98, fallback ≤ 10 %, and E1–E3 meet their thresholds on dev. Fewer moving parts make a better paper.

**Step 1: verified-distillation SFT, only if Step 0 misses.**

- **Capsules.** Take 30 % from real HRNUs (training routes and training drives only). Generate 70 % with `synth/sampler.py`: a Markov chain over surfaces, Poisson events by surface, and missingness and contradiction patterns.
- **Task mix (about 6k examples):**
  - 40 % trip narratives over varied personas and budgets;
  - 25 % QA, of which 20 % are unanswerable and the target is to abstain;
  - 20 % degraded or contradictory capsules;
  - 15 % cross-trip comparisons.
- **Teacher.** A frontier model writes plans and realizations under the same spec. Keep only outputs the verifier passes completely, and spot-check 200 by hand. Check the teacher's terms of use on training other models with its outputs.
- **Training.** QLoRA (4-bit NF4), r = 16, α = 32, dropout 0.05, all linear layers, lr 2e-4 with cosine decay, 2 epochs, max length 1536, effective batch 16, gradient checkpointing. A 3B model fits a 6 GB RTX 4050 (tight) or a Kaggle T4. I estimate 1–3 h per epoch at about 6k examples; measure it.
- **Export.** Merge the LoRA and convert to GGUF Q4_K_M.

**Step 2: DPO, only if hallucinated-missing exceeds 1 % or P_L stays below 0.98.**

- **Pairs.** About 2k. The chosen output is a verified one; the rejected output is a minimal perturbation of it:
  - a number drifted by 20–50 %;
  - a unit or anchor swapped;
  - an invented event;
  - a must-mention fact dropped;
  - a low-confidence fact left unhedged;
  - a value supplied for a MISSING fact.
- **Settings.** β = 0.1, lr 5e-6, 1 epoch (Tian et al., ICLR 2024, is the factuality-DPO precedent).

**Leakage controls (state them in the paper).**

- No test personas, no sealed questions, and no capsules from test routes or drives in any few-shot or training material.
- Freeze prompts and exemplars before touching a sealed set.

### 8.7 On-device stack and measurements

- **Runtime.** llama.cpp via the Android NDK, or a React Native binding such as `llama.rn` inside your existing RN app (check that it is maintained). Use a GBNF grammar for the plan and a cached static prefix.
- **Layer 1.** The DSP is cheap: FFT weighting per HRNU, biquads and TKEO. Port it to Kotlin or a small native module.
- **Layer 2.** Map matching against an offline OSM extract is the heavy part. For IMWUT, running Layer 2 post-trip on a laptop is acceptable if stated. For MobiSys it must run on the phone.
- **Energy.** Sample `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW` at ≥ 10 Hz, multiply by voltage, and subtract an idle baseline. Use an external power monitor if you can borrow one. The full metric list is in §6.8.

---

## 9. Schedule, gates, kill criteria

| Weeks from 2026-11-02 | Work | Gate |
|---|---|---|
| 1 | Data fixes (§8.1); `layer1/` with tests reproducing Appendix A; submit the ethics application; OSM coverage check | none |
| 2–3 | `layer2/`; pilot drive with two phones | **G1:** bump detection on held-out routes beats speed-only and RMS(a_z)/√v |
| 3–6 | Own data collection and annotation (in parallel) | **G2:** MI-PELT boundary F1 beats constant-β PELT and speed-only PELT |
| 4–5 | `layer3/`: ledger, KST, T/T+V/T→P, PGR few-shot, verifier; question elicitation | none |
| 6 | E1–E3 on dev splits | **G3:** kill gate, below |
| 6–7 | SFT if needed; on-device port; systems measurements; pilot study (n = 4–6) | The pilot SD confirms N |
| 8–10 | Study A (N = 28); field phase starts | none |
| 11–13 | Sealed-set runs, analysis, writing | none |

Feb 1, 2027 is reachable only if ethics approval arrives by week 6 and the field phase is cut. Otherwise target May 1.

**Kill criteria** (write them down now):

- **K1 (at G1).** If Layer-1 bump detection does not beat speed-only on held-out routes, the narratives are not grounded in vibration. Stop and fix Layer 1.
- **K2 (at G2).** If MI-PELT does not beat constant-β PELT, drop C2 and use plain PELT.
- **K3 (at G3).** If S-PGR misses every E1–E3 threshold on dev, the SLM claim is unsupported. The fallback is a measurement paper on when templates suffice (weaker at IMWUT), or a different venue.
- **K4.** If the fallback rate exceeds 30 % on the sealed set, report it as the main limitation.

---

## Appendix A: verified numbers

All from scripts in [verification/](verification/), run on 2026-11-01 with `C:/Users/nishk/anaconda3/envs/tf/python.exe`. Raw output is in [verification/RESULTS.md](verification/RESULTS.md).

| Quantity | Value | Script |
|---|---|---|
| PVS trips, duration, distance | 9 trips; 180.1 min; 108.6 km | pvs_facts.py |
| IMU rate, gaps | 100.0 Hz; no gap > 50 ms | pvs_facts.py |
| GNSS | 1 Hz; speed in m/s (∫v dt within 1 % of haversine path) | pvs_facts.py |
| Routes | {1,4,7} 13.8 km, {2,5,8} 11.6 km, {3,6,9} 10.7 km | pvs_facts.py |
| Labels | 54 surface transitions; 60 bump traversals (11/7/2 per route); bump speed median 3.6–5.1 m/s; quality labels flip 66–272 times per trip | pvs_facts.py |
| Axis convention | forward = +y_s (corr with GNSS dv/dt +0.41 to +0.69); z up | pvs_facts.py |
| L/R clock offsets | trip 1 −0.09 s; trip 4 +0.10 → −0.05 s; trip 5 +0.54 s; trip 6 +0.22 s; others ≤ 0.05 s; same-side ≤ 0.01 s | pvs_align_doublehit.py |
| Double-hit cue | ρ(lag, 1/v) between −0.12 and 0.00, n = 60, p ≥ 0.36 | pvs_align_doublehit.py |
| Maneuver effects | see the table in §3.5; speed elasticity 0.32 / 0.36 / 0.62 | pvs_maneuver_bands.py |
| Orientation | yaw 87.9–92.5° (vehicles 1, 3), 103.1–105.5° (vehicle 2); equivariance ≤ 0.63° | verify_orientation.py |
| ISO weights | W_k ×1000: 482 @1 Hz, 1039 @5 Hz, 769 @16 Hz; W_d: 1011 @1 Hz | verify_math.py |
| IIR error, fs = 100 Hz | ≤ 0.72 dB (W_k), ≤ 1.22 dB (W_d) to 20 Hz; 7.3–7.7 dB at 40 Hz | verify_math.py |
| IIR error, fs = 400 Hz | ≤ 0.23 dB to 40 Hz | verify_fft_weighting.py |
| FFT weighting | 0.0000 dB; VDV 1.3842 vs reference 1.3842; IIR@100 peak −3.8 % | verify_fft_weighting.py |
| MI-PELT pruning | corrected rule 0/300 mismatches vs exhaustive; textbook rule 300/300 sub-optimal | verify_math.py |
| Power, N = 28 | dz ≥ 0.549 (α .05), 0.646 (α .05/3), 0.703 (α .05/6); N = 33.4 for dz = 0.5 | verify_math.py |
| Non-inferiority, N = 28 | 82.5 % power at margin = 0.5 SD; N = 40 at 0.4 SD | inline (statsmodels) |
| Williams 4×4 | rows [[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]; balanced | verify_math.py |
| ISO comfort cut-points | 0.315, 0.561, 0.894, 1.414, 2.236 m/s² | verify_math.py |
