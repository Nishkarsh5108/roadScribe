# Revision of 2026-11-13: how each review point was handled

This note maps each point of the simulated ICDSI review (scores 3 / 4 / 3) to what changed and where. Section numbers refer to the revised paper (15 pages). The decision log is `../identifiability/WORKLOG.md` (D26–D28 and the "Night of 2026-11-13" entries).

## The paper's framing

- **New title:** *Verified Is Not True: Two Levels of Claim Identifiability in Sensor-to-Text Narration*.
- **Contribution:** a definition and an operational test for claim identifiability at two levels. The world level asks whether the sensor value is a property of the road under changing vehicle and speed. The input level asks whether each value a training target states is contained in the model's input.
- **Evidence:** a pre-registered experiment at each level (P1–P4 and P7; P6).
- **Role of RoadScribe:** the testbed, not the contribution.
- **Scope paragraph** (§1): no new estimator, model or verifier. Per-vehicle calibration is established practice (Sayers et al. 1986). One dataset.

## Must-fix items

| Review item | What changed | Where |
|---|---|---|
| "108" vs table rows of 54 | 54 = 9 drives × 2 audiences × 3 seeds; deterministic narrators have 18. "108" is gone; the calibrated-ledger repeat is one sentence | §6 Protocol, Table 4 |
| "Guarantee" while the template fails 22% | The 22% came from a **verifier bug**: summaries of "the other N stretches" were checked against an empty unit set (D26). After the fix the template passes on every drive (P_L 100%), so the guarantee holds. All numbers were rerun | §4, Table 4, README |
| 80% → 23% in abstract and contributions | Removed from both. In §5.4 it is called illustrative | §5.4 |
| Pre-registration URL, hashes, dates; "before the vibration data were analysed" | Table 1 lists P1–P7 with commit hash, time and outcome. The γ = 0.35 history is stated (fixed 2026-11-03, estimated during development on the same trips). The repository URL is in Data availability | §5.1, Table 1 |
| Data and code availability, frontier model, Disclosure of Interests | Added. **Name the teacher model yourself:** a red TBD in §6.1. The notebook says "Opus", the card says "frontier-llm", and this cannot be verified from the repository | §8, §6.1 |
| Expand S-PGR, P_L, T+V; drop "original design" / "as designed" | S-PGR no longer appears (rows are described). P_L, T and T+V are defined in §4. The other phrases are gone | §4, Table 4 |
| Blind-review rule | **Not checked.** No ICDSI-2027 CFP was found. The PDF names the authors and the GitHub URL. If review is double-blind, anonymize both | — |

## Items that would raise the odds

| Review item | What was done | Result |
|---|---|---|
| Worked example | Table 3: one place, three vehicles, wording before and after calibration (`x12`) | Three classes become one; 70 of 310 places behave like this |
| Cite Sayers 1986, AKM 1999, Maynez 2020 | Cited, plus Dhingra 2019, Dušek 2019, Nie 2019, Filippova 2020, Gekhman 2024 and Peters 2016 (ICP) | §2, §3 |
| Held-out classification without shared places | P7-H5: route **and** vehicle held out | 98.3% [97.2, 99.2] vs 90.5% [86.4, 94.5]; balanced 0.82 vs 0.66; +7.7 points [4.4, 11.4] |
| Missing CIs | κ_cal 0.96 [0.93, 0.97]; ride share [14.8, 27.4]; all-agree CIs; γ̂ 0.86 [0.68, 1.02]; accuracies | Table 2, §5.4, §7 |
| κ vs km of shared road | P7-H4 | 0.5 km: 0.95 [0.83, 0.98]; 1 km: 0.96 [0.91, 0.99] |
| Verifier extraction recall | Not hand-annotated: no human annotator was available overnight. Numbers are covered by construction (unparsed numbers fail). Descriptors outside the lexicon were counted (`x13`) | 0–0.2 per narrative for fine-tuned models, 7–10 for prompted models |
| γ̂ ≈ 0.87 vs theory 0.5 | Discussed | Equivalent waviness w ≈ 2.7; class agreement insensitive (κ 0.95 with γ = 0.5) |
| Rename "felt" | "Ride value" and "ride class"; road classes are "speed-normalized comfort classes, not a pavement condition index" | §4, §7 |
| Prompted 7–8B baseline | Prompted Qwen2.5-3B and 7B on input B | P_L 0% and 4.5%: their text is mostly unverifiable (claims name no stretch) |
| Dense sentences | Partly. The page limit forced compression | — |

## Other reviewer points

| Point | Result |
|---|---|
| No noise floor | P7-H1: left vs right dashboard module, same vehicle and pass: κ 0.98 [0.97, 0.98]. Calibration closes 92% of the gap |
| Speed filter excludes rough stretches | True: 27 of 38 "bad" places. Without a floor, the vehicle share is 15.4% and κ goes from 0.75 to 0.95 |
| Calibration results near-automatic | Stated: a near-zero vehicle share is automatic given route-stable gains; the agreement gain is not |
| The ablation | The gain gives agreement; the elasticity fixes speed dependence (§5.4) |
| Event-detector precision | 17% within 50 m of labelled bumps (a lower bound). 52.8% of events are reproduced by both other vehicles vs 6.8% by chance (§5.5, exploratory) |
| Which side's label | Left (§5.1) |
| Dashboard ≠ seat; comfort ≠ condition; agreement ≠ correctness | §7 |
| Three vehicles only | Stated in §5.1 (intervals) and §7 |
| Confounded P5 retraining (Reviewer A #2) | Replaced by P6, a controlled test with the same targets and only the input changed. 3 of 4 hypotheses held. H2 failed because of how the verifier orders failure reasons; this is explained and the per-type rates are given |
| Is mis-copying a sampling artefact or under-training? | P8 (exploratory): greedy decoding gives 76.3% vs 76.2%; a second epoch (loss 0.09) gives 75.2%, with 43% of positions still wrong |

## Still open, for you

1. **Teacher model name** (red TBD in §6.1).
2. **The repository is private.** Make it public, or give an anonymized mirror if the review is blind.
3. **ICDSI page limit.** The paper is 15 pages including references; check the CFP.
4. **Dual submission.** Make sure no IMWUT version is under review at the same time.
5. **Hand-annotation of verifier recall.** About ten narratives annotated by a person would close the remaining reviewer point.

# Second review (Gemini, 2026-11-13 morning) and the two-column version

The paper is now two-column IEEE (`IEEEtran` conference), 9 pages, as the professor asked (8–10). The title is *Same Road, Three Stories: Why Verified Sensor Narratives Can Still Be Wrong*, and the abstract and introduction open with the three-cars example (Table III).

| Review point | What changed |
|---|---|
| Red `[TBD]` for the teacher model | Claude Opus 5.5 (Anthropic), §VI-A |
| Verifier bug: "the 0.0 and 0.05 km marks" read as a length | Fixed (D29: positions may have two decimals; new test). Everything rescored. Arm B: P_L 76.2% → 76.4%, distance errors 13% → 8%, still 0 accepted. P6 verdicts unchanged |
| P5 comparison confounded by rebuilt capsules | P9 (exploratory, declared first): the first adapter on the rebuilt capsules gets P_L 46.1%, 0 accepted, 35 names and 43 map tags invented. The gain comes from the retraining |
| 86% vs 40% position error | Two measures defined (position error; wrong-position share). §VI-C now uses position error, 86% → 41% |
| Tiny speed counts; degenerate CI | Counts stated (15 vs 35 vs 34; 1 error each in B and C); the CI is removed |
| Overgeneralization from one 3B model | The policy is limited to "the 3B model we studied"; larger models and constrained decoding are named as untested |
| 300-token row has 18 runs | The protocol says so |
| 5 m/s floor removes 71% of "bad" places | Accuracy without the floor (361 places, exploratory): 94.6% vs 87.3%, balanced 0.87 vs 0.69; bad passes 50 vs 24 of 69. The abstract and introduction now use balanced accuracy |
| "Held-out routes and vehicles" overstated | The scheme is stated exactly: the test vehicle's gain comes from its own drives on the other routes; an unseen vehicle cannot be calibrated |
| "Pre-registered" over a 24 h window, self-hosted | Now "specified in advance", with the caveat that commits are in an author-controlled repository; P7 is marked as added after a review round |
| < 0.1% vehicle share is near-automatic | Table II footnote |
| Why the spectral model failed | One pass per vehicle and place leaves little within-place speed variation, so the common channel of Eq. 1 is not identified (§V-D) |
| No comparison with PVS's classifiers | Stated as out of scope (§VII) |
| Mount rigidity | §VII |
| "et al." in references | IEEE style abbreviates long author lists |

**Still open for you:**
- Make the repository public.
- Confirm the template with the organisers. The professor said two-column; IEEEtran is the usual choice, but check whether ICDSI supplies its own.
- The recall annotation.
