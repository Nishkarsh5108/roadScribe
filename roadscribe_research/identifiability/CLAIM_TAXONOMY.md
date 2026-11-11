# Claim taxonomy: roads (draft for Phase 3, written 2026-11-12)

For each kind of claim a road narrator makes, this file records what should happen to it when only a nuisance changes: the vehicle (channel), the speed (rate), the mount (gauge) or the clock. The model throughout is

log S = A_vehicle(f) + B_place(f/ρ) − log ρ

| Claim | Example | Under a vehicle change | Under a speed change | Identifiable from one traversal by one vehicle? |
|---|---|---|---|---|
| Surface type | "cobblestone from 2.1 to 2.8 km" | Invariant | Invariant | Usually. Spectral *shape* differs between surfaces, but the vehicle filter blurs it. Empirical (P1 ground truth) |
| Discrete event at a place | "a speed bump at 1.3 km" | Invariant (place); its felt severity is covariant | Invariant (place) | Yes, for the location; its severity is covariant |
| Road roughness level | "this stretch is rough" | **Invariant** | **Invariant** | **No** without a calibrated vehicle channel or speed diversity at that place (C6, `tests/test_scr.py`) |
| Relative roughness between places | "rougher than the previous stretch" | Invariant | Invariant | Only if the speeds at the two places are similar, or the common channel is known; otherwise the bias d(ν·ρ) differs between the places |
| Felt comfort | "fairly uncomfortable" (ISO a_v) | **Covariant**, through A_vehicle | **Covariant**, through B(f/ρ) and the Jacobian | Yes, as a statement about *this* ride |
| Felt severity of an event | "a hard jolt" (MTVV) | Covariant | Covariant (drivers slow before bumps) | Yes, for this ride |
| Cause attribution | "rough because of the road, not your speed" | Invariant | Invariant | Needs within-place speed variation, or a calibrated γ or channel |
| Counterfactual | "at 30 km/h it would feel …" | Model-dependent | Model-dependent | Needs B over the spatial-frequency band that speed shifts into view, plus A at those frequencies. An extrapolation; should be hedged |
| Map–signal disagreement | "the map says asphalt, but it feels like cobblestone" | Invariant | Invariant | As for surface type |

**What a pilot should test:** whether invariant claims agree across nuisance changes (P1, P2), and whether covariant claims change *as the model predicts*. That second test is P3 territory: predicted a_v for vehicle B from (Â_B, B̂), compared with B's measured a_v.

**Generation rule this implies:** a claim marked "No" in the last column must not be stated as fact unless the needed calibration or diversity is available. Otherwise the system should say what is ambiguous and what would resolve it. That is the identifiability gate of Phase 5.
