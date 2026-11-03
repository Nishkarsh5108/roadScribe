# RoadScribe

**Translating High-Frequency Vehicular IMU Telemetry and Road Topology into Grounded Natural Language Trip Narratives**

RoadScribe is a multimodal mobile sensing framework that bridges continuous, high-frequency vehicular dynamics (100 Hz IMU + GPS) and topological road graphs (OpenStreetMap) to generate grounded, section-wise natural language road quality and ride comfort reviews.

---

## Architecture Overview

1. **Layer 1 (Edge Biodynamics & Signal Processing):**
   - 3D gravity-aligned vehicle frame orientation tracking (gated geometric median + Procrustes yaw).
   - Speed-invariant kinetic normalization decoupling intrinsic surface roughness from human cabin comfort.
   - ISO 2631-1 biodynamic comfort filter bank ($W_k, W_d$, Crest Factor, VDV).
   - Modal band ratios (sprung heave 1–3 Hz, unsprung axle-hop 10–18 Hz, macro-texture chatter 18–45 Hz).
   - TKEO impulse detection and maneuver decoupling.

2. **Layer 2 (Spatial Topology & Dynamic Segmentation):**
   - HMM Viterbi map-matching to OpenStreetMap planar road network.
   - Map-Informed PELT (MI-PELT) change-point detector with corrected pruning under location-dependent penalties.
   - Dynamic trip partitioning into Homogeneous Road Narrative Units (HRNUs).

3. **Layer 3 (Grounding & Synthesis):**
   - Kinetic-Spatial Tokenizer (KST) compiling compact, multiple-choice knapsack capsules.
   - Plan–Ground–Realize (PGR) generation protocol with deterministic template fallbacks.
   - Quantized on-device Small Language Models (SLMs) with epistemic guardrails.

---

## Directory Structure

```
roadscribe/
├── roadscribe_research/
│   ├── METHODOLOGY.md            # Full formal specification and implementation plan
│   ├── paper/
│   │   ├── sec3_system.tex       # ACM IMWUT/MobiSys manuscript Section 3
│   │   └── sec4_evaluation.tex   # ACM IMWUT/MobiSys manuscript Section 4
│   └── verification/
│       ├── RESULTS.md            # Empirical verification logs on PVS dataset
│       └── *.py                  # Verification scripts (filters, PELT, orientation, etc.)
├── notebooks/
│   └── 01_prepare_pvs_data.ipynb # PVS data preparation pipeline
└── .gitignore
```
