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
roadScribe/
├── roadscribe/                   # Python package (METHODOLOGY §8.2)
│   ├── config.py                 # frozen dataclasses for every threshold; YAML + config hash
│   ├── schemas.py                # data contracts: RawTrip, VehicleFrameTrip, Event, HRNU, Fact, Capsule, ...
│   ├── io/pvs.py                 # PVS loader (one logger + labels), L/R clock offset by yaw xcorr
│   ├── layer1/                   # gnss, orientation, iso2631, spectral, speed, events
│   ├── layer2/                   # distance (5 m grid), mipelt, arbitrate (-> HRNUs)
│   ├── layer3/                   # ledger, kst, hedging, lexicon, personas, plan_grammar, templates (T / T+V)
│   ├── eval/claims.py            # closed-vocabulary claim extraction and ledger verification
│   └── synth/                    # sampler, teacher, perturb (DPO), qa, formats, build (CLI)
├── data/processed/               # synthetic SFT / DPO / QA datasets + DATASET_CARD.md
├── tests/                        # pytest port of roadscribe_research/verification/ + synth tests
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
├── pyproject.toml
└── .gitignore
```

Not implemented yet from §8.2: OSM extraction and HMM map matching (`layer2/osm.py`, `mapmatch.py`),
phone loaders, the SLM backends and PGR loop, and the evaluation metrics beyond claim verification.

## Using the package

```bash
pip install -e ".[test]"
```

```python
from roadscribe import RoadScribeConfig
from roadscribe.io.pvs import load_pvs_trip
from roadscribe.layer1.orientation import to_vehicle_frame
from roadscribe.layer1.iso2631 import weight_vehicle_axes
from roadscribe.layer1.events import detect_events

cfg = RoadScribeConfig()                      # or RoadScribeConfig.from_yaml("exp.yaml")
raw = load_pvs_trip("/data/pvs", trip=1)      # left logger, gyro in rad/s
vft = to_vehicle_frame(raw, cfg.orientation)  # gated gravity + Procrustes yaw
aw = weight_vehicle_axes(vft.a_v, vft.fs, cfg.iso)
events = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2]).events
```

`tests/test_pipeline.py` runs the whole chain (vehicle frame → events → MI-PELT → HRNUs →
ledger → capsule) on a synthetic trip.

## Tests

```bash
pytest
```

Every Appendix A number that does not need raw data is asserted (ISO weights and IIR/FFT errors,
MI-PELT pruning, comfort cut-points, power and Williams design). Tests that need the PVS dataset are
skipped unless `ROADSCRIBE_PVS_DIR` points at the folder holding `PVS 1` … `PVS 9`:

```bash
ROADSCRIBE_PVS_DIR=/path/to/multi_class_kaggle pytest -m pvs
```

## Synthetic datasets

```bash
python -m roadscribe.synth.build --out data/processed
```

builds 5,000 capsule → plan → narrative examples, 1,500 DPO pairs and 300 QA items, all verified
against their claim ledgers. Read `data/processed/DATASET_CARD.md` before training: the narratives
come from the template teacher, and two of the four personas are E2 test personas
(`--exclude-e2-test-personas` leaves them out).
