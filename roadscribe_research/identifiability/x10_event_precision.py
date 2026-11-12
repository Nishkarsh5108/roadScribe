"""EXPLORATORY (added 2026-11-13 after the simulated review): precision of RoadScribe's event detector on PVS.

PVS labels speed bumps only, so precision against the labelled bumps is a lower bound: a detected jolt at an
unlabelled pothole, kerb or braking manoeuvre counts as a false positive here. Recall is reported alongside.
A detected event matches a labelled bump if their along-track positions differ by at most 50 m (as in the paper).
Run from the repo root:  python roadscribe_research/identifiability/x10_event_precision.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from run_full_evaluation import build_trip  # noqa: E402
from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.layer3.hedging import MENTION_THRESHOLD  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer  # noqa: E402

TOL = 50.0
BANDS = ((MENTION_THRESHOLD, 0.6), (0.6, 0.85), (0.85, 1.01))


def main():
    cfg, tok = RoadScribeConfig(), RegexTokenizer()
    rows, gt_total, gt_hit = [], 0, 0
    for trip in range(1, 10):
        tr = build_trip(trip, cfg, tok)
        gts = np.array([g.s for g in tr["gt_refs"]])
        evs = [f for f in tr["ledger"] if f.kind == "EVT"]
        for f in evs:
            s = float(f.value.get("s", np.nan))
            d = float(np.min(np.abs(gts - s))) if len(gts) else np.inf
            rows.append(dict(trip=trip, type=f.value.get("type"), conf=float(f.conf), label=f.value.get("label"),
                             match=bool(d <= TOL)))
        mentionable = np.array([float(f.value.get("s", np.nan)) for f in evs if f.conf >= MENTION_THRESHOLD and f.value.get("label") != "artifact"])
        gt_total += len(gts)
        gt_hit += int(sum(1 for g in gts if len(mentionable) and np.min(np.abs(mentionable - g)) <= TOL))
        print(f"trip {trip}: {len(gts)} bumps, {len(evs)} events", flush=True)
    R = [r for r in rows if r["conf"] >= MENTION_THRESHOLD and r["label"] != "artifact"]
    out = {"label": "EXPLORATORY, not pre-registered", "tolerance_m": TOL,
           "events_total": len(rows), "events_mentionable": len(R),
           "precision_mentionable_vs_bumps": float(np.mean([r["match"] for r in R])) if R else None,
           "recall_bumps_by_mentionable": gt_hit / gt_total, "bumps": gt_total, "bumps_matched": gt_hit,
           "by_confidence": {f"[{lo:.2f}, {min(hi, 1):.2f}]": dict(n=len(b), precision=float(np.mean([r["match"] for r in b])) if b else None)
                             for lo, hi in BANDS for b in [[r for r in R if lo <= r["conf"] < hi]]},
           "by_type": {t: dict(n=len(b), precision=float(np.mean([r["match"] for r in b])))
                       for t in sorted({r["type"] for r in R}) for b in [[r for r in R if r["type"] == t]]}}
    (HERE / "results" / "x10_event_precision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
