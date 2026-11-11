"""Independent re-check of the PVS numbers in scripts/run_full_evaluation.py (2026-11-11).

Runs the repository pipeline exactly as run_full_evaluation.py does; only the
scoring changes. Prints:

1. The original pooled anomaly-grounding number for the template narrator T
   (reproduces the reported AGR_G = 1.000).
2. Per-trip AGR_G with each trip's ground-truth bumps counted once.
3. A chance control: the same mention intervals dropped uniformly at random
   along the trip. If chance ~ observed, the metric measures interval width.
4. Recall using only mentions that state a position.
5. Detector level: labelled bumps with a ledger event within the metric's
   tolerance.
6. The S-PGR generations that passed the verifier (not fallback), with their
   full text and claim counts, and the size of the must-mention set per trip.

Usage (from the repo root, with ROADSCRIBE_PVS_DIR pointing at the PVS folders):
    python roadscribe_research/verification/recheck_eval_2026_11_11.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
os.environ.setdefault(
    "ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
)

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.eval.metrics import Mention, RefEvent, anomaly_grounding  # noqa: E402
from roadscribe.io.pvs import bump_mask, load_pvs_trip, pvs_root  # noqa: E402
from roadscribe.layer1.events import detect_events, find_runs  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402

PERSONAS = ("commuter", "sports_car")
TOL = 25.0  # same delta_s and +-25 m mention half-width as run_full_evaluation.py
rng = np.random.default_rng(0)


def mentions_of(rep, L, end):
    """Event mentions exactly as run_full_evaluation.py builds them, tagged by how they were located."""
    out = []
    for c in rep.claims:
        if c.kind != "EVT":
            continue
        pos = c.value.get("pos") if isinstance(c.value, dict) else None
        if pos is not None:
            out.append((Mention(c.key, max(0.0, pos - TOL), pos + TOL), "point"))
        elif c.unit_id:
            s0, s1 = L.extent(c.unit_id)
            out.append((Mention(c.key, s0, s1), "unit"))
        else:
            out.append((Mention(c.key, 0.0, end), "trip"))
    return out


def coverage(ms, end):
    if not ms:
        return 0.0
    iv = sorted((max(0.0, m.s_lo), min(end, m.s_hi)) for m in ms)
    tot, lo0, hi0 = 0.0, iv[0][0], iv[0][1]
    for lo, hi in iv[1:]:
        if lo > hi0:
            tot += hi0 - lo0
            lo0, hi0 = lo, hi
        else:
            hi0 = max(hi0, hi)
    return (tot + hi0 - lo0) / end


def chance_agr(ms, refs, end, draws=200):
    vals = []
    for _ in range(draws):
        rnd = []
        for m in ms:
            w = m.s_hi - m.s_lo
            lo = rng.uniform(0.0, max(1e-6, end - w))
            rnd.append(Mention(m.type, lo, lo + w))
        vals.append(anomaly_grounding(rnd, refs, delta_s=TOL).agr)
    return float(np.mean(vals))


def main():
    cfg = RoadScribeConfig()
    tok = RegexTokenizer()
    preds = json.load(open(repo_root / "data/models/kaggle_run/pvs_predictions.json", encoding="utf-8"))
    pred_map = {(p["trip_id"], p["persona"]): p["generated_narrative"] for p in preds}

    pooled_m, pooled_r = [], []
    tot = {p: Counter() for p in PERSONAS}
    accepted = []
    must_sizes = {}
    print("trip persona     GT  mentions  cover  AGR_G  chance  point-only-hits  det-hits")
    for trip in range(1, 10):
        raw = load_pvs_trip(pvs_root(), trip, side="left", with_labels=True)
        vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
        aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
        det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
        grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
        res, _ = segment(grid.Y, grid.s, [], [], cfg.mipelt)
        end = float(vft.s[-1])
        bounds = arbitrate(res.changepoints, grid.s, grid.Y, [], cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
        units = build_hrnus(bounds, vft, aw, det.events, [], grid_s=grid.s, grid_Y=grid.Y,
                            iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
        meta = TripMeta(raw.trip_id, duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=0.0)
        ledger = compile_ledger(units, det.events, meta, cfg.ledger)
        kst = build_capsule(units, det.events, meta, 128, tok, cfg.kst, ledger=ledger)
        L = LedgerView(ledger)
        gt = [RefEvent("bump", float(np.mean(vft.s[a:b])), 1.0, f"GT_{k}")
              for k, (a, b) in enumerate(find_runs(bump_mask(raw.labels)))]
        led = [f for f in L.facts if f.kind == "EVT"]
        must = [f.key for f in led if f.conf >= 0.85]
        must_sizes[trip] = len(must)
        det_m = [Mention("bump", max(0.0, float(f.value.get("s", 0.0)) - TOL), float(f.value.get("s", 0.0)) + TOL)
                 for f in led]
        det_hits = len(anomaly_grounding(det_m, gt, delta_s=TOL).matches) if (det_m and gt) else 0

        for persona in PERSONAS:
            plan_T = rule_plan(ledger, kst.capsule.selection, persona, rng=None, must_mention_events=must)
            narr_T = realize(ground(plan_T, ledger), persona, vary=False)
            rep_T = verify_text(narr_T, ledger, must_mention_events=must)
            ms = mentions_of(rep_T, L, end)
            M = [m for m, _ in ms]
            pts = [m for m, k in ms if k == "point"]
            ag = anomaly_grounding(M, gt, delta_s=TOL) if (M and gt) else None
            pt_hits = len(anomaly_grounding(pts, gt, delta_s=TOL).matches) if (pts and gt) else 0
            ch = chance_agr(M, gt, end) if (M and gt) else float("nan")
            pooled_m.extend(M)
            pooled_r.extend(gt)  # once per persona, as the original script does
            hits = 0 if ag is None else len(ag.matches)
            tot[persona].update(gt=len(gt), hits=hits, chance=ch * len(gt), pt=pt_hits, det=det_hits)
            print(f"{trip:>4} {persona:<10} {len(gt):>3} {len(M):>9} {coverage(M, end):6.2f} "
                  f"{(hits / len(gt)) if gt else float('nan'):6.2f} {ch:7.2f} {pt_hits:>16} {det_hits:>9}")

            # S-PGR exactly as run_full_evaluation.py decides acceptance
            raw_gen = pred_map.get((f"pvs{trip}", persona), "")
            has_plan = "<plan>" in raw_gen and "</plan>" in raw_gen
            plan_ok = False
            if has_plan:
                try:
                    plan_ok = isinstance(json.loads(raw_gen.split("<plan>")[1].split("</plan>")[0].strip()), dict)
                except Exception:
                    plan_ok = False
            cand = raw_gen.split("</plan>")[1].strip() if has_plan else raw_gen.strip()
            rep_c = verify_text(cand, ledger, must_mention_events=must)
            if plan_ok and rep_c.passed and len(rep_c.claims) > 0:
                accepted.append((trip, persona, rep_c.n_claims, len(cand.split()), rep_T.n_claims, cand))

    pooled = anomaly_grounding(pooled_m, pooled_r, delta_s=TOL)
    print(f"\nOriginal pooled scoring, T: AGR_G={pooled.agr:.3f} AGP_G={pooled.agp:.3f} "
          f"(mentions={len(pooled_m)}, refs={len(pooled_r)})")
    for p in PERSONAS:
        t = tot[p]
        print(f"T/{p}: per-trip AGR_G={t['hits'] / t['gt']:.3f}  chance={t['chance'] / t['gt']:.3f}  "
              f"point-only={t['pt']}/{t['gt']}  detector={t['det']}/{t['gt']}")
    print("\nmust-mention events per trip:", must_sizes)
    print(f"\nS-PGR generations accepted by the verifier: {len(accepted)}/18; "
          f"SLM claims in them: {sum(a[2] for a in accepted)}")
    for trip, persona, n_claims, n_words, n_T, text in accepted:
        print(f"  pvs{trip}/{persona}: {n_claims} claims, {n_words} words (template: {n_T} claims) -> {text!r}")


if __name__ == "__main__":
    main()
