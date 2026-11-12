"""EXPLORATORY (not pre-registered): do the language model's accepted narratives contradict each other
about the same road across vehicles, with the original versus the calibrated ledgers?

Same procedure as x2/p4: claims are extracted with the verifier against the ledger the narrative was
generated from, mapped to the shared route coordinate, and compared per 100 m stretch across vehicles,
for the same persona and seed. Only narratives that passed the S-PGR acceptance rule are used.
Usage: python roadscribe_research/identifiability/x4_slm_contradictions.py data/models/kaggle_regen/spgr_regen.json
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

import x2_narrative_contradictions as x2  # noqa: E402
from eval_spgr_regen import judge  # noqa: E402
from roadscribe.config import RoadScribeConfig, SpeedConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402


def main(path):
    gens = json.loads(Path(path).read_text(encoding="utf-8"))
    p4 = json.loads((HERE / "results" / "p4_results.json").read_text(encoding="utf-8"))
    base, tok = RoadScribeConfig(), RegexTokenizer()
    out = {"label": "EXPLORATORY, not pre-registered"}
    for cond in ("orig", "cal"):
        tot = dict(pairs=0, differ=0, accepted_narratives=0, road_claims=0)
        for route, trips in ROUTES.items():
            raws = {t: load_pvs_trip(pvs_root(), t, side="left", with_labels=False) for t in trips}
            g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
            lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
            lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
            ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
            n_bins = int(ref_s[-1] // x2.BIN) + 1
            grids = {}
            for t in trips:
                cal = p4["calibration"][route]
                cfg = base if cond == "orig" else dataclasses.replace(
                    base, speed=SpeedConfig(gamma=cal["gamma"], vehicle_gain_log=cal["gains"][str(vehicle_of(t))]))
                raw = raws[t]
                th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
                theta, _ = sample_theta(raw.t, g[t].t.to_numpy(float), th_fix, d_fix)
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
                must = [f.key for f in L.facts if f.kind == "EVT" and f.conf >= 0.85]
                visible = [u for u, lv in kst.capsule.selection.items() if lv != "OMIT"]
                s_sorted = np.maximum.accumulate(vft.s)
                for gen in gens:
                    if gen["condition"] != cond or gen["trip_id"] != f"pvs{t}":
                        continue
                    if not judge(gen["generated_narrative"], ledger, must, visible)["accepted"]:
                        continue
                    text = gen["generated_narrative"].split("</plan>")[1].strip()
                    rep = verify_text(text, ledger, must_mention_events=must)
                    cl = []
                    for c in rep.claims:
                        if c.kind == "SEV" and c.key == "road_class" and c.unit_id:
                            th0, th1 = np.interp(list(L.extent(c.unit_id)), s_sorted, theta)
                            cl.append(dict(kind="road_class", value=int(c.value), th0=float(th0), th1=float(th1)))
                    tot["accepted_narratives"] += 1
                    tot["road_claims"] += len(cl)
                    grids[(t, gen["persona"], gen["seed"])] = x2.per_stretch(cl, "road_class", n_bins)
            for persona in x2.PERSONAS:
                for seed in (0, 1, 2):
                    for a, b in combinations(trips, 2):
                        ga, gb = grids.get((a, persona, seed)), grids.get((b, persona, seed))
                        if ga is None or gb is None:
                            continue
                        for xa, xb in zip(ga, gb):
                            if xa is not None and xb is not None:
                                tot["pairs"] += 1
                                tot["differ"] += int(xa != xb)
            print(cond, route, tot, flush=True)
        tot["differ_rate"] = tot["differ"] / max(1, tot["pairs"])
        out[cond] = tot
    (HERE / "results" / "x4_slm_contradictions.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
