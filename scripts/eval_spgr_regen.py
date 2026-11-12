"""Evaluate the regenerated S-PGR narratives (Kaggle run of 2026-11-12) against the template narrator.

* Acceptance rule unchanged from run_full_evaluation.py: the plan parses as a JSON object, the
  narrative passes the verifier, and it makes at least one claim; otherwise S-PGR falls back to T.
* Original capsules are verified against the original ledger; calibrated capsules against the
  ledger built with the P4 calibration (roadscribe_research/identifiability/results/p4_results.json).
* H1 (claim precision of S-PGR non-inferior to T within 2 points) is tested with the analysis the
  paper specifies: a cluster bootstrap over the 9 drives (10,000 resamples).

Usage: python scripts/eval_spgr_regen.py data/models/kaggle_regen/spgr_regen.json
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig, SpeedConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.io.pvs import load_pvs_trip, pvs_root, route_of, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.layer3.plan_grammar import validate_plan  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402

PERSONAS = ("commuter", "sports_car")


def ledger_for(trip: int, cfg: RoadScribeConfig):
    raw = load_pvs_trip(pvs_root(), trip, side="left", with_labels=False)
    tok = RegexTokenizer()
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
    return ledger, kst.capsule, must


def judge(gen: str, ledger, must, visible=None):
    has_plan = "<plan>" in gen and "</plan>" in gen
    plan_ok = False
    plan_errors = None
    if has_plan:
        try:
            p_json = json.loads(gen.split("<plan>")[1].split("</plan>")[0].strip())
            plan_ok = isinstance(p_json, dict)
            if plan_ok:
                plan_errors = len(validate_plan(p_json, ledger, visible))
        except Exception:
            plan_ok = False
    cand = gen.split("</plan>")[1].strip() if has_plan else gen.strip()
    rep = verify_text(cand, ledger, must_mention_events=must)
    return dict(plan_parses=plan_ok, plan_valid=plan_errors == 0, has_plan=has_plan, accepted=bool(plan_ok and rep.passed and len(rep.claims) > 0),
                n_claims=rep.n_claims, n_supported=rep.n_supported, words=len(cand.split()),
                missing=rep.hallucinated_missing, passed=bool(rep.passed))


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def main(path: str):
    gens = json.loads(Path(path).read_text(encoding="utf-8"))
    p4 = json.loads((repo_root / "roadscribe_research/identifiability/results/p4_results.json").read_text(encoding="utf-8"))
    base = RoadScribeConfig()
    rows = []
    for trip in range(1, 10):
        r = route_of(trip)
        cal = p4["calibration"][r]
        cfgs = {"orig": base, "cal": dataclasses.replace(base, speed=SpeedConfig(
            gamma=cal["gamma"], vehicle_gain_log=cal["gains"][str(vehicle_of(trip))]))}
        for cond, cfg in cfgs.items():
            ledger, capsule, must = ledger_for(trip, cfg)
            for persona in PERSONAS:
                plan = rule_plan(ledger, capsule.selection, persona, rng=None, must_mention_events=must)
                rep_T = verify_text(realize(ground(plan, ledger), persona, vary=False), ledger, must_mention_events=must)
                for g in gens:
                    if g["condition"] == cond and g["trip_id"] == f"pvs{trip}" and g["persona"] == persona:
                        visible = [u for u, lv in capsule.selection.items() if lv != "OMIT"]
                        j = judge(g["generated_narrative"], ledger, must, visible)
                        S = (j["n_claims"], j["n_supported"]) if j["accepted"] else (rep_T.n_claims, rep_T.n_supported)
                        rows.append(dict(condition=cond, trip=trip, persona=persona, seed=g["seed"], hit_budget=g["hit_budget"],
                                         n_new_tokens=g["n_new_tokens"], **j, T_claims=rep_T.n_claims, T_supported=rep_T.n_supported,
                                         S_claims=S[0], S_supported=S[1], must=len(must)))
        print(f"trip {trip} evaluated", flush=True)
    out = {}
    rng = np.random.default_rng(20261112)
    for cond in ("orig", "cal"):
        R = [r for r in rows if r["condition"] == cond]
        n = len(R)
        acc = sum(r["accepted"] for r in R)
        slm = [r for r in R if r["accepted"]]
        by_trip = defaultdict(list)
        for r in R:
            by_trip[r["trip"]].append(r)

        def pl(rr, kind):
            c = sum(r[f"{kind}_claims"] for r in rr)
            s = sum(r[f"{kind}_supported"] for r in rr)
            return s / c if c else float("nan")

        diffs = []
        trips = sorted(by_trip)
        for _ in range(10000):
            pick = rng.choice(trips, size=len(trips), replace=True)
            rr = [r for t in pick for r in by_trip[t]]
            diffs.append(pl(rr, "S") - pl(rr, "T"))
        out[cond] = dict(
            n_generations=n, hit_budget=sum(r["hit_budget"] for r in R), plan_parses=sum(r["plan_parses"] for r in R),
            plan_valid_per_validator=sum(r["plan_valid"] for r in R),
            verifier_passed=sum(r["passed"] for r in R), accepted=acc, acceptance_rate=acc / n, acceptance_ci95=wilson(acc, n),
            slm_claims_per_accepted=float(np.mean([r["n_claims"] for r in slm])) if slm else float("nan"),
            slm_words_per_accepted=float(np.mean([r["words"] for r in slm])) if slm else float("nan"),
            template_claims_per_narrative=float(np.mean([r["T_claims"] for r in R])),
            P_L_slm_accepted=pl(slm, "S") if slm else float("nan"),
            P_L_spgr=pl(R, "S"), P_L_T=pl(R, "T"), diff_spgr_minus_T=pl(R, "S") - pl(R, "T"),
            diff_ci95_cluster_bootstrap_drives=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            H1_noninferior_2pt=bool(np.percentile(diffs, 2.5) > -0.02),
            hallucinated_missing=sum(r["missing"] for r in R),
            must_mention_nonempty_trips=sorted({r["trip"] for r in R if r["must"] > 0}),
        )
    res = dict(per_generation=rows, summary=out)
    dst = Path(path).with_name("spgr_regen_eval.json")
    dst.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
