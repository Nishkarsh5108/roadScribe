"""Evaluate P5 (narrator trained on PLAN and REALIZE as designed) on PVS.

Uses the ledgers the Kaggle kernel used (pvs_ledgers.pkl). Acceptance rule as before: the plan is valid,
the realized narrative passes the verifier, and it makes at least one claim.
Pre-registered (PREREG_P5.md): plan validity >= 50%, acceptance >= 30%, raw claim precision >= 0.90,
wrong-position share of event claims < 40%.
Exploratory: cross-vehicle disagreement of road-surface claims in accepted narratives, original vs calibrated.

Usage: python scripts/eval_pgr_v2.py <pgr_v2_results.json> <pvs_ledgers.pkl>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "roadscribe_research" / "identifiability"))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer3.ledger import LedgerView  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402


def wilson(k, n):
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def main(res_path, led_path):
    results = json.loads(Path(res_path).read_text(encoding="utf-8"))
    ledgers = pickle.load(open(led_path, "rb"))
    rows = []
    for r in results:
        d = ledgers[r["key"]]
        row = dict(key=r["key"], condition=r["condition"], trip=r["trip_id"], persona=r["persona"], seed=r["seed"],
                   plan_parses=r["plan_parses"], plan_valid=r["frames"] is not None, accepted=False,
                   n_claims=0, n_supported=0, reasons=Counter(), event_claims=0, wrong_position=0, missing=0)
        if r.get("narrative"):
            rep = verify_text(r["narrative"], d["ledger"], must_mention_events=d["must"])
            row.update(n_claims=rep.n_claims, n_supported=rep.n_supported, missing=rep.hallucinated_missing,
                       accepted=bool(rep.passed and rep.n_claims > 0), words=len(r["narrative"].split()))
            for c in rep.claims:
                row["event_claims"] += int(c.kind == "EVT")
                if not c.supported:
                    row["reasons"][c.reason or "unspecified"] += 1
                    row["wrong_position"] += int(c.reason == "wrong_position")
        rows.append(row)
    summary = {}
    for cond in ("orig", "cal"):
        R = [r for r in rows if r["condition"] == cond]
        n = len(R)
        acc = sum(r["accepted"] for r in R)
        realized = [r for r in R if r["n_claims"] > 0]
        cl = sum(r["n_claims"] for r in realized)
        sp = sum(r["n_supported"] for r in realized)
        ev = sum(r["event_claims"] for r in realized)
        wp = sum(r["wrong_position"] for r in realized)
        reasons = Counter()
        for r in realized:
            reasons.update(r["reasons"])
        # template reference on the same prompts
        tcl = tsp = 0
        for key in {r["key"] for r in R}:
            d = ledgers[key]
            persona = key.split("|")[2]
            sel = {u: "FULL" for u in d["visible"]}
            plan = rule_plan(d["ledger"], sel, persona, rng=None, must_mention_events=d["must"])
            rep = verify_text(realize(ground(plan, d["ledger"]), persona, vary=False), d["ledger"], must_mention_events=d["must"])
            tcl += rep.n_claims
            tsp += rep.n_supported
        summary[cond] = dict(
            n=n, plan_parses=sum(r["plan_parses"] for r in R), plan_valid=sum(r["plan_valid"] for r in R),
            plan_valid_rate=sum(r["plan_valid"] for r in R) / n, accepted=acc, acceptance_rate=acc / n,
            acceptance_ci95=wilson(acc, n), realized=len(realized),
            raw_precision=sp / cl if cl else float("nan"), claims_per_realized=cl / max(1, len(realized)),
            words_per_realized=float(np.mean([r["words"] for r in realized])) if realized else float("nan"),
            wrong_position_share_of_event_claims=wp / ev if ev else float("nan"),
            hallucinated_missing=sum(r["missing"] for r in R), unsupported_by_reason=dict(reasons.most_common()),
            template_claims_per_narrative=tcl / max(1, len({r['key'] for r in R})), template_precision=tsp / max(1, tcl),
        )
    s = summary["orig"]
    summary["verdicts_orig"] = dict(P5_H1=s["plan_valid_rate"] >= 0.5, P5_H2=s["acceptance_rate"] >= 0.3,
                                    P5_H3=s["raw_precision"] >= 0.9, P5_H4=s["wrong_position_share_of_event_claims"] < 0.4)
    out = dict(summary=summary, per_generation=[{k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()} for r in rows])
    dst = Path(res_path).with_name("pgr_v2_eval.json")
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
