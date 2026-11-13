"""P9 (exploratory; declared in WORKLOG before running): the first adapter on the rebuilt PVS capsules.
Scores the narrative after </plan> exactly as the regeneration (scripts/eval_spgr_regen.py's acceptance rule:
plan parses, every claim supported, at least one claim), next to the regeneration on the old capsules.
Usage: python roadscribe_research/identifiability/p9_eval.py [ledgers.pkl]
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from eval_p6 import LEDGERS, by_type, score  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402

led = Path(sys.argv[1]) if len(sys.argv) > 1 else LEDGERS
L = pickle.load(open(led, "rb"))


def rows_of(raw, key_of):
    out = []
    for r in raw:
        g = r["generated_narrative"]
        plan_ok = False
        if "<plan>" in g and "</plan>" in g:
            try:
                plan_ok = isinstance(json.loads(g.split("<plan>")[1].split("</plan>")[0].strip()), dict)
            except Exception:
                plan_ok = False
        out.append(dict(key=key_of(r), trip_id=r["trip_id"], persona=r["persona"], seed=r["seed"],
                        narrative=g.split("</plan>")[-1].strip(), plan_ok=plan_ok))
    return out


def summarize(rows):
    per = score(rows, L)
    reasons = Counter()
    accepted = 0
    for r, p in zip(rows, per):
        reasons.update(p["reasons"])
        accepted += bool(r["plan_ok"] and p["accepted"])
    n_cl, n_sup = sum(p["n_claims"] for p in per), sum(p["n_sup"] for p in per)
    return dict(n=len(rows), accepted=accepted, precision=n_sup / n_cl, claims_per_narrative=n_cl / len(rows),
                wrong_anchor=reasons.get("wrong_anchor", 0), wrong_map_tag=reasons.get("wrong_map_tag", 0),
                hallucinated_missing=reasons.get("hallucinated_missing", 0), by_value_type=by_type(per),
                reasons=dict(reasons.most_common()))


new = json.loads((REPO / "data/models/p9/p9_first_adapter_rebuilt_caps.json").read_text(encoding="utf-8"))
old = [r for r in json.loads((REPO / "data/models/kaggle_regen/spgr_regen.json").read_text(encoding="utf-8")) if r["condition"] == "orig"]
out = {"label": "EXPLORATORY (declared in WORKLOG before running)",
       "first adapter, old capsules (regen)": summarize(rows_of(old, lambda r: f"orig|{r['trip_id']}|{r['persona']}")),
       "first adapter, rebuilt capsules (P9)": summarize(rows_of(new, lambda r: r["key"]))}
(HERE / "results" / "p9_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
for k, v in out.items():
    if isinstance(v, dict):
        bt = v["by_value_type"]
        print(f"{k}: n={v['n']} accepted={v['accepted']} P_L={v['precision']:.3f} claims={v['claims_per_narrative']:.1f} "
              f"anchors={v['wrong_anchor']} map_tags={v['wrong_map_tag']} missing={v['hallucinated_missing']} "
              f"pos_err={bt.get('position', {}).get('error_rate', float('nan')):.2f}")
