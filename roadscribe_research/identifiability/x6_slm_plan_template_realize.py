"""EXPLORATORY (not pre-registered): the P5 model's valid plans, realized by the deterministic template realizer.

The model chooses content (its plan); code writes every value. Reports acceptance under the same strict
verifier, claim precision, and the fact overlap (Jaccard) between the model's plan and the rule planner's.
Usage: python roadscribe_research/identifiability/x6_slm_plan_template_realize.py <pgr_v2_results.json> <pvs_ledgers.pkl>
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.layer3.templates import realize, rule_plan  # noqa: E402


def main(res_path, led_path):
    R = json.loads(Path(res_path).read_text(encoding="utf-8"))
    L = pickle.load(open(led_path, "rb"))
    out = {"label": "EXPLORATORY, not pre-registered"}
    for cond in ("orig", "cal"):
        Rc = [r for r in R if r["condition"] == cond]
        valid = [r for r in Rc if r.get("frames")]
        acc = cl = sp = 0
        jac = []
        for r in valid:
            d = L[r["key"]]
            rep = verify_text(realize(r["frames"], r["persona"], vary=False), d["ledger"], must_mention_events=d["must"])
            acc += bool(rep.passed and rep.n_claims > 0)
            cl += rep.n_claims
            sp += rep.n_supported
            rp = rule_plan(d["ledger"], {u: "FULL" for u in d["visible"]}, r["persona"], rng=None, must_mention_events=d["must"])
            f_rule = {f for it in rp["items"] for f in it.get("facts", [])}
            t = r["plan_text"]
            plan = json.loads(t[t.find("{"):t.rfind("}") + 1])
            f_mod = {f for it in plan["items"] for f in it.get("facts", [])}
            jac.append(len(f_rule & f_mod) / max(1, len(f_rule | f_mod)))
        out[cond] = dict(n=len(Rc), valid_plans=len(valid), accepted=acc, acceptance_of_all=acc / len(Rc),
                         acceptance_of_valid=acc / max(1, len(valid)), precision=sp / max(1, cl), claims_per=cl / max(1, len(valid)),
                         mean_fact_jaccard_vs_rule_plan=sum(jac) / max(1, len(jac)), identical_fact_sets=sum(j == 1.0 for j in jac))
    (HERE / "results" / "x6_slm_plan_template_realize.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
