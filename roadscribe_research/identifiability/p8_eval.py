"""P8 (exploratory; predictions in WORKLOG 03:46): arm B with greedy decoding and with a second epoch.
Scored with scripts/eval_p6.py's definitions, next to arm B (P6) for reference.
Usage: python roadscribe_research/identifiability/p8_eval.py [ledgers.pkl]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_p6 import LEDGERS, boot, prec, score, stats  # noqa: E402

led = Path(sys.argv[1]) if len(sys.argv) > 1 else LEDGERS
L = pickle.load(open(led, "rb"))
runs = {"B (P6)": json.loads((REPO / "data/models/p6/B/p6_B.json").read_text(encoding="utf-8"))}
for m in ("greedy", "epoch2"):
    f = REPO / f"data/models/p8/{m}/p8_{m}.json"
    if f.exists():
        runs[f"B {m}"] = json.loads(f.read_text(encoding="utf-8"))
per = {k: score(v, L) for k, v in runs.items()}
out = {"label": "EXPLORATORY (declared in WORKLOG before running)", "arms": {k: stats(v) for k, v in per.items()}}
for k in per:
    if k != "B (P6)":
        out[f"precision {k} - B (P6)"] = boot(per[k], per["B (P6)"], prec)
(HERE / "results" / "p8_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
for k, s in out["arms"].items():
    bt = s["by_value_type"]
    print(f"{k:10s} n={s['n']} P_L={s['precision']:.3f} accepted={s['accepted']} claims={s['claims_per_narrative']:.1f} "
          f"pos_err={bt.get('position', {}).get('error_rate', float('nan')):.2f} acc_err={bt.get('acceleration', {}).get('error_rate', float('nan')):.2f}")
for k, v in out.items():
    if k.startswith("precision"):
        print(k, round(v["diff"], 3), [round(x, 3) for x in v["ci95"]])
