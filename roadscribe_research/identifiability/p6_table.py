"""Print the paper's P6 table rows and key sentences from results/p6_results.json (no hand transcription).
Run after scripts/eval_p6.py:  python roadscribe_research/identifiability/p6_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "p6_results.json").read_text(encoding="utf-8"))
NAMES = {"A": "A: capsule", "B": "B: + frames", "C": "C: + frames, no positions",
         "zs-3B": "Prompted 3B, input B", "zs-7B": "Prompted 7B, input B"}


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}\\%"


def err(s, t):
    v = s["by_value_type"].get(t)
    return "--" if not v or v["claims"] == 0 else f"{100 * v['error_rate']:.0f}\\% ({v['claims']})"


for k in ("A", "B", "C", "zs-3B", "zs-7B"):
    if k not in R["arms"]:
        continue
    s = R["arms"][k]
    print(f"{NAMES[k]} & {pct(s['precision'])} & {s['accepted']} & {s['claims_per_narrative']:.1f} & "
          f"{pct(s['wrong_position_share'], 0)} & {s['non_position_per_narrative']:.1f} & "
          f"{err(s, 'position')} & {err(s, 'speed')} & {err(s, 'acceleration')} & {err(s, 'distance')} \\\\")
print()
for h, v in R["hypotheses"].items():
    print(h, json.dumps(v))
