"""EXPLORATORY (added 2026-11-13): error rate per value type of the models trained before P6, to set against
the input-level audit (x8): positions, speeds and accelerations were absent from the first adapter's input,
distances and durations present. Uses the value-type mapping of scripts/eval_p6.py.
Usage: python roadscribe_research/identifiability/x14_per_type_errors.py <pvs_ledgers.pkl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
import pickle  # noqa: E402

from eval_p6 import by_type, score  # noqa: E402
from x9_sentence_filter import sources  # noqa: E402


def main(led_path):
    L = pickle.load(open(led_path, "rb"))
    out = {"label": "EXPLORATORY, not pre-registered"}
    for name, rows in sources().items():
        if not (name.startswith("S-PGR") or name.startswith("P5")):
            continue
        per = score([dict(key=r["key"], trip_id=r["trip"], persona=r["key"].split("|")[2], seed=0, narrative=r["text"])
                     for r in rows], L)
        out[name] = by_type(per)
        print(name, json.dumps({k: (v["claims"], round(v["error_rate"], 3)) for k, v in out[name].items()}))
    (HERE / "results" / "x14_per_type_errors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
