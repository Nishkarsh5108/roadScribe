"""Build a hand-annotation sheet for the verifier's extraction recall (the one review point not closed overnight).

Eight model narratives (two each from the first adapter, the P5 realizer, P6 arm B and the prompted 7B model;
fixed seed) are split into sentences; each sentence lists the claims the verifier extracted. A human annotator
marks every factual statement about the trip that the list misses. Recall = extracted / (extracted + missed).
Usage: python roadscribe_research/identifiability/x15_recall_sheet.py <pvs_ledgers.pkl>
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from roadscribe.eval.claims import _sentences, verify_text  # noqa: E402
from x9_sentence_filter import sources  # noqa: E402


def main(led_path):
    L = pickle.load(open(led_path, "rb"))
    src = sources()
    pick = {"S-PGR (1,024 tokens)": 2, "P5 realizer": 2, "P6 arm B": 2, "prompted Qwen2.5-7B-Instruct": 2}
    rng = random.Random(20261113)
    lines = ["# Verifier recall: hand-annotation sheet", "",
             "For each sentence, the verifier's extracted claims are listed. In the `missed` line, write every factual",
             "statement about the trip (a value, place, event, severity, comparison or cause) that no listed claim covers.",
             "Recall = extracted / (extracted + missed). Do not judge truth here, only coverage.", ""]
    for name, k in pick.items():
        for r in rng.sample(src[name], k):
            d = L[r["key"]]
            rep = verify_text(r["text"], d["ledger"], must_mention_events=d["must"])
            lines += [f"## {name}: {r['key']}", ""]
            for a, b in _sentences(r["text"]):
                cl = [c for c in rep.claims if a <= c.span[0] < b]
                lines.append(f"- **S:** {r['text'][a:b].strip()}")
                for c in cl:
                    lines.append(f"  - extracted: {c.kind}/{c.key} = `{r['text'][c.span[0]:c.span[1]]}` ({'supported' if c.supported else c.reason})")
                lines.append("  - missed: ")
            lines.append("")
    out = HERE / "annotation" / "recall_sheet.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, sum(1 for l in lines if l.startswith("- **S:**")), "sentences")


if __name__ == "__main__":
    main(sys.argv[1])
