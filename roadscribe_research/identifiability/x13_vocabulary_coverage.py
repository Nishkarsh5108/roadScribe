"""EXPLORATORY (added 2026-11-13): what the closed-vocabulary verifier does not see.

The verifier extracts claims phrased in its lexicon; anything else passes unchecked. Every digit is either
parsed or counted as an unparsed (failing) number, so numbers are covered by construction. This script
measures the remaining blind spot: descriptive road, ride and traffic words outside the lexicon that are not
inside any extracted claim span. The word list was fixed before running and is printed with the result.

Usage: python roadscribe_research/identifiability/x13_vocabulary_coverage.py <pvs_ledgers.pkl>
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402
from x9_sentence_filter import filter_text, sources  # noqa: E402

DESCRIPTORS = ["smooth", "smoother", "smoothest", "bumpier", "bumpiest", "rougher", "roughest", "potholed",
               "cracked", "cracks", "uneven", "gravel", "paved", "harsh", "jarring", "shaky", "patchy", "broken",
               "worn", "damaged", "speed bump", "speed hump", "rutted", "ruts", "corrugated", "washboard", "loose",
               "slippery", "wet", "traffic", "congested", "busy", "steep", "hill", "hilly", "curvy", "winding",
               "junction", "roundabout", "intersection", "roadworks", "construction", "dangerous", "safe", "unsafe",
               "pleasant", "bumps", "jolts", "vibration", "vibrations", "jarring", "stable", "steady", "quiet"]
RX = re.compile(r"\b(" + "|".join(sorted({re.escape(d) for d in DESCRIPTORS}, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def uncovered(text, rep):
    spans = [c.span for c in rep.claims]
    return [m.group(1).lower() for m in RX.finditer(text) if not any(a <= m.start() < b for a, b in spans)]


def main(led_path):
    L = pickle.load(open(led_path, "rb"))
    out = {"label": "EXPLORATORY, not pre-registered", "descriptors": DESCRIPTORS}
    srcs = sources()
    srcs["T, template (sanity)"] = []
    for k, d in L.items():
        if k.startswith("orig|"):
            persona = k.split("|")[2]
            plan = rule_plan(d["ledger"], {u: "FULL" for u in d["visible"]}, persona, rng=None, must_mention_events=d["must"])
            srcs["T, template (sanity)"].append(dict(key=k, trip=k.split("|")[1], text=realize(ground(plan, d["ledger"]), persona, vary=True, seed=1)))
    for name, rows in srcs.items():
        words, words_f, n_with, n_with_f = Counter(), Counter(), 0, 0
        for r in rows:
            d = L[r["key"]]
            rep = verify_text(r["text"], d["ledger"], must_mention_events=d["must"])
            u = uncovered(r["text"], rep)
            words.update(u)
            n_with += bool(u)
            ft, frep = filter_text(r["text"], d["ledger"], d["must"])
            uf = uncovered(ft, frep)
            words_f.update(uf)
            n_with_f += bool(uf)
        n = max(len(rows), 1)
        out[name] = dict(narratives=len(rows), uncovered_per_narrative=sum(words.values()) / n,
                         narratives_with_uncovered=n_with,
                         after_sentence_filter_per_narrative=sum(words_f.values()) / n,
                         after_sentence_filter_narratives_with_uncovered=n_with_f,
                         top=words.most_common(8))
        print(name, json.dumps(out[name]), flush=True)
    (HERE / "results" / "x13_vocabulary_coverage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
