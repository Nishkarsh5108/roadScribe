"""EXPLORATORY (added 2026-11-13 after the simulated review): sentence-level fallback for model narratives.

Instead of rejecting a whole narrative, drop every sentence that holds an unsupported claim, re-verify the
remainder, and repeat (at most three rounds, because dropping a sentence can change what a following
"It ..." refers to). Reports how much of the model's text survives and whether the remainder passes the
strict verifier (every claim supported, must-mention events covered).

Sources: the S-PGR regeneration (1,024 tokens; text after </plan>), the P5 realizer, and the P6 arms and
prompted baselines when present. Original-ledger condition only.
Usage: python roadscribe_research/identifiability/x9_sentence_filter.py <pvs_ledgers.pkl>
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from roadscribe.eval.claims import _sentences, verify_text  # noqa: E402


def sources() -> dict:
    out = {}
    regen = json.loads((REPO / "data/models/kaggle_regen/spgr_regen.json").read_text(encoding="utf-8"))
    out["S-PGR (1,024 tokens)"] = [dict(key=f"orig|{r['trip_id']}|{r['persona']}", trip=r["trip_id"],
                                        text=r["generated_narrative"].split("</plan>")[-1].strip())
                                   for r in regen if r["condition"] == "orig"]
    p5 = json.loads((REPO / "data/models/kaggle_pgr_v2/pgr_v2_results.json").read_text(encoding="utf-8"))
    out["P5 realizer"] = [dict(key=r["key"], trip=r["trip_id"], text=(r.get("narrative") or "").strip())
                          for r in p5 if r["condition"] == "orig" and r.get("narrative")]
    p6 = REPO / "data/models/p6"
    for arm in ("A", "B", "C"):
        f = p6 / arm / f"p6_{arm}.json"
        if f.exists():
            out[f"P6 arm {arm}"] = [dict(key=r["key"], trip=r["trip_id"], text=r["narrative"].strip())
                                    for r in json.loads(f.read_text(encoding="utf-8"))]
    f = p6 / "zs" / "p6_zeroshot.json"
    if f.exists():
        for r in json.loads(f.read_text(encoding="utf-8")):
            out.setdefault("prompted " + r["model"].split("/")[-1], []).append(
                dict(key=r["key"], trip=r["trip_id"], text=r["narrative"].strip()))
    return out


def filter_text(text, ledger, must, rounds=3):
    cur = text
    for _ in range(rounds):
        rep = verify_text(cur, ledger, must_mention_events=must)
        bad = [c.span for c in rep.claims if not c.supported]
        if not bad:
            return cur, rep
        sents = _sentences(cur)
        keep = [(a, b) for a, b in sents if not any(a <= s0 < b or a < s1 <= b for s0, s1 in bad)]
        cur = " ".join(cur[a:b].strip() for a, b in keep)
    return cur, verify_text(cur, ledger, must_mention_events=must)


def main(led_path):
    L = pickle.load(open(led_path, "rb"))
    res = {"label": "EXPLORATORY, not pre-registered"}
    for name, rows in sources().items():
        stats = []
        for r in rows:
            d = L[r["key"]]
            rep0 = verify_text(r["text"], d["ledger"], must_mention_events=d["must"])
            n_sent0 = len(_sentences(r["text"]))
            out, rep1 = filter_text(r["text"], d["ledger"], d["must"])
            n_sent1 = len(_sentences(out)) if out.strip() else 0
            stats.append(dict(sent0=n_sent0, sent1=n_sent1, claims0=rep0.n_claims, sup0=rep0.n_supported,
                              claims1=rep1.n_claims, all_supported=rep1.n_supported == rep1.n_claims and rep1.n_claims > 0,
                              passed=bool(rep1.passed and rep1.n_claims > 0), passed0=bool(rep0.passed and rep0.n_claims > 0),
                              must_missed=len(rep1.must_mention_missed)))
        S = lambda k: float(np.sum([s[k] for s in stats]))  # noqa: E731
        res[name] = dict(narratives=len(stats), passed_whole=int(S("passed0")),
                         sentences_kept_share=S("sent1") / max(S("sent0"), 1),
                         supported_claims_kept_per_narrative=S("claims1") / len(stats),
                         supported_claims_before_per_narrative=S("sup0") / len(stats),
                         remainder_all_claims_supported=int(S("all_supported")),
                         remainder_passes_strict=int(S("passed")),
                         remainder_misses_must_mention=int(sum(1 for s in stats if s["must_missed"] > 0)))
        print(name, json.dumps(res[name]), flush=True)
    (HERE / "results" / "x9_sentence_filter.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
