"""EXPLORATORY: why do the regenerated SLM narratives fail verification?

For every regenerated narrative (original capsules), counts unsupported claims by kind and by the
verifier's reason, and checks place names: capitalised phrases after "near"/"on"/"along" that do not
occur in the capsule, and whether the verifier produced any claim for them.
Usage: python roadscribe_research/identifiability/x5_slm_failures.py data/models/kaggle_regen/spgr_regen.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_spgr_regen import ledger_for  # noqa: E402
from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402

PLACE = re.compile(r"\b(?:near|on|along|past|at)\s+((?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*)")


def main(path):
    gens = [g for g in json.loads(Path(path).read_text(encoding="utf-8")) if g["condition"] == "orig"]
    base = RoadScribeConfig()
    by_reason, by_kind, kinds_total = Counter(), Counter(), Counter()
    places, place_unchecked, examples = 0, 0, []
    for trip in range(1, 10):
        ledger, capsule, must = ledger_for(trip, base)
        for g in gens:
            if g["trip_id"] != f"pvs{trip}":
                continue
            text = g["generated_narrative"].split("</plan>")[1].strip() if "</plan>" in g["generated_narrative"] else g["generated_narrative"]
            rep = verify_text(text, ledger, must_mention_events=must)
            for c in rep.claims:
                kinds_total[c.kind] += 1
                if not c.supported:
                    by_reason[c.reason or "unspecified"] += 1
                    by_kind[c.kind] += 1
                    if len(examples) < 12 and c.reason not in {e[1] for e in examples}:
                        examples.append((text[c.span[0]:c.span[1]], c.reason))
            for m in PLACE.finditer(text):
                name = m.group(1)
                if name in ("I", "The", "U") or name.lower() in capsule.text.lower():
                    continue
                places += 1
                covered = any(c.kind == "ANCHOR" and c.span[0] <= m.start(1) < c.span[1] for c in rep.claims)
                place_unchecked += int(not covered)
        print(f"trip {trip} done", flush=True)
    out = {"label": "EXPLORATORY", "claims_by_kind": dict(kinds_total), "unsupported_by_kind": dict(by_kind),
           "unsupported_by_reason": dict(by_reason.most_common()),
           "place_names_not_in_capsule": places, "of_which_not_checked_by_verifier": place_unchecked,
           "examples": examples}
    (HERE / "results" / "x5_slm_failures.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
