"""Evaluate P6 (PREREG_P6.md): three fine-tuned arms (A capsule; B + frames; C + frames without
positions) and the prompted 3B / 7B baselines, on the 18 PVS inputs x 3 seeds.

Every narrative is checked by the unchanged verifier against the original ledger. Operational
definitions fixed before any P6 output was seen (WORKLOG D27):
  precision           pooled supported / extracted claims over the arm's 54 narratives
  wrong-position share EVT claims whose reason is wrong_position / all EVT claims
  non-position errors  unsupported claims with any reason other than wrong_position, per narrative
  acceptance           narrative passes strict verification and has at least one claim
Uncertainty: cluster bootstrap over the 9 drives (10,000 resamples, seed 20261113).

Usage: python scripts/eval_p6.py [p6_dir] [ledgers.pkl]
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from roadscribe.eval.claims import verify_text  # noqa: E402

P6 = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "data/models/p6"
LEDGERS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/Coding/Hackathon/GFG/ARM/kaggle_staging/pgr_v2_inputs/pvs_ledgers.pkl")
B_RES, SEED = 10_000, 20261113


def load_outputs():
    runs = {}
    for arm in ("A", "B", "C"):
        f = P6 / arm / f"p6_{arm}.json"
        if f.exists():
            runs[arm] = json.loads(f.read_text(encoding="utf-8"))
    f = P6 / "zs" / "p6_zeroshot.json"
    if f.exists():
        for r in json.loads(f.read_text(encoding="utf-8")):
            runs.setdefault("zs-" + r["model"].split("-")[1], []).append(r)
    return runs


NUM_TYPE = {"v_mean_mps": "speed", "trip_duration_s": "duration", "trip_length_m": "distance", "length_m": "distance"}


def value_type(c) -> str:
    """Map a verifier claim to the value types of x8_target_identifiability.py (exploratory, WORKLOG D28)."""
    if c.kind == "EVT":
        return "position" if isinstance(c.value, dict) and c.value.get("pos") is not None else "event_other"
    if c.kind == "NUM":
        return NUM_TYPE.get(c.key, "acceleration")
    if c.kind == "RANGE" or (c.kind == "REST" and c.key == "length_m"):
        return "distance"
    return c.kind.lower()


def score(rows, ledgers):
    per = []
    for r in rows:
        d = ledgers[r["key"]]
        text = r["narrative"].strip()
        rep = verify_text(text, d["ledger"], must_mention_events=d["must"])
        reasons = Counter(c.reason for c in rep.claims if not c.supported)
        types_n = Counter(value_type(c) for c in rep.claims)
        types_bad = Counter(value_type(c) for c in rep.claims if not c.supported)
        n_evt = sum(1 for c in rep.claims if c.kind == "EVT")
        per.append(dict(trip=r["trip_id"], persona=r["persona"], seed=r["seed"], n_claims=rep.n_claims,
                        n_sup=rep.n_supported, n_evt=n_evt, wrong_pos=reasons.get("wrong_position", 0),
                        non_pos=sum(v for k, v in reasons.items() if k != "wrong_position"),
                        missing=rep.hallucinated_missing, accepted=bool(rep.passed and rep.n_claims > 0),
                        must_missed=len(rep.must_mention_missed), reasons=dict(reasons),
                        types_n=dict(types_n), types_bad=dict(types_bad),
                        words=len(text.split()), n_new_tokens=r.get("n_new_tokens")))
    return per


def stats(per):
    tot = lambda k: sum(p[k] for p in per)  # noqa: E731
    return dict(n=len(per), claims=tot("n_claims"), supported=tot("n_sup"),
                precision=tot("n_sup") / max(tot("n_claims"), 1),
                evt_claims=tot("n_evt"), wrong_position=tot("wrong_pos"),
                wrong_position_share=tot("wrong_pos") / max(tot("n_evt"), 1),
                non_position_errors=tot("non_pos"), non_position_per_narrative=tot("non_pos") / max(len(per), 1),
                hallucinated_missing=tot("missing"), accepted=tot("accepted"),
                claims_per_narrative=tot("n_claims") / max(len(per), 1),
                words_per_narrative=tot("words") / max(len(per), 1),
                must_mention_missed=tot("must_missed"),
                hit_token_limit=sum(1 for p in per if (p["n_new_tokens"] or 0) >= 768),
                reasons=dict(sum((Counter(p["reasons"]) for p in per), Counter())),
                by_value_type=by_type(per))


def by_type(per):
    n = sum((Counter(p["types_n"]) for p in per), Counter())
    bad = sum((Counter(p["types_bad"]) for p in per), Counter())
    return {t: dict(claims=n[t], unsupported=bad[t], error_rate=bad[t] / n[t]) for t in sorted(n)}


def by_trip(per):
    g = defaultdict(list)
    for p in per:
        g[p["trip"]].append(p)
    return g


def boot(per_x, per_y, fn):
    """Cluster bootstrap over drives of fn(x) - fn(y); both arms share the same drives."""
    gx, gy = by_trip(per_x), by_trip(per_y)
    trips = sorted(set(gx) & set(gy))
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(B_RES):
        pick = rng.choice(trips, size=len(trips), replace=True)
        xs = [p for t in pick for p in gx[t]]
        ys = [p for t in pick for p in gy[t]]
        diffs.append(fn(xs) - fn(ys))
    diffs = np.array(diffs)
    return dict(diff=fn(per_x) - fn(per_y), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                p_le_0=float((diffs <= 0).mean()))


def boot_ratio(per_x, per_y, fn):
    gx, gy = by_trip(per_x), by_trip(per_y)
    trips = sorted(set(gx) & set(gy))
    rng = np.random.default_rng(SEED)
    rs = []
    for _ in range(B_RES):
        pick = rng.choice(trips, size=len(trips), replace=True)
        x, y = fn([p for t in pick for p in gx[t]]), fn([p for t in pick for p in gy[t]])
        rs.append(x / y if y > 0 else np.inf)
    rs = np.array(rs)
    y0 = fn(per_y)
    return dict(ratio=fn(per_x) / y0 if y0 > 0 else None,
                ci95=[float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))])


def prec(per):
    c = sum(p["n_claims"] for p in per)
    return sum(p["n_sup"] for p in per) / c if c else 0.0


def wps(per):
    e = sum(p["n_evt"] for p in per)
    return sum(p["wrong_pos"] for p in per) / e if e else 0.0


def nonpos(per):
    return sum(p["non_pos"] for p in per) / len(per) if per else 0.0


def main():
    ledgers = pickle.load(open(LEDGERS, "rb"))
    runs = load_outputs()
    per = {k: score(v, ledgers) for k, v in runs.items()}
    out = {"provenance": {"script": "scripts/eval_p6.py", "prereg": "PREREG_P6.md (commit 0bda5ac)",
                          "definitions": "WORKLOG D27", "bootstrap": f"cluster over drives, {B_RES} resamples"},
           "arms": {k: stats(v) for k, v in per.items()}}
    h = {}
    if {"A", "B"} <= set(per):
        b = boot(per["B"], per["A"], prec)
        h["P6-H1"] = dict(prediction="precision B - A >= 0.15", **b, holds=b["diff"] >= 0.15)
    if {"A", "B", "C"} <= set(per):
        wb, wc, wa = wps(per["B"]), wps(per["C"]), wps(per["A"])
        h["P6-H2"] = dict(prediction="wrong-position share C >= 2 B and A >= 2 B", A=wa, B=wb, C=wc,
                          C_over_B=boot_ratio(per["C"], per["B"], wps), A_over_B=boot_ratio(per["A"], per["B"], wps),
                          holds=wc >= 2 * wb and wa >= 2 * wb)
        nc, na = nonpos(per["C"]), nonpos(per["A"])
        h["P6-H3"] = dict(prediction="non-position errors per narrative C < 0.5 A", A=na, C=nc,
                          C_over_A=boot_ratio(per["C"], per["A"], nonpos), holds=nc < 0.5 * na)
        b = boot(per["B"], per["C"], prec)
        h["P6-H4"] = dict(prediction="precision B > C", **b, holds=b["diff"] > 0)
    out["hypotheses"] = h
    out["per_narrative"] = per
    path = repo_root / "roadscribe_research/identifiability/results/p6_results.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k, s in out["arms"].items():
        print(f"{k:8s} n={s['n']} precision={s['precision']:.3f} claims/narr={s['claims_per_narrative']:.1f} "
              f"accepted={s['accepted']} wrongpos={s['wrong_position_share']:.3f} nonpos/narr={s['non_position_per_narrative']:.2f} "
              f"missing={s['hallucinated_missing']} cap={s['hit_token_limit']}")
    for k, v in h.items():
        print(k, json.dumps(v, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
