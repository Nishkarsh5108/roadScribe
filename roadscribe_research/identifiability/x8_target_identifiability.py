"""Input-level identifiability audit of training targets (added 2026-11-13 after the simulated review).

For every value a training target states, is the value present in the model's input?
Value types parsed from the target: event position ("the 1.2 km mark"), speed (km/h), acceleration (m/s2),
distance (km, not a position) and duration (min).

Two matching rules, both reported:
  typed    the input holds a field of the same type whose value rounds to the stated value
           (capsule: extents and lengths in km, duration in min, peak exposure "mm"; frames: s, s0, s1,
           length, duration, a_v, mtvv, speed). Capsules hold no event positions or speeds, so for them
           those types are 0 by construction.
  lenient  any number in the input, as is, /1000 or x3.6, rounds to the stated value (an upper bound
           that also counts coincidences; the earlier, unscripted audit used a rule like this).
Tolerance: half a unit of the last digit the target shows.

Datasets: the original S-PGR training set (capsule -> plan + narrative; 2,500 template + 196 teacher records)
and the three P6 arms (same 1,200 targets; inputs A capsule, B + frames, C + frames without positions).
Usage: python roadscribe_research/identifiability/x8_target_identifiability.py [p6_inputs_dir] [train_sft.jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P6_IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/Coding/Hackathon/GFG/ARM/kaggle_staging/p6_inputs")
ORIG_TRAIN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/Coding/Hackathon/GFG/ARM/kaggle_staging/dataset/train_sft.jsonl")

NUM = r"(\d+(?:\.\d+)?)"
T_POS = re.compile(NUM + r"\s?km mark")
T_SPEED = re.compile(NUM + r"\s?km/h")
T_ACC = re.compile(NUM + r"\s?m/s")
T_RANGE = re.compile(NUM + r"\s?[\u2013\-]\s?" + NUM + r"\s?km\b")
T_KM = re.compile(NUM + r"\s?km\b(?!\s?mark|/h)")
T_MIN = re.compile(NUM + r"\s?min(?:ute)?s?\b")
TYPES = ("position", "speed", "acceleration", "distance", "duration")


def half(s: str) -> float:
    return 0.5 * 10 ** (-len(s.split(".")[1])) if "." in s else 0.5


def target_values(text: str) -> list[tuple[str, float, float]]:
    out, used = [], []
    for rx, typ in ((T_POS, "position"), (T_SPEED, "speed"), (T_ACC, "acceleration"), (T_MIN, "duration")):
        for m in rx.finditer(text):
            out.append((typ, float(m.group(1)), half(m.group(1))))
            used.append(m.span())
    for m in T_RANGE.finditer(text):
        for g in (1, 2):
            out.append(("distance", float(m.group(g)), half(m.group(g))))
        used.append(m.span())
    for m in T_KM.finditer(text):
        if any(a <= m.start() < b for a, b in used):
            continue
        out.append(("distance", float(m.group(1)), half(m.group(1))))
    return out


def split_input(user: str) -> tuple[str, list]:
    cap = user.split("CAPSULE\n", 1)[1] if "CAPSULE\n" in user else user
    frames = []
    if "\n\nFRAMES\n" in cap:
        cap, fj = cap.split("\n\nFRAMES\n", 1)
        frames = json.loads(fj)
    return cap, frames


def typed_candidates(cap: str, frames: list) -> dict:
    c = defaultdict(list)
    for m in re.finditer(NUM + r"-" + NUM + r"km", cap):
        c["distance"] += [float(m.group(1)), float(m.group(2))]
    for m in re.finditer(NUM + r" km\b", cap):
        c["distance"].append(float(m.group(1)))
    for m in re.finditer(r"@" + NUM + r"km", cap):
        c["distance"].append(float(m.group(1)))
    for m in re.finditer(NUM + r" min\b", cap):
        c["duration"].append(float(m.group(1)))
    for m in re.finditer(r"\bmm (\.\d+|1\.0|\d+\.\d+)", cap):
        c["acceleration"].append(float(m.group(1)))
    for f in frames:
        for k in ("s0", "s1", "length_m"):
            if isinstance(f.get(k), (int, float)):
                c["distance"].append(f[k] / 1000)
        if isinstance(f.get("length_km"), (int, float)):
            c["distance"].append(f["length_km"])
        if isinstance(f.get("duration_min"), (int, float)):
            c["duration"].append(f["duration_min"])
        for k in ("a_v", "mtvv", "a_dec", "a_wz"):
            if isinstance(f.get(k), (int, float)):
                c["acceleration"].append(f[k])
        if isinstance(f.get("speed_mps"), (int, float)):
            c["speed"].append(f["speed_mps"] * 3.6)
        for e in f.get("events") or []:
            if isinstance(e.get("s"), (int, float)):
                c["position"].append(e["s"] / 1000)
    return c


def lenient_candidates(user: str) -> list[float]:
    xs = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", user)]
    return xs + [x / 1000 for x in xs] + [x * 3.6 for x in xs]


def present(x: float, tol: float, cands) -> bool:
    return any(abs(x - c) <= tol + 1e-9 for c in cands)


def audit(pairs) -> dict:
    n, typed, lenient = defaultdict(int), defaultdict(int), defaultdict(int)
    for user, target in pairs:
        cap, frames = split_input(user)
        tc, lc = typed_candidates(cap, frames), lenient_candidates(user)
        for typ, x, tol in target_values(target):
            n[typ] += 1
            typed[typ] += present(x, tol, tc[typ])
            lenient[typ] += present(x, tol, lc)
    res = {t: dict(mentions=n[t], typed=typed[t], typed_share=typed[t] / n[t] if n[t] else None,
                   lenient=lenient[t], lenient_share=lenient[t] / n[t] if n[t] else None) for t in TYPES}
    tot = sum(n.values())
    res["all_values"] = dict(mentions=tot, typed_share=sum(typed.values()) / tot if tot else None,
                             lenient_share=sum(lenient.values()) / tot if tot else None)
    return res


def narrative_of(target: str) -> str:
    return target.split("</plan>", 1)[1] if "</plan>" in target else target


def main():
    out = {"label": "EXPLORATORY (added after the simulated review, 2026-11-13)", "rules": __doc__.split("Datasets:")[0]}
    # The original adapter's training file (2,696 records: capsule input, plan + narrative target).
    rows = [json.loads(l) for l in open(ORIG_TRAIN, encoding="utf-8")]
    orig = [("CAPSULE\n" + r["capsule"], r["narrative"]) for r in rows]
    out["original_training_set"] = dict(records=len(orig), **audit(orig))
    for arm in ("A", "B", "C"):
        recs = [json.loads(l) for l in open(P6_IN / f"train_{arm}.jsonl", encoding="utf-8")]
        out[f"p6_arm_{arm}"] = dict(records=len(recs), **audit([(r["messages"][1]["content"], r["messages"][2]["content"]) for r in recs]))
    (HERE / "results" / "x8_target_identifiability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict) and "records" in v:
            print(k, v["records"], {t: (v[t]["mentions"], round(v[t]["typed_share"] or 0, 3), round(v[t]["lenient_share"] or 0, 3))
                                    for t in TYPES}, "all", round(v["all_values"]["typed_share"], 3))


if __name__ == "__main__":
    main()
