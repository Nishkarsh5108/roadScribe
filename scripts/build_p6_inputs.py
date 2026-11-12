"""Build the inputs for P6 (PREREG_P6.md): same records and targets, three input variants.

Arm A: persona card + capsule. Arm B: + grounded frames. Arm C: + frames without event positions.
Records whose arm-B prompt is too long are dropped from all arms (identical record sets).
PVS prompts reuse the P5 capsules (no spurious map channel) and ledgers; frames come from the rule plan.

Usage: python scripts/build_p6_inputs.py <out_dir> <p5_inputs_dir>
"""
from __future__ import annotations

import copy
import json
import pickle
import random
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from roadscribe.layer3.templates import ground, rule_plan  # noqa: E402
from roadscribe.synth.formats import SYSTEM_REALIZE, persona_card  # noqa: E402

SYSTEM_NEUTRAL = ("You are RoadScribe's narrator. Write a short narrative of the trip for the reader described, "
                  "using only the information given.")
N, SEED, MAX_CHARS_B = 1200, 20261113, 6000  # ~2,000 tokens at ~3 characters per token


def frames_json(frames, drop_positions=False):
    fr = copy.deepcopy(frames)
    if drop_positions:
        for f in fr:
            for e in f.get("events") or []:
                e.pop("s", None)
    return json.dumps(fr, separators=(",", ":"), sort_keys=True)


def user(arm, persona, capsule, frames):
    u = f"{persona_card(persona)}\n\nCAPSULE\n{capsule}"
    if arm == "B":
        u += f"\n\nFRAMES\n{frames_json(frames)}"
    elif arm == "C":
        u += f"\n\nFRAMES\n{frames_json(frames, drop_positions=True)}"
    return u


def main(out_dir, p5_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(repo_root / "data/processed/roadscribe_sft.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["persona"] in ("commuter", "sports_car") and r["split"] == "train"]
    random.Random(SEED).shuffle(rows)
    kept = [r for r in rows if len(user("B", r["persona"], r["capsule"], r["frames"])) + len(r["narrative"]) <= MAX_CHARS_B][:N]
    for arm in ("A", "B", "C"):
        with open(out / f"train_{arm}.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                msgs = [{"role": "system", "content": SYSTEM_NEUTRAL},
                        {"role": "user", "content": user(arm, r["persona"], r["capsule"], r["frames"])},
                        {"role": "assistant", "content": r["narrative"]}]
                f.write(json.dumps({"id": r["id"], "messages": msgs}) + "\n")
    p5_prompts = json.loads((Path(p5_dir) / "pvs_prompts.json").read_text(encoding="utf-8"))
    ledgers = pickle.load(open(Path(p5_dir) / "pvs_ledgers.pkl", "rb"))
    pvs = {"A": [], "B": [], "C": [], "zeroshot": []}
    for p in p5_prompts:
        if p["condition"] != "orig":
            continue
        d = ledgers[p["key"]]
        capsule = p["messages"][1]["content"].split("CAPSULE\n")[1].split("\n\nFACT INDEX")[0]
        plan = rule_plan(d["ledger"], {u: "FULL" for u in d["visible"]}, p["persona"], rng=None, must_mention_events=d["must"])
        frames = ground(plan, d["ledger"])
        for arm in ("A", "B", "C"):
            pvs[arm].append(dict(key=p["key"], trip_id=p["trip_id"], persona=p["persona"],
                                 messages=[{"role": "system", "content": SYSTEM_NEUTRAL},
                                           {"role": "user", "content": user(arm, p["persona"], capsule, frames)}]))
        pvs["zeroshot"].append(dict(key=p["key"], trip_id=p["trip_id"], persona=p["persona"],
                                    messages=[{"role": "system", "content": SYSTEM_REALIZE},
                                              {"role": "user", "content": user("B", p["persona"], capsule, frames)}]))
    for k, v in pvs.items():
        (out / f"pvs_{k}.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    (out / "kept_ids.json").write_text(json.dumps([r["id"] for r in kept]), encoding="utf-8")
    print("records kept:", len(kept), "| pvs prompts per arm:", {k: len(v) for k, v in pvs.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
