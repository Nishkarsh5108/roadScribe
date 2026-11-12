"""Build the Kaggle inputs for P5 (narrator trained on its intended PLAN and REALIZE tasks).

Writes to the staging folder given on the command line:
  train.jsonl          chat examples (plan + realize), from the repository's SFT and teacher records
  pvs_prompts.json     plan prompts for 9 PVS trips x 2 personas x {orig, cal}
  pvs_ledgers.pkl      ledger, visible units and must-mention set for each prompt (for grounding)
  roadscribe/          a copy of the package source (grounding and formats run inside the kernel)
Usage: python scripts/build_pgr_v2_inputs.py D:/Coding/Hackathon/GFG/ARM/kaggle_staging/pgr_v2_inputs
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig, SpeedConfig  # noqa: E402
from roadscribe.io.pvs import load_pvs_trip, pvs_root, route_of, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger, format_ledger  # noqa: E402
from roadscribe.synth.formats import SYSTEM_PLAN, persona_card, to_chat  # noqa: E402

PERSONAS = ("commuter", "sports_car")
N_PLAN, N_REALIZE, SEED = 1200, 1200, 20261112


def training_examples() -> list[dict]:
    rows = [json.loads(l) for l in open(repo_root / "data/processed/roadscribe_sft.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["persona"] in PERSONAS and r["split"] == "train"]
    random.Random(SEED).shuffle(rows)
    plan = [dict(task="plan", **to_chat(r, "plan")) for r in rows[:N_PLAN]]
    real = [dict(task="realize", **to_chat(r, "realize")) for r in rows[N_PLAN:N_PLAN + N_REALIZE]]
    teach = [json.loads(l) for l in open(repo_root / "data/processed/roadscribe_teacher_sft.jsonl", encoding="utf-8")]
    real += [dict(task="realize", **to_chat(r, "realize")) for r in teach if r["persona"] in PERSONAS]
    ex = plan + real
    random.Random(SEED + 1).shuffle(ex)
    return ex


def pvs_inputs():
    p4 = json.loads((repo_root / "roadscribe_research/identifiability/results/p4_results.json").read_text(encoding="utf-8"))
    base, tok = RoadScribeConfig(), RegexTokenizer()
    prompts, ledgers = [], {}
    for trip in range(1, 10):
        raw = load_pvs_trip(pvs_root(), trip, side="left", with_labels=False)
        cal = p4["calibration"][route_of(trip)]
        for cond, cfg in (("orig", base), ("cal", dataclasses.replace(base, speed=SpeedConfig(
                gamma=cal["gamma"], vehicle_gain_log=cal["gains"][str(vehicle_of(trip))])))):
            vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
            aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
            det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
            grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
            res, _ = segment(grid.Y, grid.s, [], [], cfg.mipelt)
            end = float(vft.s[-1])
            bounds = arbitrate(res.changepoints, grid.s, grid.Y, [], cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
            units = build_hrnus(bounds, vft, aw, det.events, [], grid_s=grid.s, grid_Y=grid.Y,
                                iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
            for persona in PERSONAS:
                # PVS has no map coverage: do not advertise a map channel (the earlier capsules did)
                meta = TripMeta(raw.trip_id, persona=persona, channels=("acc", "gyro", "gps"),
                                duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=0.0)
                ledger = compile_ledger(units, det.events, meta, cfg.ledger)
                kst = build_capsule(units, det.events, meta, 128, tok, cfg.kst, ledger=ledger)
                visible = [u for u, lv in kst.capsule.selection.items() if lv != "OMIT"]
                L = LedgerView(ledger)
                must = [f.key for f in L.facts if f.kind == "EVT" and f.conf >= 0.85]
                user = f"{persona_card(persona)}\n\nCAPSULE\n{kst.capsule.text}\n\nFACT INDEX\n{format_ledger(ledger, [None, *visible], compact=True)}"
                key = f"{cond}|pvs{trip}|{persona}"
                prompts.append(dict(key=key, condition=cond, trip_id=f"pvs{trip}", persona=persona,
                                    messages=[{"role": "system", "content": SYSTEM_PLAN}, {"role": "user", "content": user}]))
                ledgers[key] = dict(ledger=list(ledger), visible=visible, must=must)
        print(f"trip {trip} done", flush=True)
    return prompts, ledgers


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = training_examples()
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    print("training examples:", len(ex), {t: sum(e["task"] == t for e in ex) for t in ("plan", "realize")})
    prompts, ledgers = pvs_inputs()
    (out / "pvs_prompts.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")
    with open(out / "pvs_ledgers.pkl", "wb") as f:
        pickle.dump(ledgers, f, protocol=4)
    dst = out / "roadscribe_src" / "roadscribe"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(repo_root / "roadscribe", dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("prompts:", len(prompts), "-> ", out)


if __name__ == "__main__":
    main(sys.argv[1])
