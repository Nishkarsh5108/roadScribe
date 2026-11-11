"""Narrative benchmark on the 9 PVS trips (paper Section 4), rewritten 2026-11-12.

What changed from commit 4b95a72, and why:

* **E1 (driver QA) and E2 (held-out personas) were never run.** No model was
  ever called on a question or on an E2 record. The earlier script counted
  S-PGR as correct straight from the gold labels and wrote the E2 scores as
  constants. Both are now reported as ``not_run``, with the reason, until
  predictions exist.
* **P_G was 0.94 x P_L, not a measurement.** It is removed. Precision against
  ground truth can only be measured for event mentions against labelled speed
  bumps, and is reported that way.
* **Anomaly grounding pooled all trips into one matching**, so a mention in one
  trip could match a bump in another, and each trip's bumps were appended once
  per persona. It is now scored per narrative, against that trip's own events,
  with a random-placement chance level and a point-only variant.
* **"plan_validity" only checked that the plan parses as JSON**, and is renamed
  to say so.
* **ECE_ling** is computed against *ledger* support with *assumed* hedge
  probabilities, and is renamed to say so.
* **The S-PGR composition is reported**: how many outputs came from the SLM and
  how many were template fallbacks. Every other S-PGR metric is mostly the
  template's.

Usage (from the repo root):
    python scripts/run_full_evaluation.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ.setdefault(
    "ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
)

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.eval.metrics import (  # noqa: E402
    Mention,
    RefEvent,
    anomaly_grounding_by_recording,
    ece_ling,
    severity_fidelity,
)
from roadscribe.io.pvs import bump_mask, load_pvs_trip, pvs_root, route_of, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events, find_runs  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402

CONDITIONS = ("T", "T+V", "S-PGR")
PERSONAS = ("commuter", "sports_car")
TOL_M = 25.0  # mention half-width and matching tolerance, as before (effective +-50 m)
CHANCE_DRAWS = 200
ASSUMED_HEDGE_P = {"assertive": 0.95, "likely": 0.75, "possibly": 0.50}  # not elicited from people
NOT_RUN_QA = (
    "No model has been run on the QA bank. The fine-tuned model was trained only on trip_summary "
    "records (no questions), and the Kaggle notebook generated PVS narratives only. The bank's 300 "
    "questions were written by the designer, so even once run it is a development set, not E1 "
    "(see data/processed/DATASET_CARD.md)."
)
NOT_RUN_E2 = (
    "No model has been run on test_e2.jsonl. The training set excluded the E2 personas, so an "
    "honest E2 is possible, but persona adherence also needs a defined measure (raters or a "
    "validated classifier), and none exists yet."
)


def mentions_of(rep, L, end):
    """Event mentions as (Mention, how-located): 'point' (+-TOL_M), 'unit' (stretch extent) or 'trip'."""
    out = []
    for c in rep.claims:
        if c.kind != "EVT":
            continue
        pos = c.value.get("pos") if isinstance(c.value, dict) else None
        if pos is not None:
            out.append((Mention(c.key, max(0.0, pos - TOL_M), pos + TOL_M), "point"))
        elif c.unit_id:
            s0, s1 = L.extent(c.unit_id)
            out.append((Mention(c.key, s0, s1), "unit"))
        else:
            out.append((Mention(c.key, 0.0, end), "trip"))
    return out


def build_trip(trip: int, cfg: RoadScribeConfig, tok: RegexTokenizer) -> dict:
    raw = load_pvs_trip(pvs_root(), trip, side="left", with_labels=True)
    vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
    aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
    det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
    grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
    res, _ = segment(grid.Y, grid.s, [], [], cfg.mipelt)
    end = float(vft.s[-1])
    bounds = arbitrate(res.changepoints, grid.s, grid.Y, [], cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
    units = build_hrnus(bounds, vft, aw, det.events, [], grid_s=grid.s, grid_Y=grid.Y,
                        iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
    meta = TripMeta(raw.trip_id, duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=0.0)
    ledger = compile_ledger(units, det.events, meta, cfg.ledger)
    kst = build_capsule(units, det.events, meta, 128, tok, cfg.kst, ledger=ledger)
    L = LedgerView(ledger)
    gt_refs = [RefEvent("bump", float(np.mean(vft.s[a:b])), 1.0, f"GT_{k + 1}")
               for k, (a, b) in enumerate(find_runs(bump_mask(raw.labels)))]
    ledger_refs, must = [], []
    for f in L.facts:
        if f.kind == "EVT":
            ledger_refs.append(RefEvent(f.value.get("type", "bump"), float(f.value.get("s", 0.0)),
                                        float(f.value.get("mtvv", 1.0)), f.key))
            if f.conf >= 0.85:
                must.append(f.key)
    return dict(route=route_of(trip), vehicle=vehicle_of(trip), ledger=ledger, L=L, end=end,
                selection=kst.capsule.selection, gt_refs=gt_refs, ledger_refs=ledger_refs, must=must)


def spgr_output(raw_gen: str, ledger, must, narr_T: str, rep_T):
    """S-PGR acceptance rule, unchanged from the earlier script."""
    has_plan = "<plan>" in raw_gen and "</plan>" in raw_gen
    plan_parses = False
    if has_plan:
        try:
            plan_parses = isinstance(json.loads(raw_gen.split("<plan>")[1].split("</plan>")[0].strip()), dict)
        except Exception:
            plan_parses = False
    cand = raw_gen.split("</plan>")[1].strip() if has_plan else raw_gen.strip()
    rep_c = verify_text(cand, ledger, must_mention_events=must)
    accepted = bool(plan_parses and rep_c.passed and len(rep_c.claims) > 0)
    if accepted:
        return cand, rep_c, plan_parses, True
    return narr_T, rep_T, plan_parses, False


def evaluate_pvs():
    cfg = RoadScribeConfig()
    tok = RegexTokenizer()
    preds = json.load(open(repo_root / "data/models/kaggle_run/pvs_predictions.json", encoding="utf-8"))
    pred_map = {(p["trip_id"], p["persona"]): p["generated_narrative"] for p in preds}

    acc = {c: dict(n_claims=0, n_supported=0, n_missing=0, x_sev=[], k_sev=[], hedges=[], hedge_ok=[],
                   n_gen=0, n_plan_parses=0, n_fallback=0,
                   rec_L={p: [] for p in PERSONAS}, rec_G={p: [] for p in PERSONAS},
                   rec_G_point={p: [] for p in PERSONAS}, kinds=Counter())
           for c in CONDITIONS}
    composition = []

    for trip in range(1, 10):
        tr = build_trip(trip, cfg, tok)
        print(f"trip {trip}: {tr['end'] / 1000:.1f} km, {len(tr['gt_refs'])} labelled bumps, "
              f"{len(tr['ledger_refs'])} ledger events, {len(tr['must'])} must-mention", flush=True)
        for persona in PERSONAS:
            plan_T = rule_plan(tr["ledger"], tr["selection"], persona, rng=None, must_mention_events=tr["must"])
            frames_T = ground(plan_T, tr["ledger"])
            narr_T = realize(frames_T, persona, vary=False)
            rep_T = verify_text(narr_T, tr["ledger"], must_mention_events=tr["must"])
            narr_TV = realize(frames_T, persona, vary=True, seed=trip * 100 + len(persona))
            rep_TV = verify_text(narr_TV, tr["ledger"], must_mention_events=tr["must"])
            raw_gen = pred_map.get((f"pvs{trip}", persona), "")
            narr_S, rep_S, plan_parses, accepted = spgr_output(raw_gen, tr["ledger"], tr["must"], narr_T, rep_T)
            composition.append(dict(trip=trip, persona=persona, accepted=accepted, plan_parses=plan_parses,
                                    slm_claims=rep_S.n_claims if accepted else 0,
                                    slm_words=len(narr_S.split()) if accepted else 0,
                                    template_claims=rep_T.n_claims,
                                    text=narr_S if accepted else None))

            for cond, rep, parses, fb in (("T", rep_T, True, False), ("T+V", rep_TV, True, False),
                                          ("S-PGR", rep_S, plan_parses, not accepted)):
                a = acc[cond]
                a["n_gen"] += 1
                a["n_plan_parses"] += int(parses)
                a["n_fallback"] += int(fb)
                a["n_claims"] += rep.n_claims
                a["n_supported"] += rep.n_supported
                a["n_missing"] += rep.hallucinated_missing
                for c in rep.claims:
                    if c.kind == "SEV" and c.unit_id:
                        meas = tr["L"].value(c.unit_id, "a_v")
                        if meas is not None:
                            a["x_sev"].append(float(meas))
                            a["k_sev"].append(int(c.value))
                    if c.kind == "EVT":
                        a["hedges"].append(c.hedge)
                        a["hedge_ok"].append(bool(c.supported))
                ms = mentions_of(rep, tr["L"], tr["end"])
                a["kinds"].update(k for _, k in ms)
                rec_id = f"pvs{trip}/{persona}"
                a["rec_L"][persona].append((rec_id, [m for m, _ in ms], tr["ledger_refs"], tr["end"]))
                a["rec_G"][persona].append((rec_id, [m for m, _ in ms], tr["gt_refs"], tr["end"]))
                a["rec_G_point"][persona].append((rec_id, [m for m, k in ms if k == "point"], tr["gt_refs"], tr["end"]))

    summary = {}
    for cond in CONDITIONS:
        a = acc[cond]
        sev = severity_fidelity(a["x_sev"], a["k_sev"]) if len(a["x_sev"]) >= 2 else None
        grounding = {}
        for persona in PERSONAS:
            gL = anomaly_grounding_by_recording(a["rec_L"][persona], delta_s=TOL_M)
            gG = anomaly_grounding_by_recording(a["rec_G"][persona], delta_s=TOL_M, chance_draws=CHANCE_DRAWS, seed=7)
            gP = anomaly_grounding_by_recording(a["rec_G_point"][persona], delta_s=TOL_M)
            grounding[persona] = {
                "AGP_L": round(gL.agp, 4), "AGR_L": round(gL.agr, 4), "n_ledger_events": gL.n_refs,
                "AGR_G": round(gG.agr, 4), "AGP_G": round(gG.agp, 4),
                "AGR_G_random_placement": round(gG.chance_agr, 4),
                "AGR_G_point_mentions_only": round(gP.agr, 4), "n_point_mentions": gP.n_mentions,
                "n_labelled_bumps": gG.n_refs, "n_mentions": gG.n_mentions,
            }
        summary[cond] = {
            "P_L": round(a["n_supported"] / max(1, a["n_claims"]), 4),
            "n_claims": a["n_claims"],
            "hallucinated_missing_rate": round(a["n_missing"] / max(1, a["n_claims"]), 4),
            "rho_KSF": None if sev is None else round(sev.rho_ksf, 4),
            "tau_b": None if sev is None else round(sev.tau_b, 4),
            "BC": None if sev is None else round(sev.bc, 4),
            "OI": None if sev is None else round(sev.oi, 4),
            "ece_ling_vs_ledger_with_assumed_hedge_p": round(ece_ling(a["hedges"], a["hedge_ok"], ASSUMED_HEDGE_P), 4)
            if a["hedges"] else None,
            "plan_json_parse_rate": round(a["n_plan_parses"] / a["n_gen"], 4),
            "fallback_rate": round(a["n_fallback"] / a["n_gen"], 4),
            "event_mentions_by_location_type": dict(a["kinds"]),
            "anomaly_grounding_per_persona": grounding,
        }
    acc_rows = [r for r in composition if r["accepted"]]
    comp = {
        "n_generations": len(composition),
        "n_accepted_slm_outputs": len(acc_rows),
        "n_template_fallbacks": len(composition) - len(acc_rows),
        "slm_claims_in_accepted_outputs": sum(r["slm_claims"] for r in acc_rows),
        "s_pgr_total_claims": summary["S-PGR"]["n_claims"],
        "accepted_outputs": [{k: r[k] for k in ("trip", "persona", "slm_claims", "slm_words", "template_claims", "text")}
                             for r in acc_rows],
        "must_mention_sets_empty_on_all_trips": True,
        "note": "S-PGR metrics are those of the template on fallback cases; compare n_accepted_slm_outputs.",
    }
    return summary, comp


def main():
    summary_pvs, composition = evaluate_pvs()
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    out = {
        "provenance": {
            "script": "scripts/run_full_evaluation.py (rewritten 2026-11-12)",
            "git_head_at_run": commit,
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": "PVS trips 1-9, left logger, dashboard IMU; data/models/kaggle_run/pvs_predictions.json",
        },
        "pvs": summary_pvs,
        "s_pgr_composition": composition,
        "qa_e1": {"status": "not_run", "reason": NOT_RUN_QA},
        "personas_e2": {"status": "not_run", "reason": NOT_RUN_E2},
        "removed": {
            "P_G": "was 0.94 x P_L; no measurement of per-claim precision against ground truth exists",
            "H1_bootstrap_CI_and_p": "never computed; with 3 routes a cluster bootstrap is degenerate",
            "H2_McNemar": "never computed; E1 was never run",
            "wAGR_G": "every labelled bump had weight 1.0, so it equalled AGR_G; dropped",
        },
    }
    path = repo_root / "data" / "evaluation_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["pvs"], indent=1))
    print(json.dumps({k: v for k, v in composition.items() if k != "accepted_outputs"}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
