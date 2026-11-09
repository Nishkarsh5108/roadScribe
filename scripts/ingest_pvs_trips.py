"""Ingest all 9 PVS trips through RoadScribe Layers 1-3 to generate real-world ground-truth capsules."""
import json
import os
import sys
from pathlib import Path

# Ensure paths
pvs_dir = os.environ.get("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")
os.environ["ROADSCRIBE_PVS_DIR"] = pvs_dir

repo_root = Path(r"d:\Coding\Hackathon\GFG\ARM\roadScribe")
sys.path.insert(0, str(repo_root))

from roadscribe.config import RoadScribeConfig
from roadscribe.io.pvs import load_pvs_trip, pvs_root, route_of, vehicle_of
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame
from roadscribe.layer1.iso2631 import weight_vehicle_axes
from roadscribe.layer1.events import detect_events
from roadscribe.layer2.distance import build_grid
from roadscribe.layer2.mipelt import segment
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus
from roadscribe.layer3.ledger import TripMeta, compile_ledger
from roadscribe.layer3.kst import build_capsule, RegexTokenizer

def ingest_all_pvs_trips():
    out_dir = repo_root / "data" / "processed" / "pvs_capsules"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = RoadScribeConfig()
    tok = RegexTokenizer()
    
    summary = []
    
    print(f"Starting ingestion of all 9 PVS trips from: {pvs_root()}...")
    for trip_id in range(1, 10):
        print(f"\n--- Processing Trip {trip_id} (Route {route_of(trip_id)}, Vehicle {vehicle_of(trip_id)}) ---")
        raw = load_pvs_trip(pvs_root(), trip_id, side="left", with_labels=True)
        dur_min = (raw.t[-1] - raw.t[0]) / 60.0
        
        # Layer 1
        vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
        aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
        det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
        
        # Layer 2
        grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
        res, beta = segment(grid.Y, grid.s, [], [], cfg.mipelt)
        end = float(vft.s[-1])
        bounds = arbitrate(res.changepoints, grid.s, grid.Y, [], cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
        units = build_hrnus(bounds, vft, aw, det.events, [], grid_s=grid.s, grid_Y=grid.Y,
                            iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
        
        # Layer 3
        meta = TripMeta(raw.trip_id, duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=0.0)
        ledger = compile_ledger(units, det.events, meta, cfg.ledger)
        kst = build_capsule(units, det.events, meta, 128, tok, cfg.kst, ledger=ledger)
        
        cap = kst.capsule
        
        # Save individual trip artifact
        trip_file = out_dir / f"pvs_{trip_id}.json"
        trip_data = {
            "trip_id": raw.trip_id,
            "route_id": route_of(trip_id),
            "vehicle_id": vehicle_of(trip_id),
            "dur_min": round(dur_min, 2),
            "path_km": round(end / 1000.0, 2),
            "n_samples": raw.n,
            "n_events": len(det.events),
            "n_units": len(units),
            "capsule_text": cap.text,
            "capsule_tokens": cap.n_tokens,
            "units": [
                {
                    "unit_id": u.unit_id,
                    "s0_km": round(u.s0 / 1000.0, 2),
                    "s1_km": round(u.s1 / 1000.0, 2),
                    "iso_felt": u.iso_felt,
                    "shock": u.shock,
                    "a_v": round(float(u.a_v), 3),
                    "event_ids": u.event_ids
                }
                for u in units
            ]
        }
        with open(trip_file, "w", encoding="utf-8") as f:
            json.dump(trip_data, f, indent=2)
            
        print(f"  Processed: {dur_min:.1f} min | {end/1000.0:.2f} km | {len(det.events)} events | {len(units)} HRNUs")
        print(f"  Capsule Tokens: {cap.n_tokens}")
        summary.append({
            "trip_id": raw.trip_id,
            "route": route_of(trip_id),
            "vehicle": vehicle_of(trip_id),
            "duration_min": round(dur_min, 1),
            "distance_km": round(end / 1000.0, 2),
            "events_detected": len(det.events),
            "units": len(units),
            "tokens": cap.n_tokens
        })
        
    summary_file = out_dir / "pvs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nAll 9 PVS trips successfully ingested into: {out_dir}")
    print(f"Summary written to: {summary_file}")

if __name__ == "__main__":
    ingest_all_pvs_trips()
