"""End to end on a synthetic trip: RawTrip -> vehicle frame -> events -> MI-PELT -> HRNUs -> ledger -> capsule."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import signal

from roadscribe.config import RoadScribeConfig
from roadscribe.layer1.events import detect_events
from roadscribe.layer1.gnss import along_track_distance
from roadscribe.layer1.iso2631 import weight_vehicle_axes
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame
from roadscribe.layer2.arbitrate import MapTransition, WaySpan, arbitrate, build_hrnus, transition_weights
from roadscribe.layer2.distance import build_grid
from roadscribe.layer2.mipelt import segment
from roadscribe.layer3.kst import RegexTokenizer, build_capsule
from roadscribe.layer3.ledger import Anchor, TripMeta, compile_ledger
from synth import bump, make_trip

ROUGH = (3000.0, 4500.0)
BUMPS_S = (1500.0, 2200.0)


@pytest.fixture(scope="module")
def pipeline():
    cfg = RoadScribeConfig()
    st = make_trip(seed=7, duration=600.0, vib_sd=0.25)
    raw = st.raw
    fs = raw.fs
    s_true = along_track_distance(raw.t, st.v)
    rng = np.random.default_rng(8)
    sos = signal.butter(2, [2 / (fs / 2), 30 / (fs / 2)], "band", output="sos")
    extra = np.zeros((raw.n, 3))
    rough = (s_true >= ROUGH[0]) & (s_true < ROUGH[1])
    extra[rough, 2] = signal.sosfiltfilt(sos, rng.normal(0, 1, raw.n))[rough] * 1.2
    b = bump(fs, amp=8.0)
    for sb in BUMPS_S:
        i = int(np.searchsorted(s_true, sb))
        extra[i : i + len(b), 2] += b
    raw = dataclasses.replace(raw, f_b=raw.f_b + extra @ st.R_vb)  # vehicle -> body is R_vb^T

    vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
    aw = weight_vehicle_axes(vft.a_v, fs, cfg.iso)
    det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
    grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
    end = float(vft.s[-1])
    transitions = [MapTransition(1000.0, "junction"), MapTransition(ROUGH[0], "surface"), MapTransition(ROUGH[1], "surface")]
    res, beta = segment(grid.Y, grid.s, [b.s for b in transitions], transition_weights(transitions, cfg.mipelt), cfg.mipelt)
    bounds = arbitrate(res.changepoints, grid.s, grid.Y, transitions, cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
    ways = [
        WaySpan(0.0, ROUGH[0], 1, {"name": "Road A", "surface": "asphalt"}),
        WaySpan(ROUGH[0], ROUGH[1], 2, {"name": "Road B", "surface": "sett"}),
        WaySpan(ROUGH[1], end + 1, 3, {"name": "Road C", "surface": "asphalt"}),
    ]
    units = build_hrnus(bounds, vft, aw, det.events, ways, grid_s=grid.s, grid_Y=grid.Y,
                        map_match_conf=np.full(vft.n, 0.97), iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
    meta = TripMeta(raw.trip_id, duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=1.0,
                    orientation_yaw_ci_deg=float(np.diff(vft.yaw_ci_deg[0])[0]))
    anchors = [Anchor("School 1", 2150.0)]
    ledger = compile_ledger(units, det.events, meta, cfg.ledger, anchors=anchors)
    kst = build_capsule(units, det.events, meta, 128, RegexTokenizer(), cfg.kst, ledger=ledger, anchors=anchors)
    return dict(vft=vft, det=det, grid=grid, beta=beta, bounds=bounds, units=units, ledger=ledger, kst=kst)


def test_bumps_detected_at_the_right_place(pipeline):
    ev = [e for e in pipeline["det"].events if e.kind == "impulse"]
    for sb in BUMPS_S:
        assert any(abs(e.s - sb) < 10 for e in ev), sb
    assert all(np.isfinite(e.lat) and np.isfinite(e.lon) for e in ev)


def test_rough_stretch_boundaries_snap_to_the_map(pipeline):
    bounds = pipeline["bounds"]
    for edge in ROUGH:
        near = [b for b in bounds if abs(b.s - edge) <= 25.0]
        assert near and near[0].kind == "map+signal" and near[0].s == edge, (edge, bounds)
    assert pipeline["beta"].min() < pipeline["beta"].max()


def test_units_describe_the_road(pipeline):
    units = pipeline["units"]
    rough = [u for u in units if u.s0 >= ROUGH[0] - 1 and u.s1 <= ROUGH[1] + 1]
    smooth = [u for u in units if u.s1 <= ROUGH[0] + 1 or u.s0 >= ROUGH[1] - 1]
    assert rough and smooth
    assert np.median([u.a_v for u in rough]) > 2.5 * np.median([u.a_v for u in smooth])
    assert min(u.iso_felt for u in rough) >= 2 > np.median([u.iso_felt for u in smooth])
    assert rough[0].osm_tags["surface"] == "sett" and rough[0].names == ["Road B"]
    for sb in BUMPS_S:  # the units holding the bumps are shock-dominated
        assert next(u for u in units if u.s0 <= sb < u.s1).shock
    assert all(u.s1 - u.s0 >= 50.0 for u in units)
    assert [u.unit_id for u in units] == [f"U{k + 1}" for k in range(len(units))]
    ids = [i for u in units for i in u.event_ids]
    assert sorted(ids) == sorted(e.event_id for e in pipeline["det"].events)


def test_ledger_and_capsule(pipeline):
    ledger, kst, units = pipeline["ledger"], pipeline["kst"], pipeline["units"]
    evt = [f for f in ledger if f.kind == "EVT"]
    assert len(evt) == len(pipeline["det"].events)
    cap = kst.capsule
    assert cap.n_tokens <= 128 or kst.budget_increased
    assert set(cap.selection) == {u.unit_id for u in units}
    for uid in kst.must_mention:
        assert cap.selection[uid] != "OMIT"
    assert cap.text.splitlines()[0].startswith(f"TRIP {cap.trip_id} |")
    assert cap.ledger == ledger
