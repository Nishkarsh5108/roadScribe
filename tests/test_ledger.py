"""Claim ledger (§5.1)."""
from __future__ import annotations

import pytest

from roadscribe.config import LedgerConfig
from roadscribe.layer3.ledger import (
    Anchor,
    MissingField,
    TripMeta,
    compile_ledger,
    event_word,
    facts_by_unit,
    read_jsonl,
    write_jsonl,
)
from synth import make_event, make_hrnu


@pytest.fixture
def trip_units():
    e1, e2, e3 = make_event(1, 2100.0, 0.93), make_event(2, 2400.0, 0.81), make_event(3, 3000.0, 0.55, "ambiguous", maneuver="stop_go")
    units = [
        make_hrnu(1, 0, 1900, a_v=0.25),
        make_hrnu(2, 1900, 2600, a_v=1.62, a_wz_dec=0.70, shock=True, event_ids=["E1", "E2"],
                  osm_tags={"name": "Road B"}, map_signal_disagreement=0.81),
        make_hrnu(3, 2600, 4000, a_v=0.9, a_wz_dec=None, event_ids=["E3"], names=[], osm_way_ids=[]),
    ]
    trip = TripMeta("pvs4", duration_s=1326.0, length_m=13800.0, map_coverage=0.6, orientation_yaw_ci_deg=0.9)
    return units, [e1, e2, e3], trip


def test_fact_ids_and_kinds(trip_units):
    units, events, trip = trip_units
    L = compile_ledger(units, events, trip)
    assert [f.fact_id for f in L] == [f"F{k + 1}" for k in range(len(L))]
    assert {f.kind for f in L} == {"SEV", "NUM", "EVT", "ANCHOR", "CMP", "MISSING", "META"}
    assert compile_ledger(units, events, trip) == L  # deterministic


def test_severity_and_numbers(trip_units):
    units, events, trip = trip_units
    fb = facts_by_unit(compile_ledger(units, events, trip))
    u2 = {f.key: f for f in fb["U2"] if f.kind in ("SEV", "NUM")}
    assert u2["felt_class"].value == 4 and u2["felt_class"].note == "very uncomfortable"
    assert u2["road_class"].value == 2
    assert u2["a_v"].value == pytest.approx(1.62) and u2["a_v"].tol == pytest.approx(0.162)
    assert u2["shock"].value is True and "peak 1-s exposure" in u2["mtvv_z"].note
    assert "never map to comfort words" in u2["vdv_z"].note
    assert u2["n_events"].value == 2
    assert u2["map_signal_disagreement"].value == pytest.approx(0.81)
    u1 = {f.key: f for f in fb["U1"] if f.kind == "NUM"}
    assert u1["a_v"].tol == pytest.approx(LedgerConfig().eps_abs_acc)  # absolute floor


def test_missing_facts_carry_reasons(trip_units):
    units, events, trip = trip_units
    L = compile_ledger(units, events, trip, missing=[MissingField("U3", "anchor", "gps_gap_48s")])
    miss = {(f.unit_id, f.key): f.note for f in L if f.kind == "MISSING"}
    assert miss[("U3", "road_class")] == "speed_low"
    assert miss[("U3", "name")] == "mapmatch_failed"
    assert miss[("U3", "anchor")] == "gps_gap_48s"
    assert miss[("U1", "map_signal_disagreement")] == "no_surface_classifier"
    assert ("U2", "map_signal_disagreement") not in miss  # it has a value
    assert all(f.value is None for f in L if f.kind == "MISSING")
    trip_nogyro = TripMeta("t", channels=("acc", "gps"))
    L2 = compile_ledger(units, events, trip_nogyro)
    assert any(f.kind == "MISSING" and f.key == "gyro" and f.note == "no_gyro" for f in L2)
    assert any(f.kind == "MISSING" and f.key == "map_coverage" for f in L2)


def test_events_anchors_comparisons(trip_units):
    units, events, trip = trip_units
    L = compile_ledger(units, events, trip, anchors=[Anchor("School 1", 2300.0)])
    evt = [f for f in L if f.kind == "EVT"]
    assert [f.key for f in evt] == ["E1", "E2", "E3"]
    assert evt[0].conf == pytest.approx(0.93) and evt[0].value["type"] == "bump"
    assert evt[2].value["type"] == "jolt" and evt[2].value["maneuver"] == "stop_go"
    assert evt[0].tol == 25.0
    anchors = [(f.unit_id, f.key, f.value) for f in L if f.kind == "ANCHOR"]
    assert ("U1", "name", "Road A") in anchors
    assert ("U2", "near", {"name": "School 1", "s": 2300.0, "kind": "poi"}) in anchors
    cmp_ = {(f.unit_id, f.key): f.value["relation"] for f in L if f.kind == "CMP"}
    assert cmp_[("U2", "a_v_vs_prev")] == "rougher"
    assert cmp_[("U3", "a_v_vs_prev")] == "smoother"
    assert ("U1", "a_v_vs_prev") not in cmp_


def test_history_comparison(trip_units):
    units, events, trip = trip_units
    L = compile_ledger(units, events, trip, history={"U1": 0.25, "U2": 1.0})
    cmp_ = {(f.unit_id, f.key): f.value["relation"] for f in L if f.kind == "CMP"}
    assert cmp_[("U1", "a_v_vs_history")] == "similar"
    assert cmp_[("U2", "a_v_vs_history")] == "rougher"
    assert any(f.kind == "MISSING" and f.unit_id == "U3" and f.key == "history" for f in L)


def test_event_word():
    assert event_word(make_event(1, 0, label="defect")) == "bump"
    assert event_word(make_event(1, 0, label="artifact")) == "artifact"


def test_jsonl_roundtrip(tmp_path, trip_units):
    units, events, trip = trip_units
    L = compile_ledger(units, events, trip)
    p = tmp_path / "ledger.jsonl"
    write_jsonl(L, p)
    back = read_jsonl(p)
    assert [f.fact_id for f in back] == [f.fact_id for f in L]
    assert back[0] == L[0]
