"""Kinetic-Spatial Tokenizer: exact multiple-choice knapsack and capsule assembly (§5.2)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from roadscribe.config import KSTConfig
from roadscribe.layer3.kst import (
    CallableTokenizer,
    RegexTokenizer,
    build_capsule,
    event_severity,
    mckp,
    must_mention,
    rest_line,
    salience,
    serialize_unit,
)
from roadscribe.layer3.ledger import Anchor, TripMeta
from synth import make_event, make_hrnu


def _brute_force(costs, gains, allowed, B):
    best, arg = -np.inf, None
    M, L = costs.shape
    for combo in itertools.product(range(L), repeat=M):
        if not all(allowed[m, c] for m, c in enumerate(combo)):
            continue
        c = sum(costs[m, l] for m, l in enumerate(combo))
        if c <= B:
            g = sum(gains[m, l] for m, l in enumerate(combo))
            if g > best + 1e-12:
                best, arg = g, combo
    return arg, best


def test_mckp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = int(rng.integers(1, 7))
        brief = rng.integers(5, 20, M)
        full = brief + rng.integers(5, 30, M)
        costs = np.column_stack([np.zeros(M, int), brief, full])
        sig = rng.uniform(0.1, 3, M)
        gains = sig[:, None] * np.array([0.0, 0.5, 1.0])
        allowed = np.ones((M, 3), bool)
        allowed[rng.random(M) < 0.3, 0] = False
        B = int(rng.integers(0, 150))
        choice, val = mckp(costs, gains, allowed, B)
        ref, ref_val = _brute_force(costs, gains, allowed, B)
        if ref is None:
            assert choice is None
            continue
        assert val == pytest.approx(ref_val)
        assert sum(costs[m, c] for m, c in enumerate(choice)) <= B
        assert all(allowed[m, c] for m, c in enumerate(choice))


def test_mckp_infeasible_and_edge_cases():
    costs = np.array([[0, 10, 20]])
    gains = np.array([[0.0, 1.0, 2.0]])
    forced = np.array([[False, True, True]])
    assert mckp(costs, gains, forced, 9)[0] is None
    assert mckp(costs, gains, forced, 10)[0].tolist() == [1]
    assert mckp(costs, gains, np.ones((1, 3), bool), 25)[0].tolist() == [2]
    assert mckp(costs, gains, np.ones((1, 3), bool), 0)[0].tolist() == [0]


def test_regex_tokenizer():
    tok = RegexTokenizer()
    assert tok.count("") == 0
    assert tok.count("U2 1.9-2.6km Road B") == len(["U", "2", "1", ".", "9", "-", "2", ".", "6", "km", "Road", "B"])
    assert tok.count("12345") == 2  # digit groups of up to three
    assert CallableTokenizer(lambda s: s.split(), "ws").count("a b c") == 3
    assert CallableTokenizer(lambda s: 7, "const").count("xyz") == 7


@pytest.fixture
def trip():
    events = [make_event(1, 2100.0, 0.93, mtvv=1.8), make_event(2, 2400.0, 0.81, mtvv=1.3),
              make_event(3, 3000.0, 0.55, "ambiguous", mtvv=0.6, maneuver="stop_go")]
    units = [make_hrnu(1, 0, 1900, a_v=0.25)]
    units.append(make_hrnu(2, 1900, 2600, a_v=1.62, a_wz_dec=0.70, shock=True, event_ids=["E1", "E2"],
                           conf={"map_match": 0.95, "surface_pred": "cobble"}, osm_tags={}))
    units.append(make_hrnu(3, 2600, 4000, a_v=0.9, event_ids=["E3"], osm_tags={"surface": "unpaved"}))
    for i in range(4, 9):
        units.append(make_hrnu(i, 4000 + (i - 4) * 1500, 5500 + (i - 4) * 1500, a_v=0.2 + 0.02 * i))
    meta = TripMeta("pvs4", duration_s=1326.0, length_m=11500.0, map_coverage=0.6)
    return units, events, meta


def test_salience_and_must_mention(trip):
    units, events, _ = trip
    sig = salience(units, events)
    assert int(np.argmax(sig)) == 1  # U2: felt 4 with two strong bumps
    assert event_severity(events[0]) == 1 + 4  # MTVV 1.8 -> comfort class 4
    # U2: 0.93 * 5 = 4.65; U3: 0.55 * sev(MTVV 0.6 -> class 2) = 0.55 * 3 = 1.65
    assert must_mention(units, events, KSTConfig(eta_star=2.0)) == {1}
    assert must_mention(units, events, KSTConfig(eta_star=1.6)) == {1, 2}
    assert must_mention(units, events, KSTConfig(eta_star=10.0)) == set()


def test_serializations(trip):
    units, events, _ = trip
    u2 = units[1]
    full = serialize_unit(u2, "FULL", events[:2], Anchor("School 1", 2300.0))
    assert full == "U2 1.9-2.6km Road B | cobble (osm -) | felt 4 SHOCK | road 2 | bump x2 p.93,.81 | near School 1 @2.3km | mm .95"
    assert serialize_unit(u2, "BRIEF", events[:2]) == "U2 1.9-2.6km Road B | felt 4 SHOCK | bump x2"
    assert serialize_unit(u2, "OMIT", events[:2]) == ""
    u3 = serialize_unit(units[2], "FULL", events[2:])
    assert "jolt x1 p.55 stop-go" in u3 and "osm unpaved" in u3
    assert rest_line(units[3:], events) == "REST 5 units, 7.5 km, asphalt, felt 0-1, no events"
    assert rest_line([], events) == ""


@pytest.mark.parametrize("B", [64, 128, 256, 512])
def test_capsule_respects_budget_and_must_mention(trip, B):
    units, events, meta = trip
    tok = RegexTokenizer()
    res = build_capsule(units, events, meta, budget=B, tokenizer=tok)
    cap = res.capsule
    assert cap.budget == B and cap.tokenizer == tok.name
    assert set(cap.selection) == {u.unit_id for u in units}
    assert all(cap.selection[u] != "OMIT" for u in res.must_mention)
    assert cap.text.startswith("TRIP pvs4 | 22.1 min | 11.5 km | persona=commuter")
    assert cap.n_tokens == tok.count(cap.text)
    if not res.budget_increased:
        assert cap.n_tokens <= B
    lines = cap.text.splitlines()
    shown = [l.split()[0] for l in lines[1:] if l.startswith("U")]
    assert shown == sorted(shown, key=lambda x: int(x[1:]))  # route order
    if "OMIT" in cap.selection.values():
        assert lines[-1].startswith("REST ")
    assert cap.ledger and cap.ledger[0].kind == "META"


def test_budget_monotone_in_detail(trip):
    units, events, meta = trip
    levels = {"OMIT": 0, "BRIEF": 1, "FULL": 2}
    detail = [sum(levels[v] for v in build_capsule(units, events, meta, budget=B).capsule.selection.values())
              for B in (64, 128, 256, 512, 2048)]
    assert detail == sorted(detail)
    assert set(build_capsule(units, events, meta, budget=4096).capsule.selection.values()) == {"FULL"}


def test_infeasible_budget_is_raised_and_logged(trip, caplog):
    units, events, meta = trip
    res = build_capsule(units, events, meta, budget=10)
    assert res.budget_increased and res.budget_effective > 10
    assert res.capsule.n_tokens <= res.budget_effective
    assert all(res.capsule.selection[u] == "BRIEF" for u in res.must_mention)
    assert any("infeasible" in m for m in res.log)


def test_capsule_is_deterministic(trip):
    units, events, meta = trip
    a = build_capsule(units, events, meta, budget=128).capsule
    b = build_capsule(units, events, meta, budget=128).capsule
    assert a.text == b.text and a.capsule_id == b.capsule_id


def test_joined_text_overflow_triggers_refit(trip):
    """A tokenizer that charges for newlines makes the joined text cost more than its parts."""
    units, events, meta = trip
    tok = CallableTokenizer(lambda s: RegexTokenizer().count(s) + 3 * s.count("\n"), "newline-heavy")
    res = build_capsule(units, events, meta, budget=128, tokenizer=tok)
    assert res.capsule.n_tokens <= 128 or res.budget_increased


def test_trip_level_missing_lines_in_header(trip):
    units, events, _ = trip
    meta = TripMeta("t1", channels=("acc", "gps"))
    text = build_capsule(units, events, meta, budget=512).capsule.text
    assert "MISSING gyro (no_gyro)" in text


def test_trip_without_units():
    res = build_capsule([], [], TripMeta("empty"), budget=64)
    assert res.capsule.selection == {} and res.capsule.text.startswith("TRIP empty")
    assert not res.budget_increased
