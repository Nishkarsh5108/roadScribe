"""Closed-vocabulary claim extraction and ledger verification (§6.1-6.4)."""
from __future__ import annotations

import pytest

from roadscribe.eval.claims import verify_text
from roadscribe.layer3.ledger import Anchor, MissingField, TripMeta, compile_ledger
from synth import make_event, make_hrnu


@pytest.fixture(scope="module")
def ledger():
    events = [
        make_event(1, 2100.0, 0.93, mtvv=1.8),
        make_event(2, 2400.0, 0.81, mtvv=1.3, maneuver="brake"),
        make_event(3, 3000.0, 0.55, "ambiguous", mtvv=0.6, maneuver="stop_go"),
    ]
    units = [
        make_hrnu(1, 0, 1900, a_v=0.25, v_mean=14.0),
        make_hrnu(2, 1900, 2600, a_v=1.62, a_wz_dec=0.70, shock=True, event_ids=["E1", "E2"], names=["Road B"],
                  osm_tags={"surface": "asphalt", "name": "Road B"}, map_signal_disagreement=0.81,
                  conf={"map_match": 0.95, "surface_pred": "cobble"}, v_mean=16.0),
        make_hrnu(3, 2600, 4000, a_v=0.9, a_wz_dec=None, event_ids=["E3"], names=["Road B"], v_mean=4.0),
        make_hrnu(4, 4000, 4700, a_v=0.2, names=[], osm_way_ids=[]),
    ]
    trip = TripMeta("t1", duration_s=1320.0, length_m=4700.0, map_coverage=0.8)
    return compile_ledger(units, events, trip, anchors=[Anchor("Fire Station", 2300.0)],
                          missing=[MissingField("U4", "anchor", "gps_gap_48s")])


def check(text, ledger, **kw):
    return verify_text(text, ledger, **kw)


def test_correct_text_passes(ledger):
    text = (
        "Here is your 22-minute, 4.7 km drive. Road A felt comfortable. "
        "Road B (1.9–2.6 km) felt very uncomfortable mostly because of speed: at about 58 km/h you were quick "
        "for a surface that is in fair condition. It was shock-dominated, with a peak 1-s exposure of 1.8 m/s². "
        "It rides like cobblestones, although the map lists it as asphalt. "
        "There was a sharp bump around the 2.1 km mark near Fire Station. There was probably a bump as you braked. "
        "Road B (2.6–4.0 km) was smoother than Road B (1.9–2.6 km) and felt uncomfortable. "
        "A jolt may have come from stop-and-go driving rather than from the road. "
        "You were moving too slowly there to separate the road from your speed. "
        "The stretch from 4.0 to 4.7 km felt calm. A 48-second GPS gap means its exact position is uncertain."
    )
    rep = check(text, ledger, must_mention_events=["E1"])
    assert rep.passed, [(c.kind, c.key, c.reason, text[c.span[0]:c.span[1]]) for c in rep.claims if not c.supported]
    kinds = {c.kind for c in rep.claims}
    assert {"SEV", "NUM", "EVT", "NAME", "RANGE", "ANCHOR", "CMP", "MISSING_ACK"} <= kinds
    assert rep.p_ledger == 1.0 and rep.event_mentions == 3 and set(rep.matched_events) == {"E1", "E2", "E3"}


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Road A felt very rough.", "severity_mismatch"),
        ("Road A felt a little uncomfortable.", "severity_mismatch"),  # 0.25 is outside the 0.315-0.63 band
        ("Road A felt comfortable at about 80 km/h.", "number_mismatch"),
        ("Road A (1.2 km) felt comfortable.", "number_mismatch"),
        ("Road A felt comfortable. There was a sharp bump.", "fabricated_event"),
        ("Road B (1.9–2.6 km) felt very uncomfortable. There were probably three bumps.", "event_count"),
        ("Road B (2.6–4.0 km) felt uncomfortable. There was definitely a jolt.", "overclaimed_hedge"),
        ("Road B (1.9–2.6 km) felt very uncomfortable. There was a bump around the 2.1 km mark as you braked.", "misattributed_maneuver"),
        ("Road B (1.9–2.6 km) felt very uncomfortable. There was a bump around the 2.5 km mark.", "wrong_position"),
        ("Road B (2.6–4.0 km) felt uncomfortable, and the surface itself is in poor condition.", "hallucinated_missing"),
        ("Road B (2.6–4.0 km) felt uncomfortable mostly because of speed.", "hallucinated_missing"),
        ("Mill Road (4.0–4.7 km) felt calm.", "hallucinated_missing"),
        ("Road B (5.0–5.5 km) felt calm.", "unknown_range"),
        ("Road B felt calm.", "ambiguous_reference"),
        ("It felt calm.", "unresolved"),
        ("Road A was rougher than Road B (1.9–2.6 km) and felt comfortable.", "comparison_mismatch"),
        ("Road A felt comfortable. That is rougher than usual on this route.", "no_history"),
        ("Road A felt comfortable near Fire Station.", "wrong_anchor"),
        ("Road B (1.9–2.6 km) felt very uncomfortable. It rides like asphalt, although the map lists it as sett.", "wrong_surface"),
        ("Road A felt comfortable for 7 seconds.", "unparsed_number"),
        ("Road A was the roughest stretch and felt comfortable.", "comparison_mismatch"),
        ("Road A felt comfortable. It was jarring in places.", "severity_mismatch"),
    ],
)
def test_errors_are_caught(ledger, text, reason):
    rep = check(text, ledger)
    assert not rep.passed
    assert reason in rep.reasons, rep.reasons


def test_must_mention_and_rest(ledger):
    rep = check("Road A felt comfortable.", ledger, must_mention_events=["E1"])
    assert rep.must_mention_missed == ["E1"] and "missed_must_mention" in rep.reasons
    ok = check("Road A felt comfortable. The other stretch was calm, with no bumps or jolts.", ledger, omitted_units=["U4"])
    assert ok.passed
    bad = check("The other two stretches were calm, with no bumps or jolts.", ledger, omitted_units=["U4"])
    assert "rest_count" in bad.reasons
    bad2 = check("The other two stretches ranged from calm to uncomfortable, with no bumps or jolts.", ledger,
                 omitted_units=["U3", "U4"])
    assert "event_count" in bad2.reasons  # U3 holds a mentionable jolt


def test_rest_units_inferred_without_plan(ledger):
    # without omitted_units, the REST sentence summarises the units no other sentence describes (D26)
    def n_units_ok(text):
        rep = check(text, ledger)
        return [c.supported for c in rep.claims if c.kind == "REST" and c.key == "n_units"]
    assert n_units_ok("Road A felt comfortable. The other three stretches were calm.") == [True]
    assert n_units_ok("Road A felt comfortable. The other two stretches were calm.") == [False]


@pytest.mark.parametrize("words", ["seven", "long", "the"])
def test_rest_count_slot_never_crashes(ledger, words):
    # a count above six or a word outside the lexicon fails the REST claim; it must not raise
    rep = check(f"The other {words} stretches were calm, with no bumps or jolts.", ledger, omitted_units=["U4"])
    assert "rest_count" in rep.reasons


def test_precision_aware_numbers(ledger):
    # 4.7 km trip shown with one decimal; 22 minutes is 1320 s
    assert check("Here is your 22-minute, 4.7 km drive.", ledger).passed
    assert not check("Here is your 30-minute, 4.7 km drive.", ledger).passed
    assert not check("Here is your 22-minute, 6.1 km drive.", ledger).passed


def test_report_lines(ledger):
    text = ("Trip report: 4.7 km drive, 22 minutes.\n"
            "- Road B (1.9–2.6 km), 700 m: ride very uncomfortable (1.6 m/s²); shock-dominated, peak 1-s exposure 1.8 m/s²; "
            "there was a hard bump around the 2.1 km mark; there was probably a bump around the 2.4 km mark.\n"
            "- Road B (2.6–4.0 km), 1.4 km: ride uncomfortable; surface not rated (speed too low).")
    rep = check(text, ledger)
    assert rep.passed, [(c.kind, c.reason) for c in rep.claims if not c.supported]


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Road A felt calm, and Road B (1.9–2.6 km) felt very uncomfortable.", True),  # compound sentence
        ("Road A and the stretch from 4.0 to 4.7 km both felt calm.", True),  # coordinated subjects
        ("Road A and Road B (1.9–2.6 km) both felt calm.", False),  # coordination checks every subject
        ("The roughest stretch was Road B (1.9–2.6 km), which felt very uncomfortable.", True),
        ("Road A felt calm, but on Road B (1.9–2.6 km) there was a sharp bump around the 2.1 km mark.", True),
    ],
)
def test_claims_attach_to_the_stretch_they_follow(ledger, text, ok):
    assert check(text, ledger).passed is ok
