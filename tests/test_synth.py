"""Synthetic sampler, template teacher, plan grammar, DPO perturbations, QA and the dataset build."""
from __future__ import annotations

import json
import logging
import random
import re

import pytest

from roadscribe.eval.claims import _ROADNAME, verify_text
from roadscribe.layer1.iso2631 import comfort_class
from roadscribe.layer3.lexicon import FELT_WORDS, ROAD_WORDS, contains_lexicon_word
from roadscribe.layer3.ledger import LedgerView
from roadscribe.layer3.personas import E2_TEST_PERSONAS, PERSONAS
from roadscribe.layer3.plan_grammar import canonical_json, plan_gbnf, validate_plan
from roadscribe.layer3.templates import ground, realize, rule_plan
from roadscribe.synth.build import main as build_main
from roadscribe.synth.formats import to_chat
from roadscribe.synth.perturb import EXPECTED_REASONS, PERTURBATIONS, perturb
from roadscribe.synth.qa import candidate_questions, qa_record
from roadscribe.synth.sampler import POI_NAMES, ROAD_NAMES, sample_trip
from roadscribe.synth.teacher import teach, visible_units

logging.getLogger("roadscribe.layer3.kst").setLevel(logging.WARNING)


@pytest.fixture(scope="module")
def trips():
    return [sample_trip(random.Random(f"test:{i}"), i) for i in range(160)]


# --------------------------------------------------------------------------- lexicon and name banks
def test_lexicon_classes_are_disjoint_and_names_are_clean():
    felt = [w for ws in FELT_WORDS.values() for w in ws]
    road = [w for ws in ROAD_WORDS.values() for w in ws]
    assert len(set(felt)) == len(felt) and not set(felt) & set(road)
    for name in ROAD_NAMES + POI_NAMES:
        assert not contains_lexicon_word(name), name
        assert not re.search(r"\d", name)
    assert all(_ROADNAME.fullmatch(n) for n in ROAD_NAMES)
    assert not any(_ROADNAME.search(n) for n in POI_NAMES)
    assert E2_TEST_PERSONAS == {"car_sick_passenger", "fleet_manager"}


# --------------------------------------------------------------------------- sampler
def test_sampler_is_deterministic():
    a = sample_trip(random.Random("x"), 7)
    b = sample_trip(random.Random("x"), 7)
    assert a.capsule.text == b.capsule.text and a.ledger == b.ledger


def test_sampler_physics_is_consistent(trips):
    for t in trips:
        assert max(u.iso_felt for u in t.units) == t.regime or t.regime > 0
        for u in t.units:
            assert u.iso_felt == comfort_class(u.a_v)
            assert u.a_wz < u.a_v
            if u.v_mean < 5.0:
                assert u.a_wz_dec is None and u.iso_road is None
            else:
                assert u.iso_road == comfort_class(u.a_wz_dec)
            evs = [e for e in t.events if e.event_id in u.event_ids]
            assert all(u.s0 <= e.s < u.s1 for e in evs)
            assert u.shock == any(e.label == "defect" and e.p_defect >= 0.7 and e.mtvv_z > 1.5 * u.a_wz for e in evs)
        assert sum(len(u.event_ids) for u in t.units) == len(t.events)
        assert t.capsule.persona == t.persona and t.capsule.text.startswith(f"TRIP {t.trip_id} |")


def test_sampler_covers_regimes_personas_and_degradations(trips):
    assert {t.regime for t in trips} == set(range(6))
    assert {t.persona for t in trips} == set(PERSONAS)
    assert {t.task_mix for t in trips} == {"narrative", "degraded", "history"}
    surfaces = {u.conf["surface_pred"] for t in trips for u in t.units}
    assert surfaces == {"asphalt", "cobble", "unpaved"}
    pats = {p for t in trips for p in t.patterns}
    assert len(pats) >= 5


def test_degradations_reach_the_ledger(trips):
    for t in trips:
        L = LedgerView(t.ledger)
        if "no_gyro" in t.patterns:
            assert L.missing(None).get("gyro") == "no_gyro"
            assert not any(e.kind == "avoidance" for e in t.events)
        if "gps_gap" in t.patterns:
            assert any(n.startswith("gps_gap_") for u in L.units for n in L.missing(u).values())
        if "no_map" in t.patterns:
            assert L.missing(None).get("map_coverage") == "no_map"
        if t.history:
            assert any(f.key == "a_v_vs_history" for f in t.ledger)


# --------------------------------------------------------------------------- teacher and grammar
def test_teacher_outputs_all_verify(trips):
    for i, t in enumerate(trips):
        out = teach(t, seed=i)
        assert out.ok, (t.trip_id, out.report.reasons, out.plan_errors, out.text)
        assert not out.report.must_mention_missed
        assert out.report.hallucinated_missing == 0


def test_template_condition_is_deterministic(trips):
    t = trips[0]
    plan = rule_plan(t.ledger, t.capsule.selection, t.persona, rng=None)
    frames = ground(plan, t.ledger)
    assert realize(frames, t.persona, vary=False) == realize(frames, t.persona, vary=False)
    assert verify_text(realize(frames, t.persona, vary=False), t.ledger, omitted_units=plan["omit"]).passed


def test_plan_validation_and_grammar(trips):
    t = next(x for x in trips if len(visible_units(x)) >= 2)
    out = teach(t, seed=1)
    vis = visible_units(t)
    assert validate_plan(out.plan, t.ledger, vis) == []
    js = canonical_json(out.plan)
    assert js.startswith('{"intent":"trip_summary","items":[') and json.loads(js)["omit"] == out.plan["omit"]
    bad = json.loads(js)
    bad["items"][-1]["facts"] = ["F99999"]
    assert validate_plan(bad, t.ledger, vis)
    other = next(f.fact_id for f in t.ledger if f.unit_id not in (None, bad["items"][0]["ref"]))
    bad2 = json.loads(js)
    bad2["items"][0]["facts"] = [other]
    assert any("do not belong" in e for e in validate_plan(bad2, t.ledger, vis))
    g = plan_gbnf(t.ledger, vis)
    assert g.startswith("root ::=") and all(f'\\"ref\\":\\"{u}\\"' in g for u in vis)


# --------------------------------------------------------------------------- DPO
@pytest.mark.parametrize("kind", list(PERTURBATIONS))
def test_perturbations_are_rejected_for_the_intended_reason(trips, kind):
    made = 0
    for i, t in enumerate(trips):
        ch = teach(t, seed=i)
        p = perturb(kind, t, ch, random.Random(f"{kind}:{i}"))
        if p is None:
            continue
        made += 1
        assert p.text != ch.text and not p.report.passed
        assert set(p.report.reasons) & EXPECTED_REASONS[kind]
        if made >= 5:
            break
    assert made >= 3, f"{kind}: too few applicable trips"


# --------------------------------------------------------------------------- QA
def test_qa_gold_answers_verify(trips):
    types = set()
    for i, t in enumerate(trips[:80]):
        for item in candidate_questions(t, random.Random(i)):
            rec = qa_record(t, item, "q", "h", i)
            assert rec is not None, (item.qtype, item.answer)
            types.add(item.qtype)
            if not item.answerable:
                assert item.answer.startswith("I can't tell that from this drive") and item.abstain_reason
    assert {"roughest", "unit_events", "out_of_scope", "history_missing"} <= types


# --------------------------------------------------------------------------- build
def test_build_writes_balanced_verified_files(tmp_path):
    m = build_main(["--out", str(tmp_path), "--n-sft", "24", "--n-dpo", "16", "--n-qa", "15", "--seed", "3"])
    rows = {name: [json.loads(x) for x in (tmp_path / f"roadscribe_{name}.jsonl").read_text().splitlines()]
            for name in ("sft", "dpo", "qa_bank")}
    assert len(rows["sft"]) == 24 and len(rows["dpo"]) == 16 and len(rows["qa_bank"]) == 15
    assert {r["persona"] for r in rows["sft"]} == set(PERSONAS)
    assert all(r["verification"]["passed"] for r in rows["sft"])
    assert {r["perturbation"] for r in rows["dpo"]} == set(PERTURBATIONS)
    assert all(not r["rejected_verification"]["passed"] and r["chosen"] != r["rejected"] for r in rows["dpo"])
    assert sum(not r["answerable"] for r in rows["qa_bank"]) == 3
    assert (tmp_path / "DATASET_CARD.md").read_text().startswith("# RoadScribe synthetic datasets")
    assert m["stats"]["sft"]["n"] == 24
    chat = to_chat(rows["sft"][0], "plan")["messages"]
    assert chat[-1]["content"] == rows["sft"][0]["plan"] and "CAPSULE" in chat[1]["content"]
    assert to_chat(rows["sft"][0], "realize")["messages"][-1]["content"] == rows["sft"][0]["narrative"]
    d = to_chat(rows["dpo"][0], "dpo")
    assert d["chosen"][0]["content"] != d["rejected"][0]["content"]
    assert to_chat(rows["qa_bank"][0], "qa")["messages"][-1]["content"] == rows["qa_bank"][0]["gold_answer"]
    again = build_main(["--out", str(tmp_path / "b"), "--n-sft", "24", "--n-dpo", "16", "--n-qa", "15", "--seed", "3"])
    assert (tmp_path / "b" / "roadscribe_sft.jsonl").read_text() == (tmp_path / "roadscribe_sft.jsonl").read_text()
    assert again["config_hash"] == m["config_hash"]


def test_build_can_exclude_e2_test_personas(tmp_path):
    build_main(["--out", str(tmp_path), "--n-sft", "8", "--n-dpo", "8", "--n-qa", "5", "--exclude-e2-test-personas"])
    rows = [json.loads(x) for x in (tmp_path / "roadscribe_sft.jsonl").read_text().splitlines()]
    assert {r["persona"] for r in rows} == {"commuter", "sports_car"}
    assert not any(r["e2_test_persona"] for r in rows)
