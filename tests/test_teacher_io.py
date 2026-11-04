"""Teacher I/O: committed teacher narratives and answers must keep verifying."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from roadscribe.layer3.plan_grammar import validate_plan
from roadscribe.synth.teacher_io import RAW, TEACHER_PERSONAS, ingest, sheet, teacher_trip
from roadscribe.synth.teacher import visible_units

logging.getLogger("roadscribe.layer3.kst").setLevel(logging.WARNING)


def test_teacher_trips_use_only_dev_personas():
    assert {teacher_trip("sft", i).persona for i in range(6)} == set(TEACHER_PERSONAS)
    assert "car_sick_passenger" not in TEACHER_PERSONAS and "fleet_manager" not in TEACHER_PERSONAS
    assert sheet("sft", 0).startswith("=== syn-t0000 |")


def test_committed_teacher_outputs_verify(tmp_path):
    if not any(Path(RAW).glob("*.txt")):
        return
    summary = ingest(RAW, tmp_path, quiet=True)
    assert summary["failed"] == 0
    for line in (tmp_path / "roadscribe_teacher_sft.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = teacher_trip("sft", int(r["id"][-4:]))
        assert validate_plan(json.loads(r["plan"]), t.ledger, visible_units(t)) == []
        assert not r["e2_test_persona"]
