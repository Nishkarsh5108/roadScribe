"""Shared fixtures. PVS-dependent tests run only when ROADSCRIBE_PVS_DIR is set."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from roadscribe.io.pvs import PVS_ENV


@pytest.fixture(scope="session")
def pvs_root() -> Path:
    p = os.environ.get(PVS_ENV)
    if not p:
        pytest.skip(f"PVS dataset not available (set {PVS_ENV} to the folder holding 'PVS 1' ... 'PVS 9')")
    root = Path(p)
    missing = [i for i in range(1, 10) if not (root / f"PVS {i}" / "dataset_gps_mpu_left.csv").exists()]
    if missing:
        pytest.skip(f"{PVS_ENV}={root} is missing trips {missing}")
    return root
