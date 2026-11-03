"""Configuration (§8.2) and data contracts (§8.3)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from roadscribe.config import EventConfig, ISOConfig, RoadScribeConfig, config_hash
from roadscribe.schemas import HRNU, Capsule, Event, Fact, RawTrip, VehicleFrameTrip, to_jsonable
from synth import make_hrnu


# --------------------------------------------------------------------------- config
def test_config_is_frozen():
    cfg = RoadScribeConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.iso.cf_threshold = 3.0  # type: ignore[misc]


def test_methodology_defaults():
    cfg = RoadScribeConfig()
    o = cfg.orientation
    assert (o.eps_g, o.eps_omega_deg, o.eps_a, o.v_min) == (0.3, 3.0, 0.3, 3.0)
    assert (o.handling_omega_deg, o.handling_theta_deg, o.handling_guard_s) == (30.0, 10.0, 5.0)
    assert (cfg.iso.cf_threshold, cfg.iso.mtvv_ratio_threshold, cfg.iso.vdv_ratio_threshold) == (9.0, 1.5, 1.75)
    assert cfg.iso.pad_s == 10.0 and cfg.iso.mtvv_tau_s == 1.0
    assert (cfg.speed.v_ref, cfg.speed.v_min) == (10.0, 5.0)
    assert (cfg.mipelt.w_surface, cfg.mipelt.w_highway, cfg.mipelt.w_junction, cfg.mipelt.sigma_b_m) == (1.0, 0.6, 0.3, 15.0)
    assert (cfg.arbitration.snap_m, cfg.arbitration.l_min_m, cfg.arbitration.hotelling_alpha) == (25.0, 50.0, 0.01)
    assert (cfg.distance.ds_m, cfg.distance.window_m) == (5.0, 25.0)
    assert cfg.events.envelope_s == 0.05 and cfg.events.threshold_window_s == 10.0 and cfg.events.merge_gap_s == 0.3


def test_hash_is_stable_and_sensitive():
    a, b = RoadScribeConfig(), RoadScribeConfig()
    assert a.config_hash() == b.config_hash() == config_hash(a)
    assert len(a.config_hash()) == 16
    c = a.replace(iso={"cf_threshold": 8.0})
    assert c.iso.cf_threshold == 8.0 and c.config_hash() != a.config_hash()
    assert a.replace(events=EventConfig(kappa=12.0)).events.kappa == 12.0


def test_dict_and_yaml_roundtrip(tmp_path):
    pytest.importorskip("yaml")
    cfg = RoadScribeConfig().replace(mipelt={"lam": 0.3, "pruning": "textbook"}, spectral={"body_band": (0.7, 2.4)})
    assert RoadScribeConfig.from_dict(cfg.to_dict()) == cfg
    p = tmp_path / "exp.yaml"
    cfg.to_yaml(p)
    back = RoadScribeConfig.from_yaml(p)
    assert back == cfg and back.config_hash() == cfg.config_hash()
    assert isinstance(back.spectral.body_band, tuple)


def test_partial_yaml_and_unknown_keys(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("iso:\n  method: iir\nkst:\n  budget: 128\n")
    cfg = RoadScribeConfig.from_yaml(p)
    assert cfg.iso.method == "iir" and cfg.kst.budget == 128 and cfg.events == EventConfig()
    with pytest.raises(KeyError):
        RoadScribeConfig.from_dict({"iso": {"cf_treshold": 9}})
    with pytest.raises(KeyError):
        RoadScribeConfig.from_dict({"bogus": {}})
    with pytest.raises(ValueError):
        RoadScribeConfig.from_dict({"iso": {"method": "wavelet"}})
    assert RoadScribeConfig.from_dict({"iso": {"pad_s": 20}}).iso.pad_s == 20.0
    assert ISOConfig().method == "fft"


# --------------------------------------------------------------------------- schemas
def _raw(**kw):
    n = 200
    base = dict(
        trip_id="t", source="phone", route_id="r", vehicle_id="v", driver_id="d",
        t=np.arange(n) / 100.0, f_b=np.zeros((n, 3)), w_b=np.zeros((n, 3)), fs=100.0,
        gnss=pd.DataFrame({"t": [0.0, 1.0], "lat": [0, 0], "lon": [0, 0], "speed_mps": [0, 0]}), labels=None,
    )
    base.update(kw)
    return RawTrip(**base)


def test_rawtrip_validation():
    r = _raw()
    assert r.n == 200 and r.has_gyro and r.meta == {}
    assert not _raw(w_b=None).has_gyro
    with pytest.raises(ValueError):
        _raw(t=np.r_[0.0, 0.0, np.arange(2, 200) / 100.0])
    with pytest.raises(ValueError):
        _raw(f_b=np.zeros((10, 3)))
    with pytest.raises(ValueError):
        _raw(source="satellite")
    with pytest.raises(ValueError):
        _raw(gnss=pd.DataFrame({"t": [0.0]}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.fs = 50.0  # type: ignore[misc]


def test_vehicle_frame_trip_validation():
    n = 10
    ok = dict(trip_id="t", t=np.arange(n) / 100.0, fs=100.0, a_v=np.zeros((n, 3)), w_v=None, v=np.ones(n),
              a_lon_gnss=np.zeros(n), s=np.arange(n, dtype=float), episode=np.zeros(n, int), R_vb=[np.eye(3)],
              yaw_deg=[0.0], yaw_ci_deg=[(0.0, 0.0)], n_quasistatic=[n])
    VehicleFrameTrip(**ok)
    with pytest.raises(ValueError):
        VehicleFrameTrip(**{**ok, "s": np.arange(n, 0, -1, dtype=float)})
    with pytest.raises(ValueError):
        VehicleFrameTrip(**{**ok, "yaw_deg": []})


def test_event_hrnu_fact_capsule_validation():
    with pytest.raises(ValueError):
        Event("E1", "impulse", "pothole", 0, 1, 0, 0, 0, 1, 1, 1, "cruise", 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Event("E1", "impulse", "defect", 0, 1, 0, 0, 0, 1, 1, 1, "cruise", 1.5)
    with pytest.raises(ValueError):
        make_hrnu(1, 0, 100, boundary_left="osm")
    with pytest.raises(ValueError):
        make_hrnu(1, 0, 100, iso_felt=7)
    with pytest.raises(ValueError):
        Fact("F1", "GUESS", None, "k", 1)  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Fact("F1", "NUM", "U1", "a_v", 1.0).value = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        Capsule("c", "t", "p", 64, "", 0, "tok", {"U1": "HALF"}, [])
    u = make_hrnu(1, 100, 350)
    assert isinstance(u, HRNU) and u.length_m == 250


def test_to_jsonable():
    out = to_jsonable({"a": np.float64(1.5), "b": np.int32(2), "c": np.array([1, 2]), "d": float("nan"),
                       "e": np.bool_(True), "f": (1, 2)})
    assert out == {"a": 1.5, "b": 2, "c": [1, 2], "d": None, "e": True, "f": [1, 2]}
    assert to_jsonable(Fact("F1", "NUM", "U1", "a_v", np.float32(0.5)))["value"] == 0.5
