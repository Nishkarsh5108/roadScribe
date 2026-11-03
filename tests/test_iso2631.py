"""ISO 2631-1 weighting and measures.

Ports verify_math.py (a), (e) and verify_fft_weighting.py; every Appendix A
number for the filters is asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from roadscribe.config import ISOConfig
from roadscribe.layer1 import iso2631 as iso

FGRID = np.geomspace(0.5, 40.0, 400)

# §3.3 unit-test table, |W| x 1000
TABLE_F = [0.5, 1, 2, 4, 5, 8, 12.5, 16, 20, 31.5, 40]
TABLE_WK = [418, 482, 531, 967, 1039, 1036, 902, 769, 636, 405, 314]
TABLE_WD = [853, 1011, 890, 512, 409, 253, 161, 125, 100, 63, 49]

# RESULTS.md, all 1/3-octave centres
THIRD_OCT = [0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40]
THIRD_WK = [418, 459, 477, 482, 485, 494, 531, 631, 804, 967, 1039, 1054, 1036, 988, 902, 769, 636, 513, 405, 314]
THIRD_WD = [853, 944, 992, 1011, 1008, 968, 890, 776, 642, 512, 409, 323, 253, 202, 161, 125, 100, 80, 63, 49]


def _iir_err_db(name: str, fs: float, prewarp: bool = True) -> np.ndarray:
    _, h = signal.sosfreqz(iso.sos_coefficients(name, fs, prewarp), worN=FGRID, fs=fs)
    return 20 * np.log10(np.abs(h) / np.abs(iso.analog_response(name, FGRID)))


# --------------------------------------------------------------------------- analog weights
@pytest.mark.parametrize("name,expected", [("Wk", TABLE_WK), ("Wd", TABLE_WD)])
def test_unit_test_table(name, expected):
    got = np.round(np.abs(iso.analog_response(name, np.array(TABLE_F))) * 1000).astype(int)
    assert got.tolist() == expected


@pytest.mark.parametrize("name,expected", [("Wk", THIRD_WK), ("Wd", THIRD_WD)])
def test_third_octave_values(name, expected):
    got = np.round(np.abs(iso.analog_response(name, np.array(THIRD_OCT))) * 1000).astype(int)
    assert got.tolist() == expected


def test_appendix_a_iso_weights():
    wk = np.abs(iso.analog_response("Wk", np.array([1.0, 5.0, 16.0]))) * 1000
    wd = np.abs(iso.analog_response("Wd", np.array([1.0]))) * 1000
    assert np.round(wk).astype(int).tolist() == [482, 1039, 769]
    assert int(round(wd[0])) == 1011


def test_weighting_is_zero_at_dc():
    assert iso.analog_response("Wk", 0.0) == 0
    assert iso.analog_response("Wd", 0.0) == 0


def test_dropping_lowpass_changes_w_by_less_than_0p11_db_below_40hz():
    full = np.abs(iso.analog_response("Wk", FGRID, include_lowpass=True))
    nolp = np.abs(iso.analog_response("Wk", FGRID, include_lowpass=False))
    diff = 20 * np.log10(nolp / full)
    assert np.max(np.abs(diff)) < 0.11
    # RESULTS.md: |H_l| at 20 / 31.5 / 40 Hz = -0.007 / -0.043 / -0.110 dB
    hl = [20 * np.log10(1 / np.sqrt(1 + (f / 100) ** 4)) for f in (20, 31.5, 40)]
    assert np.round(hl, 3).tolist() == [-0.007, -0.043, -0.110]


# --------------------------------------------------------------------------- digital SOS
def test_bilinear_formula_matches_scipy():
    """The §3.3 coefficient formulas equal scipy's bilinear transform at fs = K/2."""
    for name in ("Wk", "Wd"):
        for fs in (100.0, 400.0, 2000.0):
            mine = iso.sos_coefficients(name, fs, prewarp=True)
            lp = iso.WEIGHTINGS[name].f2 < fs / 2
            for row, sec in zip(mine, iso.analog_sections(name, lp)):
                wc = 2 * np.pi * sec.f_warp
                K = wc / np.tan(wc / (2 * fs))
                bz, az = signal.bilinear(np.trim_zeros(np.array(sec.b), "f"), np.array(sec.a), fs=K / 2.0)
                bz = np.pad(bz, (3 - len(bz), 0))
                np.testing.assert_allclose(row, np.concatenate([bz, az]), rtol=1e-10, atol=1e-12)


def test_sos_at_100hz_matches_verify_math_printout():
    expected = np.array(
        [
            [0.9824, -1.9648, 0.9824, 1.0, -1.9645, 0.9651],
            [0.3203, 0.1876, -0.1327, 1.0, -0.9059, 0.2811],
            [0.965, -1.7646, 0.8194, 1.0, -1.7547, 0.7942],
        ]
    )
    np.testing.assert_allclose(iso.sos_coefficients("Wk", 100.0), expected, atol=5e-5)


def test_lowpass_section_only_when_realizable():
    assert len(iso.sos_coefficients("Wk", 100.0)) == 3
    assert len(iso.sos_coefficients("Wk", 200.0)) == 3  # 100 Hz is not below fs/2
    assert len(iso.sos_coefficients("Wk", 400.0)) == 4
    assert len(iso.sos_coefficients("Wd", 100.0)) == 2


def test_iir_error_fs100():
    """Appendix A: <= 0.72 dB (W_k), <= 1.22 dB (W_d) to 20 Hz; 7.3-7.7 dB at 40 Hz."""
    band = FGRID <= 20
    ek, ed = _iir_err_db("Wk", 100.0), _iir_err_db("Wd", 100.0)
    assert round(np.max(np.abs(ek[band])), 3) == 0.716
    assert round(np.max(np.abs(ed[band])), 3) == 1.222
    assert np.max(np.abs(ek[band])) <= 0.72 and np.max(np.abs(ed[band])) <= 1.225
    for e, expected in ((ek, 7.332), (ed, 7.663)):
        worst = np.max(np.abs(e))
        assert round(worst, 3) == expected
        assert 7.3 <= worst <= 7.7
        assert FGRID[np.argmax(np.abs(e))] == pytest.approx(40.0)


def test_prewarping_helps_wk_at_fs100():
    band = FGRID <= 20
    assert round(np.max(np.abs(_iir_err_db("Wk", 100.0, prewarp=False)[band])), 3) == 1.157
    assert round(np.max(np.abs(_iir_err_db("Wd", 100.0, prewarp=False)[band])), 3) == 1.233


@pytest.mark.parametrize("name,fs,expected", [("Wk", 200.0, 1.068), ("Wd", 200.0, 1.150), ("Wk", 400.0, 0.210), ("Wd", 400.0, 0.231)])
def test_iir_error_higher_rates(name, fs, expected):
    worst = np.max(np.abs(_iir_err_db(name, fs)))
    assert round(worst, 3) == expected
    if fs == 400.0:
        assert round(worst, 2) <= 0.23  # Appendix A: <= 0.23 dB to 40 Hz


def test_iir_weighted_rms_of_sines():
    """verify_math.py: W_k-weighted RMS of a 1 m/s^2-rms sine through the 100 Hz IIR."""
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    got = {}
    for f0 in (1.0, 5.0, 16.0):
        y = iso.weight_iir(np.sqrt(2) * np.sin(2 * np.pi * f0 * t), fs, "Wk")[int(10 * fs) :]
        got[f0] = round(float(np.sqrt(np.mean(y**2))), 4)
    assert got == {1.0: 0.4824, 5.0: 1.0388, 16.0: 0.7470}
    assert round(float(np.abs(iso.analog_response("Wk", 16.0))), 4) == 0.7687


# --------------------------------------------------------------------------- FFT weighting
@pytest.mark.parametrize("name", ["Wk", "Wd"])
def test_fft_weighting_exact_on_tones(name):
    """Appendix A: 0.0000 dB error on test tones 0.5-40 Hz at fs = 100 Hz."""
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    worst = 0.0
    for f0 in [0.5, 1, 2, 4, 8, 12.5, 16, 25, 31.5, 40]:
        y = iso.weight_fft(np.sqrt(2) * np.sin(2 * np.pi * f0 * t), fs, name)[int(20 * fs) : -int(20 * fs)]
        e = 20 * np.log10(np.sqrt(np.mean(y**2)) / np.abs(iso.analog_response(name, f0)))
        worst = max(worst, abs(e))
    assert round(worst, 4) == 0.0


def test_fft_weighting_phase_matches_analog():
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    f0 = 4.0
    y = iso.weight_fft(np.sin(2 * np.pi * f0 * t), fs, "Wk")
    H = iso.analog_response("Wk", f0)
    expected = np.abs(H) * np.sin(2 * np.pi * f0 * t + np.angle(H))
    mid = slice(int(20 * fs), -int(20 * fs))
    np.testing.assert_allclose(y[mid], expected[mid], atol=1e-5)  # residual is edge leakage


def test_fft_weighting_axis_argument():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    a = iso.weight_fft(x, 100.0, "Wd", axis=0)
    b = iso.weight_fft(x.T, 100.0, "Wd", axis=1).T
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a[:, 1], iso.weight_fft(x[:, 1], 100.0, "Wd"), atol=1e-12)


def test_impulsive_bump_vdv_and_peak():
    """Appendix A: VDV 1.3842 vs 1.3842 (2 kHz reference); peak 3.7220 vs 3.7215; IIR@100 peak -3.8 %."""
    fs_hi = 2000.0
    th = np.arange(0, 20, 1 / fs_hi)
    pulse = np.exp(-(((th - 10) / 0.02) ** 2)) * 5.0  # 40 ms Gaussian bump
    ref = signal.sosfilt(iso.sos_coefficients("Wk", fs_hi), pulse)
    x100 = signal.resample_poly(pulse, 1, 20)
    y100 = iso.weight_fft(x100, 100.0, "Wk")
    iir100 = iso.weight_iir(x100, 100.0, "Wk")
    ref100 = signal.resample_poly(ref, 1, 20)
    assert round(iso.vdv(ref100, 100.0), 4) == 1.3842
    assert round(iso.vdv(y100, 100.0), 4) == 1.3842
    assert round(iso.vdv(iir100, 100.0), 4) == 1.3719
    pk_ref, pk_fft, pk_iir = (float(np.max(np.abs(a))) for a in (ref100, y100, iir100))
    assert round(pk_ref, 4) == 3.7215
    assert round(pk_fft, 4) == 3.7220
    assert round(pk_iir, 4) == 3.5812
    assert round(100 * (pk_iir / pk_ref - 1), 1) == -3.8


# --------------------------------------------------------------------------- measures
def test_vdv_ratio_equals_kurtosis_quarter_power():
    """VDV/(a_w T^1/4) = kappa^(1/4); the 1.75 criterion is kappa > 9.38; Gaussian gives 1.32."""
    rng = np.random.default_rng(1)
    fs = 100.0
    x = rng.normal(size=200_000)
    m = iso.axis_measures(x - x.mean(), fs)
    assert m.vdv_ratio == pytest.approx(m.kurtosis**0.25, rel=1e-9)
    assert round(iso.vdv_ratio_kurtosis_threshold(1.75), 2) == 9.38
    assert round(3.0**0.25, 2) == 1.32
    assert m.vdv_ratio == pytest.approx(1.32, abs=0.01)
    assert not m.shock


def test_running_rms_and_mtvv():
    fs = 100.0
    x = np.zeros(1000)
    x[500:600] = 2.0  # 1 s of 2 m/s^2
    assert iso.mtvv(x, fs, 1.0) == pytest.approx(2.0)
    assert iso.rms(x) == pytest.approx(np.sqrt(0.4))
    short = np.ones(50)  # shorter than tau: RMS of the whole segment
    assert iso.mtvv(short, fs, 1.0) == pytest.approx(1.0)


def test_shock_flag_rule():
    cfg = ISOConfig()
    assert iso.shock_flag(9.1, 1.0, 1.0, cfg)
    assert iso.shock_flag(3.0, 1.51, 1.0, cfg)
    assert iso.shock_flag(3.0, 1.0, 1.76, cfg)
    assert not iso.shock_flag(9.0, 1.5, 1.75, cfg)  # strict inequalities


def test_impulsive_signal_is_shock_dominated():
    fs = 100.0
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.1, 6000)
    x[3000:3010] += 8.0
    m = iso.axis_measures(x, fs)
    assert m.shock and m.cf > 9


def test_vector_sum():
    assert iso.vector_sum(3.0, 4.0, 12.0) == pytest.approx(13.0)
    assert iso.vector_sum(1.0, 1.0, 1.0, (1.4, 1.4, 1.0)) == pytest.approx(np.sqrt(1.96 * 2 + 1))


def test_weight_vehicle_axes_uses_wd_and_wk():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sqrt(2) * np.sin(2 * np.pi * 8.0 * t)
    aw = iso.weight_vehicle_axes(np.column_stack([x, x, x]), fs)
    mid = slice(int(10 * fs), -int(10 * fs))
    assert iso.rms(aw[mid, 0]) == pytest.approx(abs(iso.analog_response("Wd", 8.0)), rel=1e-4)
    assert iso.rms(aw[mid, 2]) == pytest.approx(abs(iso.analog_response("Wk", 8.0)), rel=1e-4)


# --------------------------------------------------------------------------- comfort words
def test_comfort_cutpoints():
    """Appendix A: 0.315, 0.561, 0.894, 1.414, 2.236 m/s^2."""
    assert np.round(iso.COMFORT_CUTPOINTS, 3).tolist() == [0.315, 0.561, 0.894, 1.414, 2.236]


@pytest.mark.parametrize("a,k", [(0.1, 0), (0.315, 1), (0.5, 1), (0.6, 2), (1.0, 3), (1.62, 4), (0.70, 2), (2.3, 5)])
def test_comfort_class(a, k):
    assert iso.comfort_class(a) == k


def test_comfort_bands_overlap():
    assert iso.comfort_bands_containing(0.9) == [2, 3]
    assert iso.comfort_bands_containing(2.2) == [4, 5]
    assert iso.comfort_label(4) == "very uncomfortable"
