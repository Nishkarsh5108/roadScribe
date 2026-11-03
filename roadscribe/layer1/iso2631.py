r"""ISO 2631-1 frequency weighting and evaluation measures (METHODOLOGY §3.3).

Analog weighting :math:`W=H_hH_lH_tH_s` with the Annex A parameters, applied
either

* **offline** in the frequency domain (:func:`weight_fft`), exact in magnitude
  and phase below :math:`f_s/2` — the recommended path, narratives are
  generated post-trip; or
* **streaming** as second-order sections obtained by a per-section bilinear
  transform pre-warped at each section's corner frequency
  (:func:`sos_coefficients`, :func:`weight_iir`).

The quantity is called *ISO 2631-1-weighted cabin acceleration*: a phone in a
holder does not measure the seat-occupant interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import fft as sfft
from scipy import signal

from ..config import ISOConfig

__all__ = [
    "WeightingParams",
    "WEIGHTINGS",
    "AnalogSection",
    "analog_sections",
    "analog_response",
    "bilinear_section",
    "sos_coefficients",
    "weight_fft",
    "weight_iir",
    "weight",
    "weight_vehicle_axes",
    "rms",
    "running_rms",
    "mtvv",
    "vdv",
    "crest_factor",
    "kurtosis",
    "AxisMeasures",
    "axis_measures",
    "shock_flag",
    "vector_sum",
    "COMFORT_BANDS",
    "COMFORT_LABELS",
    "COMFORT_CUTPOINTS",
    "comfort_cutpoints",
    "comfort_class",
    "comfort_label",
    "comfort_bands_containing",
    "vdv_ratio_kurtosis_threshold",
]

Q1 = 1.0 / np.sqrt(2.0)
WeightingName = Literal["Wk", "Wd"]


@dataclass(frozen=True)
class WeightingParams:
    """ISO 2631-1:1997 Annex A parameters (Hz)."""

    f1: float
    f2: float
    f3: float
    f4: float
    Q4: float
    f5: float | None = None
    Q5: float | None = None
    f6: float | None = None
    Q6: float | None = None


WEIGHTINGS: dict[str, WeightingParams] = {
    "Wk": WeightingParams(0.4, 100.0, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91),  # vertical, seat
    "Wd": WeightingParams(0.4, 100.0, 2.0, 2.0, 0.63),  # horizontal, seat
}


@dataclass(frozen=True)
class AnalogSection:
    """(b2 p^2 + b1 p + b0) / (a2 p^2 + a1 p + a0) with its pre-warp frequency (Hz)."""

    name: str
    b: tuple[float, float, float]
    a: tuple[float, float, float]
    f_warp: float

    def response(self, p: np.ndarray) -> np.ndarray:
        b2, b1, b0 = self.b
        a2, a1, a0 = self.a
        return (b2 * p**2 + b1 * p + b0) / (a2 * p**2 + a1 * p + a0)


def analog_sections(name: str, include_lowpass: bool = True) -> list[AnalogSection]:
    """Second-order analog sections of W_k or W_d (polynomials in p = j 2 pi f)."""
    P = WEIGHTINGS[name]
    w = lambda f: 2.0 * np.pi * f  # noqa: E731
    w1, w2, w3, w4 = w(P.f1), w(P.f2), w(P.f3), w(P.f4)
    secs = [AnalogSection("high-pass", (1.0, 0.0, 0.0), (1.0, w1 / Q1, w1**2), P.f1)]
    if include_lowpass:
        secs.append(AnalogSection("low-pass", (0.0, 0.0, w2**2), (1.0, w2 / Q1, w2**2), P.f2))
    secs.append(
        AnalogSection(
            "a-v transition",
            (0.0, w4**2 / w3, w4**2),
            (1.0, w4 / P.Q4, w4**2),
            float(np.sqrt(P.f3 * P.f4)),
        )
    )
    if P.f5 is not None:
        w5, w6 = w(P.f5), w(P.f6)
        secs.append(
            AnalogSection(
                "upward step",
                (1.0, w5 / P.Q5, w5**2),
                (1.0, w6 / P.Q6, w6**2),
                float(np.sqrt(P.f5 * P.f6)),
            )
        )
    return secs


def analog_response(name: str, f: np.ndarray | float, include_lowpass: bool = True) -> np.ndarray:
    """Complex analog weighting W(j 2 pi f)."""
    p = 1j * 2.0 * np.pi * np.asarray(f, dtype=float)
    H = np.ones_like(p)
    for sec in analog_sections(name, include_lowpass):
        H = H * sec.response(p)
    return H


def bilinear_section(
    b: Sequence[float], a: Sequence[float], K: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map one analog biquad with p -> K (1 - z^-1)/(1 + z^-1).

    B0 = b2 K^2 + b1 K + b0, B1 = 2 (b0 - b2 K^2), B2 = b2 K^2 - b1 K + b0, and
    the same for A; everything is divided by A0 (§3.3, Discrete realization 2).
    """
    b2, b1, b0 = b
    a2, a1, a0 = a
    K2 = K * K
    B = np.array([b2 * K2 + b1 * K + b0, 2.0 * (b0 - b2 * K2), b2 * K2 - b1 * K + b0])
    A = np.array([a2 * K2 + a1 * K + a0, 2.0 * (a0 - a2 * K2), a2 * K2 - a1 * K + a0])
    return B / A[0], A / A[0]


def sos_coefficients(
    name: str, fs: float, prewarp: bool = True, include_lowpass: bool | None = None
) -> np.ndarray:
    """Digital SOS matrix (n_sections, 6) for streaming weighting.

    ``include_lowpass=None`` includes the 100 Hz band limit H_l only when it is
    realizable, i.e. ``fs > 2 * f2`` (it cannot be realized at fs <= 200 Hz;
    omitting it changes W by less than 0.11 dB below 40 Hz).
    With ``prewarp=True``, K_i = w_c / tan(w_c T_s / 2) at each section's corner
    frequency; otherwise the plain bilinear transform K = 2 fs.
    """
    if include_lowpass is None:
        include_lowpass = WEIGHTINGS[name].f2 < fs / 2.0
    rows = []
    for sec in analog_sections(name, include_lowpass):
        if prewarp:
            wc = 2.0 * np.pi * sec.f_warp
            if sec.f_warp >= fs / 2.0:
                raise ValueError(f"section {sec.name!r} corner {sec.f_warp} Hz is not below fs/2")
            K = wc / np.tan(wc / (2.0 * fs))
        else:
            K = 2.0 * fs
        B, A = bilinear_section(sec.b, sec.a, K)
        rows.append(np.concatenate([B, A]))
    return np.array(rows)


def weight_iir(x: np.ndarray, fs: float, name: str, axis: int = 0, prewarp: bool = True) -> np.ndarray:
    """Causal (streaming) weighting with the pre-warped SOS cascade."""
    return signal.sosfilt(sos_coefficients(name, fs, prewarp), x, axis=axis)


def weight_fft(x: np.ndarray, fs: float, name: str, pad_s: float = 10.0, axis: int = 0) -> np.ndarray:
    """Exact frequency-domain weighting a_w = F^-1{ F{a} W(j 2 pi f_k) }.

    The signal is zero-padded by ``pad_s`` seconds on both sides so that the
    circular convolution does not wrap the weighting's impulse response.
    """
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    npad = int(round(pad_s * fs))
    total = n + 2 * npad
    nfft = sfft.next_fast_len(total, real=True)
    pad_width = [(npad, nfft - n - npad)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad_width)
    X = sfft.rfft(xp, axis=0)
    f = sfft.rfftfreq(nfft, 1.0 / fs)
    H = analog_response(name, f)  # H(0) = 0 exactly (high-pass)
    H = H.reshape((-1,) + (1,) * (x.ndim - 1))
    y = sfft.irfft(X * H, n=nfft, axis=0)[npad : npad + n]
    return np.moveaxis(y, 0, axis)


def weight(
    x: np.ndarray, fs: float, name: str, method: Literal["fft", "iir"] = "fft", pad_s: float = 10.0, axis: int = 0
) -> np.ndarray:
    if method == "fft":
        return weight_fft(x, fs, name, pad_s, axis)
    if method == "iir":
        return weight_iir(x, fs, name, axis)
    raise ValueError(f"unknown weighting method {method!r}")


def weight_vehicle_axes(a_v: np.ndarray, fs: float, cfg: ISOConfig | None = None) -> np.ndarray:
    """Weight a vehicle-frame (N, 3) acceleration: W_d on x and y, W_k on z."""
    cfg = cfg or ISOConfig()
    out = np.empty_like(np.asarray(a_v, dtype=float))
    out[:, :2] = weight(a_v[:, :2], fs, "Wd", cfg.method, cfg.pad_s, axis=0)
    out[:, 2] = weight(a_v[:, 2], fs, "Wk", cfg.method, cfg.pad_s)
    return out


# --------------------------------------------------------------------------- measures
def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def running_rms(x: np.ndarray, fs: float, tau: float = 1.0) -> np.ndarray:
    """a_w(t0) = sqrt(1/tau * integral_{t0-tau}^{t0} a_w^2 dt) for every full window."""
    x = np.asarray(x, dtype=float)
    w = max(1, int(round(tau * fs)))
    if len(x) < w:
        return np.array([rms(x)]) if len(x) else np.array([])
    c = np.concatenate([[0.0], np.cumsum(x**2)])
    return np.sqrt(np.maximum((c[w:] - c[:-w]) / w, 0.0))


def mtvv(x: np.ndarray, fs: float, tau: float = 1.0) -> float:
    """Maximum transient vibration value (max of the 1 s running RMS).

    For segments shorter than ``tau`` the RMS of the whole segment is returned.
    """
    r = running_rms(x, fs, tau)
    return float(r.max()) if r.size else float("nan")


def vdv(x: np.ndarray, fs: float) -> float:
    """Vibration dose value (integral a_w^4 dt)^(1/4), m s^-1.75."""
    x = np.asarray(x, dtype=float)
    return float((np.sum(x**4) / fs) ** 0.25)


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    return float(np.max(np.abs(x)) / r) if r > 0 else float("nan")


def kurtosis(x: np.ndarray) -> float:
    """Pearson (non-excess) kurtosis of the zero-mean weighted signal: E[x^4] / E[x^2]^2."""
    x = np.asarray(x, dtype=float)
    m2 = np.mean(x**2)
    return float(np.mean(x**4) / m2**2) if m2 > 0 else float("nan")


def vdv_ratio_kurtosis_threshold(ratio: float = 1.75) -> float:
    """VDV/(a_w T^(1/4)) = kappa^(1/4) for a zero-mean signal, so ratio r <=> kappa = r^4."""
    return ratio**4


@dataclass(frozen=True)
class AxisMeasures:
    a_w: float
    mtvv: float
    vdv: float
    cf: float
    kurtosis: float
    duration_s: float
    mtvv_ratio: float
    vdv_ratio: float
    shock: bool


def shock_flag(cf: float, mtvv_ratio: float, vdv_ratio: float, cfg: ISOConfig | None = None) -> bool:
    """S_j = 1[CF > 9  or  MTVV/a_w > 1.5  or  VDV/(a_w T^1/4) > 1.75] (ISO 2631-1 §6.2-6.3)."""
    cfg = cfg or ISOConfig()
    return bool(
        cf > cfg.cf_threshold or mtvv_ratio > cfg.mtvv_ratio_threshold or vdv_ratio > cfg.vdv_ratio_threshold
    )


def axis_measures(aw: np.ndarray, fs: float, cfg: ISOConfig | None = None) -> AxisMeasures:
    """All §3.3 measures for one weighted axis of one HRNU."""
    cfg = cfg or ISOConfig()
    aw = np.asarray(aw, dtype=float)
    T = len(aw) / fs
    a = rms(aw)
    m = mtvv(aw, fs, cfg.mtvv_tau_s)
    d = vdv(aw, fs)
    cf = crest_factor(aw)
    if a > 0:
        mr = m / a
        vr = d / (a * T**0.25)
    else:
        mr = vr = float("nan")
    return AxisMeasures(
        a_w=a,
        mtvv=m,
        vdv=d,
        cf=cf,
        kurtosis=kurtosis(aw),
        duration_s=T,
        mtvv_ratio=mr,
        vdv_ratio=vr,
        shock=shock_flag(cf, mr, vr, cfg) if a > 0 else False,
    )


def vector_sum(a_wx: float, a_wy: float, a_wz: float, k: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """a_v = sqrt(k_x^2 a_wx^2 + k_y^2 a_wy^2 + k_z^2 a_wz^2) (comfort: k = 1)."""
    return float(np.sqrt((k[0] * a_wx) ** 2 + (k[1] * a_wy) ** 2 + (k[2] * a_wz) ** 2))


# --------------------------------------------------------------------------- comfort words
#: ISO 2631-1 Annex C (informative) comfort reactions, m/s^2. The bands overlap.
COMFORT_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.315),
    (0.315, 0.63),
    (0.5, 1.0),
    (0.8, 1.6),
    (1.25, 2.5),
    (2.0, float("inf")),
)
COMFORT_LABELS: tuple[str, ...] = (
    "not uncomfortable",
    "a little uncomfortable",
    "fairly uncomfortable",
    "uncomfortable",
    "very uncomfortable",
    "extremely uncomfortable",
)


def comfort_cutpoints(bands: Sequence[tuple[float, float]] = COMFORT_BANDS) -> tuple[float, ...]:
    """Single-label cut-points: 0.315, then the geometric mean of each overlap."""
    cuts = [bands[0][1]]
    for i in range(1, len(bands) - 1):
        cuts.append(float(np.sqrt(bands[i + 1][0] * bands[i][1])))
    return tuple(cuts)


COMFORT_CUTPOINTS: tuple[float, ...] = comfort_cutpoints()


def comfort_class(a: float, cutpoints: Sequence[float] = COMFORT_CUTPOINTS) -> int:
    """k*(x) = sum_j 1[x >= theta_j], in 0..5."""
    if not np.isfinite(a):
        raise ValueError("comfort_class needs a finite value")
    return int(sum(a >= c for c in cutpoints))


def comfort_label(k: int) -> str:
    return COMFORT_LABELS[k]


def comfort_bands_containing(a: float) -> list[int]:
    """All Annex C classes whose band contains ``a`` (used for band consistency, §6.3)."""
    return [k for k, (lo, hi) in enumerate(COMFORT_BANDS) if lo <= a <= hi]
