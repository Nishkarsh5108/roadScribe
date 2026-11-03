r"""Welch PSD and modal band ratios (METHODOLOGY §3.4).

These ratios describe how the *vehicle's* modes are excited (body heave,
wheel hop, high band). At :math:`\lambda=v/f` and 10 m/s they correspond to
0.25-12.5 m wavelengths, i.e. unevenness and megatexture, not pavement
macrotexture. Use them within a vehicle and for segmentation only.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import SpectralConfig

__all__ = ["welch_psd", "band_energy", "wheel_hop_peak", "wheel_band", "modal_band_ratios"]


def welch_psd(x: np.ndarray, fs: float, cfg: SpectralConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD (Hann, 50 % overlap). L = 512, or 256 for units < 15 s."""
    cfg = cfg or SpectralConfig()
    x = np.asarray(x, dtype=float)
    L = cfg.nperseg if len(x) / fs >= cfg.short_unit_s else cfg.nperseg_short
    L = int(min(L, len(x)))
    if L < 8:
        raise ValueError("segment too short for a PSD estimate")
    return signal.welch(x, fs=fs, window="hann", nperseg=L, noverlap=L // 2, detrend="constant")


def band_energy(f: np.ndarray, P: np.ndarray, band: tuple[float, float]) -> float:
    """E_B = integral of P over [lo, hi] (rectangle rule on the PSD bins)."""
    lo, hi = band
    df = f[1] - f[0]
    m = (f >= lo) & (f <= hi)
    return float(np.sum(P[m]) * df)


def wheel_hop_peak(f: np.ndarray, P: np.ndarray, search: tuple[float, float] = (6.0, 20.0)) -> float:
    m = (f >= search[0]) & (f <= search[1])
    return float(f[m][np.argmax(P[m])])


def wheel_band(f_wh: float | None, cfg: SpectralConfig | None = None) -> tuple[float, float]:
    cfg = cfg or SpectralConfig()
    if f_wh is None:
        return cfg.wheel_default
    return (f_wh * cfg.wheel_rel[0], f_wh * cfg.wheel_rel[1])


def modal_band_ratios(
    az: np.ndarray, fs: float, f_wh: float | None = None, cfg: SpectralConfig | None = None, eps: float = 1e-12
) -> tuple[float, float]:
    """(R_wb, R_hb) = (log(E_wheel/E_body), log(E_high/E_body)) of vertical a^v_z.

    ``f_wh`` is the vehicle's wheel-hop peak, estimated on its rough units with
    :func:`wheel_hop_peak`; ``None`` uses the default [8, 16] Hz band.
    The high band is clipped at the Nyquist frequency.
    """
    cfg = cfg or SpectralConfig()
    f, P = welch_psd(az, fs, cfg)
    hi = (cfg.high_band[0], min(cfg.high_band[1], fs / 2.0))
    e_body = band_energy(f, P, cfg.body_band) + eps
    e_wheel = band_energy(f, P, wheel_band(f_wh, cfg)) + eps
    e_high = band_energy(f, P, hi) + eps
    return float(np.log(e_wheel / e_body)), float(np.log(e_high / e_body))
