r"""Speed decoupling (METHODOLOGY §3.2).

Model: :math:`\log a_{w,i}=\alpha_{j(i)}+\gamma\log v_i+\varepsilon_i`, with
:math:`j` indexing homogeneous segments (or trips). The decoupled intensity
:math:`\tilde a_w=a_w(v_{ref}/\bar v)^{\hat\gamma}` describes the *road*; the
raw :math:`a_w` describes how the ride *felt*.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import SpeedConfig

__all__ = ["fit_speed_elasticity", "decoupled_intensity", "counterfactual"]


def fit_speed_elasticity(log_aw: Sequence[float], log_v: Sequence[float], groups: Sequence) -> float:
    """Within-group (fixed-effects) OLS estimate of gamma."""
    y = np.asarray(log_aw, float)
    x = np.asarray(log_v, float)
    g = np.asarray(groups)
    yd = y.copy()
    xd = x.copy()
    for k in np.unique(g):
        m = g == k
        yd[m] -= y[m].mean()
        xd[m] -= x[m].mean()
    sxx = float(xd @ xd)
    if sxx <= 0:
        raise ValueError("no within-group speed variation; gamma is not identified")
    return float(xd @ yd / sxx)


def decoupled_intensity(a_w: float, v_mean: float, cfg: SpeedConfig | None = None) -> Optional[float]:
    """a_w (v_ref / v_mean)^gamma / exp(vehicle_gain_log), or ``None`` (speed_low) when v_mean < v_min.

    With the defaults (``vehicle_gain_log = 0``) this is the original decoupling.
    """
    cfg = cfg or SpeedConfig()
    if not np.isfinite(v_mean) or v_mean < cfg.v_min:
        return None
    return float(a_w * (cfg.v_ref / v_mean) ** cfg.gamma * np.exp(-cfg.vehicle_gain_log))


def counterfactual(a_w: float, v_mean: float, v_new: float, gamma: float) -> float:
    """a_w(v') ~= a_w (v'/v_mean)^gamma. A model extrapolation: mark it so in the ledger."""
    return float(a_w * (v_new / v_mean) ** gamma)
