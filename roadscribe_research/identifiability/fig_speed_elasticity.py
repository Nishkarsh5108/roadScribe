"""Figure: the speed elasticity of dashboard vibration changes sign between pooled and within-place estimates (P1-H3b).

Left: one point per (place, vehicle); pooled OLS of log a_wz on log speed.
Right: Frisch-Waugh-Lovell partial plot of the pre-registered model
(location FE + vehicle FE + log speed); its slope equals gamma_within exactly.
Run from the repo root after p1_cross_vehicle.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.paired.agreement import _demean  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 3: ("#1baf7a", "^")}  # validated categorical slots 1-3


def main():
    d = p1.triplets(pd.read_csv(HERE / "results" / "p1_stretches_100m.csv"))
    lv, ly = np.log(d["v_mean"].to_numpy()), np.log(d["a_wz"].to_numpy())
    loc, veh = d["loc"].to_numpy(), d["vehicle"].to_numpy()
    slope_pooled, icpt = np.polyfit(lv, ly, 1)
    D = (veh[:, None] == np.array([2, 3])[None, :]).astype(float)
    Z = _demean(D, loc)
    # FWL: residualise log v and log a_wz on location FE + vehicle FE
    def resid(y):
        yd = _demean(y, loc)
        b, *_ = np.linalg.lstsq(Z, yd, rcond=None)
        return yd - Z @ b
    rx, ry = resid(lv), resid(ly)
    slope_within = float(rx @ ry / (rx @ rx))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax = axes[0]
    for v, (c, m) in SERIES.items():
        k = veh == v
        ax.scatter(np.exp(lv[k]), np.exp(ly[k]), s=22, c=c, marker=m, alpha=0.75, edgecolors=SURFACE, linewidths=0.6,
                   label=f"Vehicle {v}")
    xs = np.linspace(lv.min(), lv.max(), 50)
    ax.plot(np.exp(xs), np.exp(icpt + slope_pooled * xs), color=INK, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 7, 10, 15, 20, 30])
    ax.set_xticklabels(["5", "7", "10", "15", "20", "30"])
    ax.set_yticks([0.3, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.3", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Mean speed on the 100 m stretch (m/s, log scale)")
    ax.set_ylabel(r"$W_k$-weighted vertical vibration $a_{wz}$ (m/s$^2$, log scale)")
    ax.set_title(f"Across places: slope {slope_pooled:+.2f}", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax = axes[1]
    for v, (c, m) in SERIES.items():
        k = veh == v
        ax.scatter(rx[k], ry[k], s=22, c=c, marker=m, alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(xs, slope_within * xs, color=INK, linewidth=1.8)
    ax.set_xlabel(r"$\log \bar{v}$, net of place and vehicle")
    ax.set_ylabel(r"$\log a_{wz}$, net of place and vehicle")
    ax.set_title(f"Within each place: slope {slope_within:+.2f}", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    out = HERE / "results" / "p1_speed_elasticity_flip.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(f"pooled {slope_pooled:+.3f}  within {slope_within:+.3f}  -> {out}")


if __name__ == "__main__":
    main()
