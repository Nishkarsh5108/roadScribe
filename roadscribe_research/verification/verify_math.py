"""Numerical verification of the formulas used in the RoadScribe methodology."""
import numpy as np
from scipy import signal
from statsmodels.stats.power import TTestPower

np.set_printoptions(precision=4, suppress=True)
FS = 100.0

# ---------------------------------------------------------------- (a) ISO 2631-1 weightings
PARAMS = {  # f1, f2, f3, f4, Q4, f5, Q5, f6, Q6   (ISO 2631-1:1997 Annex A)
    "Wk": (0.4, 100.0, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91),
    "Wd": (0.4, 100.0, 2.0, 2.0, 0.63, None, None, None, None),
}
Q1 = 1 / np.sqrt(2)


def sections(name, include_lowpass=True):
    """Analog second-order sections as (b, a, prewarp_hz) with polynomials in s."""
    f1, f2, f3, f4, Q4, f5, Q5, f6, Q6 = PARAMS[name]
    w = lambda f: 2 * np.pi * f
    secs = [([1, 0, 0], [1, w(f1) / Q1, w(f1) ** 2], f1)]                       # H_h high-pass
    if include_lowpass:
        secs.append(([w(f2) ** 2], [1, w(f2) / Q1, w(f2) ** 2], f2))            # H_l low-pass
    w3, w4 = w(f3), w(f4)
    secs.append(([w4 ** 2 / w3, w4 ** 2], [1, w4 / Q4, w4 ** 2], np.sqrt(f3 * f4)))  # H_t
    if f5 is not None:
        w5, w6 = w(f5), w(f6)
        secs.append(([1, w5 / Q5, w5 ** 2], [1, w6 / Q6, w6 ** 2], np.sqrt(f5 * f6)))  # H_s
    return secs


def analog_mag(name, f):
    s = 1j * 2 * np.pi * f
    H = np.ones_like(s)
    for b, a, _ in sections(name, include_lowpass=True):
        H *= np.polyval(b, s) / np.polyval(a, s)
    return np.abs(H)


def digital_sos(name, fs=FS, prewarp=True):
    sos = []
    for b, a, fp in sections(name, include_lowpass=False):   # H_l (100 Hz) not realisable at fs=100
        if prewarp:
            K = 2 * np.pi * fp / np.tan(np.pi * fp / fs)          # s = K (1 - z^-1)/(1 + z^-1)
            bz, az = signal.bilinear(b, a, fs=K / 2.0)
        else:
            bz, az = signal.bilinear(b, a, fs=fs)
        bz = np.concatenate([np.zeros(3 - len(bz)), bz]) if len(bz) < 3 else bz
        sos.append(np.concatenate([np.pad(bz, (0, 3 - len(bz))), np.pad(az, (0, 3 - len(az)))]))
    return np.array(sos)


third_oct = np.array([0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40])
fgrid = np.geomspace(0.5, 40, 400)
for name in ["Wk", "Wd"]:
    ref_t = analog_mag(name, third_oct)
    print(f"{name} analog |W| x1000 at 1/3-octave centres:", dict(zip(third_oct.tolist(), np.round(ref_t * 1000).astype(int).tolist())))
    ref = analog_mag(name, fgrid)
    for pw in [True, False]:
        sos = digital_sos(name, prewarp=pw)
        _, h = signal.sosfreqz(sos, worN=fgrid, fs=FS)
        err = 20 * np.log10(np.abs(h) / ref)
        band = fgrid <= 20
        print(f"  digital fs=100, prewarp={pw}: max|err| 0.5-20 Hz = {np.max(np.abs(err[band])):.3f} dB, "
              f"0.5-40 Hz = {np.max(np.abs(err)):.3f} dB (at {fgrid[np.argmax(np.abs(err))]:.1f} Hz)")
    if name == "Wk":
        print("  SOS (prewarped, fs=100):\n", digital_sos(name))
# effect of omitting H_l alone
for f in [20, 31.5, 40]:
    print(f"  |H_l({f} Hz)| = {20*np.log10(1/np.sqrt(1+(f/100)**4)):.3f} dB")
# sanity: RMS of weighted 1 m/s^2-rms sinusoid at 5 Hz and 1 Hz through digital Wk
t = np.arange(0, 60, 1 / FS)
for f0 in [1.0, 5.0, 16.0]:
    x = np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
    y = signal.sosfilt(digital_sos("Wk"), x)[int(10 * FS):]
    print(f"  Wk-weighted RMS of 1 m/s^2 rms sine at {f0} Hz: {np.sqrt(np.mean(y**2)):.4f} (analog {analog_mag('Wk', np.array([f0]))[0]:.4f})")

# ---------------------------------------------------------------- (b) PELT with location-dependent penalty
rng = np.random.default_rng(7)


def l2_cost_factory(Y):
    c1 = np.vstack([np.zeros(Y.shape[1]), np.cumsum(Y, 0)])
    c2 = np.concatenate([[0.0], np.cumsum(np.sum(Y ** 2, 1))])
    def C(t, s):  # cost of y_{t+1..s} (1-based) == rows t..s-1
        n = s - t
        mu = (c1[s] - c1[t]) / n
        return (c2[s] - c2[t]) - n * np.dot(mu, mu)
    return C


def op_exact(n, C, beta):
    F = np.full(n + 1, np.inf); F[0] = 0.0
    G = lambda t: F[t] + (beta[t] if t > 0 else 0.0)
    for s in range(1, n + 1):
        F[s] = min(G(t) + C(t, s) for t in range(0, s))
    return F[n]


def pelt(n, C, beta, corrected=True, minseg=1):
    F = np.full(n + 1, np.inf); F[0] = 0.0
    G = lambda t: F[t] + (beta[t] if t > 0 else 0.0)
    R = [0]
    for s in range(1, n + 1):
        F[s] = min(G(t) + C(t, s) for t in R)
        if corrected:
            R = [t for t in R if G(t) + C(t, s) < G(s)] + [s]
        else:  # classical rule assumes a constant penalty
            R = [t for t in R if F[t] + C(t, s) < F[s]] + [s]
    return F[n]


bad_naive, bad_corr, trials = 0, 0, 300
for k in range(trials):
    n = 120
    cps = np.sort(rng.choice(np.arange(10, n - 10), size=4, replace=False))
    mu = np.repeat(rng.normal(0, 1.2, (5, 2)), np.diff(np.concatenate([[0], cps, [n]])), axis=0)
    Y = mu + rng.normal(0, 1, (n, 2))
    C = l2_cost_factory(Y)
    beta = np.full(n + 1, 12.0)
    mapnodes = rng.choice(np.arange(5, n - 5), size=10, replace=False)
    beta[mapnodes] = 2.0      # cheaper change points at map transitions
    ref = op_exact(n, C, beta)
    bad_corr += not np.isclose(pelt(n, C, beta, True), ref)
    bad_naive += not np.isclose(pelt(n, C, beta, False), ref)
print(f"\n(b) location-dependent penalty, {trials} random series: corrected-PELT mismatches={bad_corr}, classical-rule mismatches={bad_naive}")

# ---------------------------------------------------------------- (c) power, N = 28 within-subject
tp = TTestPower()
for k in [1, 3, 6]:
    dz = tp.solve_power(nobs=28, alpha=0.05 / k, power=0.8, alternative="two-sided")
    print(f"(c) N=28 paired, alpha=0.05/{k}: minimum detectable dz at 80% power = {dz:.3f}")
print("    N needed for dz=0.5 at alpha .05:", round(tp.solve_power(effect_size=0.5, alpha=0.05, power=0.8), 1))

# ---------------------------------------------------------------- (d) Williams design, 4 conditions
k = 4
first = [0, 1, k - 1, 2]
W = np.array([[(x + r) % k for x in first] for r in range(k)])
pairs = {}
for row in W:
    for a, b in zip(row[:-1], row[1:]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
print("(d) Williams 4x4 rows:", W.tolist(), "| each ordered pair once:", sorted(set(pairs.values())) == [1] and len(pairs) == 12,
      "| column-balanced:", all(sorted(W[:, j]) == list(range(k)) for j in range(k)))

# ---------------------------------------------------------------- (e) ISO 2631-1 Annex C comfort cut-points
bands = [(0, 0.315), (0.315, 0.63), (0.5, 1.0), (0.8, 1.6), (1.25, 2.5), (2.0, np.inf)]
cuts = [0.315] + [float(np.sqrt(bands[i + 1][0] * bands[i][1])) for i in range(1, 5)]
print("(e) geometric-mean cut-points between overlapping ISO comfort bands:", np.round(cuts, 3).tolist())
