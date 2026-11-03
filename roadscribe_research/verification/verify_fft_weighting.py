"""Compare offline FFT weighting (exact complex analog response) vs IIR at higher fs."""
import numpy as np
from scipy import signal

# minimal helpers re-defined here (verify_math.py runs all its checks at import)
Q1 = 1 / np.sqrt(2)
P = {"Wk": (0.4, 100.0, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91), "Wd": (0.4, 100.0, 2.0, 2.0, 0.63, None, None, None, None)}


def secs(name, lp):
    f1, f2, f3, f4, Q4, f5, Q5, f6, Q6 = P[name]
    w = lambda f: 2 * np.pi * f
    S = [([1, 0, 0], [1, w(f1) / Q1, w(f1) ** 2], f1)]
    if lp:
        S.append(([w(f2) ** 2], [1, w(f2) / Q1, w(f2) ** 2], f2))
    S.append(([w(f4) ** 2 / w(f3), w(f4) ** 2], [1, w(f4) / Q4, w(f4) ** 2], np.sqrt(f3 * f4)))
    if f5:
        S.append(([1, w(f5) / Q5, w(f5) ** 2], [1, w(f6) / Q6, w(f6) ** 2], np.sqrt(f5 * f6)))
    return S


def H(name, f, lp=True):
    s = 1j * 2 * np.pi * f
    out = np.ones_like(s)
    for b, a, _ in secs(name, lp):
        out *= np.polyval(b, s) / np.polyval(a, s)
    return out


def sos_at(name, fs):
    lp = 100.0 < fs / 2
    rows = []
    for b, a, fp in secs(name, lp):
        K = 2 * np.pi * fp / np.tan(np.pi * fp / fs)
        bz, az = signal.bilinear(b, a, fs=K / 2.0)
        rows.append(np.concatenate([np.pad(bz, (3 - len(bz), 0)) if len(bz) < 3 else bz, az]))
    return np.array(rows)


fgrid = np.geomspace(0.5, 40, 400)
for name in ["Wk", "Wd"]:
    for fs in [200.0, 400.0]:
        _, h = signal.sosfreqz(sos_at(name, fs), worN=fgrid, fs=fs)
        err = 20 * np.log10(np.abs(h) / np.abs(H(name, fgrid)))
        print(f"{name} IIR fs={fs:.0f}: max|err| 0.5-40 Hz = {np.max(np.abs(err)):.3f} dB")


def fft_weight(x, fs, name, pad_s=10.0):
    npad = int(pad_s * fs)
    xp = np.concatenate([np.zeros(npad), x, np.zeros(npad)])
    N = int(2 ** np.ceil(np.log2(len(xp))))
    X = np.fft.rfft(xp, N)
    f = np.fft.rfftfreq(N, 1 / fs)
    Hf = H(name, np.maximum(f, 1e-9))
    Hf[0] = 0.0
    y = np.fft.irfft(X * Hf, N)[npad:npad + len(x)]
    return y


fs = 100.0
t = np.arange(0, 120, 1 / fs)
for name in ["Wk", "Wd"]:
    worst = 0
    for f0 in [0.5, 1, 2, 4, 8, 12.5, 16, 25, 31.5, 40]:
        x = np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
        y = fft_weight(x, fs, name)[int(20 * fs):-int(20 * fs)]
        e = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.abs(H(name, np.array([f0])))[0])
        worst = max(worst, abs(e))
    print(f"{name} FFT weighting at fs=100: max|err| over test tones 0.5-40 Hz = {worst:.4f} dB")
# phase check vs analog for VDV: compare FFT-weighted transient with a high-rate IIR reference
fs_hi = 2000.0
th = np.arange(0, 20, 1 / fs_hi)
pulse = np.exp(-((th - 10) / 0.02) ** 2) * 5.0          # 40 ms Gaussian bump (impulsive)
ref = signal.sosfilt(sos_at("Wk", fs_hi), pulse)
x100 = signal.resample_poly(pulse, 1, 20)
y100 = fft_weight(x100, 100.0, "Wk")
iir100 = signal.sosfilt(sos_at("Wk", 100.0), x100)
ref100 = signal.resample_poly(ref, 1, 20)
vdv = lambda y, fs: (np.sum(y ** 4) / fs) ** 0.25
print(f"VDV of impulsive bump: high-rate reference={vdv(ref100,100):.4f}, FFT@100={vdv(y100,100):.4f}, IIR@100={vdv(iir100,100):.4f}")
print(f"peak |aw|: reference={np.max(np.abs(ref100)):.4f}, FFT@100={np.max(np.abs(y100)):.4f}, IIR@100={np.max(np.abs(iir100)):.4f}")
