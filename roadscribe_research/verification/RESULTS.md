# Verification scripts and raw outputs (run 2026-11-01)

Interpreter: `C:/Users/nishk/anaconda3/envs/tf/python.exe` (numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, statsmodels).
Data: `ml_model_work/data/raw/multi_class_kaggle/PVS 1..9` (read-only).
`pvs_align_doublehit.py` writes `pvs_doublehit.csv` next to itself. No other script writes anything.

| Script | Tests | Runtime |
|---|---|---|
| `pvs_facts.py` | sampling, GNSS, units, routes, labels, axis convention, L/R zero-lag correlation | ~2 min |
| `pvs_align_doublehit.py` | L/R clock offsets via yaw-rate cross-correlation; wheelbase double-hit cue | ~2 min |
| `pvs_maneuver_bands.py` | effect of maneuvers on vertical vibration by band (asphalt, OLS, trip-clustered SE) | ~1 min |
| `verify_orientation.py` | gated gravity + Rodrigues + Procrustes yaw on PVS; equivariance under random rotations | ~3 min |
| `verify_math.py` | ISO W_k/W_d analog values and IIR error at 100 Hz; MI-PELT pruning vs exhaustive OP; power; Williams design; ISO cut-points | ~1 min |
| `verify_fft_weighting.py` | IIR at 200/400 Hz; exact FFT weighting; VDV/peak of an impulsive bump vs 2 kHz reference | <1 min |

## pvs_facts.py

```
trip 1: n=144036 fs=100.0 gaps>50ms=0 gps_dt=1.0 dur=24.0min path=13.83km int_v=13.77km vmax=26.87 corr_ax_dvdt=-0.06 corr_ay_dvdt=0.43 corr_gz_courserate=-0.559 surf_changes=9 bumps=11 sb_v_med=3.62 quality_changes=230
trip 2: n=124684 dur=20.8 path=11.63 corr_ay_dvdt=0.412 surf_changes=4 bumps=7 sb_v_med=4.60 quality_changes=194
trip 3: n=105816 dur=17.6 path=10.74 corr_ay_dvdt=0.405 surf_changes=5 bumps=2 sb_v_med=4.52 quality_changes=115
trip 4: n=132492 dur=22.1 path=13.83 corr_ay_dvdt=0.692 surf_changes=9 bumps=11 sb_v_med=3.81 quality_changes=272
trip 5: n=133877 dur=22.3 path=11.64 corr_ay_dvdt=0.561 surf_changes=4 bumps=7 sb_v_med=5.09 quality_changes=138
trip 6: n=96279  dur=16.0 path=10.74 corr_ay_dvdt=0.531 surf_changes=5 bumps=2 sb_v_med=4.70 quality_changes=66
trip 7: n=128548 dur=21.4 path=13.79 corr_ay_dvdt=0.663 surf_changes=9 bumps=11 sb_v_med=4.04 quality_changes=202
trip 8: n=123618 dur=20.6 path=11.64 corr_ay_dvdt=0.572 surf_changes=4 bumps=7 sb_v_med=5.13 quality_changes=90
trip 9: n=91555  dur=15.3 path=10.76 corr_ay_dvdt=0.583 surf_changes=5 bumps=2 sb_v_med=4.13 quality_changes=68
L/R zero-lag correlation, 0.2-5 Hz (acc_z): T1 0.491, T2 0.87, T3 0.851, T4 0.641, T5 -0.09, T6 -0.192, T7 0.699, T8 0.689, T9 0.5
TOTAL dur_min=180.1 path_km=108.60 speed_bumps=60 surf_changes=54
```

## pvs_align_doublehit.py

```
trip yaw_lag_s yaw_peak az_lag_s az_peak az_r_at0 yaw_lag_first_third yaw_lag_last_third sameside_lag
1    -0.09     0.996    -0.09    0.908   0.489    -0.06               -0.12              0.00
2    -0.01     0.992    -0.01    0.914   0.908     0.00               -0.02              0.00
3     0.03     0.998     0.03    0.919   0.858     0.03                0.03              0.01
4     0.03     0.992     0.03    0.687   0.641     0.10               -0.05              0.00
5     0.54     0.986     0.51    0.555  -0.091     0.62                0.46              0.00
6     0.22     0.995     0.21    0.722  -0.234     0.26                0.15              0.01
7    -0.01     0.994    -0.01    0.719   0.714    -0.01                0.00              0.00
8     0.02     0.998     0.02    0.724   0.703     0.01                0.02              0.00
9     0.04     0.997     0.04    0.648   0.517     0.04                0.05              0.00

acc_z_above_suspension tau_ac: n=60 spearman(tau,1/v)=-0.121 p=0.36
acc_z_above_suspension tau_pk: n=60 spearman(tau,1/v)=-0.002 p=0.99
acc_z_dashboard        tau_ac: n=60 spearman(tau,1/v)=-0.002 p=0.99
acc_z_dashboard        tau_pk: n=60 spearman(tau,1/v)=-0.034 p=0.79
```

## pvs_maneuver_bands.py

```
windows: accel 23, brake 75, cruise 225, turn 92
lo (dash 0.5-4)   logv=0.323  accel -0.175 (SE 0.097, p .070)  brake -0.179 (0.053, p .0006)  turn +0.066 (0.080, p .41)
hi (dash 4-25)    logv=0.363  accel -0.057 (0.079, p .47)      brake -0.128 (0.057, p .025)   turn -0.044 (0.031, p .15)
wheel_hi (4-25)   logv=0.615  accel -0.367 (0.122, p .0028)    brake -0.160 (0.084, p .058)   turn +0.055 (0.047, p .25)
dash - wheel      accel +0.310 (0.087, p .0004)  brake +0.032 (0.033, p .34)  turn -0.099 (0.026, p .0001)
```

## verify_orientation.py

```
trip veh yaw_deg tilt_deg n_quasistatic equivariance_med_deg equivariance_max_deg
1    1   92.19   1.25     50791         0.231                0.389
2    1   91.30   3.74     47066         0.120                0.299
3    1   87.89   2.03     41968         0.185                0.401
4    2  104.14   0.95     47067         0.193                0.284
5    2  105.48   3.03     39102         0.279                0.633
6    2  103.05   2.41     38259         0.090                0.206
7    3   90.60   0.90     37941         0.208                0.409
8    3   92.52   3.17     37928         0.204                0.500
9    3   89.14   2.80     30216         0.125                0.277
```

The equivariance error comes from the componentwise median. The methodology specifies a geometric median, which is equivariant.

## verify_math.py

```
Wk analog |W|x1000: 0.5:418 0.63:459 0.8:477 1:482 1.25:485 1.6:494 2:531 2.5:631 3.15:804 4:967 5:1039 6.3:1054 8:1036 10:988 12.5:902 16:769 20:636 25:513 31.5:405 40:314
  digital fs=100 prewarp=True : max|err| 0.5-20 Hz = 0.716 dB, 0.5-40 Hz = 7.332 dB (at 40 Hz)
  digital fs=100 prewarp=False: max|err| 0.5-20 Hz = 1.157 dB, 0.5-40 Hz = 7.799 dB
Wd analog |W|x1000: 0.5:853 0.63:944 0.8:992 1:1011 1.25:1008 1.6:968 2:890 2.5:776 3.15:642 4:512 5:409 6.3:323 8:253 10:202 12.5:161 16:125 20:100 25:80 31.5:63 40:49
  digital fs=100 prewarp=True : max|err| 0.5-20 Hz = 1.222 dB, 0.5-40 Hz = 7.663 dB
|H_l| at 20 / 31.5 / 40 Hz: -0.007 / -0.043 / -0.110 dB
Wk-weighted RMS of 1 m/s^2 sine: 1 Hz 0.4824 (analog 0.4825); 5 Hz 1.0388 (1.0388); 16 Hz 0.7470 (0.7687)
(b) location-dependent penalty, 300 random series: corrected-PELT mismatches=0, classical-rule mismatches=300
(c) N=28 paired: min dz at 80% power = 0.549 (alpha .05), 0.646 (.05/3), 0.703 (.05/6); N for dz=0.5: 33.4
(d) Williams 4x4: [[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]  each ordered pair once: True  column-balanced: True
(e) ISO comfort cut-points: [0.315, 0.561, 0.894, 1.414, 2.236]
```

Non-inferiority power was computed inline with statsmodels `TTestPower` (one-sided, α = .05): N = 26.1 needed at d = 0.5; power 0.825 at N = 28; N = 40.0 at d = 0.4.

## verify_fft_weighting.py

```
Wk IIR fs=200: max|err| 0.5-40 Hz = 1.068 dB      Wk IIR fs=400: 0.210 dB
Wd IIR fs=200: max|err| 0.5-40 Hz = 1.150 dB      Wd IIR fs=400: 0.231 dB
Wk FFT weighting at fs=100: max|err| over test tones 0.5-40 Hz = 0.0000 dB
Wd FFT weighting at fs=100: max|err| over test tones 0.5-40 Hz = 0.0000 dB
VDV of impulsive bump: 2 kHz reference=1.3842, FFT@100=1.3842, IIR@100=1.3719
peak |aw|:              reference=3.7215, FFT@100=3.7220, IIR@100=3.5812
```

## run_full_evaluation.py (Narrative Benchmark, Paper §4)

Interpreter: `C:/Users/nishk/anaconda3/python.exe` (numpy 1.26.4, scipy 1.16.3).
Data: 9 PVS trips (108.6 km), 300 question-answering pairs (`roadscribe_qa_bank.jsonl`), 2,500 held-out persona samples (`test_e2.jsonl`).

```
=== PVS Benchmark (108.6 km, 9 trips) ===
Condition [T]:
  P_L: 99.15% | P_G: 93.21% | Halluc Missing: 0.00%
  rho_KSF: 0.483 | tau_b: 0.413 | BC: 68.9% | OI: -0.743
  AGP_L: 100.0% | AGR_L: 92.4%
  AGP_G: 18.2% | AGR_G: 100.0% | wAGR_G: 100.0%
  ECE_ling: 0.2875 | Plan Valid: 100.0% | Fallback: 0.0%

Condition [T+V]:
  P_L: 99.15% | P_G: 93.20% | Halluc Missing: 0.00%
  rho_KSF: 0.481 | tau_b: 0.411 | BC: 68.3% | OI: -0.759
  AGP_L: 100.0% | AGR_L: 92.4%
  AGP_G: 18.2% | AGR_G: 100.0% | wAGR_G: 100.0%
  ECE_ling: 0.2875 | Plan Valid: 100.0% | Fallback: 0.0%

Condition [S-PGR]:
  P_L: 98.90% | P_G: 92.97% | Halluc Missing: 0.00%
  rho_KSF: 0.499 | tau_b: 0.427 | BC: 70.2% | OI: -0.728
  AGP_L: 100.0% | AGR_L: 69.3%
  AGP_G: 24.2% | AGR_G: 100.0% | wAGR_G: 100.0%
  ECE_ling: 0.2914 | Plan Valid: 61.1% | Fallback: 72.2%

=== E1 Driver Question Answering (300 pairs) ===
Total Questions: 300 (Answerable: 240, Unanswerable: 60)
Baseline T Correct Rate: 59.7% | Abstention R: 75.0%
S-PGR Correct Rate:      100.0% | Abstention R: 100.0%

=== E2 Held-Out Personas (2,500 samples) ===
Evaluated 2,500 samples across held-out personas: {'car_sick_passenger', 'fleet_manager'}
Baseline T Nearest Persona Fit: 52.0%
S-PGR Grounded Persona Fit:    96.4% (P_L: 98.8%)
```
