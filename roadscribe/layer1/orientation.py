r"""Body-to-vehicle orientation (METHODOLOGY §3.1).

Pipeline per mount episode :math:`e`:

0. *Mount episodes.* Candidate handling where
   :math:`\lVert\bar\omega^b\rVert-|\dot\psi^G|>\Omega_h`; a candidate splits
   the trip when the quasi-static gravity directions on its two sides differ
   by more than :math:`\theta_h`. Samples within ±5 s of a split are
   discarded (episode index ``-1``).
1. *Gated gravity.* Geometric median of the 0.5 Hz low-passed specific force
   over the quasi-static set :math:`\mathcal Q_e`. Unlike a componentwise
   median, the geometric median is rotation-equivariant.
2. *Tilt.* Rodrigues rotation :math:`\mathbf R_1` carrying the up vector onto
   :math:`\mathbf e_z`.
3. *Yaw.* Closed-form 2-D Procrustes fit of the leveled horizontal specific
   force against GNSS longitudinal and centripetal acceleration.
4. :math:`\mathbf R_{vb}=\mathbf R_z(\hat\psi)\mathbf R_1` and
   :math:`\mathbf a^v=\mathbf R_{vb}\mathbf f^b-g\mathbf e_z`.

Sign convention. ``yaw_matrix(psi)`` is the 3-D embedding of the methodology's
:math:`\mathbf R_2(\psi)=[[\cos\psi,\sin\psi],[-\sin\psi,\cos\psi]]`, which maps
leveled sensor axes to (forward, left). A sensor whose forward axis is
:math:`+y` (PVS) therefore has :math:`\hat\psi=+90^\circ`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..config import OrientationConfig
from ..schemas import RawTrip, VehicleFrameTrip
from .gnss import along_track_distance, clean_fixes, course_rate_deg_s, speed_and_alon

__all__ = [
    "OrientationError",
    "EpisodeOrientation",
    "OrientationResult",
    "lowpass",
    "geometric_median",
    "rodrigues_to_z",
    "yaw_matrix",
    "procrustes_yaw",
    "quasi_static_mask",
    "bootstrap_yaw_ci",
    "detect_mount_episodes",
    "estimate_orientation",
    "to_vehicle_frame",
    "rotation_angle_deg",
]

log = logging.getLogger(__name__)
E_Z = np.array([0.0, 0.0, 1.0])


class OrientationError(RuntimeError):
    """Raised when an episode has too little quasi-static data or GNSS excitation."""


@dataclass(frozen=True)
class EpisodeOrientation:
    episode: int
    i0: int  # first IMU sample of the episode (inclusive)
    i1: int  # last IMU sample of the episode (exclusive)
    R_vb: np.ndarray  # 3x3, vehicle <- body
    R_tilt: np.ndarray  # R_1 (leveling)
    up_b: np.ndarray  # unit up vector in the body frame
    g_norm: float  # |g-hat|, a check on accelerometer scale
    tilt_deg: float
    yaw_deg: float
    yaw_ci_deg: tuple[float, float]
    n_quasistatic: int
    n_fixes: int
    excitation: float  # sum_k ||r_k||^2, (m/s^2)^2
    yaw_reference: str  # "gyro" or "gnss_course"


@dataclass(frozen=True)
class OrientationResult:
    episodes: list[EpisodeOrientation]
    episode_index: np.ndarray  # (N,) episode per IMU sample, -1 = discarded
    splits_t: list[float]  # times of confirmed handling splits


# --------------------------------------------------------------------------- primitives
def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    sos = signal.butter(order, fc / (fs / 2.0), "low", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def geometric_median(X: np.ndarray, tol: float = 1e-12, max_iter: int = 1000) -> np.ndarray:
    """Geometric median (Weiszfeld iteration with the Vardi-Zhang fix).

    Starts from the mean, and every update is rotation-equivariant, so the
    estimate rotates exactly with the data (to within ``tol``).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a non-empty (n, d) array")
    if len(X) == 1:
        return X[0].copy()
    y = X.mean(axis=0)
    scale = max(1.0, float(np.linalg.norm(y)))
    for _ in range(max_iter):
        diff = X - y
        d = np.linalg.norm(diff, axis=1)
        nz = d > 1e-15 * scale
        w = 1.0 / d[nz]
        T = (w[:, None] * X[nz]).sum(axis=0) / w.sum()
        eta = int((~nz).sum())
        if eta == 0:
            y_new = T
        else:
            r = float(np.linalg.norm((w[:, None] * diff[nz]).sum(axis=0)))
            gamma = 1.0 if r == 0.0 else min(1.0, eta / r)
            y_new = (1.0 - gamma) * T + gamma * y
        step = float(np.linalg.norm(y_new - y))
        y = y_new
        if step <= tol * scale:
            break
    return y


def _skew(k: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def rodrigues_to_z(u: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Rotation R_1 with R_1 u/|u| = e_z (Rodrigues), including both degenerate cases."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    k = np.cross(u, E_Z)
    s, c = float(np.linalg.norm(k)), float(u @ E_Z)
    if s < eps:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = _skew(k / s)
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def yaw_matrix(psi: float) -> np.ndarray:
    """3-D embedding of R_2(psi) = [[cos, sin], [-sin, cos]] (see module docstring)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def procrustes_yaw(h: np.ndarray, r: np.ndarray) -> float:
    """argmin_psi sum_k ||R_2(psi) h_k - r_k||^2 in closed form (radians).

    With M = sum_k h_k r_k^T, psi = atan2(M_21 - M_12, M_11 + M_22).
    """
    h = np.asarray(h, float)
    r = np.asarray(r, float)
    M = h.T @ r
    return float(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1]))


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def bootstrap_yaw_ci(
    h: np.ndarray,
    r: np.ndarray,
    psi_hat: float,
    n_boot: int,
    block: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Moving-block bootstrap CI of the Procrustes yaw over fixes (degrees).

    Consecutive GNSS fixes are autocorrelated, so fixes are resampled in
    blocks of ``block`` fixes; ``block=1`` is the plain bootstrap.
    """
    n = len(h)
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    deltas = np.empty(n_boot)
    offs = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        deltas[b] = _wrap(procrustes_yaw(h[idx], r[idx]) - psi_hat)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return float(np.degrees(psi_hat + lo)), float(np.degrees(psi_hat + hi))


def quasi_static_mask(
    f_lp: np.ndarray,
    w_lp: np.ndarray | None,
    a_lon: np.ndarray,
    v: np.ndarray,
    cfg: OrientationConfig,
) -> np.ndarray:
    """The quasi-static set Q_e of §3.1, Step 1 (gyro gate skipped without a gyroscope)."""
    q = np.abs(np.linalg.norm(f_lp, axis=1) - cfg.g) < cfg.eps_g
    if w_lp is not None:
        q &= np.linalg.norm(w_lp, axis=1) < np.radians(cfg.eps_omega_deg)
    q &= np.abs(a_lon) < cfg.eps_a
    q &= v > cfg.v_min
    return q


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the relative rotation Ra Rb^T, degrees."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


# --------------------------------------------------------------------------- Step 0
def detect_mount_episodes(
    t: np.ndarray,
    f_lp: np.ndarray,
    w_lp: np.ndarray | None,
    course_rate_rad_s: np.ndarray,
    qs: np.ndarray,
    cfg: OrientationConfig,
) -> tuple[np.ndarray, list[float]]:
    """Episode index per sample (-1 = discarded) and the confirmed split times."""
    n = len(t)
    episode = np.zeros(n, dtype=int)
    if w_lp is None:
        return episode, []
    cr = np.nan_to_num(np.abs(course_rate_rad_s), nan=0.0)
    cand = np.linalg.norm(w_lp, axis=1) - cr > np.radians(cfg.handling_omega_deg)
    runs = _runs(cand)
    if not runs:
        return episode, []
    # merge candidates closer than the guard interval
    merged: list[list[int]] = [list(runs[0])]
    for a, b in runs[1:]:
        if t[a] - t[merged[-1][1] - 1] < cfg.handling_guard_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    splits: list[tuple[int, int]] = []
    for k, (a, b) in enumerate(merged):
        t_a, t_b = t[a], t[b - 1]
        prev_end = t[merged[k - 1][1] - 1] + cfg.handling_guard_s if k > 0 else -np.inf
        next_start = t[merged[k + 1][0]] - cfg.handling_guard_s if k + 1 < len(merged) else np.inf
        left = qs & (t < t_a - cfg.handling_guard_s) & (t >= max(prev_end, t_a - cfg.handling_guard_s - cfg.handling_side_window_s))
        right = qs & (t > t_b + cfg.handling_guard_s) & (t <= min(next_start, t_b + cfg.handling_guard_s + cfg.handling_side_window_s))
        if left.sum() < cfg.min_quasistatic or right.sum() < cfg.min_quasistatic:
            log.info("handling candidate at t=%.1f s: not enough quasi-static data to confirm", t_a)
            continue
        g_l = geometric_median(f_lp[left], cfg.geomedian_tol, cfg.geomedian_max_iter)
        g_r = geometric_median(f_lp[right], cfg.geomedian_tol, cfg.geomedian_max_iter)
        if _angle_between_deg(g_l, g_r) > cfg.handling_theta_deg:
            splits.append((a, b))

    split_times: list[float] = []
    ep = 0
    last = 0
    for a, b in splits:
        lo = int(np.searchsorted(t, t[a] - cfg.handling_guard_s, "left"))
        hi = int(np.searchsorted(t, t[b - 1] + cfg.handling_guard_s, "right"))
        episode[last:lo] = ep
        episode[lo:hi] = -1
        ep += 1
        last = hi
        split_times.append(float(0.5 * (t[a] + t[b - 1])))
    episode[last:] = ep
    # guard windows of nearby splits can swallow a whole episode: renumber 0..K-1
    kept = np.unique(episode[episode >= 0])
    remap = np.full(int(episode.max()) + 2, -1)
    remap[kept] = np.arange(len(kept))
    episode = np.where(episode >= 0, remap[episode], -1)
    return episode, split_times


# --------------------------------------------------------------------------- Steps 1-3
def _estimate_episode(
    ep: int,
    sel: np.ndarray,
    t: np.ndarray,
    f_lp: np.ndarray,
    w_lp: np.ndarray | None,
    qs: np.ndarray,
    fix_t: np.ndarray,
    fix_v: np.ndarray,
    fix_alon: np.ndarray,
    fix_course_rate: np.ndarray,
    cfg: OrientationConfig,
    rng: np.random.Generator,
) -> EpisodeOrientation:
    idx = np.flatnonzero(sel)
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    q = qs & sel
    n_qs = int(q.sum())
    if n_qs < cfg.min_quasistatic:
        raise OrientationError(
            f"episode {ep}: only {n_qs} quasi-static samples (< {cfg.min_quasistatic})"
        )
    g_hat = geometric_median(f_lp[q], cfg.geomedian_tol, cfg.geomedian_max_iter)
    up = g_hat / np.linalg.norm(g_hat)
    R1 = rodrigues_to_z(up)

    # GNSS fixes inside the episode (excluding discarded samples), moving fast enough
    t_sel = t[idx]
    inside = (fix_t >= t_sel[0]) & (fix_t <= t_sel[-1])
    # drop fixes whose IMU neighbourhood is discarded (episode boundary guard)
    nearest = np.clip(np.searchsorted(t, fix_t), 0, len(t) - 1)
    inside &= sel[nearest]
    use = inside & (fix_v > cfg.v_min_yaw) & np.isfinite(fix_alon)

    fl = f_lp[i0:i1] @ R1.T
    tt = t[i0:i1]
    if w_lp is not None:
        wz = (w_lp[i0:i1] @ R1.T)[:, 2]
        wz_k = np.interp(fix_t, tt, wz)
        yaw_ref = "gyro"
    else:
        # left turn = positive yaw rate = negative (clockwise) course rate
        wz_k = -fix_course_rate
        use &= np.isfinite(wz_k)
        yaw_ref = "gnss_course"
    n_fix = int(use.sum())
    if n_fix < cfg.min_fixes:
        raise OrientationError(f"episode {ep}: only {n_fix} usable GNSS fixes (< {cfg.min_fixes})")
    H = np.column_stack([np.interp(fix_t[use], tt, fl[:, 0]), np.interp(fix_t[use], tt, fl[:, 1])])
    r = np.column_stack([fix_alon[use], fix_v[use] * wz_k[use]])
    psi = procrustes_yaw(H, r)
    dt_fix = float(np.median(np.diff(fix_t))) if len(fix_t) > 1 else 1.0
    block = int(round(cfg.bootstrap_block_s / max(dt_fix, 1e-9)))
    ci = bootstrap_yaw_ci(H, r, psi, cfg.bootstrap_n, block, cfg.bootstrap_level, rng)
    R_vb = yaw_matrix(psi) @ R1
    tilt = float(np.degrees(np.arccos(np.clip(up @ E_Z, -1.0, 1.0))))
    return EpisodeOrientation(
        episode=ep,
        i0=i0,
        i1=i1,
        R_vb=R_vb,
        R_tilt=R1,
        up_b=up,
        g_norm=float(np.linalg.norm(g_hat)),
        tilt_deg=tilt,
        yaw_deg=float(np.degrees(psi)),
        yaw_ci_deg=ci,
        n_quasistatic=n_qs,
        n_fixes=n_fix,
        excitation=float(np.sum(r**2)),
        yaw_reference=yaw_ref,
    )


def estimate_orientation(raw: RawTrip, cfg: OrientationConfig | None = None) -> OrientationResult:
    """Estimate R_vb per mount episode for a :class:`RawTrip`."""
    cfg = cfg or OrientationConfig()
    rng = np.random.default_rng(cfg.seed)
    t, fs = raw.t, raw.fs
    f_lp = lowpass(raw.f_b, fs, cfg.lp_hz, cfg.lp_order)
    w_lp = lowpass(raw.w_b, fs, cfg.lp_hz, cfg.lp_order) if raw.w_b is not None else None

    fixes = clean_fixes(raw.gnss)
    v, a_lon = speed_and_alon(t, fixes.t, fixes.speed)
    _, fix_alon = speed_and_alon(fixes.t, fixes.t, fixes.speed)
    # the first and last fix have one-sided derivatives; leave them out of the fit
    fix_alon = fix_alon.copy()
    fix_alon[[0, -1]] = np.nan
    course_rate = np.radians(course_rate_deg_s(fixes, v_min=cfg.v_min_yaw))
    cr_imu = np.interp(t, fixes.t, np.nan_to_num(course_rate, nan=0.0))

    qs = quasi_static_mask(f_lp, w_lp, a_lon, v, cfg)
    episode, splits = detect_mount_episodes(t, f_lp, w_lp, cr_imu, qs, cfg)
    episodes = []
    for ep in range(int(episode.max()) + 1):
        sel = episode == ep
        if not sel.any():
            continue
        episodes.append(
            _estimate_episode(
                ep, sel, t, f_lp, w_lp, qs, fixes.t, fixes.speed, fix_alon, course_rate, cfg, rng
            )
        )
    return OrientationResult(episodes=episodes, episode_index=episode, splits_t=splits)


# --------------------------------------------------------------------------- Step 4
def to_vehicle_frame(
    raw: RawTrip,
    cfg: OrientationConfig | None = None,
    result: OrientationResult | None = None,
    s: np.ndarray | None = None,
) -> VehicleFrameTrip:
    """Rotate a :class:`RawTrip` into the vehicle frame (§3.1, Step 4).

    Samples discarded around a handling split (episode ``-1``) are rotated with
    the nearest episode's ``R_vb`` so that downstream filters see a continuous
    signal; their episode index stays ``-1`` so they can be masked.
    ``s`` defaults to the integral of GNSS speed (replace it with the
    map-matched arc length once Layer 2 has run).
    """
    cfg = cfg or OrientationConfig()
    result = result or estimate_orientation(raw, cfg)
    t = raw.t
    n = len(t)
    fixes = clean_fixes(raw.gnss)
    v, a_lon = speed_and_alon(t, fixes.t, fixes.speed)

    eps = result.episodes
    centres = np.array([0.5 * (t[e.i0] + t[e.i1 - 1]) for e in eps])
    which = np.empty(n, dtype=int)
    for k, e in enumerate(eps):
        which[result.episode_index == e.episode] = k
    disc = np.flatnonzero(result.episode_index < 0)
    if len(disc):
        # nearest episode by time
        which[disc] = np.argmin(np.abs(t[disc, None] - centres[None, :]), axis=1)
    R = np.stack([e.R_vb for e in eps])  # (K,3,3)
    a_v = np.einsum("nij,nj->ni", R[which], raw.f_b) - cfg.g * E_Z
    w_v = np.einsum("nij,nj->ni", R[which], raw.w_b) if raw.w_b is not None else None
    if s is None:
        s = along_track_distance(t, v)
    return VehicleFrameTrip(
        trip_id=raw.trip_id,
        t=t,
        fs=raw.fs,
        a_v=a_v,
        w_v=w_v,
        v=v,
        a_lon_gnss=a_lon,
        s=s,
        episode=result.episode_index.copy(),
        R_vb=[e.R_vb for e in eps],
        yaw_deg=[e.yaw_deg for e in eps],
        yaw_ci_deg=[e.yaw_ci_deg for e in eps],
        n_quasistatic=[e.n_quasistatic for e in eps],
    )
