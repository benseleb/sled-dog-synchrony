"""Windowed pairwise phase-locking value with a pooled time-shift surrogate
null (manuscript Eq. 3, SI S4), ported from
furriest_frequency_analysis.ipynb cell 72. Reproduces the notebook's
primary-window result exactly (22 of 2,070 pair-windows significant)."""
import numpy as np


def _plv_windows(t, window_s, step_s):
    """Window index arrays exactly as the manuscript notebook builds them
    (furriest_frequency_analysis.ipynb, make_windows)."""
    out = []
    for s in np.arange(t[0], t[-1] - window_s + 1e-9, step_s):
        idx = np.flatnonzero((t >= s) & (t < s + window_s))
        if idx.size and t[idx[-1]] - t[idx[0]] >= 0.98 * window_s:
            out.append(idx)
    return out


def _window_plvs(dphi, wins, min_valid_frac):
    """PLV per window, NaN where < min_valid_frac of samples are finite.
    Cumulative sums: identical to the notebook's per-window nanmean, but
    O(n) per surrogate lag instead of O(n * windows)."""
    ok = np.isfinite(dphi)
    z = np.where(ok, np.exp(1j * np.where(ok, dphi, 0.0)), 0.0)
    cz = np.concatenate([[0], np.cumsum(z)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    a = np.array([w[0] for w in wins]); b = np.array([w[-1] + 1 for w in wins])
    n_ok, n_all = cn[b] - cn[a], b - a
    with np.errstate(invalid="ignore", divide="ignore"):
        plv = np.abs((cz[b] - cz[a]) / n_ok)
    plv[(n_ok == 0) | (n_ok / n_all < min_valid_frac)] = np.nan
    return plv


def windowed_plv_pooled(t, phase_i, phase_j, window_s=60.0, step_s=15.0, n_surrogates=800,
                        plv_min=0.70, min_valid_frac=0.90, rng=None):
    """Manuscript PLV method (SI S4; furriest_frequency_analysis.ipynb cell
    72), ported: phases on a shared uniform grid t (s), NaN where the team
    is below the speed gate. One pooled null per pair: n_surrogates circular
    shifts of phase_j, lag uniform in [W/2, L - W/2], PLV of every window
    under every lag pooled; tau95 = its 95th percentile. A window is
    significant if PLV > tau95 and PLV > plv_min. Windows with < 90% valid
    samples get NaN PLV (not tested). Returns (rows, tau95)."""
    rng = np.random.default_rng(0) if rng is None else rng
    wins = _plv_windows(t, window_s, step_s)
    if not wins:
        return [], np.nan
    dt, length, n = np.median(np.diff(t)), t[-1] - t[0], phase_i.size
    tmin = window_s / 2.0
    if length - tmin <= tmin:
        return [], np.nan
    pool = []
    for _ in range(n_surrogates):
        k = int(np.round(rng.uniform(tmin, length - tmin) / dt)) % n
        k = k or 1
        pool.append(_window_plvs(np.angle(np.exp(1j * (phase_i - np.roll(phase_j, k)))), wins, min_valid_frac))
    pool = np.concatenate(pool)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return [], np.nan
    tau95 = np.percentile(pool, 95.0)
    obs = _window_plvs(np.angle(np.exp(1j * (phase_i - phase_j))), wins, min_valid_frac)
    rows = [dict(t_start=t[w[0]], t_end=t[w[-1]], plv=p, surrogate_95=tau95,
                 significant=bool(np.isfinite(p) and p > tau95 and p > plv_min))
            for w, p in zip(wins, obs)]
    return rows, tau95
