"""GPS speed latency, one value per run.

Run-level: high-passed GPS speed correlated with the median high-passed
activity envelope of the dogs in each qualifying window (data inside the
window only, each dog filtered within its own boot segments), window curves
combined duration-weighted. The median makes it robust to one dog with a bad
clock; dogs verified in sync are used when at least two are present.

Used if its peak is clear (margin >= MIN_MARGIN, r >= MIN_R), else
gps.GPS_LAG_S. One lag per run because no GPS log has a gap (> 1 s) within
any analysed span, so the logger's latency has no mechanism to change within
a run; per-window estimates are kept as diagnostics only (their scatter,
e.g. Dec 11 0.4-2.0 s, matches the ~+-1 s estimator noise of 2-4 min
windows).

Output: config.GPS_LAGS_CSV. Run after sync_check.
"""
import numpy as np
import pandas as pd

from . import config, sync_check as sc
from .gps import GPS_LAG_S
from .windows import window_csv

OUT = config.GPS_LAGS_CSV
SHIFTS = np.arange(-8.0, 8.01, sc.GRID_STEP)   # symmetric, so the margin rule isn't biased by the range
MIN_MARGIN = 0.2
MIN_R = 0.3
DAYS = [(d, g) for d, g, tier in sc.DAYS if tier in (1, 2, 3)]


def window_curve(day, gps_file, lo, hi, dogs, boot):
    """Correlation of team activity with GPS speed vs shift for [lo, hi], or None."""
    grid = np.arange(lo, hi, sc.GRID_STEP)
    t_gps, v_gps = sc.read_gps(sc.DATA / day / gps_file)
    speed = sc.highpass_envelope(np.interp(grid, t_gps, v_gps, left=np.nan, right=np.nan))
    files = sc.dog_files(day)
    verified = [d for d in dogs if ((boot.day == day) & (boot.dog == d) & (boot.boot_start < hi)
                                    & (boot.boot_end > lo) & (boot.verdict == "in sync")).any()]
    use = verified if len(verified) >= 2 else dogs
    envs = [sc.envelope_pieces(files[d], lo, hi, grid) for d in use if d in files]
    envs = [e for e in envs if np.isfinite(e).sum() * sc.GRID_STEP >= sc.WIN_MIN_OVERLAP_S]
    if not envs:
        return None, 0
    with np.errstate(all="ignore"):
        team = np.nanmedian(np.array(envs), axis=0)
    # Same call and sign convention as sync_check (GPS vs dog): the lag to add
    # to GPS timestamps (GPS_LAG_S convention) is MINUS the peak shift.
    # Reproduces Dec 11 = +1.2 s.
    return sc.lag_curve(speed, team, SHIFTS, sc.WIN_MIN_OVERLAP_S), len(envs)


def estimate_run_lags():
    boot = pd.read_csv(sc.OUT / "sync_check_boot_segments.csv")
    rows = []
    for day, gps_file in DAYS:
        w = pd.read_csv(window_csv(day))
        w = w[w.duration_s >= 120].sort_values("start_epoch")
        if w.empty:
            continue
        curves = []
        for r in w.itertuples():
            curve, n = window_curve(day, gps_file, r.start_epoch, r.end_epoch, r.dogs.split("|"), boot)
            if curve is None:
                rows.append(dict(day=day, window_start=r.start_epoch, duration_s=r.duration_s, n_dogs=0))
                continue
            curves.append((curve, r.duration_s))
            lag, rr, margin = sc.peak_stats(curve, SHIFTS)
            rows.append(dict(day=day, window_start=r.start_epoch, duration_s=r.duration_s, n_dogs=n,
                             lag_window_s=-lag, r=rr, margin=margin))
        # run level: duration-weighted mean of the window curves
        if curves:
            wsum = sum(d for _, d in curves)
            run_curve = sum(np.nan_to_num(c) * d for c, d in curves) / wsum
            run_lag, run_r, run_margin = sc.peak_stats(run_curve, SHIFTS)
            for row in rows:
                if row["day"] == day:
                    row.update(lag_run_s=-run_lag, r_run=run_r, margin_run=run_margin)
    df = pd.DataFrame(rows)
    use_run = (df.margin_run >= MIN_MARGIN) & (df.r_run >= MIN_R)
    df["gps_lag_s"] = np.where(use_run, df.lag_run_s, GPS_LAG_S)
    df["source"] = np.where(use_run, "run", "global default")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def lag_for_day(day, table=None):
    """GPS lag (s, added to GPS timestamps) for a run; GPS_LAG_S if unknown."""
    t = pd.read_csv(OUT) if table is None else table
    m = t[t.day == day]
    return float(m.gps_lag_s.iloc[0]) if len(m) else GPS_LAG_S
