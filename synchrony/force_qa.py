"""Standing load-cell QA for every dog-day in datasets.QUEUE, over
its qualifying windows. Run before trusting any force-phase regression on a
new day/dog, and whenever the force pipeline changes.

Each check below either found a real failure in the 2022 data (2026-10-01)
or was checked and came up clean -- kept so a new day can't slip one past:

  sentinel_s     load == -999 / 404 (firmware "no reading" / "abandoned").
                 Found: Alice (~23 min) and Sitka (~8.5 min) on Dec 11.
  dead           constant load throughout (or no load column). Found: Blaise,
                 Chuck on Dec 05, whose firmware (dec04b) logged no load.
  max_drift      slow-baseline excursion within a window, in stride-pulse
                 units (see force.force_valid_mask). Found: Trillium Dec 07
                 (3.6; load cell collapses from ~145k to -120k counts in the
                 last ~5 min). Normal <=0.55; masked above FORCE_DRIFT_MAX.
  baseline_step  jump BETWEEN a dog's windows (handled by per-window
                 baseline removal in analysis.load_dog_windows). Found: Tiki Dec 11.
  clip_frac      samples pinned at the series max/min. Clean everywhere.
  flat_s         longest non-sentinel constant run (expected ~0.016 s: the
                 ADC updates every other sample). Clean everywhere.
  pulse_skew     skew of 1 s-detrended force while moving; >0 means tension
                 pulses point up, which signed-argmax peak picking assumes.
                 +0.45..+1.96 everywhere -- no inverted load cells.
  lock_frac      fraction of force variance explained by the mean per-stride
                 waveform (force that's actually stride-locked); 0.47-0.78
                 everywhere, nothing near noise.
  usable_s       moving, in-window seconds with valid force.

verdict: dead / exclude (<120 s usable) / partial (>10 s masked) / ok.
Isolated sentinel samples (a few seconds total) are common and harmless once
masked -- but unmasked they were real hazards: 404 - offset is a large
POSITIVE spike for a dog with a negative offset (Tizzy: +50k counts), which
signed-argmax peak picking would select.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew

from . import config
from .analysis import load_dog_windows
from .datasets import QUEUE, MIN_WINDOW_S, SPEED_MIN

OUT = config.FORCE_SCAN_CSV
FS = 125.0
MIN_USABLE_S = MIN_WINDOW_S
PARTIAL_MASKED_S = 10.0


def stride_lock_fraction(t, zc, force, ok_stride, n_bins=20):
    W = []
    T = np.diff(zc)
    for k in np.flatnonzero(ok_stride):
        W.append(np.interp(zc[k] + np.linspace(0, 1, n_bins, endpoint=False) * T[k], t, force))
    W = np.array(W)
    W = W[np.isfinite(W).all(axis=1)]
    if len(W) < 20:
        return np.nan
    W -= W.mean(axis=1, keepdims=True)
    return W.mean(axis=0).var() / W.var()


def scan(day, dog, dog_file, gps_file):
    d = load_dog_windows(day, dog, dog_file, gps_file)
    diag, valid = d["force_diag"], d["force_valid"]
    row = dict(day=day, dog=dog, sentinel_s=diag["sentinel_s"], dead=diag["dead"],
               max_drift=diag["max_drift"], drift_masked_s=diag["drift_s"],
               baseline_step=d["baseline_step"])
    moving = d["speed_gated"] >= SPEED_MIN
    use = moving & valid
    row["usable_s"] = use.sum() / FS
    if diag["dead"] or use.sum() < FS * 10:
        row["verdict"] = "dead" if diag["dead"] else "exclude"
        return row

    load = d["load"][use]
    row["clip_frac"] = np.mean((load == load.max()) | (load == load.min()))
    good_load = d["load"][valid]
    change = np.flatnonzero(np.diff(good_load) != 0)
    row["flat_s"] = np.diff(np.r_[-1, change, good_load.size - 1]).max() / FS

    x = pd.Series(d["force_aligned"])
    x[~valid] = np.nan
    pulse = x - x.rolling(int(FS), center=True, min_periods=int(FS) // 2).median()
    row["pulse_skew"] = skew(pulse[use].dropna())

    zc, t = d["zc"], d["t"]
    T = np.diff(zc)
    sp = np.nan_to_num(d["speed_gated"])
    ok = (T > 0.2) & (T < 0.6) & (np.interp(zc[:-1], t, sp) >= SPEED_MIN) & (np.interp(zc[1:], t, sp) >= SPEED_MIN)
    row["lock_frac"] = stride_lock_fraction(t, zc, x.to_numpy(), ok)

    masked = diag["sentinel_s"] + diag["drift_s"]
    if row["usable_s"] < MIN_USABLE_S:
        row["verdict"] = "exclude"
    elif masked > PARTIAL_MASKED_S:
        row["verdict"] = "partial"
    else:
        row["verdict"] = "ok"
    return row


def scan_all():
    rows = []
    for item in QUEUE:
        rows.append(scan(*item))
        print(f"  {item[0]:17s} {item[1]:9s} {rows[-1]['verdict']}", flush=True)
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"\nSaved -> {OUT}")
    return res
