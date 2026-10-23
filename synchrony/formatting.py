"""Dec 11 primary-window data, formatted from the raw logger files.

Reproduces the manuscript's data prep (data_prep.ipynb, then the window crop
in furriest_data_trim.ipynb), so Table I and the Dec 11 figures need
nothing but the raw data:

  1. epoch in ms = epoch * 1000 + ms
  2. cut the file at the first change of the load-cell offset (a reboot)
  3. 0.5 Hz high-pass (2nd-order Butterworth, zero-phase) on ax, ay, az over
     the WHOLE remaining file -- filtfilt output depends on the span it sees,
     so this must not be applied to the cropped window
  4. crop to the primary window

Harper is the exception. Her device rebooted into three part files with
corrupted stretches, and the manuscript used a hand-stitched file. That file
is exactly: every row of all three parts with a Dec 11 timestamp, minus
firmware-corrupted rows (error > 2) and minus the row at each load-cell
offset change, in time order -- and step 3 then ran across its gaps. Both
are reproduced here.

The manuscript's files also carried a calibrated `force` column (N), which
the pipeline never uses (it works from load - offset), so it is not built.
"""
import numpy as np
import pandas as pd

from . import config
from .datasets import DEC11_DOGS, DEC11_RAW_FILES, RUN_START_MS, WINDOW_END_MS
from .phase import butter_filter

DAY = "DEC_11_TEST"
COLUMNS = ["epoch", "ax", "ay", "az", "gx", "gy", "gz", "load", "offset", "error"]
HARPER_PARTS = ["Harper_Device10_part1.csv", "Harper_Device10_part2.csv", "Harper_Device10_part3.csv"]
# Dec 11, 2022 00:00-24:00 UTC, in s; corrupted stretches carry other days' timestamps (e.g. Dec 7)
DAY_EPOCH_S = (1670716800, 1670803200)


def _read(name):
    raw = pd.read_csv(config.RAW_DIR / DAY / name, skiprows=4, low_memory=False)
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def _cut_at_offset_change(raw):
    """Keep rows up to (not including) the first change of `offset`."""
    change = np.flatnonzero(np.diff(raw["offset"].to_numpy()) != 0)
    return raw.iloc[:change[0] + 1] if change.size else raw


def _harper():
    rows = []
    for name in HARPER_PARTS:
        raw = _read(name)
        on_day = raw.epoch.between(*DAY_EPOCH_S) & (raw.error <= 2)
        # the first row after an offset change belongs to the next boot
        offset_change = raw.offset.ne(raw.offset.shift()) & (raw.index > 0)
        rows.append(raw[on_day & ~offset_change])
    out = pd.concat(rows).dropna()
    out = out.assign(_t=out.epoch * 1000 + out.ms).sort_values("_t").drop(columns="_t")
    return out.reset_index(drop=True)


def format_dog(dog):
    """One dog's formatted primary-window DataFrame (columns as COLUMNS)."""
    raw = _harper() if dog == "Harper" else _cut_at_offset_change(_read(DEC11_RAW_FILES[dog]))
    raw = raw.copy()
    raw["epoch"] = raw["epoch"] * 1000 + raw["ms"]
    for a in ("ax", "ay", "az"):
        raw[a] = butter_filter(raw[a].to_numpy(), 0.5, 125.0, "high", 2)
    win = raw[(raw.epoch >= RUN_START_MS - 243500) & (raw.epoch <= WINDOW_END_MS)]
    return win[COLUMNS].reset_index(drop=True)


def build_all(out_dir=None):
    """Write data_{i}.csv for every Dec 11 dog (i indexes DEC11_DOGS)."""
    out_dir = config.DEC11_FORMATTED_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, dog in enumerate(DEC11_DOGS):
        format_dog(dog).to_csv(out_dir / f"data_{i}.csv", index=False)
        print(f"  {dog:8s} -> {out_dir / f'data_{i}.csv'}")
