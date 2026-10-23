"""Raw sensor CSV loading and corruption trimming.

Raw logger CSV: 4 metadata lines, then header
epoch,ms,ax,ay,az,gx,gy,gz,load,offset,error at 125 Hz. `epoch` is whole
seconds; sub-second time is in `ms`, so always use epoch + ms/1000.
"""
import csv
from pathlib import Path

import numpy as np

GAP_SECONDS = 1.0
ERROR_THRESHOLD = 2.0
NEG_LEAP_THRESHOLD_S = -0.5
POS_LEAP_THRESHOLD_S = 60.0
# Corruption sends load to +-1e9-scale garbage; real strong-pull tension
# reaches ~2e5 counts, so the floor sits well above that. epoch == 0 and the
# uint32/uint16 maxima are hardware fill patterns seen in corrupted tails.
# load == 404 is the firmware's intentional "load cell abandoned" marker, not
# corruption (see force.LOAD_SENTINELS).
GARBAGE_MAG = 1.0e7
UINT32_MAX = 4294967295
UINT16_MAX = 65535
COLS = ["epoch", "ms", "ax", "ay", "az", "gx", "gy", "gz", "load", "offset", "error"]
HEADER_SKIP = 4


def infer_dog_id(path: Path) -> str:
    stem = Path(path).stem
    return stem.split("_")[0] if "_" in stem else stem


def load_rows(path):
    data = {c: [] for c in COLS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        # Metadata lines precede the column header: HEADER_SKIP (4) on most
        # firmware, 5 on nov18a/nov19a (Nov 17-20). Find the header row
        # instead of assuming its position.
        header = None
        for _ in range(HEADER_SKIP + 4):
            row = next(reader, None)
            if row and row[0].strip().lower() == "epoch":
                header = row
                break
        if header is None:
            return {c: np.array([]) for c in COLS}
        header = [h.strip().lower() for h in header]
        idx = {c: (header.index(c) if c in header else None) for c in COLS}
        if idx["epoch"] is None:
            idx["epoch"] = 0
        for row in reader:
            try:
                vals = {c: (float(row[idx[c]]) if idx[c] is not None else 0.0) for c in COLS}
            except (ValueError, IndexError):
                break
            for c in COLS:
                data[c].append(vals[c])
    return {c: np.array(v) for c, v in data.items()}


def _first_sanity_fail(data):
    n = data["epoch"].size
    if n == 0:
        return None
    nan_mask = np.zeros(n, dtype=bool)
    for c in ("ax", "ay", "az", "gx", "gy", "gz", "load"):
        nan_mask |= np.isnan(data[c])
    epoch, ms, load, offset = data["epoch"], data["ms"], data["load"], data["offset"]
    bad = (nan_mask | (epoch <= 0) | (epoch >= UINT32_MAX - 1) | (ms >= UINT16_MAX - 1)
           | ((np.abs(load) > GARBAGE_MAG) & (load != 404)) | (np.abs(offset) > GARBAGE_MAG))
    idx = np.where(bad)[0]
    return int(idx[0]) if idx.size else None


def trim_corruption(data):
    epochs = data["epoch"]
    if epochs.size == 0:
        return data
    errors = data["error"]
    bad = np.where(errors > ERROR_THRESHOLD)[0]
    err_pos = int(bad[0]) if bad.size else None
    diffs = np.diff(epochs)
    leap_idx = np.where((diffs < NEG_LEAP_THRESHOLD_S) | (diffs > POS_LEAP_THRESHOLD_S))[0]
    leap_pos = int(leap_idx[0] + 1) if leap_idx.size else None
    sanity_pos = _first_sanity_fail(data)
    candidates = [p for p in (err_pos, leap_pos, sanity_pos) if p is not None]
    cut = min(candidates) if candidates else epochs.size
    return {c: v[:cut] for c, v in data.items()}


def load_dog(csv_path, run_start_epoch=None):
    """Load one dog's raw CSV, corruption-trim, and (optionally) cut to the
    true run-start. Returns dict with full-precision t (seconds from
    run-start), az, load, offset, and t0 (the absolute epoch t=0 corresponds
    to, in seconds -- needed to align GPS speed onto the same t axis)."""
    data = trim_corruption(load_rows(csv_path))
    if data["epoch"].size < 200:
        return None
    t_full = data["epoch"] + data["ms"] / 1000.0
    if run_start_epoch is not None:
        mask = data["epoch"] >= run_start_epoch
        if mask.sum() < 200:
            return None
        data = {c: v[mask] for c, v in data.items()}
        t_full = t_full[mask]
    return dict(t=t_full - t_full[0], t0=t_full[0], az=data["az"], load=data["load"], offset=data["offset"])


def contiguous_segments(epoch_s: np.ndarray, gap_s: float = GAP_SECONDS):
    if epoch_s.size == 0:
        return []
    diffs = np.diff(epoch_s)
    cuts = np.where(diffs > gap_s)[0]
    starts = [epoch_s[0]] + [epoch_s[i + 1] for i in cuts]
    ends = [epoch_s[i] for i in cuts] + [epoch_s[-1]]
    return [(float(s), float(e)) for s, e in zip(starts, ends) if e >= s]


def coalesce(segs, eps=1e-9):
    if not segs:
        return []
    segs = sorted(segs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        cs, ce = merged[-1]
        if s <= ce + eps:
            merged[-1] = (cs, max(ce, e))
        else:
            merged.append((s, e))
    return merged
