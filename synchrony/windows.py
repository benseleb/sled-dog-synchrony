"""Qualifying windows: stretches where >= 2 dogs record at once.

Accounts for each dog's own gaps and reconnections (a dog with 40 fragmented
segments only counts during its covered stretches). Windows are kept whole,
not clipped to high-speed sub-portions; a window is dropped only if it never
reaches 3 m/s. Each kept window reports frac_moving (fraction of its
duration >= 3 m/s) and max_speed_mps. Downstream, only windows
>= datasets.MIN_WINDOW_S are analysed.

One CSV per run in config.WINDOWS_DIR: start_epoch, end_epoch, duration_s,
n_dogs, dogs (|-joined), frac_moving, max_speed_mps.
"""
import csv

import numpy as np
import pandas as pd

from . import config
from .datasets import RUN, MIN_WINDOW_S, WINDOW_PAD_S, SPEED_MIN
from .gps import gps_speed_interp
from .io import load_rows, trim_corruption, contiguous_segments, coalesce, infer_dog_id

MIN_OVERLAP_S = 5.0
SPEED_GRID_DT = 1.0  # seconds; GPS speed is a smoothed spline, no need for sample-rate resolution


def window_csv(day):
    return config.WINDOWS_DIR / f"{day.replace('/', '_')}.csv"


def per_dog_segments(day_dir):
    segments_by_dog = {}
    for p in sorted(day_dir.glob("*.csv")):
        if "GPSLogger" in p.name:
            continue
        dog = infer_dog_id(p)
        epochs = trim_corruption(load_rows(p))["epoch"]
        segs = contiguous_segments(epochs)
        segments_by_dog.setdefault(dog, []).extend(segs)
    return {d: coalesce(segs) for d, segs in segments_by_dog.items()}


def sweep_active_dogs(segments_by_dog):
    """Returns merged (start, end, frozenset(active_dogs)) intervals with >=2
    dogs active, by checking which dogs cover the midpoint of each interval
    between consecutive segment-boundary times."""
    times = sorted({t for segs in segments_by_dog.values() for pair in segs for t in pair})
    raw = []
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if t1 - t0 < 1e-6:
            continue
        mid = (t0 + t1) / 2
        active = frozenset(dog for dog, segs in segments_by_dog.items()
                            if any(s <= mid < e for s, e in segs))
        if len(active) >= 2:
            raw.append((t0, t1, active))
    merged = []
    for t0, t1, active in raw:
        if merged and merged[-1][2] == active and t0 - merged[-1][1] < 1e-6:
            merged[-1] = (merged[-1][0], t1, active)
        else:
            merged.append((t0, t1, active))
    return merged


def speed_profile(gps_path, t0, t1, speed_min=SPEED_MIN, dt=SPEED_GRID_DT):
    """(max_speed, frac_time_moving) over [t0,t1] -- doesn't clip/split the
    window, just characterizes it. A window entirely below speed_min has zero
    chance of real running gait (max_speed check); anything else is kept
    whole and left for the actual segmentation/stride-finding step to
    determine which portions carry useful (running or walking) gait info."""
    t_grid = np.arange(t0, t1, dt)
    if t_grid.size < 2:
        return 0.0, 0.0
    speed = gps_speed_interp(gps_path, t_grid, 0.0)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return 0.0, 0.0
    return float(np.max(speed)), float(np.mean(speed >= speed_min))


def build_windows(day):
    """Build and save one run's window table; returns its rows (None if no
    2-dog overlap at all)."""
    run = RUN[day]
    day_dir = config.RAW_DIR / day
    print(f"\n{'='*70}\n{day}\n{'='*70}")
    segments_by_dog = per_dog_segments(day_dir)
    active_windows = sweep_active_dogs(segments_by_dog)
    if not active_windows:
        print("  no 2+-dog recording overlap at all")
        return None

    rows = []
    n_dropped_stationary = 0
    for t0, t1, dogs in active_windows:
        duration = t1 - t0
        if duration < MIN_OVERLAP_S:
            continue
        max_speed, frac_moving = speed_profile(day_dir / run.gps_file, t0, t1)
        if max_speed < SPEED_MIN:
            n_dropped_stationary += 1
            continue
        rows.append(dict(start_epoch=t0, end_epoch=t1, duration_s=duration,
                         n_dogs=len(dogs), dogs="|".join(sorted(dogs)),
                         frac_moving=round(frac_moving, 3), max_speed_mps=round(max_speed, 2)))

    # rank by "dog-seconds" (duration * n_dogs) -- rewards both longer
    # concurrency and more dogs included, not just one or the other
    rows.sort(key=lambda r: -(r["duration_s"] * r["n_dogs"]))
    config.WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = window_csv(day)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_epoch", "end_epoch", "duration_s", "n_dogs",
                                          "dogs", "frac_moving", "max_speed_mps"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(segments_by_dog)} dogs, {len(rows)} window(s) with >=2 dogs and some point >={SPEED_MIN} m/s; "
          f"dropped {n_dropped_stationary} that never reached {SPEED_MIN} m/s -> {out_path}")
    return rows


def windows_for(day):
    """The run's qualifying (>= MIN_WINDOW_S) windows."""
    w = pd.read_csv(window_csv(day))
    return w[w.duration_s >= MIN_WINDOW_S].reset_index(drop=True)


def qualifying_ranges(day_dir, dog, min_window_s=MIN_WINDOW_S, pad_s=WINDOW_PAD_S):
    """(start, end) absolute-epoch ranges, each padded by pad_s, of the
    qualifying windows this dog appears in. A dog can have several disjoint
    ranges on one day (most Dec 11 dogs do)."""
    csv_path = window_csv(day_dir)
    ranges = []
    if not csv_path.exists():
        return ranges
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["duration_s"]) < min_window_s:
                continue
            if dog not in row["dogs"].split("|"):
                continue
            ranges.append((float(row["start_epoch"]) - pad_s, float(row["end_epoch"]) + pad_s))
    return ranges
