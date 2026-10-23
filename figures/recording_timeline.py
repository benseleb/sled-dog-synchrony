#!/usr/bin/env python3
"""Why the data are choppy: per run, every instrumented dog's recording
(after corruption trimming), its device reboots, where its force signal is
missing, and the qualifying windows (>= 2 dogs recording, >= 120 s) that the
analysis is restricted to.

Usage: python3 figures/recording_timeline.py
Output: docs/figures/00_recording_timelines.png
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from synchrony import config
from synchrony.datasets import ANALYSED_DOGS, RUN, SPEED_MIN
from synchrony.force import LOAD_SENTINELS
from synchrony.gps import gps_speed_interp
from synchrony.gps_lag import lag_for_day
from synchrony.io import load_rows, trim_corruption, GAP_SECONDS
from synchrony.sync_check import dog_files
from synchrony.windows import windows_for
from pipeline_figures import C, INK2, MUTED, RUN_LABEL

PAD_S = 120.0          # context shown before the first and after the last window
MIN_DROPOUT_S = 1.0    # force-invalid stretches shorter than this aren't drawn
FS = 125.0


def runs_of(mask, t, min_len_s):
    """(start, end) times of True runs in mask lasting >= min_len_s."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(t[a], t[b - 1]) for a, b in zip(edges[::2], edges[1::2]) if t[b - 1] - t[a] >= min_len_s]


def firmware_summary(files):
    """'firmware a (IMU x Hz, load y Hz) ...' from the device-file headers."""
    seen = Counter()
    for paths in files.values():
        with open(paths[0], encoding="utf-8-sig", errors="replace") as f:
            head = "".join(next(f, "") for _ in range(4))
        ino = re.search(r"DoggyLogger_(\w+)\.ino", head)
        rate = re.search(r"Sampling IMU at (\d+) Hz and load cell at (\d+) Hz", head)
        if ino:
            seen[f"{ino.group(1)} (IMU {rate.group(1)} Hz, load {rate.group(2)} Hz)" if rate
                 else f"{ino.group(1)} (no load column)"] += 1
    return "; ".join(f"{k} ×{n}" if len(seen) > 1 else k for k, n in seen.items())


def dog_timeline(paths, lo, hi):
    """Recorded stretches, reboot times and force-invalid stretches in [lo, hi]."""
    stretches, reboots, dropouts = [], [], []
    dead = True
    for p in paths:
        d = trim_corruption(load_rows(p))
        if d["epoch"].size < 10:
            continue
        t = d["epoch"] + d["ms"] / 1000.0
        load = d["load"]
        if np.ptp(load[~np.isin(load, LOAD_SENTINELS)]) > 0:
            dead = False
        breaks = np.flatnonzero(np.diff(t) > GAP_SECONDS)
        for idx in np.split(np.arange(t.size), breaks + 1):
            a, b = t[idx[0]], t[idx[-1]]
            if b < lo or a > hi:
                continue
            stretches.append((a, b))
            if lo <= a <= hi and idx[0] > 0:
                reboots.append(a)
            bad = np.isin(load[idx], LOAD_SENTINELS)
            dropouts += runs_of(bad, t[idx], MIN_DROPOUT_S)
    return stretches, reboots, dropouts, dead


def main():
    days = list(ANALYSED_DOGS)
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 2.1 + 0.32 * sum(len(dog_files(d)) for d in days)),
                             gridspec_kw=dict(height_ratios=[len(dog_files(d)) + 1.5 for d in days]))
    for ax, day in zip(axes, days):
        w = windows_for(day)
        lo, hi = w.start_epoch.min() - PAD_S, w.end_epoch.max() + PAD_S
        files = dog_files(day)
        dogs = sorted(files)
        analysed = {dog for dog, _ in ANALYSED_DOGS[day]}
        for r in w.itertuples():
            ax.axvspan((r.start_epoch - lo) / 60, (r.end_epoch - lo) / 60, color=C[0], alpha=0.10, lw=0)

        grid = np.arange(lo, hi, 1.0)
        moving = gps_speed_interp(config.RAW_DIR / day / RUN[day].gps_file, grid, 0.0,
                                  shift_s=lag_for_day(day)) >= SPEED_MIN
        for a, b in runs_of(moving, grid, 1.0):
            ax.plot([(a - lo) / 60, (b - lo) / 60], [-1, -1], color=C[2], lw=4, solid_capstyle="butt")

        for y, dog in enumerate(dogs):
            stretches, reboots, dropouts, dead = dog_timeline(files[dog], lo, hi)
            col = C[0] if dog in analysed else MUTED
            for a, b in stretches:
                ax.plot([(max(a, lo) - lo) / 60, (min(b, hi) - lo) / 60], [y, y], color=col, lw=6,
                        solid_capstyle="butt", alpha=0.35 if dead else 1.0)
            for a, b in dropouts:
                ax.plot([(max(a, lo) - lo) / 60, (min(b, hi) - lo) / 60], [y, y], color=C[1], lw=6,
                        solid_capstyle="butt")
            ax.plot((np.array(reboots) - lo) / 60, np.full(len(reboots), y), "|", color=INK2, ms=11, mew=1.2)
            if dead:
                ax.text((hi - lo) / 60, y, " no force", va="center", fontsize=7, color=C[1])
        ax.set_yticks([-1] + list(range(len(dogs))), [f"≥{SPEED_MIN:g} m/s"] + dogs, fontsize=7)
        ax.set_ylim(len(dogs) - 0.4, -1.6)
        ax.set_xlim(0, (hi - lo) / 60)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{RUN_LABEL[day]}: {len(w)} qualifying window(s), {w.duration_s.sum() / 60:.0f} min"
                     f"   |   firmware {firmware_summary(files)}", fontsize=9)
    axes[-1].set_xlabel("minutes")
    handles = [Line2D([], [], color=C[0], lw=6, label="recording (analysed dog)"),
               Line2D([], [], color=MUTED, lw=6, label="recording (dog not analysed)"),
               Line2D([], [], color=C[1], lw=6, label="force missing (firmware sentinel)"),
               Line2D([], [], color=INK2, marker="|", ls="", ms=11, mew=1.2, label="device reboot (data gap)"),
               Line2D([], [], color=C[2], lw=4, label=f"team ≥ {SPEED_MIN:g} m/s (GPS)"),
               Patch(color=C[0], alpha=0.10, label="qualifying window (≥ 2 dogs, ≥ 120 s)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    out = config.FIGURES_DIR / "00_recording_timelines.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
