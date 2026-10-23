#!/usr/bin/env python3
"""Figures for SYNC_CHECK_WALKTHROUGH.md (whole-second check per boot
segment, 2026-10-10): the boot-segment map, why tugline force rather than the
activity envelope, how a whole-second call is made, the
Dec-11 primary window, and the one confident mismatch (Captain).

Reads the sync check's outputs (scripts/02_sync_check.py); recomputes correlation curves
only where a figure needs the full curve.

Usage: python3 figures/sync_figures.py [fig_name ...]
Output: docs/figures/sync/n*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synchrony import config, sync_check as sc
from synchrony.windows import window_csv
from pipeline_figures import C, INK, INK2, MUTED, BAND, RUN_LABEL, RUN_ORDER  # shared style + rcParams

OUT = config.FIGURES_DIR / "sync"
RES = config.SYNC_DIR
VERDICT_COLOR = {"in sync": C[0], "can't tell": "#c9c8c0"}
OFFSET_COLOR = C[1]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png", flush=True)


def force_curve(day, dog, lo, hi, mates):
    """Full force correlation curve of `dog` vs its teammates' median over [lo, hi]."""
    files = sc.dog_files(day)
    grid = np.arange(lo, hi, sc.FORCE_STEP)
    mine = sc.force_on_grid(files[dog], lo, hi, grid)
    refs = [f for f in (sc.force_on_grid(files[o], lo, hi, grid) for o in mates) if f is not None]
    with np.errstate(all="ignore"):
        ref = np.nanmedian(np.array(refs), axis=0) if len(refs) > 1 else refs[0]
    return sc.force_lag_curve(mine, ref)


def envelope_curve(day, dog, lo, hi, mates):
    files = sc.dog_files(day)
    grid = np.arange(lo, hi, sc.GRID_STEP)
    mine, _ = sc.window_envelope(files[dog], lo, hi, grid)
    refs = [sc.envelope_pieces(files[o], lo, hi, grid) for o in mates]
    with np.errstate(all="ignore"):
        ref = np.nanmedian(np.array(refs), axis=0)
    return sc.lag_curve(mine, ref, sc.WIN_SHIFTS, sc.WIN_MIN_OVERLAP_S)


# ---------------------------------------------------------------- figures

def fig_boot_map():
    """Every analysed boot segment, coloured by verdict, one panel per run."""
    seg = pd.read_csv(RES / "sync_check_segments.csv")
    seg = seg[seg.window == "segment"]
    boot = pd.read_csv(RES / "sync_check_boot_segments.csv")
    nrow = -(-len(RUN_ORDER) // 2)
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3 * nrow))
    for ax, day in zip(axes.T.ravel(), RUN_ORDER):
        w = pd.read_csv(window_csv(day))
        w = w[w.duration_s >= 120]
        t0, t1 = w.start_epoch.min(), w.end_epoch.max()
        for r in w.itertuples():
            ax.axvspan((r.start_epoch - t0) / 60, (r.end_epoch - t0) / 60, color=BAND, alpha=0.45, lw=0)
        s = seg[seg.day == day]
        dogs = sorted(s.dog.unique())
        files = sc.dog_files(day)
        for y, dog in enumerate(dogs):
            # reboots (data gaps) inside the run, as ticks
            for a, _, _ in sc.gapfree_segments(files[dog]):
                if t0 < a < t1:
                    ax.plot((a - t0) / 60, y, "|", color=INK2, ms=9, mew=1)
            for r in s[s.dog == dog].itertuples():
                v = boot[(boot.day == day) & (boot.dog == dog) & np.isclose(boot.boot_start, r.boot_start, rtol=0, atol=0.01)].verdict.iloc[0]
                col = OFFSET_COLOR if v.startswith("OFFSET") else VERDICT_COLOR.get(v, C[7])
                ax.barh(y, r.span_s / 60, left=(r.start_epoch - t0) / 60, height=0.55, color=col, lw=0)
                if v.startswith("OFFSET"):
                    ax.annotate(v.replace("OFFSET ", ""), ((r.start_epoch - t0 + r.span_s / 2) / 60, y),
                                ha="center", va="center", fontsize=7, color="white", weight="bold")
        ax.set_yticks(range(len(dogs)), dogs, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        b = boot[boot.day == day]
        n_ct = (b.verdict == "can't tell").sum()
        ax.set_title(f"{RUN_LABEL[day]}: {(b.verdict == 'in sync').sum()} in sync, {n_ct} can't tell, "
                     f"{b.verdict.str.startswith('OFFSET').sum()} offset", fontsize=9)
        ax.set_xlabel("minutes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR["in sync"]),
               plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR["can't tell"]),
               plt.Rectangle((0, 0), 1, 1, color=OFFSET_COLOR),
               plt.Line2D([], [], marker="|", ls="", color=INK2, ms=9),
               plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.45)]
    fig.legend(handles, ["in sync", "can't tell", "whole-second offset", "reboot (data gap)", "qualifying window"],
               loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "n01_boot_segment_map")


def fig_force_vs_envelope():
    day, _, lo, hi = sc.PRIMARY
    files = sc.dog_files(day)
    dogs = sorted(files)
    a = lo + 300  # 40 s excerpt start
    show = ["Tango", "Tiki", "Tizzy", "Winnie"]
    gf, ge = np.arange(lo, hi, sc.FORCE_STEP), np.arange(lo, hi, sc.GRID_STEP)
    fig = plt.figure(figsize=(11, 7.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    for i, dog in enumerate(show):
        # filtered over the whole primary window, then shown as an excerpt
        ax1.plot(gf - a, sc.force_on_grid(files[dog], lo, hi, gf) - 4 * i, color=C[i], lw=0.9)
        e, _ = sc.window_envelope(files[dog], lo, hi, ge)
        ax2.plot(ge - a, e - 4 * i, color=C[i], lw=1.2)
    for ax, title in [(ax1, "A  Tugline force, 0.2–2 Hz (one row per dog)"),
                      (ax2, "B  Activity envelope, 30 s high-passed (same dogs)")]:
        ax.set_xlim(0, 40)
        ax.set_ylim(-4 * len(show) + 1.5, 3.5)
        ax.set_yticks([-4 * i for i in range(len(show))], show, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("seconds (Dec 11 primary window excerpt)")
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, sharey=ax3)
    for dog in dogs:
        mates = [o for o in dogs if o != dog]
        ax3.plot(sc.FORCE_SHIFTS, force_curve(day, dog, lo, hi, mates), color=C[0], lw=0.8, alpha=0.7)
        ax4.plot(sc.WIN_SHIFTS, envelope_curve(day, dog, lo, hi, mates), color=C[1], lw=0.8, alpha=0.7)
    for ax, title in [(ax3, "C  Force: correlation vs teammates (each line a dog)"), (ax4, "D  Envelope: same")]:
        for k in range(-4, 5):
            ax.axvline(k, color=MUTED, lw=0.5, ls=":")
        ax.axvline(0, color=INK2, ls="--", lw=0.8)
        ax.set_xlim(-4, 4)
        ax.set_xlabel("clock offset tested (s); dotted = whole seconds")
        ax.set_title(title, fontsize=9)
    ax3.set_ylabel("correlation r")
    fig.tight_layout()
    save(fig, "n02_force_vs_envelope")


def fig_whole_second_examples():
    seg = pd.read_csv(RES / "sync_check_segments.csv")
    day, _, plo, phi = sc.PRIMARY
    cap = seg[(seg.day == "DEC_10_TEST/Run2") & (seg.dog == "Captain") & seg.verdict.str.startswith("OFFSET")].iloc[0]
    tomi = seg[(seg.day == "DEC_09_TEST") & (seg.dog == "Tomi")].iloc[0]
    cases = [
        ("A  Tizzy, Dec 11 primary window: in sync", day, "Tizzy", plo, phi),
        ("B  Captain, Dec 10 run 2, one boot segment: +1 s", cap.day, "Captain", cap.start_epoch,
         cap.start_epoch + cap.span_s),
        ("C  Tomi, Dec 9 (team mostly stopped): can't tell", tomi.day, "Tomi", tomi.start_epoch,
         tomi.start_epoch + tomi.span_s),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (title, d, dog, lo, hi) in zip(axes, cases):
        mates = sorted(set(seg[seg.day == d].dog) - {dog}) if d != day else [o for o in sc.dog_files(d) if o != dog]
        r = force_curve(d, dog, lo, hi, mates)
        bins = sc.whole_second_bins(r)
        best = max(bins, key=bins.get)
        second = max(v for k, v in bins.items() if k != best)
        for k, v in bins.items():
            ax.axvspan(k - 0.5, k + 0.5, color=C[1] if k == best else BAND, alpha=0.25 if k == best else 0.35, lw=0)
            ax.plot([k - 0.45, k + 0.45], [v, v], color=INK2, lw=1)
        ax.plot(sc.FORCE_SHIFTS, r, color=C[0], lw=1.2)
        ax.axhline(second, color=INK2, ls=":", lw=0.8)
        ax.annotate(f"margin {bins[best] - second:.2f}", (best, bins[best]), (8, 6), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.set_xlim(-3.5, 3.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("clock offset tested (s)")
    axes[0].set_ylabel("force correlation vs teammates")
    fig.suptitle("Each whole second k scores the best r within ±0.5 s of k (grey bars). k is called only if it "
                 "beats the next-best whole second (dotted) by ≥ 0.10", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "n03_whole_second_call")


def fig_primary():
    s = pd.read_csv(RES / "sync_check_segments.csv")
    p = s[s.window == "primary"].sort_values("dog")
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in (-1, 1):
        ax.axvspan(k - 0.5, k + 0.5, color=C[7], alpha=0.06, lw=0)
    ax.axvspan(-0.5, 0.5, color=C[0], alpha=0.08, lw=0)
    y = np.arange(len(p))
    ax.plot(p.force_peak_s, y - 0.12, "o", color=C[0], ms=7, mec="white", mew=1, label="force peak")
    ax.plot(p.lag_team_s, y + 0.12, "s", color=C[1], ms=6, mec="white", mew=1, label="envelope peak")
    ax.set_yticks(y, p.dog)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("lag vs teammates (s); blue band = 0 s bin, red bands = ±1 s bins a clock error would fall in")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Dec 11 primary window: every dog in the 0 s bin", fontsize=9)
    fig.tight_layout()
    save(fig, "n05_primary_window")


def fig_captain():
    seg = pd.read_csv(RES / "sync_check_segments.csv")
    cap = seg[(seg.day == "DEC_10_TEST/Run2") & (seg.dog == "Captain") & seg.verdict.str.startswith("OFFSET")].iloc[0]
    lo, hi = cap.start_epoch, cap.start_epoch + cap.span_s
    files = sc.dog_files(cap.day)
    grid = np.arange(lo, hi, sc.FORCE_STEP)
    F = {d: sc.force_on_grid(files[d], lo, hi, grid) for d in ("Captain", "Chips", "Glow")}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (x, y), col in zip([("Captain", "Chips"), ("Captain", "Glow"), ("Chips", "Glow")], [C[1], C[3], C[0]]):
        r = sc.force_lag_curve(F[x], F[y])
        ax.plot(sc.FORCE_SHIFTS, r, color=col, lw=1.4,
                label=f"{x} vs {y}: peak {sc.FORCE_SHIFTS[np.nanargmax(r)]:+.2f} s")
    for k in range(-3, 4):
        ax.axvline(k, color=MUTED, lw=0.5, ls=":")
    ax.axvline(0, color=INK2, ls="--", lw=0.8)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("clock offset tested (s)")
    ax.set_ylabel("force correlation (pairwise)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Dec 10 run 2, Captain's boot segment ({cap.span_s:.0f} s)", fontsize=9)
    fig.tight_layout()
    save(fig, "n06_captain_triangle")


FIGS = dict(map=fig_boot_map, force=fig_force_vs_envelope, call=fig_whole_second_examples,
            primary=fig_primary, captain=fig_captain)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
