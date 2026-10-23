#!/usr/bin/env python3
"""Manuscript-Fig-4-style phase-locking figure for one replication run
(default: Nov 23, focal dog Arrow), using only dogs whose clocks are verified
in sync (sync_check_boot_segments.csv), the same PLV settings as
synchrony/analysis.py (manuscript method: 60 s windows, 15 s step, pooled
800-lag null per pair, PLV_min 0.70, < 3 m/s masked), and the same loading/segmentation path.

Panels:
  A  stride frequency per dog (violins, mean bar)            ~ Fig 4B
  B  pairwise |mean frequency difference|                    ~ Fig 4C
  C  cumulative phase difference, focal vs each teammate;
     significant 60-s PLV windows in bold black, slow spans shaded   ~ Fig 4D
  D  wrapped phase difference in a locked vs an unlocked window      ~ Fig 4E
  E  PLV for every pair over time, significant windows outlined (new)

Team positions are unknown for the November runs (no field notes), so C
shows the focal dog against every teammate rather than gangline neighbours.

Usage: python3 figures/phase_locking_figure.py [DAY] [FOCAL]
Output: docs/figures/phase_locking_<day>.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from synchrony import analysis as ra, config, datasets as ds
from synchrony.plv import windowed_plv_pooled
from synchrony.windows import windows_for
from pipeline_figures import C, INK, INK2, MUTED, BAND, RUN_LABEL

OUT = config.FIGURES_DIR
FS = 125.0
SPEED_MIN = ds.SPEED_MIN


def verified(boot, day, dog, lo, hi):
    s = boot[(boot.day == day) & (boot.dog == dog) & (boot.boot_start < hi) & (boot.boot_end > lo)]
    return not s.empty and (s.verdict == "in sync").all()


def main(day="NOV_23_TEST", focal="Arrow"):
    boot = pd.read_csv(config.SYNC_DIR / "sync_check_boot_segments.csv")
    windows = windows_for(day).sort_values("start_epoch").reset_index(drop=True)
    items = {d: (f, g) for dd, d, f, g in ds.QUEUE if dd == day}
    data = {d: ra.load_dog_windows(day, d, *fg) for d, fg in items.items()}
    data = {d: v for d, v in data.items() if v is not None}

    # dogs verified in every window they appear in (Trillium on Nov 23 isn't)
    dogs = sorted(d for d in data if all(verified(boot, day, d, w.start_epoch, w.end_epoch)
                                         for w in windows.itertuples() if d in w.dogs.split("|")))
    excluded = sorted(set(data) - set(dogs))
    assert focal in dogs, f"{focal} not verified on {day}"
    color = {d: C[i % len(C)] for i, d in enumerate([focal] + [d for d in dogs if d != focal])}
    t_ref = windows.start_epoch.min()

    # ---- stride frequencies over moving strides inside the windows
    freq = {}
    for d in dogs:
        zc, v = data[d]["zc_abs"], data[d]
        sp0 = np.interp(zc[:-1], v["t_abs"], v["speed_gated"])
        sp1 = np.interp(zc[1:], v["t_abs"], v["speed_gated"])
        f = 1 / np.diff(zc)
        freq[d] = f[(sp0 >= SPEED_MIN) & (sp1 >= SPEED_MIN) & (f > 1.5) & (f < 4.5)]
    mean_f = {d: freq[d].mean() for d in dogs}

    # ---- per-window PLV for every verified pair (same call and pair order as synchrony/analysis.py)
    plv = []
    for wi, w in windows.iterrows():
        names = [d for d in w.dogs.split("|") if d in dogs]
        grid, ph = ra.window_phases(day, w, names, data, items[names[0]][1])
        rng = np.random.default_rng(123)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows, _ = windowed_plv_pooled(grid, ph[a], ph[b], rng=rng)
                plv += [dict(window=wi, a=a, b=b, t_start=r["t_start"] + w.start_epoch, t_end=r["t_end"] + w.start_epoch,
                             **{k: r[k] for k in ("plv", "significant")}) for r in rows]
    plv = pd.DataFrame(plv)
    plv = plv[plv.plv.notna()]  # windows < 90% moving aren't tested (manuscript rule)
    n_sig = int(plv.significant.sum())

    # ---- figure
    fig = plt.figure(figsize=(12, 13))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1.5, 1, 1.1], hspace=0.55, wspace=0.35,
                  top=0.94, bottom=0.04)

    # A: stride frequency violins
    ax = fig.add_subplot(gs[0, :2])
    parts = ax.violinplot([freq[d] for d in dogs], showextrema=False, widths=0.8)
    for body, d in zip(parts["bodies"], dogs):
        body.set_facecolor(color[d]); body.set_alpha(0.45); body.set_edgecolor("none")
    for x, d in enumerate(dogs, 1):
        ax.plot([x - 0.3, x + 0.3], [mean_f[d]] * 2, color=INK, lw=1.5)
    ax.set_xticks(range(1, len(dogs) + 1), dogs)
    ax.set_ylabel("stride frequency (Hz)")
    ax.set_ylim(2.2, 4.0)
    allf = np.concatenate([freq[d] for d in dogs])
    ax.set_title(f"A  Stride frequency, moving strides (team mean {np.mean([mean_f[d] for d in dogs]):.2f} "
                 f"± {np.std([mean_f[d] for d in dogs]):.2f} Hz across dogs)", fontsize=9)
    ax.grid(axis="x", visible=False)

    # B: |mean frequency difference|
    ax = fig.add_subplot(gs[0, 2])
    m = np.array([[abs(mean_f[a] - mean_f[b]) for b in dogs] for a in dogs])
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(0.3, m.max()))
    ax.set_xticks(range(len(dogs)), dogs, rotation=60, fontsize=7)
    ax.set_yticks(range(len(dogs)), dogs, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|Δ mean freq| (Hz)", fontsize=8)
    ax.set_title("B  Pairwise |Δ mean frequency|", fontsize=9)

    # C: cumulative phase difference focal vs teammates
    ax = fig.add_subplot(gs[1, :])
    for w in windows.itertuples():
        ax.axvspan((w.start_epoch - t_ref) / 60, (w.end_epoch - t_ref) / 60, color=BAND, alpha=0.25, lw=0)
        grid = np.arange(w.start_epoch, w.end_epoch, 1 / FS)
        fa = data[focal]
        pa = np.interp(grid, fa["t_abs"], np.unwrap(fa["phase"]))
        sp = np.interp(grid, fa["t_abs"], np.nan_to_num(fa["speed_gated"]))
        slow = sp < SPEED_MIN
        # shade slow spans (stride phase not meaningful there)
        edges = np.flatnonzero(np.diff(np.r_[0, slow.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            ax.axvspan((grid[s0] - t_ref) / 60, (grid[min(s1, grid.size - 1)] - t_ref) / 60,
                       color=MUTED, alpha=0.25, lw=0)
        for b in dogs:
            if b == focal or b not in w.dogs.split("|"):
                continue
            pb = np.interp(grid, data[b]["t_abs"], np.unwrap(data[b]["phase"]))
            cyc = (pa - pb) / (2 * np.pi)
            cyc -= cyc[0]  # each window starts at 0: clocks only verified within windows
            x = (grid - t_ref) / 60
            ax.plot(x, cyc, color=color[b], lw=1.0, label=b if w.Index == 0 else None)
            sig = plv[(plv.window == w.Index) & (((plv.a == focal) & (plv.b == b)) | ((plv.a == b) & (plv.b == focal)))
                      & plv.significant]
            for r in sig.itertuples():
                k = (grid >= r.t_start) & (grid <= r.t_end)
                ax.plot(x[k], cyc[k], color=INK, lw=3.2, solid_capstyle="butt", zorder=3)
    ax.axhline(0, color=INK2, ls="--", lw=0.8)
    # Zoom on the near-zero region, where locking can happen (as in the
    # manuscript, which showed only similar-frequency neighbours); dogs whose
    # stride frequency differs a lot drift off-scale within a minute or two
    ylim = 60
    ax.set_ylim(-ylim, ylim)
    off = [b for b in dogs if b != focal and abs(mean_f[focal] - mean_f[b]) > 0.15]
    if off:
        ax.text(0.99, 0.03, "off-scale (drift > ±60 cycles): " +
                ", ".join(f"{b} (Δf {mean_f[focal] - mean_f[b]:+.2f} Hz)" for b in off),
                transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("minutes")
    ax.set_ylabel(f"{focal} − teammate (cycles)")
    ax.legend(ncol=len(dogs) - 1, fontsize=8, loc="upper left")
    ax.set_title(f"C  Cumulative phase difference, focal dog {focal} vs each teammate (reset to 0 at each "
                 f"window start). Bold black: significant 60-s PLV windows. Grey: team below 3 m/s",
                 fontsize=9)

    # D: wrapped phase-difference histograms (locked vs unlocked window for the focal's top pair)
    fp = plv[(plv.a == focal) | (plv.b == focal)].copy()
    fp["partner"] = np.where(fp.a == focal, fp.b, fp.a)
    top = fp[fp.significant].partner.value_counts()
    partner = top.index[0] if len(top) else fp.groupby("partner").plv.max().idxmax()
    pair = fp[fp.partner == partner]
    locked = pair.sort_values("plv").iloc[-1]
    unlocked = pair.sort_values("plv").iloc[len(pair) // 2]
    other = plv.sort_values("plv").iloc[len(plv) // 2]
    for col, (r, title) in enumerate([(locked, f"{focal}–{partner}, most locked window"),
                                      (unlocked, f"{focal}–{partner}, median window"),
                                      (other, f"{other.a}–{other.b}, median window (team)")]):
        ax = fig.add_subplot(gs[2, col], projection="polar")
        a, b = (focal, partner) if col < 2 else (other.a, other.b)
        grid = np.arange(r.t_start, r.t_end, 1 / FS)
        dphi = np.mod(np.interp(grid, data[a]["t_abs"], np.unwrap(data[a]["phase"]))
                      - np.interp(grid, data[b]["t_abs"], np.unwrap(data[b]["phase"])), 2 * np.pi)
        h, e = np.histogram(dphi, bins=24, range=(0, 2 * np.pi), density=True)
        ax.bar(e[:-1], h, width=2 * np.pi / 24, align="edge", color=C[0] if r.significant else MUTED,
               edgecolor="white", lw=0.5)
        ax.set_yticks([])
        ax.set_title(f"{'D' if col == 0 else ''}  {title}\nPLV {r.plv:.2f} "
                     f"({'significant' if r.significant else 'n.s.'}), t = {(r.t_start - t_ref) / 60:.1f} min",
                     fontsize=8)

    # E: PLV raster, all verified pairs
    ax = fig.add_subplot(gs[3, :])
    pairs = sorted({(r.a, r.b) for r in plv.itertuples()}, key=lambda p: (p != tuple(sorted((focal, partner))), p))
    for y, (a, b) in enumerate(pairs):
        s = plv[(plv.a == a) & (plv.b == b)]
        for r in s.itertuples():
            x0 = (r.t_start - t_ref) / 60
            ax.add_patch(plt.Rectangle((x0, y - 0.4), 0.25, 0.8, color=plt.cm.Blues(r.plv), lw=0))
            if r.significant:
                ax.add_patch(plt.Rectangle((x0, y - 0.4), 0.25, 0.8, fill=False, ec=C[1], lw=1.5))
    ax.set_xlim(0, (windows.end_epoch.max() - t_ref) / 60)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_yticks(range(len(pairs)), [f"{a}–{b}" for a, b in pairs], fontsize=6)
    ax.grid(False)
    ax.set_xlabel("minutes (window start; each tile = one 60-s window, 15-s steps overlap)")
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("PLV", fontsize=8)
    ax.set_title(f"E  PLV for all {len(pairs)} verified pairs; orange outline = significant "
                 f"({n_sig} of {len(plv)} windows, {100 * n_sig / len(plv):.2f}%)", fontsize=9)

    note = f"; not shown (clock not verified): {', '.join(excluded)}" if excluded else ""
    fig.suptitle(f"{RUN_LABEL.get(day, day)}: phase locking among {len(dogs)} dogs with verified clocks{note}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    out = OUT / f"phase_locking_{day.replace('/', '_')}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  ({n_sig}/{len(plv)} significant; focal {focal}, top partner {partner})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
