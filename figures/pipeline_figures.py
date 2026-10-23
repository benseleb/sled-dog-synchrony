#!/usr/bin/env python3
"""Figures for docs/PIPELINE.md and the replication summary, all drawn
from the real data and the saved result CSVs -- nothing is re-fitted here
except where a figure needs per-stride detail the CSVs don't keep.

Usage: python3 figures/pipeline_figures.py [fig_name ...]
Output: docs/figures/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synchrony import config, datasets as ds
from synchrony.force import force_phase_regression, smooth_force, LOAD_SENTINELS
from synchrony.gps import estimate_gps_lag, gps_speed_interp, read_gps, GPS_LAG_S
from synchrony.io import load_rows, trim_corruption
from synchrony.phase import butter_filter, marker_event, segment_and_phase, znormalize, time_interpolated_phase
from synchrony.windows import window_csv

DATA = config.RAW_DIR
OUT = config.FIGURES_DIR

RUN_START_MS, WINDOW_END_MS, DEC11_DOGS = ds.RUN_START_MS, ds.WINDOW_END_MS, ds.DEC11_DOGS

# Reference categorical palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#1f1f1e", "#55544f", "#9a9990", "#e4e3dd"
BAND = "#cde2fb"

RUN_LABEL = {"DEC_11_TEST": "Dec 11", "DEC_10_TEST/Run1": "Dec 10 run 1", "DEC_10_TEST/Run2": "Dec 10 run 2",
             "DEC_05_TEST": "Dec 5", "DEC_07_TEST": "Dec 7", "DEC_09_TEST": "Dec 9",
             "NOV_23_TEST": "Nov 23", "NOV_29_TEST": "Nov 29"}
RUN_ORDER = list(RUN_LABEL)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png", flush=True)


def raw_dog(day, fname):
    d = trim_corruption(load_rows(DATA / day / fname))
    d["t"] = d["epoch"] + d["ms"] / 1000.0
    return d


def formatted_dec11(i):
    df = pd.read_csv(config.DEC11_FORMATTED_DIR / f"data_{i}.csv")
    df = df[(df.epoch >= RUN_START_MS) & (df.epoch <= WINDOW_END_MS)]
    t = (df.epoch.to_numpy() - RUN_START_MS) / 1000.0
    return t, df


# ---------------------------------------------------------------- figures

def fig_dataset():
    """Which dogs recorded together, when: one panel per run."""
    nrow = -(-len(RUN_ORDER) // 2)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 2.85 * nrow))
    for ax, day in zip(axes.T.ravel(), RUN_ORDER):
        w = pd.read_csv(window_csv(day))
        w = w[w.duration_s >= 120]
        dogs = sorted({d for ds in w.dogs for d in ds.split("|")})
        t0 = w.start_epoch.min()
        for _, r in w.iterrows():
            for d in r.dogs.split("|"):
                y = dogs.index(d)
                ax.barh(y, r.duration_s / 60, left=(r.start_epoch - t0) / 60, height=0.6,
                        color=C[0], alpha=0.35 + 0.65 * r.frac_moving, linewidth=0)
        ax.set_yticks(range(len(dogs)), dogs, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{RUN_LABEL[day]}: {len(w)} window(s), {w.duration_s.sum() / 60:.0f} min")
        ax.set_xlabel("minutes from first window")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Qualifying windows (≥120 s, ≥2 dogs, team reaches 3 m/s); darker = more of the window above 3 m/s",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "01_dataset_windows")


def fig_preprocessing():
    """Raw az -> gravity removed -> z-scored with crossings -> phase."""
    d = raw_dog("DEC_11_TEST", "Tiki_Device05.csv")
    a = RUN_START_MS / 1000 + 300
    k = (d["t"] >= a - 30) & (d["t"] <= a + 34)            # filter on a margin, show 4 s
    t, az = d["t"][k] - a, d["az"][k]
    seg = segment_and_phase(t, az)
    az_hp = butter_filter(az, 0.5, 125.0, "high", 2)
    z, _, _ = znormalize(az_hp)
    show = (t >= 0) & (t <= 4)
    zc = seg["zero_crossings"][(seg["zero_crossings"] >= 0) & (seg["zero_crossings"] <= 4)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(t[show], az[show], color=INK2)
    axes[0].set_ylabel("raw $a_z$ (g)")
    axes[0].set_title("A  Raw dorsoventral acceleration (includes gravity, ~0.85 g at rest)")
    axes[1].plot(t[show], z[show], color=C[0])
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].plot(zc, np.zeros_like(zc), "o", ms=6, color=C[1], mec="white", mew=1.5, zorder=3)
    axes[1].set_ylabel("$a_z$ (z-score)")
    axes[1].set_title("B  0.5 Hz high-pass, z-scored; dots = automatic upward zero-crossings (stride boundaries $t_k$)")
    axes[2].plot(t[show], np.mod(seg["phase"][show], 2 * np.pi), color=C[0])
    axes[2].set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    axes[2].set_ylabel("phase φ")
    axes[2].set_xlabel("seconds")
    axes[2].set_title("C  Phase: rises linearly 0→2π between consecutive crossings (manuscript Eq. 1)")
    fig.tight_layout()
    save(fig, "02_preprocessing_phase")


def fig_segmentation_error():
    """What a segmentation error looks like, and why beta_omega catches it."""
    i = DEC11_DOGS.index("Shifter")
    t, df = formatted_dec11(i)
    z, _, _ = znormalize(df.az.to_numpy())
    auto = np.array(marker_event(t, z))
    cur = np.load(config.DEC11_CURATED_ZC_DIR / f"{DEC11_DOGS[i]}.npy")
    # first curated crossing with no automatic crossing within 40 ms, while running
    speed = gps_speed_interp(DATA / "DEC_11_TEST/GPSLogger00.csv", t, RUN_START_MS / 1000)
    missed = [c for c in cur if np.min(np.abs(auto - c)) > 0.15 and np.interp(c, t, speed) > 3.5]
    c0 = missed[len(missed) // 2]
    w = (t >= c0 - 1.5) & (t <= c0 + 1.5)

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t[w] - c0, z[w], color=INK2, lw=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    cw, aw = cur[np.abs(cur - c0) < 1.5] - c0, auto[np.abs(auto - c0) < 1.5] - c0
    ax.plot(cw, np.full_like(cw, 0.35), "v", ms=8, color=C[2], label="hand-curated")
    ax.plot(aw, np.full_like(aw, -0.35), "^", ms=8, color=C[1], label="automatic")
    ax.annotate("missed by automatic\n→ two strides merged into one", (0, 0), (0.35, 1.6), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("seconds from the missed crossing")
    ax.set_ylabel("$a_z$ (z)")
    ax.set_title("A  Shifter, Dec 11: one missed crossing")
    ax.legend(loc="lower left", ncol=2)

    for j, (zc, name, col) in enumerate([(cur, "hand-curated", C[2]), (auto, "automatic", C[1])]):
        ax = fig.add_subplot(2, 2, 3 + j)
        T = np.diff(zc)
        om = 2 * np.pi / T
        moving = np.interp(zc[:-1], t, speed) >= 3
        om_prev, dom = om[:-1][moving[1:]], np.diff(om)[moving[1:]]
        ax.plot((om_prev - om_prev.mean()) / om_prev.std(), dom, ".", ms=2, color=col, alpha=0.5)
        b = np.polyfit((om_prev - om_prev.mean()) / om_prev.std(), dom, 1)[0]
        ax.set_title(f"{'BC'[j]}  {name}: slope of Δω on ω_(k−1) {b:.2f}", fontsize=9)
        ax.set_xlabel("$ω_{k-1}$ (z-scored)")
        ax.set_ylabel("$Δω_k$ (rad/s)")
        ax.set_ylim(-25, 25)
    fig.tight_layout()
    save(fig, "03_segmentation_error")


def fig_gps_lag():
    dogs = []
    lo, hi = RUN_START_MS / 1000, WINDOW_END_MS / 1000
    for name in ["Glow_Device06.csv", "Healy_Device01.csv", "Sitka_Device03.csv", "Tiki_Device05.csv",
                 "Tizzy_Device02.csv", "Winnie_Device09.csv", "Alice_Device07.csv", "Tango_Device11.csv"]:
        d = raw_dog("DEC_11_TEST", name)
        k = (d["t"] >= lo) & (d["t"] <= hi)
        dogs.append((d["t"][k], d["ax"][k], d["ay"][k], d["az"][k]))
    gps = DATA / "DEC_11_TEST/GPSLogger00.csv"
    best, rbest, shifts, r = estimate_gps_lag(gps, dogs)

    t_gps, v_gps = read_gps(gps)
    grid = np.arange(lo + 60, lo + 360, 0.2)
    act = []
    for t, ax_, ay, az in dogs:
        mag2 = sum(butter_filter(x, 0.5, 125.0, "high", 2) ** 2 for x in (ax_, ay, az))
        act.append(np.interp(grid, t, np.sqrt(np.convolve(mag2, np.ones(125) / 125, "same"))))
    act = np.mean(act, axis=0)

    fig = plt.figure(figsize=(10, 6))
    a1 = fig.add_subplot(2, 2, (1, 2))
    tt = grid - grid[0]
    zs = lambda x: (x - x.mean()) / x.std()
    a1.plot(tt, zs(act), color=INK2, lw=0.8, label="team IMU activity")
    a1.plot(tt, zs(np.interp(grid, t_gps, v_gps)), color=C[0], lw=1.2, label="GPS speed, as recorded")
    a1.plot(tt, zs(np.interp(grid, t_gps + best, v_gps)), color=C[1], lw=1.2, ls="--",
            label=f"GPS speed shifted +{best:.1f} s")
    a1.set_xlabel("seconds")
    a1.set_ylabel("z-score")
    a1.set_title("A  GPS speed vs team activity (Dec 11, 5 min excerpt)")
    a1.legend(ncol=3, loc="lower right", fontsize=8)
    a2 = fig.add_subplot(2, 2, 3)
    a2.plot(shifts, r, color=C[0])
    a2.axvline(best, color=C[1], ls="--", lw=1)
    a2.annotate(f"peak +{best:.1f} s (r = {rbest:.2f})", (best, rbest), (best + 1.2, rbest - 0.08), fontsize=8)
    a2.set_xlabel("shift added to GPS timestamps (s)")
    a2.set_ylabel("correlation r")
    a2.set_title("B  Correlation vs shift")
    a3 = fig.add_subplot(2, 2, 4)
    g = pd.read_csv(config.GPS_LAGS_CSV)
    for y, day in enumerate(RUN_ORDER):
        r = g[(g.day == day) & g.lag_window_s.notna()]
        a3.plot(r.lag_window_s, np.full(len(r), y) + np.random.default_rng(y).uniform(-0.15, 0.15, len(r)), "o",
                ms=4, color=C[0], alpha=0.5, mec="white", mew=0.5)
        if len(g[g.day == day]):
            used = g[g.day == day].iloc[0]
            a3.plot(used.gps_lag_s, y, "D" if used.source == "run" else "s", ms=7, color=C[1], mec="white", zorder=3)
    a3.axvline(GPS_LAG_S, color=MUTED, ls="--", lw=1)
    a3.set_yticks(range(len(RUN_ORDER)), [RUN_LABEL[d] for d in RUN_ORDER])
    a3.invert_yaxis()
    a3.grid(axis="y", visible=False)
    a3.set_xlabel("GPS lag (s)")
    a3.set_title("C  Per run: windows (dots), lag used (diamond; square = default)")
    fig.tight_layout()
    save(fig, "05_gps_lag")


def fig_force_qa():
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    cases = [
        ("DEC_11_TEST", "Alice_Device07.csv", "A  Alice, Dec 11: −999 'no reading' sentinel for ~23 min (error column stays 0)"),
        ("DEC_11_TEST", "Tiki_Device05.csv", "B  Tiki, Dec 11: baseline steps between windows (each window's median removed before pooling)"),
        ("DEC_07_TEST", "Trillium_Device08.csv", "C  Trillium, Dec 7: failing cell drifts within the window (masked where 20 s median departs > 1 pulse size)"),
    ]
    for ax, (day, f, title) in zip(axes, cases):
        d = raw_dog(day, f)
        w = pd.read_csv(window_csv(day))
        dog = f.split("_")[0]
        w = w[(w.duration_s >= 120) & w.dogs.str.contains(dog)]
        lo, hi = w.start_epoch.min() - 60, w.end_epoch.max() + 60
        k = (d["t"] >= lo) & (d["t"] <= hi)
        t, load, off = d["t"][k], d["load"][k], d["offset"][k]
        force = (load - off).astype(float)
        sentinel = np.isin(load, LOAD_SENTINELS)
        tt = (t - lo) / 60
        for _, r in w.iterrows():
            ax.axvspan((r.start_epoch - lo) / 60, (r.end_epoch - lo) / 60, color=BAND, alpha=0.5, lw=0)
        good = ~sentinel
        shown = np.where(good, force, np.nan)  # NaN, not dropped, so masked spans stay visible gaps
        ax.plot(tt[::5], shown[::5] / 1e3, color=INK2, lw=0.4)
        if sentinel.any():
            ax.plot(tt[sentinel][::25], np.full(sentinel.sum(), np.nanmin(force[good]) / 1e3)[::25], "|",
                    color=C[7], ms=8, label="sentinel samples (masked)")
            ax.legend(loc="upper right", fontsize=8)
        med = pd.Series(shown).rolling(2500, center=True, min_periods=500).median().to_numpy()
        ax.plot(tt, np.where(good, med, np.nan) / 1e3, color=C[1], lw=1.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("load − offset (k counts)")
    axes[-1].set_xlabel("minutes (pale blue = qualifying windows; orange = 20 s rolling median)")
    fig.tight_layout()
    save(fig, "06_force_qa")


def fig_regression():
    i = DEC11_DOGS.index("Tango")
    t, df = formatted_dec11(i)
    zc = np.load(config.DEC11_CURATED_ZC_DIR / f"{DEC11_DOGS[i]}.npy")
    speed = gps_speed_interp(DATA / "DEC_11_TEST/GPSLogger00.csv", t, RUN_START_MS / 1000)
    res = force_phase_regression(t, zc, df.load.to_numpy(), df.offset.to_numpy(), speed_mps=speed,
                                    exclude_low_speed_strides=True, return_design=True)
    X, y = res["X"], res["y"]
    phase = time_interpolated_phase(zc, t)
    fs = smooth_force(t, df.load.to_numpy(), df.offset.to_numpy())
    fs = fs[0] if isinstance(fs, tuple) else fs

    fig = plt.figure(figsize=(10, 6.5))
    a = fig.add_subplot(2, 1, 1)
    w = (t >= 300) & (t <= 302.5)
    a.plot(t[w] - 300, np.cos(phase[w]), color=C[0], label="cos φ (stride phase)")
    fw = fs[w]
    a.plot(t[w] - 300, 2 * (fw - np.nanmin(fw)) / (np.nanmax(fw) - np.nanmin(fw)) - 1, color=C[1],
           label="tugline force (smoothed, rescaled)")
    for c in zc[(zc >= 300) & (zc <= 302.35)]:
        s = (t >= c - 0.15) & (t <= c + 0.15)
        j = np.nanargmax(fs[s])
        a.plot(t[s][j] - 300, 2 * (fs[s][j] - np.nanmin(fw)) / (np.nanmax(fw) - np.nanmin(fw)) - 1, "o",
               ms=7, color=C[1], mec="white", mew=1.5, zorder=3)
        a.axvline(c - 300, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("seconds")
    a.set_title("A  Tango, Dec 11: per-stride force peak $F_k$ (dots) found within ±0.15 s of each stride boundary (dotted)")
    a.legend(loc="lower right", ncol=2, fontsize=8)

    labels = [("cos $φ_{F_k}$", 0), ("$F_k$ (z)", 2), ("$ω_{k-1}$ (z)", 3)]
    for j, (lab, col) in enumerate(labels):
        b = fig.add_subplot(2, 3, 4 + j)
        b.plot(X[:, col], y, ".", ms=2, color=C[0], alpha=0.35)
        coef = {0: res["beta_cos"], 2: res["beta_F"], 3: res["beta_omega"]}[col]
        xx = np.linspace(np.percentile(X[:, col], 1), np.percentile(X[:, col], 99), 10)
        b.plot(xx, res["beta0"] + coef * xx, color=C[1], lw=2)
        b.set_xlabel(lab)
        b.set_ylabel("$Δω_k$ (rad/s)" if j == 0 else "")
        b.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
        b.set_title(f"{'BCD'[j]}  coefficient {coef:+.2f}", fontsize=9)
    fig.tight_layout()
    save(fig, "07_regression")


def fig_beta_screen():
    f = pd.read_csv(config.ANALYSIS_DIR / "force_phase_regression.csv").dropna(subset=["beta_omega"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(-0.77, -0.33, color=C[0], alpha=0.08, lw=0)
    ax.text(-0.55, 4.85, "screening band (curated Dec-11 CI envelope)", ha="center", fontsize=8, color=INK2)
    for y, day in enumerate(RUN_ORDER):
        sub = f[f.day == day]
        jit = np.random.default_rng(y).uniform(-0.15, 0.15, len(sub))
        ok = ~sub.hand_check.astype(bool)
        ax.plot(sub.beta_omega[ok], y + jit[ok.to_numpy()], "o", ms=6, color=C[0], mec="white", mew=1)
        ax.plot(sub.beta_omega[~ok], y + jit[~ok.to_numpy()], "o", ms=7, color=C[1], mec="white", mew=1)
        for _, r in sub[~ok].iterrows():
            ax.annotate(r.dog, (r.beta_omega, y), (4, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.set_yticks(range(len(RUN_ORDER)), [RUN_LABEL[d] for d in RUN_ORDER])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("β_ω (automatic segmentation; orange = flagged for hand-checking)")
    ax.set_title("β_ω screen across all 28 fitted dog-runs (Dec 9: no fits, too little running)")
    fig.tight_layout()
    save(fig, "08_beta_omega_screen")


def fig_plv():
    p = pd.read_csv(config.ANALYSIS_DIR / "pairwise_plv.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvline(0.70, color=INK2, ls="--", lw=0.8)
    ax.text(0.71, 5.6, "PLV$_{min}$ = 0.70", fontsize=8, color=INK2)
    for y, day in enumerate(RUN_ORDER):
        sub = p[p.day == day]
        if sub.empty:  # verified-clock rule left fewer than 2 verified dogs in every window
            ax.text(1.02, y, "no pair with both clocks verified", va="center", fontsize=8, color=INK2,
                    transform=ax.get_yaxis_transform())
            continue
        jit = np.random.default_rng(y).uniform(-0.18, 0.18, len(sub))
        ax.plot(sub.max_plv, y + jit, "o", ms=4, color=C[0], alpha=0.55, mec="white", mew=0.5)
        pct = 100 * sub.n_significant.sum() / sub.n_windows.sum()
        ax.text(1.02, y, f"{pct:.2f}% of {sub.n_windows.sum():,} windows significant", va="center", fontsize=8,
                color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(RUN_ORDER)), [RUN_LABEL[d] for d in RUN_ORDER])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(RUN_ORDER) - 0.2, -0.5)  # fixed, so empty runs keep their rows
    ax.set_xlabel("highest 60-s PLV reached by each dog pair")
    ax.set_title("Pairwise phase locking by run (verified clocks only): one dot per dog pair per window")
    fig.tight_layout()
    save(fig, "10_plv_by_run")


def fig_force_by_run():
    f = pd.read_csv(config.ANALYSIS_DIR / "force_phase_regression.csv").dropna(subset=["beta_omega"])
    f = f[~f.hand_check.astype(bool)]
    t1 = pd.read_csv(config.TABLES_DIR / "table1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (m, lab) in zip(axes, [("r2_model", "$R^2_{model}$"), ("beta_omega", "β_ω"), ("beta_F", "β_F")]):
        ax.axvspan(t1[m].min(), t1[m].max(), color=C[2], alpha=0.12, lw=0)
        for y, day in enumerate(RUN_ORDER):
            sub = f[f.day == day]
            ax.plot(sub[m], y + np.random.default_rng(y).uniform(-0.15, 0.15, len(sub)), "o", ms=5,
                    color=C[0], mec="white", mew=0.8)
        ax.set_title(lab)
        ax.grid(axis="y", visible=False)
    axes[1].axvline(0, color=MUTED, lw=0.6)
    axes[2].axvline(0, color=MUTED, lw=0.6)
    axes[0].set_yticks(range(len(RUN_ORDER)), [RUN_LABEL[d] for d in RUN_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Force–phase regression by run (unflagged dog-runs). Green band = Table I range (primary run, hand-curated)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "11_force_by_run")


FIGS = dict(dataset=fig_dataset, preprocessing=fig_preprocessing, segmentation=fig_segmentation_error,
            gps=fig_gps_lag, force_qa=fig_force_qa,
            regression=fig_regression, beta=fig_beta_screen, plv=fig_plv,
            force_by_run=fig_force_by_run)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
