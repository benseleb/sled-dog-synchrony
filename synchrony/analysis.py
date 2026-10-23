"""Both analyses on every qualifying window of every analysed run.

Segmentation is automatic (phase.segment_and_phase); a hand-corrected
zero-crossing file is used instead wherever one exists
(hand_corrected_path). Justification: on the Dec 11 primary window,
automatic segmentation reproduces the hand-curated Dec-11 Table I
inside the bootstrap 95% CI for 9/10 dogs, and the exception (Shifter) shows
up as an out-of-band beta_omega. So everything runs automatically and dogs
outside BETA_OMEGA_OK are flagged for hand correction.

Analysis is restricted to the qualifying windows themselves (padding and
gaps between a dog's windows excluded) by setting speed to NaN outside them;
force_phase_regression's speed gate (exclude_low_speed_strides=True) then
drops every stride/force peak not fully inside a window.

GPS lag: one value per run (gps_lag.py). PLV uses only boot segments that
sync_check verified in sync (dropped dogs are logged to sync_trimmed.csv).
Force regression is per dog and doesn't need cross-dog sync.

Outputs (config.ANALYSIS_DIR): force_phase_regression.csv, pairwise_plv.csv,
sync_trimmed.csv.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config, gps_lag, sync_check as sc
from .datasets import QUEUE, TIER, MIN_WINDOW_S, SPEED_MIN
from .force import force_valid_mask, force_phase_regression
from .gps import gps_speed_interp
from .io import load_dog
from .phase import segment_and_phase, time_interpolated_phase
from .plv import windowed_plv_pooled
from .windows import qualifying_ranges, windows_for

OUT = config.ANALYSIS_DIR
# GPS lag: one value per run from gps_lag.py (run gps_lag.py after
# sync_check.py). Only the force regression's speed gate uses it.
# Dec-11 curated bootstrap 95% CI envelope for beta_omega, Shifter excluded
# (automatic vs hand-curated segmentation) -- outside this, hand-check the dog
BETA_OMEGA_OK = (-0.77, -0.33)


def hand_corrected_path(day_dir, dog):
    """Where the segmentation GUI saves a dog-run's corrected zero crossings
    (seconds from the dog's crop start, see load_dog_windows)."""
    return config.HAND_CORRECTED_DIR / day_dir.replace("/", "_") / f"{dog}_curated.npy"


def load_dog_windows(day, dog, dog_file, gps_file):
    """Mirror load_for_editing's crop exactly, then segment + gate to windows."""
    crop = qualifying_ranges(day, dog)                 # padded
    spans = qualifying_ranges(day, dog, pad_s=0)       # exact
    lo, hi = min(s for s, _ in crop), max(e for _, e in crop)
    d = load_dog(config.RAW_DIR / day / dog_file, lo)
    if d is None:
        return None
    keep = d["t"] <= (hi - d["t0"])
    d = {k: (v[keep] if k != "t0" else v) for k, v in d.items()}
    seg = segment_and_phase(d["t"], d["az"])
    if seg is None:
        return None
    zc, source = seg["zero_crossings"], "auto"
    curated = hand_corrected_path(day, dog)
    if curated.exists():
        zc, source = np.load(curated), "hand"

    speed = gps_speed_interp(config.RAW_DIR / day / gps_file, d["t"], d["t0"], shift_s=gps_lag.lag_for_day(day))
    t_abs = d["t"] + d["t0"]
    in_window = np.zeros_like(t_abs, dtype=bool)
    for s, e in spans:
        in_window |= (t_abs >= s) & (t_abs <= e)
    speed_gated = np.where(in_window, speed, np.nan)

    # Per-window force baseline: the load cell's zero can step between
    # windows (Tiki, Dec 11: ~155k-count jump between t+1157s and t+1464s,
    # every window individually beta_F>0 but pooled beta_F<0). Pooling
    # across such a step makes F_k encode "which side of the step" instead
    # of pull strength, so remove each window's own median before pooling.
    # Equivalent to the manuscript's single-window normalization when a dog
    # has one window. baseline_step reports the largest jump between window
    # baselines relative to the typical within-window p1-p99 force range.
    # Invalid load-cell samples (sentinels, dead cell, in-window baseline
    # drift) -- see force_valid_mask / force_scan.py
    spans_rel = [(s - d["t0"], e - d["t0"]) for s, e in spans]
    force_valid, force_diag = force_valid_mask(d["t"], d["load"], d["offset"], spans=spans_rel,
                                                  moving=speed_gated >= SPEED_MIN)

    force = (d["load"] - d["offset"]).astype(float)
    force_aligned = force.copy()
    medians, ranges = [], []
    for s, e in spans:
        m = (t_abs >= s) & (t_abs <= e) & force_valid
        if m.sum() < 2:
            continue
        medians.append(np.median(force[m]))
        ranges.append(np.subtract(*np.percentile(force[m], [99, 1])))
        force_aligned[(t_abs >= s) & (t_abs <= e)] -= medians[-1]
    baseline_step = (np.ptp(medians) / np.median(ranges)) if len(medians) > 1 and np.median(ranges) > 0 else 0.0

    T = np.diff(zc)
    zc_abs = zc + d["t0"]
    stride_in = np.interp(zc[:-1], d["t"], speed_gated) >= SPEED_MIN
    stride_in &= np.interp(zc[1:], d["t"], speed_gated) >= SPEED_MIN
    irregular = (T < 1 / 3.5) | (T > 1 / 2.5)
    pct_flag = 100 * irregular[stride_in].mean() if stride_in.any() else np.nan

    return dict(t=d["t"], t0=d["t0"], t_abs=t_abs, load=d["load"], offset=d["offset"],
                force_aligned=force_aligned, baseline_step=baseline_step,
                force_valid=force_valid, force_diag=force_diag,
                zc=zc, zc_abs=zc_abs, source=source, speed_gated=speed_gated,
                phase=time_interpolated_phase(zc, d["t"]), pct_flag_moving=pct_flag,
                n_moving_strides=int(stride_in.sum()))


def window_phases(day, w, names, loaded, gps_file):
    """Phases of `names` on a shared 125 Hz grid over window w, for the
    manuscript PLV method (SI S4; validate_plv_port.py reproduces the
    notebook exactly): NaN below 3 m/s and outside each dog's own data (the
    notebook extrapolated, harmless on Dec 11 where every dog spans the run).
    Returns (grid in s from window start, {dog: phase})."""
    grid_abs = np.arange(w.start_epoch, w.end_epoch, 1 / 125.0)
    moving = gps_speed_interp(config.RAW_DIR / day / gps_file, grid_abs, 0.0,
                                 shift_s=gps_lag.lag_for_day(day)) >= SPEED_MIN
    ph = {}
    for n in names:
        d = loaded[n]
        p = np.interp(grid_abs, d["t_abs"], np.unwrap(d["phase"]), left=np.nan, right=np.nan)
        p[~moving] = np.nan
        ph[n] = p
    return grid_abs - w.start_epoch, ph


def run(force_only=False, plv_days=None):
    """plv_days: rerun only these days (both threads) and keep every other
    day's rows from the existing CSVs -- e.g. after a sync-trim change that
    only affects one run."""
    by_day = defaultdict(list)
    for day, dog, dog_file, gps_file in QUEUE:
        if plv_days is None or day in plv_days:
            by_day[day].append((dog, dog_file, gps_file))
    # Sync rule (Ben, 2026-10-10): a dog enters PLV in a window only if every
    # boot segment of its covering that window is verified "in sync" by
    # sync_check.py. Undecidable and offset segments are both excluded, even
    # where the offset could be guessed. Force regression is per dog and
    # doesn't need cross-dog sync, so it's unaffected.
    boot = sc.verified_in_sync()
    trimmed = []

    force_rows, plv_rows = [], []
    for day, items in by_day.items():
        print(f"\n=== {day} (tier {TIER[day]}) ===", flush=True)
        loaded = {}
        for dog, dog_file, gps_file in items:
            d = load_dog_windows(day, dog, dog_file, gps_file)
            if d is None:
                print(f"  {dog:9s} no usable data")
                continue
            loaded[dog] = d
            kw = dict(speed_mps=d["speed_gated"], speed_min=SPEED_MIN, exclude_low_speed_strides=True,
                      force_valid=d["force_valid"])
            # same rule as force_scan.py: <120 s of moving, valid force (e.g. all
            # of Dec 09, ~16 s) gives a ~45-row fit that is pure noise
            usable_s = np.sum((d["speed_gated"] >= SPEED_MIN) & d["force_valid"]) / 125.0
            if usable_s < MIN_WINDOW_S:
                fr = fr_unaligned = None
            else:
                fr = force_phase_regression(d["t"], d["zc"], d["force_aligned"], np.zeros_like(d["force_aligned"]), **kw)
                fr_unaligned = force_phase_regression(d["t"], d["zc"], d["load"], d["offset"], **kw)
            row = dict(day=day, tier=TIER[day], dog=dog, zc_source=d["source"],
                       n_moving_strides=d["n_moving_strides"], pct_flag_moving=d["pct_flag_moving"],
                       baseline_step=d["baseline_step"], force_sentinel_s=d["force_diag"]["sentinel_s"],
                       force_drift_s=d["force_diag"]["drift_s"], force_max_drift=d["force_diag"]["max_drift"])
            if fr_unaligned is not None:
                row.update(beta_F_unaligned=fr_unaligned["beta_F"], beta_omega_unaligned=fr_unaligned["beta_omega"])
            if fr is None:
                row.update(note=f"no regression: {usable_s:.0f} s usable force (dead cell, or < {MIN_WINDOW_S:.0f} s)")
            else:
                row.update({k: fr[k] for k in ["n_strides", "r2_model", "beta_F", "beta_omega", "A_phi", "theta0"]})
                row["hand_check"] = not (BETA_OMEGA_OK[0] <= fr["beta_omega"] <= BETA_OMEGA_OK[1])
            force_rows.append(row)
            print(f"  {dog:9s} [{d['source']}] moving strides={d['n_moving_strides']:5d} step={d['baseline_step']:4.1f} "
                  f"masked={d['force_diag']['sentinel_s'] + d['force_diag']['drift_s']:5.0f}s "
                  f"flag={d['pct_flag_moving']:4.1f}%  " +
                  (f"R2={fr['r2_model']:.3f} bF={fr['beta_F']:.3f} bW={fr['beta_omega']:.3f} "
                   f"Aphi={fr['A_phi']:.3f}{'  <-- HAND CHECK' if row['hand_check'] else ''}"
                   if fr else row["note"]), flush=True)

        if force_only:  # PLV doesn't use force; keep the existing pairwise_plv.csv
            continue
        for wi, w in windows_for(day).iterrows():
            names = []
            for n in w.dogs.split("|"):
                if n not in loaded:
                    continue
                k = (loaded[n]["t_abs"] >= w.start_epoch) & (loaded[n]["t_abs"] <= w.end_epoch)
                covered = np.ptp(loaded[n]["t_abs"][k]) if k.sum() > 1 else 0.0
                # shared_windows built coverage from every file a dog has that day
                # (e.g. Harper's part2/3 on Dec 11); the queue loads one file, so
                # a listed dog can have little or no data here
                if covered < MIN_WINDOW_S:
                    print(f"  window {wi}: {n} skipped, only {covered:.0f}s of data in the loaded file")
                    continue
                segs = boot[(boot.day == day) & (boot.dog == n) & (boot.boot_start < w.end_epoch)
                            & (boot.boot_end > w.start_epoch)]
                if segs.empty or (segs.verdict != "in sync").any():
                    verdict = "not checked" if segs.empty else ", ".join(sorted(set(segs.verdict) - {"in sync"}))
                    print(f"  window {wi}: {n} dropped from PLV, sync {verdict}")
                    trimmed.append(dict(day=day, window=wi, window_start=w.start_epoch, dog=n, verdict=verdict))
                    continue
                names.append(n)
            grid, ph = window_phases(day, w, names, loaded, next(g for dg, _, g in items if dg in loaded))
            rng = np.random.default_rng(123)  # one stream per window, pairs in order, as in the notebook
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    res, tau95 = windowed_plv_pooled(grid, ph[names[a]], ph[names[b]], rng=rng)
                    plv = np.array([r["plv"] for r in res])
                    tested = np.isfinite(plv)
                    if not tested.any():
                        continue
                    n_sig = sum(r["significant"] for r in res)
                    plv_rows.append(dict(day=day, tier=TIER[day], window=wi, window_start=w.start_epoch,
                                         window_s=w.duration_s, frac_moving=w.frac_moving,
                                         dog_a=names[a], dog_b=names[b], n_windows=int(tested.sum()),
                                         n_windows_all=len(res), tau95=tau95,
                                         n_significant=n_sig, pct_significant=100 * n_sig / tested.sum(),
                                         mean_plv=plv[tested].mean(), max_plv=plv[tested].max()))
            print(f"  window {wi} ({w.duration_s:.0f}s, {len(names)} dogs): PLV done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    force_df, plv_df = pd.DataFrame(force_rows), pd.DataFrame(plv_rows)
    if plv_days is not None:  # splice the rerun days into the existing results
        old_f = pd.read_csv(OUT / "force_phase_regression.csv")
        old_p = pd.read_csv(OUT / "pairwise_plv.csv")
        force_df = pd.concat([old_f[~old_f.day.isin(plv_days)], force_df], ignore_index=True)
        plv_df = pd.concat([old_p[~old_p.day.isin(plv_days)], plv_df], ignore_index=True)
        order = {d: i for i, d in enumerate(TIER)}
        force_df = force_df.sort_values("day", key=lambda c: c.map(order), kind="stable")
        plv_df = plv_df.sort_values("day", key=lambda c: c.map(order), kind="stable")
    force_df.to_csv(OUT / "force_phase_regression.csv", index=False)
    if not force_only:
        plv_df.to_csv(OUT / "pairwise_plv.csv", index=False)
        pd.DataFrame(trimmed, columns=["day", "window", "window_start", "dog", "verdict"]).to_csv(
            OUT / "sync_trimmed.csv", index=False)
    print(f"\nSaved -> {OUT}")
