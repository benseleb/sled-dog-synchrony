"""Cross-dog clock-sync check, one verdict per boot segment.

Firmware facts (Ben, 2026-10-10): every data gap is a hard-coded device
reboot. On restart, logging begins on a GPS PPS edge and the loop is locked
to the top of the GPS second, so sub-second alignment across dogs is
guaranteed by hardware. The only possible error is a mislabelled second (the
first GPS packet after a reset not matching the latest pulse): an INTEGER
offset, constant within a boot segment (a dog's stretch between gaps). So
the question per boot segment is "0, or +-k s?"; sub-second residual lags
(+-0.3 s on Dec 11) are reaction time / position, not clock error.

Method, per dog per boot segment (intersected with its qualifying windows):
  1. Tugline force, band-passed 0.2-2 Hz (below the ~3 Hz stride, so stride
     echoes don't alias onto whole seconds), cross-correlated against the
     median of the teammates' force. Shared jolts from the rig reach every
     line almost at once, giving a ~0.35 s wide peak.
  2. Each whole second k in -7..7 scores the best r within +-0.5 s of k; a
     call needs a margin >= FORCE_MARGIN_MIN over the runner-up. Calibrated
     on the field-confirmed Dec-11 primary window cut into pieces: 1.0%
     wrong calls at 120 s, 0% from 180 s.
  3. Spans >= 10 min also get calls on ~300 s pieces, pooled with the
     whole-span call (a long span's peak can be real but blurred).
  4. Fallback for spans >= 10 min only: the activity envelope (1 s RMS of
     high-passed |a|, 30 s high-passed), whose ~3 s wide peak only resolves
     whole seconds over long spans.

PLV rule (Ben, 2026-10-10): a dog enters a PLV window only if its covering
boot segment is "in sync"; undecidable and offset segments are excluded.

Outputs (config.SYNC_DIR): sync_check_segments.csv (every span and piece),
sync_check_boot_segments.csv (one verdict per boot segment).
"""
import numpy as np
import pandas as pd

from . import config
from .datasets import SYNC_CHECK_RUNS, RUN_START_MS, WINDOW_END_MS
from .io import load_rows, trim_corruption, GAP_SECONDS
from .force import LOAD_SENTINELS
from .gps import read_gps
from .phase import butter_filter
from .windows import window_csv

DATA = config.RAW_DIR
OUT = config.SYNC_DIR

FS = 125.0
GRID_STEP = 0.1
SHIFTS = np.arange(-60.0, 60.01, GRID_STEP)
MIN_MARGIN = 0.2          # envelope peak margin below this = lag not trustworthy either way
MIN_OVERLAP_S = 120.0
# The raw envelope is dominated by slow, shared trends (the whole run is
# "more active" than idle), which makes every correlation peak broad
# (margin ~0.07). Subtracting a 30 s moving mean leaves the transitions --
# launches, stops, slowdowns -- and gives sharp, unique peaks (margin
# ~0.4-0.6 on Dec 11; offsets of 0.3-20 s injected into one dog are
# recovered against the team within 0.3 s).
ENVELOPE_HP_S = 30.0

# (day_dir, gps_file, tier). Dec 06 (tier 0) gets windows but is skipped here.
DAYS = [(r.day, r.gps_file, r.tier) for r in SYNC_CHECK_RUNS]

WIN_SHIFTS = np.arange(-15.0, 15.01, GRID_STEP)   # windows are 2-35 min; +-15 s still catches Healy-sized offsets
WIN_MIN_OVERLAP_S = 60.0
# The manuscript's analysis window on Dec 11: the field-confirmed initial test
PRIMARY = ("DEC_11_TEST", "primary", RUN_START_MS / 1000, WINDOW_END_MS / 1000)


def dog_files(day_dir):
    """{dog: [csv paths]} -- multi-part files (Harper_Device10_part1/2/3) grouped."""
    out = {}
    for p in sorted((DATA / day_dir).glob("*_Device*.csv")):
        out.setdefault(p.name.split("_")[0], []).append(p)
    return out


def zscore(x):
    m = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if m.sum() > 2 and np.nanstd(x) > 0:
        out[m] = (x[m] - np.nanmean(x)) / np.nanstd(x)
    return out


def highpass_envelope(a, width_s=ENVELOPE_HP_S):
    """a minus its NaN-aware moving mean, re-z-scored."""
    w = int(width_s / GRID_STEP)
    m = np.isfinite(a)
    num = np.convolve(np.where(m, a, 0.0), np.ones(w), "same")
    den = np.convolve(m.astype(float), np.ones(w), "same")
    return zscore(a - num / np.maximum(den, 1))


def lag_curve(x, ref, shifts=SHIFTS, min_overlap_s=MIN_OVERLAP_S):
    """r(shift) = corr(x(t + shift), ref(t)), over finite overlap. A positive
    best shift means x's events are stamped `shift` s LATER than ref's."""
    n = x.size
    r = np.full(shifts.size, np.nan)
    for i, s in enumerate(shifts):
        k = int(round(s / GRID_STEP))
        a = x[max(k, 0): n + min(k, 0)]
        b = ref[max(-k, 0): n - max(k, 0)]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() * GRID_STEP >= min_overlap_s:
            r[i] = np.corrcoef(a[m], b[m])[0, 1]
    return r


def peak_stats(r, shifts=SHIFTS):
    """Best shift, its r, and the peak's prominence: best r minus the best r
    more than 5 s away (a sharp unique peak has a large margin)."""
    if np.all(np.isnan(r)):
        return np.nan, np.nan, np.nan
    j = int(np.nanargmax(r))
    far = np.abs(shifts - shifts[j]) > 5.0
    margin = r[j] - np.nanmax(r[far]) if np.isfinite(r[far]).any() else np.nan
    return shifts[j], r[j], margin


# ---------------------------------------------------------------- envelopes
#
# A dog's clock can only change where THAT dog's recording breaks (dropout,
# reconnect, new file); window boundaries are mostly set by other dogs
# joining or dropping out. So the unit is each dog's own gap-free stretch,
# intersected with the qualifying windows it belongs to.
#
# Envelope lag precision depends on length. Calibrated on the field-confirmed
# Dec-11 primary window cut into sub-windows:
# 99% of in-sync |lag| < ~2 s at 120-180 s, < ~1 s at 240-400 s, 0.6 s at
# 600 s, 0.3 s over the full 766 s. Tolerances below are those, rounded up.
TOLERANCE_BY_LENGTH = [(600.0, 0.5), (240.0, 1.0), (120.0, 2.0)]  # (min stretch s, tolerance s)

_rows_cache = {}


def _load(path):
    if path not in _rows_cache:
        d = trim_corruption(load_rows(path))
        d["t"] = d["epoch"] + d["ms"] / 1000.0
        _rows_cache[path] = d
    return _rows_cache[path]


def window_envelope(paths, lo, hi, grid):
    """High-passed activity envelope from samples inside [lo, hi] only.
    Returns (envelope, note); envelope is None if the dog's data inside the
    window isn't one gap-free stretch covering it."""
    for p in paths:
        d = _load(p)
        k = (d["t"] >= lo) & (d["t"] <= hi)
        if k.sum() < 10 * FS:
            continue
        t = d["t"][k]
        gaps = np.diff(t) > GAP_SECONDS
        if gaps.any() or t[0] - lo > GAP_SECONDS or hi - t[-1] > GAP_SECONDS:
            return None, f"{p.name}: gap inside window ({np.diff(t).max():.1f} s) or partial coverage"
        mag2 = sum(butter_filter(d[a][k].astype(float), 0.5, FS, "high", 2) ** 2 for a in ("ax", "ay", "az"))
        rms = np.sqrt(np.convolve(mag2, np.ones(int(FS)) / FS, mode="same"))
        return highpass_envelope(np.interp(grid, t, rms)), p.name
    return None, "no data in window"


def check_window(day_dir, gps_file, label, lo, hi, dogs):
    grid = np.arange(lo, hi, GRID_STEP)
    t_gps, v_gps = read_gps(DATA / day_dir / gps_file)
    speed = highpass_envelope(np.interp(grid, t_gps, v_gps, left=np.nan, right=np.nan))
    files = dog_files(day_dir)
    acts, notes = {}, {}
    for dog in dogs:
        env, note = window_envelope(files.get(dog, []), lo, hi, grid)
        notes[dog] = note
        if env is not None:
            acts[dog] = env
    rows = []
    for dog in dogs:
        row = dict(day=day_dir, window=label, start_epoch=lo, duration_s=round(hi - lo), dog=dog,
                   n_dogs_used=len(acts), source=notes[dog])
        if dog in acts and len(acts) >= 2:
            others = np.array([b for d, b in acts.items() if d != dog])
            with np.errstate(all="ignore"):
                ref = np.nanmedian(others, axis=0) if len(others) > 1 else others[0]
            lt, rt, mt = peak_stats(lag_curve(acts[dog], ref, WIN_SHIFTS, WIN_MIN_OVERLAP_S), WIN_SHIFTS)
            lg, rg, mg = peak_stats(lag_curve(speed, acts[dog], WIN_SHIFTS, WIN_MIN_OVERLAP_S), WIN_SHIFTS)
            row.update(lag_team_s=lt, r_team=rt, margin_team=mt, lag_gps_s=-lg, r_gps=rg, margin_gps=mg)
        rows.append(row)
    return rows


def tolerance_for(length_s):
    for min_len, tol in TOLERANCE_BY_LENGTH:
        if length_s >= min_len:
            return tol
    return np.nan


def gapfree_segments(paths):
    """[(t_start, t_end, path)] for every gap-free stretch in a dog's files."""
    out = []
    for p in paths:
        d = _load(p)
        if d["t"].size < 10 * FS:
            continue
        br = np.flatnonzero(np.diff(d["t"]) > GAP_SECONDS)
        for idx in np.split(np.arange(d["t"].size), br + 1):
            out.append((d["t"][idx[0]], d["t"][idx[-1]], p))
    return out


def merge_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + GRID_STEP:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def envelope_pieces(paths, lo, hi, grid, min_piece_s=30.0):
    """A teammate's envelope over [lo, hi] built from its own gap-free pieces
    (each filtered separately, NaN across its gaps)."""
    env = np.full(grid.size, np.nan)
    for a, b, p in gapfree_segments(paths):
        a, b = max(a, lo), min(b, hi)
        if b - a < min_piece_s:
            continue
        k = (grid >= a) & (grid <= b)
        e, _ = window_envelope([p], a, b, grid[k])
        if e is not None:
            env[k] = e
    return env


def main_segments():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Initial test: the field-confirmed Dec-11 primary window, as is
    day, label, lo, hi = PRIMARY
    pdogs = sorted(dog_files(day))
    for r in check_window(day, "GPSLogger00.csv", label, lo, hi, pdogs):
        r.update(tier=1, span_s=r["duration_s"], tolerance_s=tolerance_for(r["duration_s"]))
        add_force_call(r, dog_files(day), r["dog"], [o for o in pdogs if o != r["dog"]], lo, hi)
        rows.append(r)
    for day_dir, gps_file, tier in DAYS:
        if tier == 0:
            continue
        print(f"=== {day_dir}", flush=True)
        w = pd.read_csv(window_csv(day_dir))
        w = w[w.duration_s >= 120]
        t_gps, v_gps = read_gps(DATA / day_dir / gps_file)
        files = dog_files(day_dir)
        dogs = sorted({d for ds in w.dogs for d in ds.split("|")})
        for dog in dogs:
            my_windows = merge_spans([(r.start_epoch, r.end_epoch) for r in w.itertuples() if dog in r.dogs.split("|")])
            for a, b, p in gapfree_segments(files[dog]):
                for wa, wb in my_windows:
                    lo, hi = max(a, wa), min(b, wb)
                    if hi - lo < 120:
                        continue
                    grid = np.arange(lo, hi, GRID_STEP)
                    mine, _ = window_envelope([p], lo, hi, grid)
                    mates = [envelope_pieces(files[o], lo, hi, grid) for o in dogs if o != dog]
                    mates = [m for m in mates if np.isfinite(m).sum() * GRID_STEP >= WIN_MIN_OVERLAP_S]
                    row = dict(day=day_dir, tier=tier, window="segment", dog=dog, start_epoch=lo, boot_start=a, boot_end=b,
                               duration_s=round(hi - lo), span_s=round(hi - lo), source=p.name,
                               n_dogs_used=len(mates) + 1, tolerance_s=tolerance_for(hi - lo))
                    if mine is not None and mates:
                        with np.errstate(all="ignore"):
                            ref = np.nanmedian(np.array(mates), axis=0) if len(mates) > 1 else mates[0]
                        speed = highpass_envelope(np.interp(grid, t_gps, v_gps, left=np.nan, right=np.nan))
                        lt, rt, mt = peak_stats(lag_curve(mine, ref, WIN_SHIFTS, WIN_MIN_OVERLAP_S), WIN_SHIFTS)
                        lg, rg, mg = peak_stats(lag_curve(speed, mine, WIN_SHIFTS, WIN_MIN_OVERLAP_S), WIN_SHIFTS)
                        row.update(lag_team_s=lt, r_team=rt, margin_team=mt, lag_gps_s=-lg, r_gps=rg, margin_gps=mg)
                    add_force_call(row, files, dog, [o for o in dogs if o != dog], lo, hi)
                    rows.append(row)
                    rows.extend(piece_rows(row, files, [o for o in dogs if o != dog], lo, hi))
    df = pd.DataFrame(rows)
    df["envelope_verdict"] = np.select(
        [df.lag_team_s.isna(), df.margin_team < MIN_MARGIN, df.lag_team_s.abs() > df.tolerance_s],
        ["not tested", "inconclusive", "OFFSET"], default="ok")
    calls = [final_call(r) for r in df.to_dict("records")]
    df["verdict"], df["verdict_from"] = [c[0] for c in calls], [c[1] for c in calls]
    df.to_csv(OUT / "sync_check_segments.csv", index=False, float_format="%.3f")
    boot = summarize_boot_segments(df[df.window.isin(["segment", "piece"])])
    boot.to_csv(OUT / "sync_check_boot_segments.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 240, "display.max_rows", 500):
        cols = ["day", "window", "dog", "span_s", "lag_team_s", "margin_team", "force_peak_s", "force_k_s",
                "force_margin", "verdict", "verdict_from"]
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ---------------------------------------------------------------- whole-second (force) check

FORCE_STEP = 0.02
FORCE_BAND = (0.2, 2.0)
FORCE_SHIFTS = np.arange(-8.0, 8.001, FORCE_STEP)  # +-7 whole seconds compete; widened from +-4 after Nov 28
FORCE_MARGIN_MIN = 0.10
ENVELOPE_FALLBACK_MIN_S = 600.0   # envelope only resolves whole seconds over >= 10 min
PIECE_SPLIT_MIN_S = 600.0         # spans this long also get calls on ~300 s pieces (Ben, 2026-10-12)
PIECE_S = 300.0


def force_on_grid(paths, lo, hi, grid, min_piece_s=30.0):
    """0.2-2 Hz tugline force on grid, built from each gap-free piece
    separately (sentinels interpolated over). None if the cell is dead."""
    out = np.full(grid.size, np.nan)
    for a, b, p in gapfree_segments(paths):
        a, b = max(a, lo), min(b, hi)
        if b - a < min_piece_s:
            continue
        d = _load(p)
        k = (d["t"] >= a) & (d["t"] <= b)
        t, load = d["t"][k], d["load"][k]
        good = ~np.isin(load, LOAD_SENTINELS)
        if good.sum() < 10 * FS:
            continue
        f = np.interp(t, t[good], (load - d["offset"][k])[good].astype(float))
        if np.ptp(f) == 0:
            return None
        f = butter_filter(butter_filter(f, FORCE_BAND[0], FS, "high", 2), FORCE_BAND[1], FS, "low", 4)
        g = (grid >= a) & (grid <= b)
        out[g] = zscore(np.interp(grid[g], t, f))
    return out if np.isfinite(out).any() else None


def force_lag_curve(x, ref):
    """r(shift) on FORCE_SHIFTS, same convention as lag_curve."""
    n = x.size
    r = np.full(FORCE_SHIFTS.size, np.nan)
    for i, sft in enumerate(FORCE_SHIFTS):
        k = int(round(sft / FORCE_STEP))
        a, b = x[max(k, 0): n + min(k, 0)], ref[max(-k, 0): n - max(k, 0)]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() * FORCE_STEP >= WIN_MIN_OVERLAP_S:
            r[i] = np.corrcoef(a[m], b[m])[0, 1]
    return r


def whole_second_bins(r):
    """{k: best r within +-0.5 s of whole second k}, k = -7..7."""
    bins = {}
    for k in range(-7, 8):
        rk = r[np.abs(FORCE_SHIFTS - k) < 0.5]
        if np.isfinite(rk).any():
            bins[k] = np.nanmax(rk)
    return bins


def whole_second_call(x, ref):
    """(best whole-second offset, margin over the next best, peak lag).
    Each whole second k scores the best r within +-0.5 s of k."""
    r = force_lag_curve(x, ref)
    bins = whole_second_bins(r)
    if len(bins) < 2:
        return np.nan, np.nan, np.nan
    peak = FORCE_SHIFTS[np.nanargmax(r)]
    # A peak at the edge of the search range means the true offset may lie
    # beyond it (Nov 28: pairwise peaks at +5 and +8 s); the edge bin would
    # then win by a spurious margin, so make no call.
    if abs(peak) > FORCE_SHIFTS.max() - 0.5:
        return np.nan, np.nan, peak
    best = max(bins, key=bins.get)
    return best, bins[best] - max(v for k, v in bins.items() if k != best), peak


def add_force_call(row, files, dog, mates, lo, hi):
    grid = np.arange(lo, hi, FORCE_STEP)
    mine = force_on_grid(files[dog], lo, hi, grid)
    refs = [force_on_grid(files[o], lo, hi, grid) for o in mates]
    refs = [f for f in refs if f is not None and np.isfinite(f).sum() * FORCE_STEP >= WIN_MIN_OVERLAP_S]
    if mine is None or not refs:
        row.update(force_note="no usable force" if mine is None else "no teammate force")
        return
    with np.errstate(all="ignore"):
        ref = np.nanmedian(np.array(refs), axis=0) if len(refs) > 1 else refs[0]
    k, margin, peak = whole_second_call(mine, ref)
    row.update(force_k_s=k, force_margin=margin, force_peak_s=peak, force_mates=len(refs))
    if np.isnan(k) and np.isfinite(peak):
        row.update(force_note="peak at search edge: offset may exceed +-7 s")


def final_call(row):
    """Whole-second verdict per boot segment: force first, envelope only for
    >= 10-min segments, otherwise can't tell."""
    if row.get("force_margin", np.nan) >= FORCE_MARGIN_MIN:
        k = int(row["force_k_s"])
        return ("in sync" if k == 0 else f"OFFSET {k:+d} s"), "force"
    lag = row.get("lag_team_s", np.nan)
    # Only round a lag that sits near a whole second: -0.5 s is as close to
    # -1 as to 0, and sub-second residuals are behaviour, not clock
    if (row["span_s"] >= ENVELOPE_FALLBACK_MIN_S and row.get("margin_team", 0) >= MIN_MARGIN
            and abs(lag - np.round(lag)) <= 0.3):
        k = int(np.round(lag))
        return ("in sync" if k == 0 else f"OFFSET {k:+d} s"), "envelope"
    return "can't tell", ""


def piece_rows(seg_row, files, mates, lo, hi):
    """Extra force calls on 300-600 s pieces of a long (>= PIECE_SPLIT_MIN_S)
    span, pooled with the whole-span call by summarize_boot_segments. Over a
    long span the force peak can be real but blurred below the margin (Dec
    07 Baby/Stevie, 35 min); pieces match the calibrated lengths (0 wrong
    calls from 180 s). Equal pieces, so no short remainder."""
    if hi - lo < PIECE_SPLIT_MIN_S:
        return []
    edges = np.linspace(lo, hi, int((hi - lo) // PIECE_S) + 1)
    out = []
    for pa, pb in zip(edges[:-1], edges[1:]):
        row = {k: seg_row[k] for k in ("day", "tier", "dog", "boot_start", "boot_end", "source")}
        row.update(window="piece", start_epoch=pa, duration_s=round(pb - pa), span_s=round(pb - pa),
                   tolerance_s=tolerance_for(pb - pa))
        add_force_call(row, files, row["dog"], mates, pa, pb)
        out.append(row)
    return out


def summarize_boot_segments(df):
    """One verdict per boot segment (a dog's stretch between resets), pooling
    the calls from every analysed piece of it: in sync / OFFSET if all
    confident calls agree, CONFLICT if they don't, can't tell if none."""
    out = []
    for (day, dog, boot, boot_end), g in df.groupby(["day", "dog", "boot_start", "boot_end"], sort=False):
        calls = [v for v in g.verdict if v != "can't tell"]
        verdict = ("can't tell" if not calls else calls[0] if len(set(calls)) == 1 else
                   "CONFLICT: " + ", ".join(sorted(set(calls))))
        whole = g[g.window != "piece"] if "window" in g else g
        out.append(dict(day=day, dog=dog, boot_start=boot, boot_end=boot_end, n_pieces=len(whole),
                        analysed_s=whole.span_s.sum(), n_calls=len(calls), verdict=verdict))
    return pd.DataFrame(out)


def verified_in_sync(path=OUT / "sync_check_boot_segments.csv"):
    """Boot-segment table for the PLV inclusion rule (Ben, 2026-10-10):
    only boot segments verified "in sync" may enter pairwise analysis;
    undecidable and offset segments are both excluded."""
    return pd.read_csv(path)
