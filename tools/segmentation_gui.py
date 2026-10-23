"""
Manual zero-crossing correction tool (generalized from the Dec-11-only
ZeroCrossingEditor in the manuscript's furriest_segmentation_extended.ipynb).
hand_correct_segmentation.ipynb drives it; saved files land where
synchrony.analysis picks them up automatically.

Typical use, in a local Jupyter session (needs an interactive backend):

    %matplotlib qt
    from segmentation_gui import load_for_editing, ZeroCrossingEditor, verify_saved
    from synchrony import config
    from synchrony.analysis import hand_corrected_path

    t, z, auto_zc, run_start, speed_mps, spans = load_for_editing(
        config.RAW_DIR / "DEC_10_TEST/Run1/Shifter_Device09.csv",
        config.RAW_DIR / "DEC_10_TEST/Run1/GPSLogger00.csv",
    )
    save_path = hand_corrected_path("DEC_10_TEST/Run1", "Shifter")
    editor = ZeroCrossingEditor(t, z, zero_crossings=auto_zc, save_path=save_path,
                                 speed_mps=speed_mps, speed_min=3.0)
    # speed_mps shades stretches below speed_min light blue in the plot --
    # those are excluded from analysis regardless of segmentation quality
    # there (see exclude_low_speed_strides in synchrony.force), so skip them.
    # ... correct in the GUI, press 's' to save, close the window when done ...

Then in a later cell (after closing the figure):

    verify_saved(save_path, auto_n=len(auto_zc))
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synchrony.gps import gps_run_start, gps_speed_interp
from synchrony.io import load_dog
from synchrony.phase import butter_filter, znormalize, marker_event


def load_for_editing(dog_csv, gps_csv=None, run_start=None, fs=125.0, crop_to=None, highlight=None):
    """Load+trim+gravity-filter+z-normalize one dog's signal exactly as
    segment_and_phase does (so what you see/correct here is what the
    analysis pipeline will actually use), plus an automatic first-pass
    zero-crossing array to pre-seed the editor and (when gps_csv is given)
    GPS speed on the same time axis, for the editor to shade low-speed
    stretches -- those never feed force_phase_regression at all (excluded
    both by the force-covariate speed gate and, now, by the stride-timing
    speed gate too -- see exclude_low_speed_strides in synchrony.force), so
    there's no need to correct segmentation precisely there.

    Pass either run_start (an epoch, if you already know it) or gps_csv (to
    detect it via GPS speed onset, matching the GPS run-start onset). If neither is
    given, no run-start trim is applied (uses the whole corruption-trimmed
    file), and speed_mps comes back as None (no shading).

    crop_to: optional list of (start,end) ABSOLUTE-epoch ranges (e.g. from
    synchrony.windows.qualifying_ranges) -- loads only the union span
    [min(starts), max(ends)] instead of the whole file. Only >=120s windows
    count as worth hand-correcting (datasets.MIN_WINDOW_S);
    for a dog with several disjoint qualifying windows in one day (common on
    Dec 11), this still loads one contiguous span covering all of them rather
    than stitching disjoint segments together, which would fake a
    discontinuity right where marker_event needs a real, continuous signal."""
    if crop_to:
        lo = min(s for s, _ in crop_to)
        hi = max(e for _, e in crop_to)
        run_start = lo if run_start is None else max(run_start, lo)
    if run_start is None and gps_csv is not None:
        run_start = gps_run_start(Path(gps_csv))
    d = load_dog(Path(dog_csv), run_start)
    if d is None:
        raise ValueError(f"{dog_csv}: insufficient data (after run-start trim, if any)")
    if crop_to:
        keep = d["t"] <= (hi - d["t0"])
        if keep.sum() < 200:
            raise ValueError(f"{dog_csv}: crop_to range has too little data ({keep.sum()} samples)")
        d = {k: (v[keep] if k != "t0" else v) for k, v in d.items()}
    az_hp = butter_filter(d["az"], 0.5, fs, "high", 2)
    z, _, _ = znormalize(az_hp)
    auto_zc = np.array(marker_event(d["t"], z))
    speed_mps = gps_speed_interp(Path(gps_csv), d["t"], d["t0"]) if gps_csv is not None else None
    qualifying_spans = [(s - d["t0"], e - d["t0"]) for s, e in highlight] if highlight else None
    return d["t"], z, auto_zc, run_start, speed_mps, qualifying_spans


def verify_saved(save_path, auto_n=None):
    """Reload the saved file and print sanity stats -- run this right after
    closing the editor to confirm the save actually landed correctly."""
    save_path = Path(save_path)
    if not save_path.exists():
        print(f"NOT FOUND: {save_path} -- did you press 's' before closing the figure?")
        return None
    zc = np.load(save_path)
    dt = np.diff(zc)
    n_flagged = int(np.sum((dt < 1 / 3.5) | (dt > 1 / 2.5))) if dt.size else 0
    print(f"{save_path}")
    print(f"  n_crossings = {zc.size}" + (f"  (automatic pass had {auto_n})" if auto_n is not None else ""))
    print(f"  span = {zc[-1] - zc[0]:.1f}s" if zc.size else "  (empty)")
    print(f"  still-flagged-irregular = {n_flagged}/{max(dt.size,1)} ({100*n_flagged/max(dt.size,1):.1f}%)")
    if not np.all(np.diff(zc) > 0):
        print("  WARNING: not strictly increasing -- something's wrong with this file")
    return zc


class ZeroCrossingEditor:
    """
    Left-click: add (snaps to nearest zero)
    'd': delete mode (click near a crossing to remove)
    'a': add mode
    's': save crossings to save_path
    'm': stop deleting and adding
    'z': zoom out to full

    Pass speed_mps (same length/alignment as t, e.g. from load_for_editing)
    to shade stretches below speed_min light blue -- those never reach
    force_phase_regression regardless of segmentation quality there (see
    exclude_low_speed_strides), so treat shaded regions as skippable.

    Pass qualifying_spans (t-relative (start,end) pairs, e.g.
    load_for_editing's highlight output) to additionally shade, in gray,
    whatever's outside them -- i.e. the padding margin and any gap between
    disjoint qualifying windows within one crop. Only the un-shaded stretches
    are in scope for the current analysis; no separate label needed, the
    shading says it.
    """
    def __init__(self, t, z, zero_crossings, save_path, tol=0.5, speed_mps=None, speed_min=3.0,
                 qualifying_spans=None):
        self.t = np.asarray(t, float)
        self.z = np.asarray(z, float)
        self.cross = np.array([] if zero_crossings is None else zero_crossings, dtype=float)
        self.cross.sort()
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.tol = float(tol)
        self.delete_tol = 0.08
        self._low_speed_spans = _low_speed_spans(self.t, speed_mps, speed_min) if speed_mps is not None else []
        self._inactive_spans = _complement_spans(self.t, qualifying_spans) if qualifying_spans else []

        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 6))
        self.mode = 'mute'
        self._sc_ok = None
        self._sc_bad = None

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        print(f"Editing -> will save to {self.save_path}")
        print("Controls: left-click add (snaps). 'd' delete mode, 'a' add mode, 's' save. Toolbar pan/zoom works.")
        if speed_mps is not None:
            print(f"Shaded blue = speed < {speed_min} m/s (excluded from analysis regardless -- skippable).")
        if qualifying_spans:
            print("Shaded gray = outside any qualifying window (padding/gap -- skippable).")
        self._draw()

    def _compute_bad_pts(self):
        xc = self.cross
        bad_pts = np.zeros_like(xc, dtype=bool)
        if xc.size < 2:
            return bad_pts
        dt = np.diff(xc)
        bad_intervals = np.where((dt < 1 / 3.5) | (dt > 1 / 2.5))[0]
        if bad_intervals.size:
            bad_pts[bad_intervals] = True
            bad_pts[bad_intervals + 1] = True
        return bad_pts

    def _draw(self):
        if hasattr(self, "_have_limits") and self._have_limits:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = ylim = None

        self.ax.cla()
        for s, e in self._inactive_spans:
            self.ax.axvspan(s, e, color='0.5', alpha=0.15, zorder=0, lw=0)
        for s, e in self._low_speed_spans:
            self.ax.axvspan(s, e, color='tab:blue', alpha=0.12, zorder=0, lw=0)
        self.ax.plot(self.t, self.z, lw=1, color='0.3')
        bad_mask = self._compute_bad_pts()

        if self.cross.size:
            ok = ~bad_mask
            self._sc_ok = self.ax.scatter(self.cross[ok], np.zeros(np.sum(ok)),
                                           s=28, color='tab:green', label='OK', zorder=5)
            self._sc_bad = self.ax.scatter(self.cross[bad_mask], np.zeros(np.sum(bad_mask)),
                                            s=42, color='tab:red', label='Irregular spacing', zorder=6)
            self.ax.legend(loc='upper right')

        self.ax.set_ylabel('signal z')
        self.ax.set_xlabel('time (s)')

        if xlim is not None and ylim is not None:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        else:
            self._have_limits = True

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if self.mode == 'mute':
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        t_click = event.xdata
        if t_click is None:
            return

        if self.mode == 'add':
            tz = _snap_to_zero(self.t, self.z, t_click)
            self.cross = np.unique(np.r_[self.cross, tz])
            self.cross.sort()
        elif self.mode == 'delete':
            if self.cross.size:
                k = np.argmin(np.abs(self.cross - t_click))
                if np.abs(self.cross[k] - t_click) <= self.delete_tol:
                    self.cross = np.delete(self.cross, k)
        self._draw()

    def _on_key(self, event):
        if event.key == 'd':
            self.mode = 'delete'
            print("Mode: delete (click near a crossing to remove)")
        elif event.key == 'a':
            self.mode = 'add'
            print("Mode: add (click to add/snap)")
        elif event.key == 'm':
            self.mode = 'mute'
            print("Mode: mute (clicks disabled)")
        elif event.key == 'z':
            self.ax.set_xlim(self.t[0], self.t[-1])
            self.ax.set_ylim(self.z.min() - 2, self.z.max() + 2)
            self.mode = 'mute'
            print("Zoom reset to full view")
        elif event.key == 's':
            np.save(self.save_path, self.cross)
            print(f"Saved {self.cross.size} crossings -> {self.save_path}")

        self.fig.canvas.draw_idle()


def _low_speed_spans(t, speed_mps, speed_min):
    """Contiguous (start, end) time spans where speed_mps < speed_min,
    computed once at init (speed doesn't change as you edit crossings)."""
    low = np.asarray(speed_mps) < speed_min
    if not low.any():
        return []
    edges = np.diff(low.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if low[0]:
        starts = np.r_[0, starts]
    if low[-1]:
        ends = np.r_[ends, len(low) - 1]
    return [(t[s], t[min(e, len(t) - 1)]) for s, e in zip(starts, ends)]


def _complement_spans(t, spans):
    """Gaps in t's own [t[0], t[-1]] range not covered by any (start,end) in
    spans -- used to shade whatever's outside the qualifying window(s)."""
    if t.size == 0:
        return []
    spans = sorted(spans)
    out = []
    cur = t[0]
    for s, e in spans:
        if s > cur:
            out.append((cur, min(s, t[-1])))
        cur = max(cur, e)
        if cur >= t[-1]:
            break
    if cur < t[-1]:
        out.append((cur, t[-1]))
    return out


def _snap_to_zero(t, z, t_click, window=0.05):
    i0 = np.searchsorted(t, t_click)
    left = np.searchsorted(t, t_click - window)
    right = np.searchsorted(t, t_click + window)
    tt = t[max(0, left):min(len(t), right)]
    zz = z[max(0, left):min(len(t), right)]
    if tt.size < 2:
        return t_click

    sgn = np.sign(zz)
    changes = np.where(np.diff(sgn) != 0)[0]
    cand_ts = []
    for k in changes:
        t1, t2 = tt[k], tt[k + 1]
        z1, z2 = zz[k], zz[k + 1]
        if z2 != z1:
            cand_ts.append(t1 - z1 * (t2 - t1) / (z2 - z1))
    if cand_ts:
        cand_ts = np.asarray(cand_ts)
        return cand_ts[np.argmin(np.abs(cand_ts - t_click))]

    j = np.clip(i0, 1, len(t) - 2)
    idx = slice(max(0, j - 3), min(len(t), j + 4))
    m, b = np.polyfit(t[idx], z[idx], 1)
    return (-b / m) if m != 0 else t_click
