"""GPS logger reading: run start, smoothed speed on the IMU time axis, and
the GPS-speed latency estimate.

GPS logger CSV: 2 metadata lines, then epoch,ms,lat,lon,alt,SIV,speed,
heading,pDOP at 5 Hz; speed in mm/s.
"""
import csv

import numpy as np
from scipy.interpolate import splrep, BSpline

from .phase import butter_filter

# GPS speed lags the IMU: estimated on Dec 11 by matching GPS speed against
# team IMU activity (estimate_gps_lag) -- 9-dog peak +1.2 s, flat over
# +1.0..+1.6 s. The manuscript's SFCT get_speed hand-tuned +1.5 s, inside
# that band. Added to GPS timestamps before interpolating onto the IMU axis.
# Default only: the analysis uses one estimated lag per run (gps_lag.py).
GPS_LAG_S = 1.2


def gps_run_start(gps_path, thresh=300.0, sustain=5.0, header_skip=2):
    epochs, speeds = [], []
    with open(gps_path, newline="") as f:
        for _ in range(header_skip):
            next(f, None)
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ei, si = header.index("epoch"), header.index("speed")
        for row in reader:
            try:
                epochs.append(float(row[ei])); speeds.append(float(row[si]))
            except (ValueError, IndexError):
                break
    epochs, speeds = np.array(epochs), np.array(speeds)
    moving = speeds > thresh
    for i in range(len(epochs)):
        if moving[i]:
            j = i
            while j < len(epochs) and epochs[j] - epochs[i] < sustain:
                j += 1
            if moving[i:j].all() and j > i:
                return epochs[i]
    return None


def gps_speed_interp(gps_path, t_target, t0, header_skip=2, artifact_speed_max=10.0, smooth_s=900,
                     shift_s=GPS_LAG_S):
    """Smoothed GPS speed (m/s), spline-interpolated onto t_target (seconds
    relative to t0, an absolute epoch in seconds -- e.g. load_dog's `t0`).
    Matches the manuscript's own speed-gating approach (SFCT_functions.get_speed):
    drop obvious GPS speed artifacts (>10 m/s) before spline-smoothing, since
    raw GPS speed is noisy sample-to-sample. shift_s is added to GPS
    timestamps to correct GPS speed latency (see GPS_LAG_S)."""
    epochs, speeds = [], []
    with open(gps_path, newline="") as f:
        for _ in range(header_skip):
            next(f, None)
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ei, si = header.index("epoch"), header.index("speed")
        mi = header.index("ms") if "ms" in header else None
        for row in reader:
            try:
                e = float(row[ei]) + (float(row[mi]) / 1000.0 if mi is not None else 0.0)
                epochs.append(e); speeds.append(float(row[si]) / 1000.0)  # mm/s -> m/s
            except (ValueError, IndexError):
                break
    epochs, speeds = np.array(epochs), np.array(speeds)
    t_gps = epochs + shift_s - t0
    keep = speeds <= artifact_speed_max
    if keep.sum() < 4:
        return np.full_like(t_target, np.nan)
    tck = splrep(t_gps[keep], speeds[keep], s=smooth_s)
    return BSpline(*tck)(t_target)


def estimate_gps_lag(gps_path, dogs, fs=125.0, shifts=np.arange(-5.0, 8.01, 0.1),
                     grid_step=0.2, header_skip=2, artifact_speed_max=10.0):
    """Estimate GPS speed latency vs. the IMUs for one day: the shift (s, added
    to GPS timestamps, same convention as gps_speed_interp's shift_s) that
    maximizes the correlation between raw GPS speed and team IMU activity.

    Activity = 1 s RMS of the dynamic acceleration magnitude, averaged across
    dogs. Not integrated to velocity -- raw IMU integration drifts within
    seconds -- but it rises and falls with running intensity, which is all a
    timing match needs. Gravity is removed first with the same 0.5 Hz
    high-pass as segment_and_phase, or the ~1 g constant swamps the signal.

    dogs: list of (t_abs, ax, ay, az), t_abs in absolute epoch seconds
    (epoch + ms/1000). Pass corruption-trimmed data, ideally excluding known
    segmentation outliers. Returns (best_shift, r_at_best, shifts, r_curve);
    check r_curve for a clear single peak before trusting best_shift.
    Raw (unsmoothed) GPS is used here so transitions stay sharp."""
    epochs, speeds = [], []
    with open(gps_path, newline="") as f:
        for _ in range(header_skip):
            next(f, None)
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ei, si, mi = header.index("epoch"), header.index("speed"), header.index("ms")
        for row in reader:
            try:
                epochs.append(float(row[ei]) + float(row[mi]) / 1000.0)
                speeds.append(float(row[si]) / 1000.0)
            except (ValueError, IndexError):
                break
    t_gps, speed = np.array(epochs), np.array(speeds)
    keep = speed <= artifact_speed_max
    t_gps, speed = t_gps[keep], speed[keep]

    t_lo = max(max(d[0][0] for d in dogs), t_gps[0] + shifts.max())
    t_hi = min(min(d[0][-1] for d in dogs), t_gps[-1] + shifts.min())
    grid = np.arange(t_lo, t_hi, grid_step)
    win = int(fs)  # 1 s RMS window
    activity = []
    for t, ax, ay, az in dogs:
        mag2 = sum(butter_filter(a, 0.5, fs, 'high', 2) ** 2 for a in (ax, ay, az))
        rms = np.sqrt(np.convolve(mag2, np.ones(win) / win, mode='same'))
        activity.append(np.interp(grid, t, rms))
    activity = np.mean(activity, axis=0)

    r = np.array([np.corrcoef(np.interp(grid, t_gps + s, speed), activity)[0, 1] for s in shifts])
    j = int(np.argmax(r))
    return shifts[j], r[j], shifts, r


def read_gps(path, header_skip=2, artifact_speed_max=10.0):
    epochs, speeds = [], []
    with open(path, newline="") as f:
        for _ in range(header_skip):
            next(f, None)
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ei, si, mi = header.index("epoch"), header.index("speed"), header.index("ms")
        for row in reader:
            try:
                epochs.append(float(row[ei]) + float(row[mi]) / 1000.0)
                speeds.append(float(row[si]) / 1000.0)
            except (ValueError, IndexError):
                break
    t, v = np.array(epochs), np.array(speeds)
    keep = v <= artifact_speed_max
    return t[keep], v[keep]
