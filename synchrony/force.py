"""Load-cell QA and the force-phase Huber regression (manuscript Eq. 4,
Table I)."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.linear_model import HuberRegressor

from .phase import znormalize, time_interpolated_phase

# Load-cell firmware sentinels, not real readings: 404 = load cell abandoned
# after 5 comm-error resets; -999 = no reading (found 2026-10-01: Alice ~23 min
# and Sitka ~8.5 min of Dec 11, mid-run, error column still 0).
LOAD_SENTINELS = (-999, 404)
# Max allowed slow-baseline excursion (20 s rolling median vs. the window's own
# median), in units of the dog's stride-pulse size (p1-p99 of 1 s-detrended
# force). Normal dogs sit <=0.55 on every 2022 day checked; Trillium (Dec 07,
# load cell collapsing from ~145k to -120k counts) hit 3.6.
FORCE_DRIFT_MAX = 1.0


def smooth_force(t, load, offset, smooth_window=11, smooth_poly=3):
    """Percentile-normalize + Savitzky-Golay smooth the force signal at its
    true sample rate, then linearly upsample back onto t. The load ADC on
    this hardware only actually updates every other IMU sample (confirmed
    duplicate consecutive raw counts on every 2022 field day checked, not a
    Dec-11-only artifact) -- smoothing at full rate would just be smoothing
    already-duplicated step data instead of the real underlying signal.
    Matches SFCT_functions.get_smooth_force/normalize_forces exactly (offset
    is a per-file constant here, so subtracting it first vs. not is
    equivalent after percentile-normalizing)."""
    force_raw = (load - offset).astype(float)
    t_half, force_half = t[::2], force_raw[::2]
    offset_load = force_half - np.percentile(force_half, 1)
    norm_load = offset_load / np.percentile(offset_load, 99)
    smooth_half = savgol_filter(norm_load, smooth_window, smooth_poly)
    return interp1d(t_half, smooth_half, kind="linear", fill_value="extrapolate")(t)


def force_valid_mask(t, load, offset, spans=None, moving=None, drift_max=FORCE_DRIFT_MAX, fs=125.0):
    """Per-sample validity of the load-cell signal, plus per-reason diagnostics.

    Invalid: firmware sentinels (LOAD_SENTINELS), a dead cell (constant value
    throughout), and -- within each span (t-relative (start, end) pairs; whole
    series if None) -- stretches where the slow baseline drifts more than
    drift_max stride-pulse sizes from that span's median (a failing or
    shifting load cell, not pulling). Baseline/pulse statistics use only
    `moving` samples when given, since stops legitimately change tension.
    Pass the result to force_phase_regression(force_valid=...)."""
    load = np.asarray(load)
    sentinel = np.isin(load, LOAD_SENTINELS)
    valid = ~sentinel
    diag = dict(sentinel_s=sentinel.sum() / fs, dead=False, drift_s=0.0, max_drift=0.0)
    if not valid.any() or np.ptp(load[valid]) == 0:
        diag["dead"] = True
        return np.zeros(load.size, dtype=bool), diag
    x = pd.Series((load - offset).astype(float))
    x[~valid] = np.nan
    use = valid if moving is None else valid & moving
    pulse = x - x.rolling(int(fs), center=True, min_periods=int(fs) // 2).median()
    if np.isfinite(pulse[use]).sum() < fs * 10:
        return valid, diag
    pulse_size = np.subtract(*np.nanpercentile(pulse[use], [99, 1]))
    if spans is None:
        spans = [(t[0], t[-1])]
    for s0, s1 in spans:
        w = (t >= s0) & (t <= s1)
        xw = x.where(w & use)
        if xw.notna().sum() < fs * 10:
            continue
        rm = xw.rolling(int(fs * 20), center=True, min_periods=int(fs * 5)).median()
        dev = ((rm - np.nanmedian(xw)).abs() / pulse_size).to_numpy()
        diag["max_drift"] = max(diag["max_drift"], np.nanmax(dev))
        bad = w & (np.nan_to_num(dev) > drift_max)
        diag["drift_s"] += bad.sum() / fs
        valid &= ~bad
    return valid, diag


def force_phase_regression(t, zero_crossings, load, offset, speed_mps=None, speed_min=3.0,
                            smooth=True, phase_zc=None, exclude_flagged=False,
                            flag_lo_hz=2.5, flag_hi_hz=3.5, exclude_low_speed_strides=False,
                            return_design=False, force_valid=None):
    """Per-dog robust regression: Delta_omega_k ~ cos(phi_Fk) + sin(phi_Fk) + F_k + omega_{k-1}
    Matches the manuscript's Eq. 4 / Table I exactly. Force proxy defaults to
    the smoothed/normalized signal (see smooth_force); pass smooth=False for
    raw load-offset (z-scoring downstream makes scale irrelevant either way,
    but smoothing materially changes which sample gets picked as "the" peak
    -- confirmed against Table I).

    speed_mps: optional GPS speed (m/s), same length/alignment as t (see
    gps_speed_interp). Below speed_min the dog isn't in a real gait, so those
    samples are excluded from force-peak selection -- matches the manuscript's
    own speed>=3 m/s gate (SFCT_functions.SFCT_values); without it, low-speed/
    standing periods can contaminate the force-timing (A_phi) estimate.

    zero_crossings defines the stride boundaries (T_k, omega_k, Delta_omega).
    phase_zc, if given, is a SEPARATE array used only to anchor phi_Fk (via
    time_interpolated_phase, exactly as the manuscript computes phase) --
    matches the original pipeline, which anchors phase to accelerometer
    zero-crossings but computes stride timing from a distinct curated
    stride-boundary timeline (stride_labels/*_stride_times.npy on Dec 11).
    These two arrays can differ by only a few ms and still measurably shift
    A_phi (a small timing offset near a ~0.33s stride is a proportionally
    large phase error) -- confirmed against Table I. Defaults to
    zero_crossings itself when not given (the two roles collapse into one
    array, e.g. on days without a separate curated stride-label timeline).

    exclude_flagged: if True, also drop any Delta_omega row where the current
    or previous stride period falls outside [1/flag_hi_hz, 1/flag_lo_hz] --
    the same implausible-frequency band used elsewhere (segment_and_phase's
    n_flagged, the ZeroCrossingEditor GUI's own flag) as a rough proxy for
    "this interval likely needs manual correction." Use it to sensitivity-
    check whether uncorrected automatic-segmentation errors are actually
    driving a result, without doing the correction itself.

    exclude_low_speed_strides: speed_mps above only gates the force
    covariate -- a stride spanning a stationary-to-moving transition would
    otherwise still contribute its real (non-gait) duration to
    omega_k/Delta_omega uncorrected, since T_k is driven by zero-crossing
    timing alone. Set this to additionally require both endpoints of a
    stride period to be above speed_min before it enters the regression at
    all. Defaults to False and is a SEPARATE opt-in from speed_mps itself,
    because the original manuscript pipeline has this identical gap (it
    speed-gates force and phase but never gates T_prev/T_curr) -- turning
    this on by default would silently break the exact Table I match.

    force_valid: optional per-sample bool mask (see force_valid_mask) --
    invalid load-cell samples (sentinels, drift) are excluded from force-peak
    selection, so strides there drop out instead of contributing a
    meaningless peak picked from a constant or drifting signal."""
    zc = zero_crossings
    if len(zc) < 20:
        return None
    if phase_zc is None:
        phase_zc = zc
    if np.ptp(load) == 0:
        return None  # load cell never produced real readings for this dog/day (e.g. constant 0)
    force_raw = smooth_force(t, load, offset) if smooth else (load - offset).astype(float)
    if force_valid is not None:
        # widen invalid stretches by the smoothing support (11-pt Savitzky-Golay
        # at half rate = 22 samples) so smoothed edges near a sentinel/drift
        # can't be picked as a peak
        bad = np.convolve(~np.asarray(force_valid), np.ones(23), mode="same") > 0
        force_raw = force_raw.copy()
        force_raw[bad] = np.nan
    stride_speed_ok = None
    if speed_mps is not None:
        force_raw = force_raw.copy()
        force_raw[~(speed_mps >= speed_min)] = np.nan
        if exclude_low_speed_strides:
            speed_at_zc = np.interp(zc, t, speed_mps)
            stride_speed_ok = speed_at_zc >= speed_min

    T_k = np.diff(zc)  # stride periods
    omega_k = 2 * np.pi / T_k

    phi_Fk, F_k = [], []
    for k in range(len(zc) - 1):
        w0, w1 = zc[k] - 0.15, zc[k] + 0.15
        mask = (t >= w0) & (t < w1)  # matches SFCT_values' half-open window exactly
        if mask.sum() < 2:
            phi_Fk.append(np.nan); F_k.append(np.nan)
            continue
        seg_t, seg_f = t[mask], force_raw[mask]
        finite = np.isfinite(seg_f)
        if not finite.any():
            phi_Fk.append(np.nan); F_k.append(np.nan)
            continue
        seg_t, seg_f = seg_t[finite], seg_f[finite]
        # signed argmax, not abs(): abs() lets a raw-ADC noise spike or a big
        # negative dip masquerade as "the" tension peak just as readily as a
        # real one, which produced wildly unstable phase-effect (A_phi)
        # estimates on noisier dogs -- confirmed against the manuscript's own
        # Table I (Huber regression on the original curated Dec-11 window).
        i_peak = np.argmax(seg_f)
        t_peak = seg_t[i_peak]
        phi_Fk.append(time_interpolated_phase(phase_zc, np.array([t_peak]))[0])
        F_k.append(seg_f[i_peak])
    phi_Fk, F_k = np.array(phi_Fk), np.array(F_k)

    # Delta_omega_k = omega_k - omega_{k-1}; align F_k/phi_Fk (stride k) and omega_{k-1}
    valid = np.isfinite(phi_Fk[1:]) & np.isfinite(F_k[1:])
    if stride_speed_ok is not None:
        stride_ok = stride_speed_ok[:-1] & stride_speed_ok[1:]  # both endpoints of T_k above speed_min
        valid &= stride_ok[1:] & stride_ok[:-1]
    if exclude_flagged:
        lo_T, hi_T = 1 / flag_hi_hz, 1 / flag_lo_hz
        plausible = (T_k >= lo_T) & (T_k <= hi_T)
        valid &= plausible[1:] & plausible[:-1]
    dOmega = omega_k[1:][valid] - omega_k[:-1][valid]
    phiF = phi_Fk[1:][valid]
    F = F_k[1:][valid]
    omega_prev = omega_k[:-1][valid]
    if dOmega.size < 20:
        return None

    F_z, _, _ = znormalize(F)
    omega_prev_z, _, _ = znormalize(omega_prev)

    X = np.column_stack([np.cos(phiF), np.sin(phiF), F_z, omega_prev_z])
    y = dOmega

    # sklearn's HuberRegressor, not scipy least_squares(loss='huber'): matches
    # the manuscript's own SFCT_functions.HuberAnalysis exactly (default
    # epsilon=1.35), rather than an independently-tuned f_scale that turned
    # out to shrink beta_omega ~15-25% vs Table I even with everything else
    # (speed masking, peak-picking) already aligned.
    reg = HuberRegressor().fit(X, y)
    beta_cos, beta_sin, beta_F, beta_w = reg.coef_
    beta0_ = reg.intercept_
    r2_model = reg.score(X, y)

    A_phi = np.hypot(beta_cos, beta_sin)
    theta0 = np.arctan2(beta_sin, beta_cos)

    res = dict(n_strides=int(dOmega.size), beta0=beta0_, beta_cos=beta_cos,
               beta_sin=beta_sin, beta_F=beta_F, beta_omega=beta_w,
               r2_model=r2_model, A_phi=A_phi, theta0=theta0)
    if return_design:
        # lets callers bootstrap the fit (e.g. for a noise band) without
        # re-deriving the per-stride rows
        res.update(X=X, y=y)
    return res
