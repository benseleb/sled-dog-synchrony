"""Stride segmentation and phase reduction (manuscript Eq. 1).

butter_filter, marker_event, znormalize and time_interpolated_phase are the
manuscript's own primitives (furriest_functions.py), unchanged.
"""
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import numpy as np

def butter_filter(data, cutoff_freq, sampling_freq, typ, order):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff, btype=typ, analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def marker_event(t, orig_Z):
    int_fs = 125
    jerk = np.gradient(orig_Z)
    jerkf = butter_filter(jerk, 8, int_fs, 'low', 4)
    
    peaks, _ = find_peaks(jerkf, distance=int_fs/4)
    window_width = 5
    
    zero_crossings = []
    for peak in peaks:
        start = max(0, peak - window_width)
        end = min(len(orig_Z)-1, peak + window_width)
        segment = orig_Z[start:end]
        segment_t = t[start:end]

        # Find zero-crossings in the segment
        approx_crossings = np.where(np.diff(np.sign(segment)) > 0)[0]
        if approx_crossings.size:
            i = approx_crossings[0]
            f_interp = interp1d(segment[i:i+2], segment_t[i:i+2], assume_sorted=True)
            x0 = f_interp(0.0)
            zero_crossings.append(x0)

    return zero_crossings

def znormalize(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return (data - data_mean) / data_std, data_mean, data_std

# generate a phase representation by setting the start and end of a segment as a single oscillation
def time_interpolated_phase(zero_crossings, t):
    phase_T = np.arange(0, len(zero_crossings), 1)

    # Create a linear interpolation function for the phase based on the peak times
    phase_interp_func = interp1d(zero_crossings, phase_T, kind='linear', fill_value='extrapolate')

    # Compute the interpolated phase values at the new time points
    interpolated_phase = phase_interp_func(t)

    # only unwrap just before plotting
    phase_T = (interpolated_phase * 2* np.pi) % (2 * np.pi)
    return phase_T


def segment_and_phase(t, az, fs=125.0):
    """Automatic stride segmentation + phase reduction (Eq. 1). No manual
    correction -- caller should check `n_flagged` before trusting this.

    Raw az carries a large gravity-reaction DC/slow-drift
    component (~0.8-0.9 in the idle baseline) that plain mean-subtraction
    (znormalize) does not remove -- it only kills a *constant* offset, not
    drift as the dog's orientation shifts through a run. The trusted Dec-11
    formatted data (config.DEC11_FORMATTED_DIR) already has this removed
    via a high-pass Butterworth filter during data prep (citing Dewhirst et
    al. 2017); this replicates that exact step (0.5 Hz cutoff, order 2,
    zero-phase via filtfilt) for data read straight from the raw CSVs."""
    az_hp = butter_filter(az, 0.5, fs, 'high', 2)
    z, _, _ = znormalize(az_hp)
    zero_crossings = np.array(marker_event(t, z))
    if zero_crossings.size < 10:
        return None
    dt = np.diff(zero_crossings)
    n_flagged = int(np.sum((dt < 1/3.5) | (dt > 1/2.5)))
    phase = time_interpolated_phase(zero_crossings, t)
    return dict(zero_crossings=zero_crossings, phase=phase, n_flagged=n_flagged,
                pct_flagged=100 * n_flagged / len(dt))
