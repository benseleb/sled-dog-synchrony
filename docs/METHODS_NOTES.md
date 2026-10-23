# Method decisions

These are the non-obvious choices in the pipeline, the evidence behind each,
and approaches that were tried and dropped. Read this before changing a
method. The checks quoted here were run during development.

## Table I

`scripts/06_table1.py` computes Table I on the Dec 11 primary window. It
uses the hand-curated zero crossings for both phase and stride timing, plus
the same speed gate and force masking as every other run.

| | Range over the 10 dogs |
|---|---|
| R² | 0.24–0.43 |
| β_ω | −0.40 to −0.63 |
| β_F | 0.07–0.21 |
| A_φ | 0.05–0.60 |

`force.force_phase_regression` reproduces the original analysis code (the
manuscript's `SFCT_functions.py`) to ~1e-10, with identical stride counts,
on identical inputs. Those inputs
include the original code's own stride-boundary timeline, which was rounded
up to the next 125 Hz sample. A boundary is shared by ω_{k−1} and ω_k, so
that rounding biases β_ω negative, which is why the pipeline uses the
crossings directly. Matching the original code exactly required:

- **Signed argmax** for the force peak. `abs()` lets a noise spike or a deep
  negative dip be picked as "the" tension peak.
- **Speed gate ≥ 3 m/s** on force-peak selection.
- **Savitzky–Golay smoothing of force at its true sample rate.** The load
  ADC updates every other IMU sample, so force is smoothed at half rate and
  then upsampled (`force.smooth_force`).
- **sklearn `HuberRegressor`** (default ε = 1.35). `scipy.optimize.least_squares(loss="huber")`
  with a hand-picked `f_scale` shrank β_ω by 15–25%.
- **0.5 Hz high-pass gravity filter** (2nd-order Butterworth, zero-phase) on
  raw `az`. The formatted Dec 11 data already had it applied.

## Automatic segmentation, with a β_ω screen

Hand-checking every stride (~2,200 per dog) for 56 dog-runs is not
practical. The Dec 11 primary window was run three ways: hand-curated crossings, automatic crossings on the same
signal, and the full automatic path from raw files. They were compared against
a 300-replicate moving-block bootstrap (20-stride blocks) of the curated fit.

- 9 of 10 dogs need ≤ 15 edits out of ~2,100–2,400 strides. Shared
  crossings match to 0 ms, and every regression parameter falls inside the
  curated 95% CI.
- The exception is Shifter, whose β_ω is −1.00 automatic vs −0.57 curated.
  His waveform fails segmentation on every day he runs.

A missed or extra crossing creates a pair of stride-frequency outliers, and
those bias β_ω specifically. **The screen:** any dog-run with β_ω outside
[−0.77, −0.33] (the curated Dec 11 CI envelope, Shifter excluded) is flagged
for a manual pass with `tools/hand_correct_segmentation.ipynb`. Flagged
dog-runs are kept out of the reported ranges.

Rejected screens:

- **R².** Automatic misses cluster where the speed gate opens (launches,
  restarts), and a single outlier there swings R² (Tiki: 0.33 vs 0.26) while
  the Huber coefficients barely move.
- **Irregular-stride rate.** It doesn't separate good dogs from bad ones.
  Shifter is at 3.0%, but Tiki at 2.1% needed only ~5 edits.

Caveat: the screen was calibrated on one day. An error that doesn't move β_ω
(e.g. one shifted crossing) would slip through. PLV is robust to that,
because one bad crossing perturbs phase for about one stride.

## GPS speed lag

GPS speed lags the IMUs. The original code used a hand-tuned +1.5 s.

- **Dropped first attempt:** choose the shift that minimizes irregular
  strides inside the speed gate. This is confounded, because the irregular
  in-gate strides are real 3.5–4 Hz launch strides, which any later gate
  removes regardless of the true lag.
- **Current method:** cross-correlate GPS speed with team IMU activity (1 s
  RMS of high-passed |a|; integrating acceleration drifts within seconds).
  On Dec 11 this gives +1.2 s (flat over +1.0 to +1.6). `gps.GPS_LAG_S = 1.2`
  is the default, and Table I uses it.
- **One lag per run** (`synchrony/gps_lag.py`). No GPS log has a gap
  within an analysed span, so the latency can't change within a run.
  Per-window estimates scatter by about ±1 s, which is estimator noise.
  Values: 0.3 s (Dec 5) to 2.3 s (Dec 9). Runs without a clear peak (both
  Dec 10 runs) fall back to 1.2 s; those windows are 100% moving, so the lag
  has no effect there. Switching from a single 1.2 s to per-run lags moved
  every regression parameter by ≤ 0.03.

## Load-cell QA

Force problems are independent of segmentation, so hand-correcting
crossings never fixes them. `synchrony/force_qa.py` checks every dog-run,
and `force.force_valid_mask` masks bad samples before peaks are picked.

| Problem | Where found | Handling |
|---|---|---|
| `load = −999` ("no reading"; `error` stays 0) | Alice Dec 11 (~23 min), Sitka Dec 11 (~8.5 min); isolated samples elsewhere | Masked |
| `load = 404` (cell abandoned) | Isolated samples | Masked. Unmasked, `404 − offset` is a large positive spike for dogs with a negative offset (Tizzy: +50k counts), which signed-argmax would pick |
| No load reading at all | Blaise, Chuck, Dec 5 (firmware `dec04b` logged no load column) | Not fitted. Still in PLV |
| Baseline step between windows | Tiki Dec 11 (~155k counts) | Each window's median removed before pooling. Before the fix, Tiki's pooled β_F flipped to −0.10 while every window was positive |
| Baseline drift within a window | Trillium Dec 7 (~145k → −120k counts over ~5 min) | Masked where the 20 s rolling median departs > 1 stride-pulse size (normal ≤ 0.55). Trillium's β_F went 0.05 → 0.22 |
| Too little running | All of Dec 9 | Not fitted (< 120 s usable) |

Checks that came up clean everywhere are kept as standing checks: ADC
clipping, non-sentinel flat-lines, pulse polarity (skew +0.45 to +1.96), and
stride-locking (0.47–0.78 of force variance explained by the mean stride
waveform).

## Clock sync

See [SYNC_CHECK.md](SYNC_CHECK.md) for the method. Dropped versions:

- **Whole-run pooled envelope check.** It reported Julie +1.0 s, Captain
  +0.8 s and Healy −5.4 s, all artifacts of pooling: data outside the
  windows, and averaging across reboots. Since an offset can change at every
  reboot, the unit has to be the boot segment.
- **Dog-vs-dog pairwise force check.**
  On the field-confirmed window it was wrong in 16–25% of confident calls,
  because single-dog force has side-lobes near ±1 s that a team median
  averages out. It never excluded anything.

Direction of bias: an undetected 1 s mislabel lowers PLV, which biases toward
"no synchrony". That's why undecidable segments are excluded rather than
assumed fine.

## PLV

`plv.windowed_plv_pooled` is a port of the manuscript's own code
(`furriest_frequency_analysis.ipynb`, cell 72): 60 s windows with a 15 s step,
phases masked below 3 m/s, windows < 90% valid not tested, one pooled null
per pair from 800 circular shifts (lag in [W/2, L − W/2]), and τ95 its 95th
percentile. A window is significant if PLV > τ95 **and** PLV > 0.70.
The port reproduces the notebook exactly: 22 of
2,070 pair-windows, with τ95 differing by 4e-14. An earlier version written
from the main-text paragraph alone (200 surrogates per window, no speed
mask) is superseded.

The notebook extrapolated phases past each dog's data. Here, phases are NaN
outside a dog's data, which makes no difference on Dec 11, where every dog
spans the run.

## Interpretation notes

- **β_F > 0 in all 47 fits**, but small: the median of 0.14 rad/s per SD of
  force is about 0.8% of stride frequency. Raw corr(F_k, Δω_k) ≈ 0, but F_k
  correlates with the previous stride's frequency (r 0.2–0.5). So the
  positive β_F appears only once ω_{k−1} is controlled for. It reads as
  force indexing the dog's own effort, not entrainment. Untested: adding GPS
  team acceleration as a covariate.
- **Legacy bug, for anyone reading the old notebooks:**
  `furriest_frequency_analysis_extended.ipynb` allocates a 10-row phase
  array but fills only 4 rows. Its saved 10×10 heatmaps include six all-zero
  "dogs" and are not trustworthy. A clean rerun of that 4-dog, 27-min window
  agrees with the weak-coupling result.
