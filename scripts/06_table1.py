#!/usr/bin/env python3
"""Step 6: Table I -> results/tables/table1.csv

Force-phase regression on the Dec-11 primary window, with the hand-curated
zero crossings for both phase anchoring and stride timing.
Computed exactly as every other run is (speed gate, force masking), so the
primary window gets no special-casing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import pandas as pd

from synchrony import config, datasets as ds
from synchrony.force import force_valid_mask, force_phase_regression
from synchrony.gps import gps_speed_interp

DOGS = ds.DEC11_DOGS
RUN_START = ds.RUN_START_MS
GPS = config.RAW_DIR / "DEC_11_TEST" / "GPSLogger00.csv"

rows = []
print(f"{'dog':8s} {'n':>5s} {'r2':>7s} {'beta0':>8s} {'beta_cos':>9s} {'beta_sin':>9s} "
      f"{'beta_F':>7s} {'beta_omega':>10s} {'A_phi':>7s} {'theta0':>7s}")

for focal, dog in enumerate(DOGS):
    focal_zc = np.load(config.DEC11_CURATED_ZC_DIR / f'{dog}.npy')
    df = pd.read_csv(config.DEC11_FORMATTED_DIR / f'data_{focal}.csv')
    t_raw = (df['epoch'].to_numpy() - RUN_START) / 1000.0
    # pipeline speed (gps.GPS_LAG_S shift), not SFCT get_speed's hand-tuned
    # +1.5 s, so this matches what every other dataset gets
    speed_on_focal = gps_speed_interp(GPS, t_raw, RUN_START / 1000)

    # Same load-cell masking as the production path (synchrony/analysis.py):
    # only isolated 404 sentinels fall in this window, but unmasked they can
    # become the stride's "peak" for dogs with a negative offset
    load, offset = df['load'].to_numpy(), df['offset'].to_numpy()
    force_valid, _ = force_valid_mask(t_raw, load, offset, moving=speed_on_focal >= 3.0)
    res = force_phase_regression(t_raw, focal_zc, load, offset, force_valid=force_valid,
                                     speed_mps=speed_on_focal, speed_min=3.0, exclude_low_speed_strides=True)
    res.update(day="DEC_11_TEST", dog=dog)
    rows.append(res)
    print(f"{dog:8s} {res['n_strides']:5d} {res['r2_model']:7.3f} {res['beta0']:8.3f} "
          f"{res['beta_cos']:9.3f} {res['beta_sin']:9.3f} {res['beta_F']:7.3f} "
          f"{res['beta_omega']:10.3f} {res['A_phi']:7.3f} {res['theta0']:7.3f}")

config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.TABLES_DIR / "table1.csv"
fieldnames = ["day", "dog", "n_strides", "beta0", "beta_cos", "beta_sin",
              "beta_F", "beta_omega", "r2_model", "A_phi", "theta0"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in fieldnames})
print(f"\nSaved -> {out_path}")
