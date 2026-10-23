#!/usr/bin/env python3
"""Step 3: GPS speed latency per run -> results/gps_lags.csv (needs step 2)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.gps_lag import estimate_run_lags

if __name__ == "__main__":
    estimate_run_lags()
