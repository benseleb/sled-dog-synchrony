#!/usr/bin/env python3
"""Step 5: force-phase regression + pairwise PLV -> results/analysis/

Usage: 05_run_analysis.py [--force-only] [--days=DEC_10_TEST/Run2,...]
  --force-only  skip the slow PLV step (keeps the existing pairwise_plv.csv)
  --days        rerun only these runs and splice them into the existing CSVs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.analysis import run

if __name__ == "__main__":
    days = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--days=")), None)
    run(force_only="--force-only" in sys.argv, plv_days=days)
