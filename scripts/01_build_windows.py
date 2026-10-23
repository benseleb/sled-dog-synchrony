#!/usr/bin/env python3
"""Step 1: qualifying multi-dog windows per run -> results/windows/<run>.csv

Usage: 01_build_windows.py [DAY ...]   (default: every run in datasets.RUNS)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.datasets import RUNS
from synchrony.windows import build_windows

if __name__ == "__main__":
    for day in sys.argv[1:] or [r.day for r in RUNS]:
        build_windows(day)
