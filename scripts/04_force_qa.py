#!/usr/bin/env python3
"""Step 4: load-cell QA for every analysed dog-run -> results/force_scan.csv

Run on any new data before trusting its force regression (needs steps 1, 3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.force_qa import scan_all

if __name__ == "__main__":
    scan_all()
