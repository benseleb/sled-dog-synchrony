#!/usr/bin/env python3
"""Step 0: Dec 11 primary-window data from the raw files -> data/dec11_formatted/
(data_{i}.csv, i indexing datasets.DEC11_DOGS). Only Table I (step 6) and the
Dec 11 figures read these; steps 1-5 work from the raw files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.formatting import build_all

if __name__ == "__main__":
    build_all()
