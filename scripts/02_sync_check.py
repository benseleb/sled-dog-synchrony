#!/usr/bin/env python3
"""Step 2: per-boot-segment clock-sync verdicts -> results/sync_check/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony.sync_check import main_segments

if __name__ == "__main__":
    main_segments()
