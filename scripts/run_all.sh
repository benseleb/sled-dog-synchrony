#!/usr/bin/env bash
# Full pipeline from raw data. PLV (step 5) is the slow part.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY 00_format_dec11.py
$PY 01_build_windows.py
$PY 02_sync_check.py
$PY 03_gps_lag.py
$PY 04_force_qa.py
$PY 05_run_analysis.py
$PY 06_table1.py
