"""Filesystem locations. Raw data is not in the repo; point these at it with
environment variables, or symlink it into data/ (see README)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Per-test-day folders copied from the field SSD (csv/txt only), e.g.
# RAW_DIR / "DEC_11_TEST" / "Tango_Device11.csv"
RAW_DIR = Path(os.environ.get("SLEDDOG_RAW", REPO / "data" / "raw"))
# Dec 11 per-dog CSVs as formatted for the manuscript (data_{i}.csv, i indexes
# datasets.DEC11_DOGS), built from the raw files by scripts/00_format_dec11.py.
# Only Table I and the Dec 11 figures read these.
DEC11_FORMATTED_DIR = Path(os.environ.get("SLEDDOG_DEC11_FORMATTED", REPO / "data" / "dec11_formatted"))

CURATED_DIR = REPO / "data" / "curated"
DEC11_CURATED_ZC_DIR = CURATED_DIR / "dec11_primary_zero_crossings"
HAND_CORRECTED_DIR = CURATED_DIR / "hand_corrected"

RESULTS_DIR = Path(os.environ.get("SLEDDOG_RESULTS", REPO / "results"))
WINDOWS_DIR = RESULTS_DIR / "windows"
SYNC_DIR = RESULTS_DIR / "sync_check"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
TABLES_DIR = RESULTS_DIR / "tables"
GPS_LAGS_CSV = RESULTS_DIR / "gps_lags.csv"
FORCE_SCAN_CSV = RESULTS_DIR / "force_scan.csv"

FIGURES_DIR = REPO / "docs" / "figures"
