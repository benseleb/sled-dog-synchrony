"""Which recordings exist, which are used, and why. Single source of truth for
every script; docs/DATA_VETTING.md explains the decisions in prose.

Tiers (cross-dog clock sync):
  1  confirmed in the field (LED blink check in the run notes): Dec 10, Dec 11
  2  no field confirmation; verified from the data by sync_check
  3  November runs: no field notes found; verified from the data by sync_check
  0  sync check only, not analysed (Dec 06: unattributed desync, unusable GPS)
  None  windows built only: no usable window (Nov 20) or not yet checked (Nov 18)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    day: str          # folder under config.RAW_DIR
    gps_file: str
    tier: int | None
    note: str = ""


RUNS = [
    Run("DEC_11_TEST", "GPSLogger00.csv", 1, "Manuscript run; includes the primary window"),
    Run("DEC_10_TEST/Run1", "GPSLogger00.csv", 1, "Sitka, Winnie: device failures; Julie +1 s (field notes)"),
    Run("DEC_10_TEST/Run2", "GPSLogger01.csv", 1, "Gobi corrupted early; Captain +1 s from ~1/3 through"),
    Run("DEC_05_TEST", "GPSLogger01.csv", 2, "Blaise, Chuck: firmware logged no load column"),
    Run("DEC_07_TEST", "GPSLogger00.csv", 2, "All 10 dogs, one 35-min window, no resets"),
    Run("DEC_09_TEST", "GPSLogger00.csv", 2, "Above 3 m/s only 6% of the window"),
    Run("DEC_06_TEST", "GPSLogger00.csv", 0, "Unattributed desync in notes; GPS unusable"),
    Run("NOV_18_TEST", "GPSLogger00.csv", None, "One 46-min 3-dog window; not yet sync-checked or analysed"),
    Run("NOV_20_TEST", "GPSLogger00.csv", None, "Files corrupt ~7 min in, before the team runs"),
    Run("NOV_22_TEST", "GPSLogger00.csv", 3, "Fails sync check (2 dogs ~1 s apart)"),
    Run("NOV_23_TEST", "GPSLogger00.csv", 3, "8 dogs, passes sync check"),
    Run("NOV_26_TEST", "GPSLogger00.csv", 3, "Too fragmented by resets to check"),
    Run("NOV_28_TEST", "GPSLogger00.csv", 3, "Fails sync check (multi-second disagreements)"),
    Run("NOV_29_TEST", "GPSLogger00.csv", 3, "8 dogs, passes sync check"),
]
RUN = {r.day: r for r in RUNS}

# Runs whose windows the sync check and GPS-lag estimate visit (sync_check.DAYS)
SYNC_CHECK_RUNS = [r for r in RUNS if r.tier is not None]

# Dogs analysed per run: every dog that appears in at least one qualifying
# (>= MIN_WINDOW_S, >= 2 dogs) window. Dogs left out, and why:
#   Dec 10 Run1: Sitka, Winnie (device failures, almost no in-run data);
#                Alice, Healy (only in sub-120 s windows)
#   Dec 10 Run2: Gobi (corrupted early)
#   Dec 09:      Baby (no data after run start); Stevie, Trillium (sub-120 s only)
#   Harper, Dec 11: only part1 of her three files is loaded (see docs/DATA_VETTING.md)
ANALYSED_DOGS = {
    "DEC_11_TEST": [("Alice", "Alice_Device07.csv"), ("Glow", "Glow_Device06.csv"),
                    ("Harper", "Harper_Device10_part1.csv"), ("Healy", "Healy_Device01.csv"),
                    ("Shifter", "Shifter_Device04.csv"), ("Sitka", "Sitka_Device03.csv"),
                    ("Tango", "Tango_Device11.csv"), ("Tiki", "Tiki_Device05.csv"),
                    ("Tizzy", "Tizzy_Device02.csv"), ("Winnie", "Winnie_Device09.csv")],
    "DEC_10_TEST/Run1": [("Elanor", "Elanor_Device05.csv"), ("Julie", "Julie_Device03.csv"),
                         ("Sassy", "Sassy_Device04.csv"), ("Shifter", "Shifter_Device09.csv"),
                         ("Tiki", "Tiki_Device02.csv"), ("Tizzy", "Tizzy_Device06.csv")],
    "DEC_10_TEST/Run2": [("Captain", "Captain_Device04.csv"), ("Chips", "Chips_Device05.csv"),
                         ("Glow", "Glow_Device02.csv")],
    "DEC_05_TEST": [("Blaise", "Blaise_Device02.csv"), ("Ceaser", "Ceaser_Device03.csv"),
                    ("Chuck", "Chuck_Device09.csv"), ("Sky", "Sky_Device04.csv")],
    "DEC_07_TEST": [("Aly", "Aly_Device09.csv"), ("Arrow", "Arrow_Device07.csv"),
                    ("Baby", "Baby_Device06.csv"), ("Blaise", "Blaise_Device02.csv"),
                    ("Ceaser", "Ceaser_Device01.csv"), ("Siren", "Siren_Device10.csv"),
                    ("Sky", "Sky_Device04.csv"), ("Sophie", "Sophie_Device05.csv"),
                    ("Stevie", "Stevie_Device03.csv"), ("Trillium", "Trillium_Device08.csv")],
    "DEC_09_TEST": [("Arrow", "Arrow_Device07.csv"), ("Blaise", "Blaise_Device02.csv"),
                    ("Ceaser", "Ceaser_Device04.csv"), ("Siren", "Siren_Device03.csv"),
                    ("Sky", "Sky_Device06.csv"), ("Sophie", "Sophie_Device05.csv"),
                    ("Tomi", "Tomi_Device01.csv")],
    "NOV_23_TEST": [("Arrow", "Arrow_Device03.csv"), ("Ceaser", "Ceaser_Device04.csv"),
                    ("Charger", "Charger_Device01.csv"), ("Sassy", "Sassy_Device05.csv"),
                    ("Sky", "Sky_Device08.csv"), ("Sophie", "Sophie_Device06.csv"),
                    ("Tony", "Tony_Device07.csv"), ("Trillium", "Trillium_Device02.csv")],
    "NOV_29_TEST": [("Arrow", "Arrow_Device08.csv"), ("Baby", "Baby_Device09.csv"),
                    ("Blaise", "Blaise_Device06.csv"), ("Cassidy", "Cassidy_Device01.csv"),
                    ("Siren", "Siren_Device03.csv"), ("Sky", "Sky_Device04.csv"),
                    ("Sophie", "Sophie_Device05.csv"), ("Tony", "Tony_Device07.csv")],
}
TIER = {day: RUN[day].tier for day in ANALYSED_DOGS}

# (day, dog, dog_file, gps_file) for every analysed dog-run, in run order
QUEUE = [(day, dog, f, RUN[day].gps_file) for day, dogs in ANALYSED_DOGS.items() for dog, f in dogs]

# A qualifying window: >= 2 dogs recording at once, >= MIN_WINDOW_S long, and
# the team reaches 3 m/s somewhere in it. 120 s is the shortest span giving
# more than one 60 s / 15 s-step PLV estimate.
MIN_WINDOW_S = 120.0
WINDOW_PAD_S = 20.0   # context margin added around a window when loading a dog
SPEED_MIN = 3.0       # m/s; running-gait gate used throughout (manuscript Methods)

# ---- Dec 11 primary window (the manuscript's analysis window) ----
RUN_START_MS = 1670781044993 + 243500
WINDOW_END_MS = 1670782054863
# Index order of the formatted data_{i}.csv files
DEC11_DOGS = ['Tango', 'Sitka', 'Shifter', 'Healy', 'Glow',
              'Harper', 'Tizzy', 'Tiki', 'Alice', 'Winnie']
DEC11_RAW_FILES = dict(ANALYSED_DOGS["DEC_11_TEST"])
