# Data vetting: which recordings are used, and why

This covers every folder of the 2022 Michigan field season (`Team Run Data/`).
The machine-readable version is [`synchrony/datasets.py`](../synchrony/datasets.py).
Every script reads its run list, tiers, dog files and exclusions from that file.

## Recording format

| File | Layout |
|---|---|
| `<Dog>_Device<NN>.csv` (one per dog; `_part1/2/3` after a file restart) | 4 metadata lines, then `epoch,ms,ax,ay,az,gx,gy,gz,load,offset,error` at 125 Hz |
| `GPSLogger<NN>.csv` (one per run) | 2 metadata lines, then `epoch,ms,lat,lon,alt,SIV,speed,heading,pDOP` at 5 Hz; `speed` in mm/s |

`epoch` holds whole seconds only. Time is always `epoch + ms/1000`. Using
`epoch` alone collapses 125 samples onto one timestamp.

## What makes a recording usable

A recording has to pass three independent checks. Each is a separate step of
the pipeline.

1. **Concurrency.** A *qualifying window* needs ≥ 2 dogs recording at once
   for ≥ 120 s, and the team must reach 3 m/s somewhere in it. 120 s is the
   shortest span that gives more than one 60 s / 15 s-step PLV estimate.
   (`scripts/01_build_windows.py`, after corruption trimming in
   `synchrony/io.py`.)
2. **Clock sync**, needed for pairwise PLV only. Each dog's boot segment
   (its stretch between device reboots) is called *in sync*, *offset by k
   s* or *can't tell*. Only *in sync* segments enter PLV.
   (`scripts/02_sync_check.py`; method in [SYNC_CHECK.md](SYNC_CHECK.md).)
3. **Load cell**, needed for the force regression only. Sentinels, missing
   cells, baseline steps and drift are found and masked, and a dog needs
   ≥ 120 s of valid, moving force to be fitted. (`scripts/04_force_qa.py`.)

The force regression is per dog, so it needs no cross-dog sync. That's why
some runs contribute to one analysis but not the other.

### Corruption trimming

Each file is cut at its first fault: an error-code burst (`error > 2`), a
timestamp that jumps backwards (> 0.5 s) or forwards (> 60 s), NaNs, or
fill/overflow values (`epoch` ≤ 0 or at the uint32 max, `ms` at the uint16
max, `|load|` or `|offset|` > 1e7). Real hard-pull tension reaches ~2e5
counts, so magnitude alone is not a corruption signal. `load == 404` is the
firmware's intentional "load cell abandoned after 5 comm resets" marker, not
corruption. It is masked as a force sentinel instead (along with `-999`, "no
reading").

## Runs

Tier = how cross-dog clock sync is established: **1** confirmed in the field
(LED blink check in the run notes); **2** not confirmed in the field,
verified from the data; **3** November, no field notes found, verified from
the data.

### Analysed (8 runs, 30 dogs, 56 dog-runs, 154 min of qualifying windows)

| Run | Tier | Dogs | Windows | Minutes | In PLV | In force regression | Notes |
|---|---|---:|---:|---:|---|---|---|
| Dec 11 | 1 | 10 | 8 | 45 | 10 dogs | 10 | The manuscript's run. Contains the primary window (below) |
| Dec 10, run 1 | 1 | 6 | 1 | 2 | 1 pair (Sassy–Tiki) | 6 | Julie +1 s per field notes, confirmed by the sync check |
| Dec 10, run 2 | 1 | 3 | 2 | 5 | — | 3 | Captain +1 s from ~1/3 through (notes + sync check). No two verified dogs share a window |
| Dec 7 | 2 | 10 | 1 | 35 | 10 dogs | 10 | Cleanest run: one 35-min window, no device resets, near-perfect load cells |
| Dec 5 | 2 | 4 | 3 | 19 | — | 2 | Blaise and Chuck's firmware logged no load. Too few verified dogs for PLV |
| Dec 9 | 2 | 7 | 1 | 4 | — | — | Above 3 m/s only 6% of the window (~16 s of usable force per dog) |
| Nov 23 | 3 | 8 | 2 | 26 | 7 dogs (Trillium unverified) | 8 | No field notes found in the copied data |
| Nov 29 | 3 | 8 | 3 | 17 | 8 dogs | 8 | No field notes found in the copied data |

Dogs left out of analysed runs:

- **Dec 10 run 1:** Sitka and Winnie had device failures, with almost no
  in-run data. Alice and Healy appear only in sub-120 s windows. (Healy also
  sits at −5 s in the sync check. That doesn't matter, since he's in no
  analysed window.)
- **Dec 10 run 2:** Gobi's file is corrupted early.
- **Dec 9:** Baby has no data after the run starts. Stevie and Trillium
  appear only in sub-120 s windows.
- **Dec 11, Harper:** only `Harper_Device10_part1.csv` is loaded. Her
  part 2/3 files count toward window building but aren't analysed, so she
  is skipped from one window (logged at run time).

### Checked and excluded

| Run | Why |
|---|---|
| Dec 6 | The notes say "a couple sensors blinked out of sync" without naming them. The sync check finds Mikey ~1 s off in run 1 (Freedom, Harper and Sassy agree at 0 s), so about an hour with three verified dogs could be rescued. But the GPS log is unusable (impossible speeds, uncorrelated with the dogs), so there is no speed gate or GPS lag. Windows and sync check are built, not analysed (tier 0 in `datasets.py`) |
| Nov 22 | Fails the sync check: the only two dogs (Roz, Sally) disagree by ~1 s |
| Nov 26 | Too fragmented by device resets: one 128 s window, 3 verified dogs |
| Nov 28 | Fails the sync check: pairwise disagreements of +5 to +8 s |
| Nov 20 | Every dog's file is corrupted within ~7 min, before the team starts running |
| Nov 18 | **Not yet analysed.** One 46-min window with 3 dogs (Captain, Mikey, Mirage), 79% above 3 m/s. It was missed earlier because its files have one extra header line. It still needs the sync check (pre-Dec 11 firmware), and its GPS speed spline is unstable at one edge |

### Not usable at all

| Folder | Why |
|---|---|
| Dec 2, Dec 3, Dec 4 | Device failures (file corruption, GPS time jumps). The `DEC_04_TEST` folder is mislabeled: its notes are for 12/03 |
| Nov 16 | No sensors run |
| Nov 17 | Only one dog instrumented |
| Oct 2, 27, 28, 29; Nov 1, 12; `Calibration/`; `Flight_Records_11_28_22/` | No dog device files (drone flight records, calibration runs) |

## The Dec 11 primary window

The manuscript's analysis window: `RUN_START_MS = 1670781044993 + 243500` to
`WINDOW_END_MS = 1670782054863` (766 s), all 10 dogs. It has field-confirmed
sync and hand-curated stride boundaries
(`data/curated/dec11_primary_zero_crossings/<Dog>.npy`, seconds from
`RUN_START_MS`), and it serves as the ground truth for the pipeline's
calibrations.

Table I and the Dec 11 figures read per-dog formatted CSVs, which
`scripts/00_format_dec11.py` builds from the raw files (`synchrony/formatting.py`):
cut at the first load-cell offset change, 0.5 Hz high-pass over the whole
file, crop to the window. Harper's three part files are stitched (Dec 11
rows only, corrupted rows and offset-change rows dropped), as for the
manuscript. The rebuilt files match the manuscript's to floating-point
precision, and Table I computed from them is identical.

## Limitations of the recordings

The loggers were developed during the season, and that shapes what each run
can support.

**Firmware changed from day to day.** The file format, sample rates and
timing all vary with firmware version (from each file's header):

| Runs | Firmware | IMU / load cell | Format |
|---|---|---|---|
| Nov 18, Nov 20 | `nov18a`, `nov19a` | 125 / 62 Hz | One extra header line; `…,load` (no offset or error columns) |
| Nov 22 | `nov21b` | 125 / 62 Hz | |
| Nov 23, Nov 26, Dec 6, one Dec 5 device | `nov23a`, `dec06a` | **120** / 60 Hz | `…,load,offset,error` |
| Nov 28, Nov 29, Dec 2 | `nov28a`, `nov29a` | 125 / 62 Hz | `…,load,offset,error` |
| Dec 3, Dec 4, one Dec 5 device | `dec02a`, `dec03b`, `dec04a` | 125 / 25–125 Hz | `…,load,code` (no offset or error columns) |
| Two Dec 5 devices (Blaise, Chuck) | `dec04b` | 125 Hz, no load cell | `…,gz,code` (no load column) |
| Dec 7, Dec 9, Dec 10 | `dec06b` | 125 / 62 Hz | `…,load,offset,error` |
| Dec 10, Dec 11 | `dec06c` | 125 / 62 Hz | `…,load,offset,error` |

Consequences:

- The 120 Hz firmware doesn't sample on a fixed grid: intervals jitter
  between 6 and 10 ms. The pipeline uses the true timestamps for stride
  boundaries and phase, but its filters assume a uniform 125 Hz. Nov 23 is
  the analysed run affected. Its effect hasn't been quantified (resampling
  to a uniform grid before filtering would test it).
- On the `dec04a` device (Ceaser, Dec 5), the load cell updates at 25 Hz, not
  every other sample as `force.smooth_force` assumes.
- Missing columns are read as zeros, so a file with no `load` looks like a
  constant (dead) load cell and is excluded from the force analysis.
- The number of metadata lines before the column header varies (4, or 5 on
  `nov18a`/`nov19a`). The loader finds the header row itself. An earlier
  version assumed 4 and silently read Nov 17–20 as empty.

**Clock sync wasn't reliable until Dec 11.** The loggers lock sampling to the
GPS pulse, but until the firmware was fully corrected on Dec 11, a device
could label its data with the wrong GPS second after a reboot: an offset of
a whole number of seconds that persists until the next reboot. The Dec 10
field notes name two dogs this happened to. For every run before Dec 11,
each boot segment's sync therefore has to be verified from the data
([SYNC_CHECK.md](SYNC_CHECK.md)), and unverified segments are left out of
the pairwise analysis.

**Recordings are fragmented.** Each logger reboots itself after a
communication or force-data error, leaving a gap of a few seconds. Load
cells can also drop out mid-run (firmware sentinel values), sometimes for
the rest of the run (Alice and Sitka on Dec 11). Some dogs reboot every few
minutes, and others never do. Pairwise analysis needs two dogs recording at
once with verified clocks, so it is restricted to qualifying windows, which
are much shorter than the runs themselves.

![Recording timelines](figures/00_recording_timelines.png)

## Open data questions

These are things only the field record can settle:

- Nov 23 and Nov 29 field notes (conditions, rig, team positions). They
  weren't in the csv/txt copy. Check the original SSD.
- Vehicle, load and surface for each analysed run.
- Why Stevie, Siren and Blaise on Dec 7 have weak activity signals (loose
  mounts?).
