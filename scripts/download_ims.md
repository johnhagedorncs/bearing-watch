# Obtaining the NASA IMS bearing dataset

This repo ships no data and downloads nothing. The synthetic generator exists so
the pipeline is runnable immediately; this file is how you get the real thing.

## What the dataset is

Generated at the Center for Intelligent Maintenance Systems, University of
Cincinnati, and distributed through the NASA Prognostics Data Repository.

Four double-row bearings on a single shaft. The shaft is held at a constant
2000 RPM and a 6000 lb radial load is applied to the shaft and bearings. The rig
was run continuously until a bearing failed.

Each file is one snapshot: **20,480 points sampled at 20 kHz**, which is one
second of signal, recorded **every 10 minutes**.

| Set | Files | Channels | Outcome |
|-----|-------|----------|---------|
| 1 | 2,156 | 8 (two accelerometers per bearing) | inner-race failure on bearing 3, roller-element failure on bearing 4 |
| 2 | 984 | 4 (one per bearing) | outer-race failure on bearing 1 |
| 3 | 6,324 | 4 (one per bearing) | outer-race failure on bearing 3 |

## On-disk format

* ASCII, **tab-separated**, one column per channel.
* **No header row**, no index column.
* Exactly **20,480 rows** per file.
* The **filename is the timestamp**, formatted `YYYY.MM.DD.HH.MM.SS`
  (e.g. `2004.02.12.10.32.39`), with no extension.

`src/bearingwatch/data/ims_parser.py` implements exactly this and asserts the
row count and channel count on every read. A short file aborts the run rather
than being padded or accepted, because a silently truncated record produces a
subtly wrong spectrum and no error anywhere downstream.

## Getting it

The dataset is distributed as part of the NASA Prognostics Data Repository,
hosted by the Prognostics Center of Excellence at NASA Ames Research Center.
Search the repository for the IMS bearing dataset and download the archive.
Mirrors exist (Kaggle among them) but verify the file counts in the table above
against whatever you download -- partial mirrors are common and the preflight
gate is there to catch them.

Expect several gigabytes.

## Laying it out

Unpack so that each set is its own directory of snapshot files:

```
data/raw/
  set1/     2156 files, 8 channels
  set2/      984 files, 4 channels
  set3/     6324 files, 4 channels
```

Some distributions nest the files one level deeper (e.g. `1st_test/`,
`2nd_test/`, `4th_test/` -- note that the third archive is commonly named
`4th_test`). Flatten or rename so each directory contains snapshot files
directly.

## Verifying before you trust anything

Run the preflight gate first. It checks that the data exists, at the density
claimed, in the format expected, and it fails loudly rather than warning:

```bash
bearingwatch preflight --data data/raw/set2 \
  --set data.set_name=set2 \
  --set data.expected_channels=4 \
  --set data.expected_files=984
```

Setting `expected_files` turns "I downloaded most of it" into an error. A
partial download trains fine, scores fine, and produces a plausible number that
means nothing.

For set 1, the channel count differs:

```bash
bearingwatch preflight --data data/raw/set1 \
  --set data.set_name=set1 \
  --set data.expected_channels=8 \
  --set data.expected_files=2156
```

## The declared split

Fixed before any results were looked at, and repeated here so it is not silently
revised later:

* **set 2 to develop** — build features, choose the model, set thresholds.
* **set 3 to validate** — check that the choices made on set 2 transfer.
* **set 1 as hold-out** — touched once, at the end, and reported whatever it says.

Set 1 is the hold-out for a specific reason beyond being unseen: it has 8
channels rather than 4 and it fails in two different modes (inner race and
roller element) rather than one. So it tests generalisation across channel
layout and across failure mode, not just across time.

## Which channel is which

The `data.target_channel` config option selects the bearing under study
(0-based). For set 2 the failure is on bearing 1, which is channel 0. For set 3
the failure is on bearing 3. For set 1, with two accelerometers per bearing, the
channel-to-bearing mapping is documented with the dataset -- **check it rather
than assuming**, because getting it wrong means training on a healthy bearing
and reporting that nothing was detected.

The healthy bearings in each set are the negative controls: same shaft, same
load, same speed, same duration, no fault. If a feature rises on every channel
at once, the rig or the instrumentation changed, not a bearing.

## Bearing geometry

`configs/features.yaml` carries the Rexnord ZA-2115 geometry used to compute the
defect frequencies (BPFO, BPFI, BSF, FTF). **Verify those numbers against the
dataset documentation before trusting any result that depends on them.**
Geometry tables get copied between papers without being rechecked, and a wrong
pitch diameter shifts every defect frequency without producing an error.

The sanity check, at 2000 RPM: BPFO ≈ 236 Hz, BPFI ≈ 297 Hz, BSF ≈ 140 Hz,
FTF ≈ 15 Hz. `tests/test_features.py` asserts these.

## Attribution

Cite the IMS dataset and the NASA Prognostics Data Repository as the
distributors require. The dataset is theirs; this repository only reads it.
