#!/usr/bin/env python3
"""Generate synthetic run-to-failure data in the NASA IMS on-disk format.

    python scripts/make_synthetic_data.py --out data/raw/synthetic

Why this exists: the real IMS dataset is a multi-gigabyte download, so without
it a reviewer cloning this repo cannot run anything. The generator writes files
in the EXACT format the real parser expects -- 20,480 tab-separated ASCII rows,
one column per channel, no header, filename is the timestamp -- so the whole
pipeline runs end to end with no downloads, and the code being exercised is the
real parser rather than a test double.

What it is NOT: a validated bearing model. It reproduces the qualitative shape
of a run-to-failure record and nothing more. Every number this repo reports from
synthetic data is labelled as such, and none of it substitutes for a result on
the real dataset. See scripts/download_ims.md.

Disk: roughly 0.8 MB per snapshot, so the default 320 snapshots is about 250 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bearingwatch.data.synthetic import SyntheticSpec, generate_run  # noqa: E402

BYTES_PER_SNAPSHOT_MB = 0.8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/raw/synthetic", help="output directory")
    parser.add_argument("--n-snapshots", type=int, default=320)
    parser.add_argument(
        "--n-channels",
        type=int,
        default=4,
        help="4 matches IMS sets 2 and 3; 8 matches set 1",
    )
    parser.add_argument(
        "--healthy-fraction",
        type=float,
        default=0.40,
        help=(
            "fraction of the run with defect severity EXACTLY zero. Deliberately "
            "larger than data.healthy_fraction so the declared reference window sits "
            "safely inside the truly healthy phase."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gain", type=float, default=1.0, help="global sensor gain")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output dir")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        print(
            f"{out} already exists and is not empty. Pass --force to overwrite, or pick "
            "another --out. Refusing to mix two generated runs in one directory: the "
            "timestamps would interleave and the result would be neither run.",
            file=sys.stderr,
        )
        return 1

    spec = SyntheticSpec(
        n_snapshots=args.n_snapshots,
        n_channels=args.n_channels,
        healthy_fraction=args.healthy_fraction,
        seed=args.seed,
    )
    spec.validate()

    healthy_end = int(round(spec.healthy_fraction * spec.n_snapshots))
    estimated_mb = spec.n_snapshots * BYTES_PER_SNAPSHOT_MB

    print(f"generating {spec.n_snapshots} snapshots into {out}")
    print(f"  format          {spec.n_points} points x {spec.n_channels} channels, tab-separated")
    seconds = spec.n_points / spec.sample_rate_hz
    print(f"  sample rate     {spec.sample_rate_hz} Hz ({seconds:.0f} s per snapshot)")
    print(f"  cadence         every {spec.interval_minutes} minutes")
    print(f"  healthy phase   snapshots 0..{healthy_end - 1} (severity exactly 0)")
    print(f"  degradation     snapshots {healthy_end}..{spec.n_snapshots - 1}")
    print(f"  faulted channel {spec.faulted_channel} (the others are negative controls)")
    print(f"  defect rate     {spec.defect_freq_hz:.1f} Hz (BPFO at {spec.shaft_rpm:.0f} RPM)")
    print(f"  estimated size  ~{estimated_mb:.0f} MB")
    print()

    if args.gain != 1.0:
        spec.noise_rms *= args.gain
        spec.shaft_amplitude *= args.gain
        print(f"  applying a global sensor gain of {args.gain}x")

    summary = generate_run(out, spec, progress=True)
    summary["truth_degradation_index"] = healthy_end

    print()
    print(json.dumps(summary, indent=2))
    print()
    print("next:")
    print(f"  bearingwatch preflight --data {out}")
    print(f"  bearingwatch train     --data {out}")
    print(f"  bearingwatch infer     --data {out} --truth-degradation-index {healthy_end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
