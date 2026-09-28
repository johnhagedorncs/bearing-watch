#!/usr/bin/env python3
"""Why a distribution-drift monitor cannot tell you what changed.

Run:
    python scripts/drift_demo.py --run runs/<run_id> --data data/raw/synthetic

Four steps and a fix:

  1. Establish the reference distribution from the declared healthy window.
  2. Replay the run chronologically through the scoring path, as if live, and
     watch the drift monitor fire before failure. Report how long before.
  3. Inject a FAKE drift -- rescale the raw signal by a constant, which is what
     an accelerometer swap or a changed amplifier setting does -- and watch the
     monitor fire IDENTICALLY. It cannot tell a degrading bearing from a
     changed sensor, because at the level of "has this distribution moved" the
     two are the same event.
  4. Implement something that can. A broadband gain g scales every band energy
     by g^2, which in log space is the same constant added to every band. So
     the LEVEL of the log band-energy vector moves and its SHAPE does not.
     Mechanical degradation moves the shape too, because a developing defect
     puts energy in particular places rather than everywhere at once.

Step 3 is the point. A drift alert says "something changed" and stops there,
and "something changed" sends an engineer to inspect a machine when the actual
event was a maintenance action on an instrument. Step 4 turns that into "the
spectrum changed shape, which a gain change cannot do", which is a claim worth
paging someone about.

The honest limits of step 4 are printed by the script and set out in full in
src/bearingwatch/monitor/band_split.py. The short version: it only handles
BROADBAND gain changes. A loosening mount or a damaged cable changes the
response differently at different frequencies, which is a shape change, and
this method calls that mechanical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bearingwatch.config import Config  # noqa: E402
from bearingwatch.data.cache import SCALED_METADATA_KEY, FeatureCache  # noqa: E402
from bearingwatch.data.ims_parser import Snapshot, list_snapshot_files, read_snapshot  # noqa: E402
from bearingwatch.detect.onset import detect_onset  # noqa: E402
from bearingwatch.features.pipeline import (  # noqa: E402
    build_feature_table,
    extract_snapshot_features,
    feature_columns,
)
from bearingwatch.monitor.band_split import (  # noqa: E402
    VERDICT_GAIN,
    VERDICT_MECHANICAL,
    BandSplitDiscriminant,
    band_columns,
)
from bearingwatch.monitor.drift import compute_drift  # noqa: E402
from bearingwatch.serve.deps import load_run  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def heading(number: int | None, text: str) -> None:
    print(f"\n{RULE}")
    print(f"STEP {number}. {text}" if number else text)
    print(RULE)


def unscaled(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag a frame as unscaled so the double-scaling guard lets it through."""
    frame = frame.copy()
    frame.attrs["bearingwatch"] = {SCALED_METADATA_KEY: False, "source": "drift-demo"}
    return frame


def featurise_with_gain(
    paths: list[Path], cfg: Config, gain: float, progress_every: int = 100
) -> pd.DataFrame:
    """Re-featurise raw snapshots with the signal multiplied by ``gain``.

    Applied to the RAW WAVEFORM, not to the features, so the gain propagates
    through exactly the path a real sensor change would: the same code, the same
    Welch estimate, the same band integration. Faking it at the feature level
    would be assuming the very thing the demo is trying to show.
    """
    rows: list[dict[str, float]] = []
    index: list[Any] = []
    for i, path in enumerate(paths):
        snap = read_snapshot(
            path,
            expected_points=cfg.data.expected_points,
            expected_channels=cfg.data.expected_channels,
            timestamp_format=cfg.data.filename_timestamp_format,
        )
        scaled = Snapshot(timestamp=snap.timestamp, data=snap.data * gain, path=snap.path)
        feats = extract_snapshot_features(scaled, cfg)
        feats["snapshot_index"] = float(i)
        rows.append(feats)
        index.append(snap.timestamp)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    ...{i + 1}/{len(paths)}", flush=True)

    frame = pd.DataFrame(rows, index=pd.Index(index, name="timestamp"))
    frame["source_file"] = [p.name for p in paths]
    frame["snapshot_index"] = frame["snapshot_index"].astype(int)
    return unscaled(frame.sort_index())


def first_firing_window(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    columns: list[str],
    cfg: Config,
    start: int,
) -> tuple[int | None, dict[str, Any] | None]:
    """Slide a window forward and report where the drift monitor first fires."""
    size = cfg.monitor.window_size
    for end in range(start + size, len(frame) + 1):
        window = frame.iloc[end - size : end]
        report = compute_drift(reference, window, cfg.monitor, columns=columns)
        if report.fired:
            return end - 1, report.to_dict()
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", help="run folder (default: latest complete under --runs-root)")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--data", help="data root (default: the run's own data root)")
    parser.add_argument("--cache-dir", default=".cache/features")
    parser.add_argument(
        "--gain",
        type=float,
        default=1.8,
        help="multiplicative sensor gain to inject in step 3",
    )
    parser.add_argument(
        "--truth-degradation-index",
        type=int,
        help="index where degradation actually starts (synthetic data only)",
    )
    parser.add_argument("--out", help="write the results as JSON")
    args = parser.parse_args(argv)

    loaded = load_run(args.run, runs_root=args.runs_root)
    cfg = loaded.config
    data_root = args.data or cfg.data.root
    cfg = cfg.copy_with({"data.root": str(data_root)})

    print(RULE)
    print("DRIFT DEMONSTRATION")
    print(RULE)
    print(f"run            {loaded.run_id}  ({loaded.model.kind})")
    print(f"data           {data_root}")
    print(f"injected gain  {args.gain}x  (applied to the raw waveform)")

    paths = list_snapshot_files(cfg.data.root)
    cache = FeatureCache(args.cache_dir)
    frame, _ = build_feature_table(cfg, cache=cache, paths=paths)
    columns = feature_columns(frame)
    n = len(frame)
    n_healthy = int(round(cfg.data.healthy_fraction * n))
    interval_h = cfg.data.snapshot_interval_minutes / 60.0
    results: dict[str, Any] = {"run_id": loaded.run_id, "n_snapshots": n, "gain": args.gain}

    # ---------------------------------------------------------------- step 1
    heading(1, "Establish the reference distribution from the healthy window")

    reference = unscaled(frame.iloc[:n_healthy][columns].copy())
    monitored = band_columns(columns) or columns
    print(f"  healthy window     snapshots 0..{n_healthy - 1} of {n}")
    print("  declared in        configs/data.yaml (healthy_fraction)")
    print(f"  monitored features {len(monitored)} spectral band energies")
    print(
        "\n  The reference is a period DECLARED stationary before any results were "
        "looked at,\n  not a rolling window over the series being judged. "
        "src/bearingwatch/detect/\n  reference_threshold.py explains why the rolling "
        "version cannot detect a ramp."
    )
    results["healthy_window"] = {"n": n_healthy, "n_monitored_features": len(monitored)}

    # ---------------------------------------------------------------- step 2
    heading(2, "Replay the run chronologically and watch the monitor fire")

    scores = loaded.score_frame(frame)
    detection = detect_onset(
        scores,
        loaded.threshold,
        window=cfg.detect.persistence.window,
        consecutive=cfg.detect.persistence.consecutive,
        timestamps=list(frame.index),
        healthy_end_index=n_healthy,
        snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
    )

    drift_index, drift_report = first_firing_window(
        reference, unscaled(frame), monitored, cfg, n_healthy
    )

    print(f"  anomaly detector fired at snapshot   {detection.onset_index}")
    print(f"  drift monitor fired at snapshot      {drift_index}")
    if drift_index is not None:
        lead_h = (n - 1 - drift_index) * interval_h
        print(f"  lead time to end of run              {lead_h:.1f} h")
        print(f"  drifted feature fraction             {drift_report['drifted_fraction']:.0%}")
        print(f"  mean PSI                             {drift_report['mean_psi']:.3f}")
        results["degradation"] = {
            "drift_fired_at": drift_index,
            "lead_hours": round(lead_h, 2),
            "drifted_fraction": drift_report["drifted_fraction"],
            "mean_psi": drift_report["mean_psi"],
            "detector_onset": detection.onset_index,
        }
    else:
        print("  drift monitor never fired")
        results["degradation"] = {"drift_fired_at": None}

    early = False
    if args.truth_degradation_index is not None and drift_index is not None:
        offset = drift_index - args.truth_degradation_index
        early = offset < 0
        print(
            f"  degradation truly began at           {args.truth_degradation_index} "
            f"({offset:+d} snapshots)"
        )
        results["degradation"]["onset_minus_truth"] = int(offset)

    print(
        "\n  So far this looks like a working early-warning system. It fired well before\n"
        "  the end of the run, on features that genuinely moved."
    )
    if early:
        # Reported rather than tuned away. The threshold could be raised until
        # this went away, but that would be fitting the monitor to a known
        # answer, and on real data there is no known answer to fit to.
        print(
            f"\n  EXCEPT: it fired {abs(offset)} snapshots BEFORE degradation actually began.\n"
            "  Those snapshots are statistically identical to the reference period -- the\n"
            "  synthetic generator sets defect severity to exactly zero there. This is a\n"
            "  false positive, produced by comparing a "
            f"{cfg.monitor.window_size}-snapshot window against a\n"
            f"  {n_healthy}-snapshot reference and calling "
            f"{drift_report['drifted_fraction']:.0%} drifted features a regime change.\n"
            "  It is left in rather than tuned away: the threshold could be raised until it\n"
            "  disappeared, but that is fitting the monitor to an answer that only exists\n"
            "  because this data is synthetic. On the real dataset there is nothing to fit to."
        )
        results["degradation"]["fired_before_degradation_began"] = True

    # ---------------------------------------------------------------- step 3
    heading(3, f"Inject a fake drift: rescale the raw signal by {args.gain}x")

    print("  Re-featurising the HEALTHY window with a gain applied to the waveform.")
    print("  Nothing is wrong with this bearing. Only the sensor gain changed.\n")

    healthy_paths = paths[:n_healthy]
    gained = featurise_with_gain(healthy_paths, cfg, args.gain)

    gain_report = compute_drift(reference, gained[monitored], cfg.monitor, columns=monitored)
    degraded_window = unscaled(frame.iloc[-cfg.monitor.window_size :][columns].copy())
    degraded_report = compute_drift(reference, degraded_window, cfg.monitor, columns=monitored)

    print(f"  {'':22}{'DEGRADING BEARING':>20}{'GAIN CHANGE ONLY':>20}")
    print(f"  {THIN[:62]}")
    print(f"  {'monitor fires':22}{degraded_report.fired!s:>20}{gain_report.fired!s:>20}")
    print(
        f"  {'drifted feature frac':22}{degraded_report.drifted_fraction:>19.0%}"
        f"{gain_report.drifted_fraction:>19.0%}"
    )
    print(f"  {'mean PSI':22}{degraded_report.mean_psi:>20.3f}{gain_report.mean_psi:>20.3f}")
    print(f"  {'max PSI':22}{degraded_report.max_psi:>20.3f}{gain_report.max_psi:>20.3f}")

    results["gain_change"] = {
        "fired": gain_report.fired,
        "drifted_fraction": gain_report.drifted_fraction,
        "mean_psi": gain_report.mean_psi,
    }
    results["degraded_window"] = {
        "fired": degraded_report.fired,
        "drifted_fraction": degraded_report.drifted_fraction,
        "mean_psi": degraded_report.mean_psi,
    }

    identical = gain_report.fired and degraded_report.fired
    print()
    if identical:
        print("  THE PUNCHLINE")
        print("  Both fire. Both flag most of the monitored features. The monitor reports")
        print("  a failing bearing and a recalibrated accelerometer as the same event,")
        print("  because at the level of 'has this distribution moved' they ARE the same")
        print("  event. A drift alert cannot tell you WHICH of them happened, and sending")
        print("  someone to inspect a machine when a technician changed a sensor is how")
        print("  monitoring loses its audience.")
    else:
        print("  NOTE: the two did not both fire at this gain. Try a larger --gain;")
        print("  the point holds whenever the gain is big enough to move the distribution.")

    # ---------------------------------------------------------------- step 4
    heading(4, "Something that CAN tell them apart: the band-split discriminant")

    print("  A broadband gain g multiplies power by g^2, so in log space it adds the SAME")
    print("  constant to every band. Split the log band-energy vector into:")
    print("     level = mean over bands      -> moves under both causes")
    print("     shape = vector minus level   -> invariant under gain, moves under wear")
    print()

    discriminant = loaded.band_split or BandSplitDiscriminant.fit(
        frame.iloc[:n_healthy],
        band_names=monitored,
        level_z_threshold=cfg.monitor.band_split.level_z_threshold,
        shape_z_threshold=cfg.monitor.band_split.shape_z_threshold,
    )

    gain_verdict = discriminant.classify(gained)
    degraded_verdict = discriminant.classify(frame.iloc[-cfg.monitor.window_size :])

    print(f"  {'':22}{'DEGRADING BEARING':>20}{'GAIN CHANGE ONLY':>20}")
    print(f"  {THIN[:62]}")
    print(f"  {'level z':22}{degraded_verdict.level_z:>20.2f}{gain_verdict.level_z:>20.2f}")
    print(f"  {'shape z':22}{degraded_verdict.shape_z:>20.2f}{gain_verdict.shape_z:>20.2f}")
    print(
        f"  {'implied gain':22}{degraded_verdict.implied_gain:>19.2f}x"
        f"{gain_verdict.implied_gain:>19.2f}x"
    )
    print(f"  {'verdict':22}{degraded_verdict.verdict:>20}{gain_verdict.verdict:>20}")

    results["band_split"] = {
        "degraded": degraded_verdict.to_dict(),
        "gain": gain_verdict.to_dict(),
        "injected_gain": args.gain,
        "recovered_gain": gain_verdict.implied_gain,
    }

    print()
    print(f"  degrading bearing: {degraded_verdict.explain()}")
    print()
    print(f"  gain change:       {gain_verdict.explain()}")

    separated = (
        degraded_verdict.verdict == VERDICT_MECHANICAL and gain_verdict.verdict == VERDICT_GAIN
    )
    print()
    if separated:
        error = abs(gain_verdict.implied_gain - args.gain) / args.gain
        print("  SEPARATED. Same drift signal, two different causes, correctly named.")
        print(
            f"  The recovered gain is {gain_verdict.implied_gain:.3f}x against "
            f"{args.gain:.3f}x injected ({error:.1%} error),"
        )
        print("  which is a number a technician can check against the instrument record.")
    else:
        print("  NOT SEPARATED at these thresholds. Reported as-is rather than tuned away.")

    # ------------------------------------------------------------ the limits
    heading(None, "WHERE THIS DOES NOT WORK")
    print(
        """  1. FREQUENCY-DEPENDENT instrumentation faults defeat it. A loosening mount, a
     damaged cable, or a sensor with a shifted resonance changes the response
     differently at different frequencies. That is a shape change, and this
     method will call it mechanical. It only handles BROADBAND gain changes.

  2. LATE-STAGE degradation can read as a gain change. Once a spall spreads
     across a race the response becomes broadband rather than a discrete impulse
     train, the shape change flattens out, and the level keeps climbing. The
     discriminant is strongest early -- which is fortunately when the answer is
     worth more.

  3. CLIPPING is not multiplicative. An over-ranged ADC distorts the waveform
     and generates harmonics, so it registers as a shape change. Correctly
     not-a-gain-change, but it is not a bearing fault either and this cannot
     say so.

  4. SIMULTANEOUS changes are not decomposed. A sensor swap during a period of
     real degradation gives one level shift and one shape shift, and this
     reports "mechanical" without noticing the instrumentation event underneath.

  5. It compares LOG ENERGIES OF THE SAME BANDS. Change
     features.spectral.n_bands and the reference is void -- which is one more
     reason the feature cache key covers it.

  What this buys is not certainty. It is the difference between "something
  changed" and "the spectrum changed shape, which a gain change cannot do".
  The second is worth waking someone up for."""
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {out_path}")

    print()
    return 0 if separated else 1


if __name__ == "__main__":
    raise SystemExit(main())
