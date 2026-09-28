"""Command line interface.

``train`` and ``infer`` are INDEPENDENT subcommands with no shared process
state. They communicate only through a run folder on disk.

That separation is not stylistic. A pipeline where training and scoring are two
halves of one long-running script cannot be orchestrated: the steps cannot be
scheduled apart, retried apart, given different resources, or run on different
machines, and scoring cannot happen at all without re-running training. Splitting
them at a durable artifact boundary is what makes the Argo WorkflowTemplate in
deploy/argo/ a description of the real pipeline rather than a wrapper around a
monolith.

The test of the separation: ``infer`` must work in a process that has never seen
``train``, given only a path. It does.

Subcommands:
    synth      generate synthetic IMS-format data
    preflight  run the data gates and stop
    train      fit on the healthy window, write a run folder
    infer      score a dataset with a run folder, detect onset
    compare    train all three models on the same split, print the table
    ablate     feature ablation over kurtosis and the defect bands
    runs       list run folders
    serve      start the API
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigError
from .data.cache import FeatureCache
from .data.ims_parser import fingerprint_files, list_snapshot_files
from .data.preflight import PreflightError, run_preflight
from .detect.onset import detect_onset, sweep_lead_time
from .detect.reference_threshold import ReferenceThreshold
from .features.pipeline import (
    apply_scaler,
    build_feature_table,
    constant_features,
    feature_columns,
)
from .models.base import build_model
from .monitor.band_split import BandSplitDiscriminant, band_columns
from .runs.run_folder import (
    METRICS_FILE,
    REFERENCE_FILE,
    SCORES_FILE,
    THRESHOLD_FILE,
    RunFolder,
    RunFolderError,
    list_runs,
)
from .serve.deps import BAND_SPLIT_FILE, load_run
from .tracking.mlflow_tracking import build_tracker

log = logging.getLogger("bearingwatch")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PREFLIGHT = 2
EXIT_CONFIG = 3


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _parse_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_config(args: argparse.Namespace) -> Config:
    overrides = _parse_overrides(getattr(args, "set", None))
    if getattr(args, "data", None):
        overrides["data.root"] = args.data
    if getattr(args, "model", None):
        overrides["model.kind"] = args.model
    return Config.load(args.config_dir, overrides)


def _healthy_split(frame: pd.DataFrame, healthy_fraction: float) -> tuple[pd.DataFrame, int]:
    """First ``healthy_fraction`` of the CHRONOLOGICALLY SORTED run.

    Chronological, never random. A random split would put snapshots from the
    degraded phase into the training set, the model would learn to reconstruct
    them, and the score would stop rising -- and the run would report excellent
    validation loss while detecting nothing.
    """
    frame = frame.sort_index()
    n = len(frame)
    n_healthy = int(round(healthy_fraction * n))
    if n_healthy < 2:
        raise ConfigError(
            f"healthy window is {n_healthy} snapshot(s) out of {n}. "
            f"Increase data.healthy_fraction (currently {healthy_fraction}) or use more data."
        )
    return frame.iloc[:n_healthy], n_healthy


def _calibration_split(n_healthy: int, calibration_fraction: float) -> tuple[int, int]:
    """Split the healthy window into a fit slice and a calibration slice.

    The model is fitted on the FIRST part and the threshold is calibrated on the
    LAST part, which the model has never seen. Calibrating on the fitting data
    measures in-sample residuals, which are optimistically low for any
    reconstruction model, and yields a threshold tight enough to fire on healthy
    data the model simply has not encountered.

    Returns ``(n_fit, n_calibration)``. A calibration_fraction of 0 disables the
    split and reproduces the in-sample failure deliberately.
    """
    if calibration_fraction <= 0:
        return n_healthy, 0
    n_calibration = int(round(calibration_fraction * n_healthy))
    n_calibration = max(2, min(n_calibration, n_healthy - 2))
    return n_healthy - n_calibration, n_calibration


def _print_kv(title: str, rows: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    width = max((len(k) for k in rows), default=0)
    for key, value in rows.items():
        print(f"  {key:<{width}}  {value}")


# --------------------------------------------------------------------------
# synth
# --------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from .data.synthetic import SyntheticSpec, generate_run

    spec = SyntheticSpec(
        n_snapshots=args.n_snapshots,
        n_channels=args.n_channels,
        healthy_fraction=args.healthy_fraction,
        seed=args.seed,
    )
    print(f"generating {spec.n_snapshots} synthetic snapshots into {args.out}")
    print(
        f"  healthy phase: snapshots 0..{int(spec.healthy_fraction * spec.n_snapshots) - 1}; "
        f"degradation from {int(spec.healthy_fraction * spec.n_snapshots)}"
    )
    summary = generate_run(args.out, spec, progress=True)
    print(json.dumps(summary, indent=2))
    return EXIT_OK


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    try:
        report = run_preflight(cfg.data, sample_files=args.sample_files)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    _print_kv("preflight passed", report.to_dict())
    return EXIT_OK


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_config(args)

    # Gate 1: the data. Before anything else, and fatal.
    try:
        preflight = run_preflight(cfg.data, sample_files=args.sample_files)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    print(
        f"preflight ok: {preflight.n_files} snapshots, {preflight.n_points} points x "
        f"{preflight.n_channels} channels, span {preflight.span_hours} h"
    )

    files = list_snapshot_files(cfg.data.root)
    fingerprint = fingerprint_files(files)

    cache = FeatureCache(args.cache_dir, enabled=not args.no_cache)
    frame, entry = build_feature_table(cfg, cache=cache, paths=files, progress=True)
    print(
        f"features: {len(frame)} snapshots x {len(feature_columns(frame))} features "
        f"[cache key {entry.key} {'HIT' if entry.hit else 'MISS -> computed'}]"
    )

    healthy, n_healthy = _healthy_split(frame, cfg.data.healthy_fraction)
    columns = feature_columns(frame)
    print(
        f"healthy window: snapshots 0..{n_healthy - 1} of {len(frame)} "
        f"({cfg.data.healthy_fraction:.0%}), declared in configs/data.yaml"
    )

    # A feature that never moves across the WHOLE run is an extraction bug, not
    # a quiet sensor. It is invisible downstream -- it standardises to zero and
    # contributes nothing to any score -- so this is the only place it surfaces.
    dead = constant_features(frame, columns)
    if dead:
        print(
            f"WARNING: {len(dead)} feature(s) are constant across all {len(frame)} snapshots "
            f"and carry no information: {dead[:6]}{' ...' if len(dead) > 6 else ''}\n"
            "  A constant feature is almost always an extraction bug (a band narrower than "
            "the spectral resolution, a frequency above Nyquist), not a quiet channel.",
            file=sys.stderr,
        )

    run, manifest = RunFolder.create(
        args.runs_root, cfg, command=" ".join(["bearingwatch", "train", *sys.argv[2:]])
    )
    manifest.data_fingerprint = fingerprint
    manifest.preflight = preflight.to_dict()
    manifest.healthy_window = {
        "fraction": cfg.data.healthy_fraction,
        "n_snapshots": n_healthy,
        "first": str(healthy.index[0]),
        "last": str(healthy.index[-1]),
    }
    print(f"run folder: {run.path}")

    n_fit, n_calibration = _calibration_split(n_healthy, cfg.detect.calibration_fraction)
    fit_slice = healthy.iloc[:n_fit]
    calibration_slice = healthy.iloc[n_fit:] if n_calibration else fit_slice
    print(
        f"  model fits on snapshots 0..{n_fit - 1}; threshold calibrates on "
        f"{n_calibration or n_fit} "
        f"{'held-out' if n_calibration else 'IN-SAMPLE (calibration_fraction=0)'} snapshots"
    )

    try:
        # --- scale, fitted on the FIT SLICE only ---------------------------
        # Nothing that touches the calibration slice, so the calibration
        # residuals are genuinely out of sample.
        if cfg.model.standardise:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(fit_slice[columns].to_numpy(dtype=np.float64))
            x_fit = apply_scaler(fit_slice, scaler, columns)[columns].to_numpy(dtype=np.float64)
            x_calibration = apply_scaler(calibration_slice, scaler, columns)[columns].to_numpy(
                dtype=np.float64
            )
            manifest.add_output("scaler", run.save_scaler(scaler, columns), run.path)
        else:
            x_fit = fit_slice[columns].to_numpy(dtype=np.float64)
            x_calibration = calibration_slice[columns].to_numpy(dtype=np.float64)

        # --- fit ----------------------------------------------------------
        model = build_model(cfg.model.kind, cfg.model)
        print(f"training {cfg.model.kind} on {x_fit.shape[0]} healthy snapshots...")
        model.fit(x_fit, columns)
        for path in model.save(run.path):
            manifest.add_output(path.stem, path, run.path)

        # --- threshold, from the HELD-OUT part of the declared reference ---
        in_sample_scores = model.score(x_fit)
        healthy_scores = model.score(x_calibration)
        threshold = ReferenceThreshold.fit(
            healthy_scores,
            z_threshold=cfg.detect.z_threshold,
            mad_scale=cfg.detect.mad_scale,
            min_mad=cfg.detect.min_mad,
            reference=cfg.detect.reference,
            reference_span=(
                str(calibration_slice.index[0]),
                str(calibration_slice.index[-1]),
            ),
        )
        manifest.add_output("threshold", threshold.save(run.file(THRESHOLD_FILE)), run.path)

        # The gap between these two is the size of the mistake that calibrating
        # in-sample would make. Recorded, because it is the evidence for the
        # split rather than an argument for it.
        optimism = float(np.median(healthy_scores) / max(np.median(in_sample_scores), 1e-30))
        print(
            f"  in-sample healthy median {np.median(in_sample_scores):.4g}, "
            f"held-out {np.median(healthy_scores):.4g} "
            f"(held-out is {optimism:.1f}x higher)"
        )

        # --- band-split discriminant --------------------------------------
        bands = band_columns(columns)
        if cfg.monitor.band_split.enabled and len(bands) >= 2:
            discriminant = BandSplitDiscriminant.fit(
                healthy,
                band_names=bands,
                level_z_threshold=cfg.monitor.band_split.level_z_threshold,
                shape_z_threshold=cfg.monitor.band_split.shape_z_threshold,
            )
            manifest.add_output(
                "band_split", discriminant.save(run.file(BAND_SPLIT_FILE)), run.path
            )

        # --- reference features, for the drift monitor ---------------------
        # Stored UNSCALED. The drift monitor compares raw feature distributions,
        # and storing them post-transform would make every later comparison
        # depend on a scaler the monitor has no reason to know about.
        reference_out = healthy[columns].copy()
        manifest.add_output("reference", run.write_table(REFERENCE_FILE, reference_out), run.path)

        # --- metrics ------------------------------------------------------
        metrics = {
            "model_kind": cfg.model.kind,
            "n_snapshots": len(frame),
            "n_features": len(columns),
            "n_healthy": n_healthy,
            "n_fit": n_fit,
            "n_calibration": n_calibration,
            "calibration_fraction": cfg.detect.calibration_fraction,
            "healthy_fraction": cfg.data.healthy_fraction,
            "train": model.train_metrics_,
            "threshold": threshold.to_dict(),
            "healthy_score_median": float(np.median(healthy_scores)),
            "healthy_score_max": float(np.max(healthy_scores)),
            "in_sample_score_median": float(np.median(in_sample_scores)),
            "calibration_optimism_ratio": optimism,
            "feature_cache_key": entry.key,
            "feature_cache_hit": entry.hit,
        }
        manifest.metrics = metrics
        manifest.add_output("metrics", run.write_metrics(metrics), run.path)

        # --- tracking (optional, never fatal) -----------------------------
        tracker = build_tracker(
            cfg.tracking.enabled, cfg.tracking.tracking_uri, cfg.tracking.experiment_name
        )
        with tracker.run(run_name=run.run_id, tags={"model_kind": cfg.model.kind}):
            tracker.log_params(
                {
                    "model": asdict(cfg.model),
                    "data": asdict(cfg.data),
                    "features": asdict(cfg.features),
                    "config_hash": cfg.config_hash,
                    "feature_cache_key": cfg.feature_cache_key,
                }
            )
            tracker.log_metrics(
                {
                    **{f"train.{k}": v for k, v in model.train_metrics_.items()},
                    "n_healthy": n_healthy,
                    "n_features": len(columns),
                    "score_threshold": threshold.score_threshold,
                }
            )
            tracker.log_artifacts(run.path, artifact_path="run")
        if tracker.enabled and tracker.run_id:
            manifest.notes.append(f"mlflow run_id={tracker.run_id}")

        manifest.finalise("complete")
    except Exception:
        # Leave the manifest marked failed. A run folder without a 'complete'
        # manifest cannot be loaded for serving, which is the intended outcome.
        manifest.finalise("failed")
        manifest.save(run.path)
        raise

    manifest.save(run.path)

    _print_kv(
        "training complete",
        {
            "run_id": run.run_id,
            "run_dir": str(run.path),
            "model": cfg.model.kind,
            "features": len(columns),
            "healthy snapshots": n_healthy,
            "healthy score median": f"{np.median(healthy_scores):.6g}",
            "score threshold": f"{threshold.score_threshold:.6g}",
            "z threshold": cfg.detect.z_threshold,
            "reference": cfg.detect.reference,
        },
    )
    print(f"\nnext:  bearingwatch infer --run {run.path} --data {cfg.data.root}")
    return EXIT_OK


# --------------------------------------------------------------------------
# infer
# --------------------------------------------------------------------------


def cmd_infer(args: argparse.Namespace) -> int:
    loaded = load_run(args.run, runs_root=args.runs_root)
    cfg = loaded.config
    print(f"loaded run {loaded.run_id} ({loaded.model.kind}) from {loaded.path}")

    # The run's OWN config decides how features are computed. Not configs/ on
    # disk -- those describe the next run, not this one.
    data_root = args.data or cfg.data.root
    cfg = cfg.copy_with({"data.root": str(data_root)})

    try:
        preflight = run_preflight(cfg.data, sample_files=args.sample_files)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    files = list_snapshot_files(cfg.data.root)
    cache = FeatureCache(args.cache_dir, enabled=not args.no_cache)
    frame, entry = build_feature_table(cfg, cache=cache, paths=files, progress=True)
    print(
        f"features: {len(frame)} snapshots [cache key {entry.key} "
        f"{'HIT' if entry.hit else 'MISS -> computed'}]"
    )

    scores = loaded.score_frame(frame)

    healthy_end = int(round(cfg.data.healthy_fraction * len(frame)))
    result = detect_onset(
        scores,
        loaded.threshold,
        window=cfg.detect.persistence.window,
        consecutive=cfg.detect.persistence.consecutive,
        timestamps=list(frame.index),
        healthy_end_index=healthy_end,
        snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
    )

    lead = sweep_lead_time(
        scores,
        cfg.detect.lead_time_baseline_fractions,
        z_threshold=cfg.detect.z_threshold,
        mad_scale=cfg.detect.mad_scale,
        min_mad=cfg.detect.min_mad,
        window=cfg.detect.persistence.window,
        consecutive=cfg.detect.persistence.consecutive,
        snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
        max_healthy_fraction=cfg.data.healthy_fraction,
    )

    # Optional ground truth. Synthetic data knows exactly where degradation
    # begins; the real IMS sets do not carry a labelled fault time, and
    # inventing one to measure against would be circular. So this is an opt-in
    # flag, reported alongside the config-derived split rather than replacing
    # it, and the README says which numbers came from which.
    truth = None
    if args.truth_degradation_index is not None:
        t = int(args.truth_degradation_index)
        truth_fired = result.fired_mask[t:] if result.fired_mask is not None else np.array([])
        truth_quiet = result.fired_mask[:t] if result.fired_mask is not None else np.array([])
        truth = {
            "degradation_start_index": t,
            "detection_rate_after_truth": (
                float(np.mean(truth_fired)) if truth_fired.size else 0.0
            ),
            "false_positive_rate_before_truth": (
                float(np.mean(truth_quiet)) if truth_quiet.size else 0.0
            ),
            "onset_minus_truth_snapshots": (
                None if result.onset_index is None else int(result.onset_index - t)
            ),
            "max_z_before_truth": (float(np.max(result.z_scores[:t])) if t > 0 else float("nan")),
        }

    out_dir = Path(args.out) if args.out else loaded.path / "inference" / _infer_id(entry.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_frame = pd.DataFrame(
        {
            "score": scores,
            "z": result.z_scores,
            "exceeds": result.z_scores > loaded.threshold.z_threshold,
            "fired": result.fired_mask,
            "snapshot_index": frame["snapshot_index"].to_numpy(),
            "source_file": frame["source_file"].to_numpy(),
        },
        index=frame.index,
    )
    scores_frame.to_parquet(out_dir / SCORES_FILE, compression="snappy")

    payload = {
        "run_id": loaded.run_id,
        "model_kind": loaded.model.kind,
        "data_root": str(cfg.data.root),
        "n_snapshots": len(frame),
        "feature_cache_key": entry.key,
        "preflight": preflight.to_dict(),
        "detection": result.to_dict(),
        "lead_time_range": lead.to_dict(),
        "threshold": loaded.threshold.to_dict(),
        "healthy_end_index": healthy_end,
        "ground_truth": truth,
    }
    with (out_dir / METRICS_FILE).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)

    _print_kv(
        "detection",
        {
            "fired": result.fired,
            "onset index": result.onset_index,
            "onset timestamp": result.onset_timestamp,
            "max z (whole run)": f"{result.max_z:.2f}",
            "max z (healthy phase)": f"{result.healthy_max_z:.2f}",
            "healthy false-positive rate": f"{result.healthy_false_positive_rate:.3%}",
            "degradation detection rate": f"{result.degraded_detection_rate:.1%}",
            "lead time range": lead.summary(),
        },
    )
    if truth is not None:
        _print_kv(
            "against declared ground truth (synthetic data only)",
            {
                "degradation starts at index": truth["degradation_start_index"],
                "detection rate after truth": f"{truth['detection_rate_after_truth']:.1%}",
                "false positives before truth": (
                    f"{truth['false_positive_rate_before_truth']:.3%}"
                ),
                "onset - truth (snapshots)": truth["onset_minus_truth_snapshots"],
                "max z before truth": f"{truth['max_z_before_truth']:.2f}",
            },
        )
    print(f"\nwrote {out_dir / SCORES_FILE}")
    print(f"wrote {out_dir / METRICS_FILE}")

    if args.fail_if_quiet and not result.fired:
        print("detector did not fire anywhere in the run", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def _infer_id(cache_key: str) -> str:
    from datetime import datetime, timezone

    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{cache_key[:8]}"


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    """Train every model on the SAME healthy window and score the SAME run.

    Same features, same split, same threshold rule, same persistence rule. The
    only thing that varies is the model. Anything else and a difference in the
    table would be a difference in the harness.
    """
    base = _load_config(args)
    try:
        run_preflight(base.data, sample_files=args.sample_files)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    files = list_snapshot_files(base.data.root)
    cache = FeatureCache(args.cache_dir, enabled=not args.no_cache)
    frame, entry = build_feature_table(base, cache=cache, paths=files, progress=True)
    columns = feature_columns(frame)
    healthy, n_healthy = _healthy_split(frame, base.data.healthy_fraction)

    from sklearn.preprocessing import StandardScaler

    # Identical split for every model: fit on the first part of the healthy
    # window, calibrate the threshold on the held-out remainder.
    n_fit, n_calibration = _calibration_split(n_healthy, base.detect.calibration_fraction)
    scaler = StandardScaler().fit(healthy[columns].iloc[:n_fit].to_numpy(dtype=np.float64))
    x_all = scaler.transform(frame[columns].to_numpy(dtype=np.float64))
    x_fit = x_all[:n_fit]
    x_calibration = x_all[n_fit:n_healthy] if n_calibration else x_fit
    print(
        f"  fit on {n_fit} healthy snapshots, threshold calibrated on "
        f"{n_calibration or n_fit} held-out"
    )

    kinds = args.models or ["autoencoder", "pca", "kurtosis"]
    rows: list[dict[str, Any]] = []

    for kind in kinds:
        cfg = base.copy_with({"model.kind": kind})
        try:
            model = build_model(kind, cfg.model)
            model.fit(x_fit, columns)
            scores = model.score(x_all)
        except Exception as exc:
            rows.append({"model": kind, "error": str(exc)[:80]})
            print(f"  {kind}: FAILED ({exc})", file=sys.stderr)
            continue

        threshold = ReferenceThreshold.fit(
            model.score(x_calibration),
            z_threshold=cfg.detect.z_threshold,
            mad_scale=cfg.detect.mad_scale,
            min_mad=cfg.detect.min_mad,
        )
        result = detect_onset(
            scores,
            threshold,
            window=cfg.detect.persistence.window,
            consecutive=cfg.detect.persistence.consecutive,
            timestamps=list(frame.index),
            healthy_end_index=n_healthy,
            snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
        )
        lead = sweep_lead_time(
            scores,
            cfg.detect.lead_time_baseline_fractions,
            z_threshold=cfg.detect.z_threshold,
            window=cfg.detect.persistence.window,
            consecutive=cfg.detect.persistence.consecutive,
            snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
            max_healthy_fraction=cfg.data.healthy_fraction,
        )
        row = {
            "model": kind,
            "fired": result.fired,
            "onset_index": result.onset_index,
            "detect_rate": result.degraded_detection_rate,
            "healthy_fp": result.healthy_false_positive_rate,
            "healthy_max_z": result.healthy_max_z,
            "max_z": result.max_z,
            "lead_min_h": lead.min_hours,
            "lead_max_h": lead.max_hours,
            "separation": _separation(scores, n_healthy),
        }

        # Ground truth, when it exists. The config split calls everything after
        # the healthy window "degradation", but the healthy window is
        # deliberately shorter than the true healthy phase so the reference
        # period stays clean. Snapshots between the two ARE still healthy, and a
        # model that fires there is producing false positives that the
        # config-split columns score as successes. On synthetic data the true
        # boundary is known, so both are reported.
        if args.truth_degradation_index is not None:
            t = int(args.truth_degradation_index)
            mask = result.fired_mask
            row["truth_fp"] = float(np.mean(mask[:t])) if t > 0 else 0.0
            row["truth_detect"] = float(np.mean(mask[t:])) if mask[t:].size else 0.0
            row["onset_minus_truth"] = (
                None if result.onset_index is None else int(result.onset_index - t)
            )
        rows.append(row)

    _print_comparison(
        rows, n_healthy, len(frame), len(columns), entry.key, args.truth_degradation_index
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "data_root": str(base.data.root),
                    "n_snapshots": len(frame),
                    "n_features": len(columns),
                    "n_healthy": n_healthy,
                    "feature_cache_key": entry.key,
                    "rows": rows,
                },
                fh,
                indent=2,
                sort_keys=True,
                default=str,
            )
        print(f"\nwrote {args.out}")
    return EXIT_OK


def _separation(scores: np.ndarray, n_healthy: int) -> float:
    """Median degraded score minus median healthy score, in healthy MADs.

    A scale-free way to compare models whose raw scores are in different units:
    a PCA reconstruction MSE and a kurtosis deviation are not comparable
    numbers, but "how many healthy-window MADs apart are the two phases" is.
    """
    healthy = scores[:n_healthy]
    degraded = scores[n_healthy:]
    if degraded.size == 0:
        return float("nan")
    med = float(np.median(healthy))
    mad = max(float(np.median(np.abs(healthy - med))), 1e-12)
    return float((np.median(degraded) - med) / (1.4826 * mad))


def _print_comparison(
    rows: list[dict[str, Any]],
    n_healthy: int,
    n_total: int,
    n_features: int,
    key: str,
    truth: int | None = None,
) -> None:
    print(
        f"\nmodel comparison  ({n_total} snapshots, {n_features} features, "
        f"healthy window = first {n_healthy}, cache key {key})"
    )
    header = (
        f"{'model':<14}{'fired':<7}{'onset':<8}{'detect%':<10}{'healthyFP%':<12}"
        f"{'maxZ':<10}{'sep(MAD)':<10}{'lead h':<14}"
    )
    if truth is not None:
        header += f"{'trueFP%':<10}{'trueDet%':<10}{'onset-truth':<12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<14}ERROR: {r['error']}")
            continue
        lead = (
            f"{r['lead_min_h']:.1f}-{r['lead_max_h']:.1f}" if r["lead_min_h"] is not None else "-"
        )
        line = (
            f"{r['model']:<14}"
            f"{r['fired']!s:<7}"
            f"{r['onset_index']!s:<8}"
            f"{r['detect_rate'] * 100:<10.1f}"
            f"{r['healthy_fp'] * 100:<12.1f}"
            f"{r['max_z']:<10.1f}"
            f"{r['separation']:<10.1f}"
            f"{lead:<14}"
        )
        if truth is not None:
            line += (
                f"{r.get('truth_fp', 0) * 100:<10.1f}"
                f"{r.get('truth_detect', 0) * 100:<10.1f}"
                f"{r.get('onset_minus_truth')!s:<12}"
            )
        print(line)

    if truth is not None:
        print(
            f"\ntrueFP%/trueDet% are measured against the known degradation start "
            f"(index {truth}), which only synthetic data has. healthyFP%/detect% use the "
            f"config split at {n_healthy}. A model can look good on the second pair and "
            "bad on the first by firing in the gap between them, where the bearing is "
            "still healthy."
        )


# --------------------------------------------------------------------------
# ablate
# --------------------------------------------------------------------------


def cmd_ablate(args: argparse.Namespace) -> int:
    """Do kurtosis and the defect-band features earn their place?

    Each ablation gets its own feature cache key, so an ablated run physically
    cannot read the full-feature cache. That is the cache-key correctness
    property being used rather than merely asserted.
    """
    base = _load_config(args)
    try:
        run_preflight(base.data, sample_files=args.sample_files)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return EXIT_PREFLIGHT

    files = list_snapshot_files(base.data.root)
    cache = FeatureCache(args.cache_dir, enabled=not args.no_cache)

    variants = [
        ("full", {}),
        ("no_kurtosis", {"features.ablation.drop_kurtosis": True}),
        ("no_defect_bands", {"features.ablation.drop_defect_bands": True}),
        (
            "neither",
            {
                "features.ablation.drop_kurtosis": True,
                "features.ablation.drop_defect_bands": True,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []
    seen_keys: dict[str, str] = {}

    for name, overrides in variants:
        cfg = base.copy_with(overrides) if overrides else base
        frame, entry = build_feature_table(cfg, cache=cache, paths=files)
        columns = feature_columns(frame)

        if entry.key in seen_keys:
            raise RuntimeError(
                f"ablation {name!r} produced the same cache key as {seen_keys[entry.key]!r}. "
                "The cache key does not cover the ablation switches, which means an ablated "
                "run would read full-feature values out of the cache and report them as an "
                "ablation result."
            )
        seen_keys[entry.key] = name

        healthy, n_healthy = _healthy_split(frame, cfg.data.healthy_fraction)

        from sklearn.preprocessing import StandardScaler

        n_fit, n_calibration = _calibration_split(n_healthy, cfg.detect.calibration_fraction)
        scaler = StandardScaler().fit(healthy[columns].iloc[:n_fit].to_numpy(dtype=np.float64))
        x_all = scaler.transform(frame[columns].to_numpy(dtype=np.float64))

        model = build_model(cfg.model.kind, cfg.model)
        model.fit(x_all[:n_fit], columns)
        scores = model.score(x_all)

        threshold = ReferenceThreshold.fit(
            scores[n_fit:n_healthy] if n_calibration else scores[:n_fit],
            z_threshold=cfg.detect.z_threshold,
        )
        result = detect_onset(
            scores,
            threshold,
            window=cfg.detect.persistence.window,
            consecutive=cfg.detect.persistence.consecutive,
            healthy_end_index=n_healthy,
            snapshot_interval_minutes=cfg.data.snapshot_interval_minutes,
        )
        rows.append(
            {
                "variant": name,
                "cache_key": entry.key,
                "n_features": len(columns),
                "fired": result.fired,
                "onset_index": result.onset_index,
                "detect_rate": result.degraded_detection_rate,
                "healthy_fp": result.healthy_false_positive_rate,
                "separation": _separation(scores, n_healthy),
                "max_z": result.max_z,
            }
        )

    print(f"\nfeature ablation  (model={base.model.kind})")
    header = (
        f"{'variant':<18}{'feats':<8}{'cache key':<20}{'fired':<7}{'onset':<8}"
        f"{'detect%':<10}{'healthyFP%':<12}{'sep(MAD)':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['variant']:<18}{r['n_features']:<8}{r['cache_key']:<20}"
            f"{r['fired']!s:<7}{r['onset_index']!s:<8}"
            f"{r['detect_rate'] * 100:<10.1f}{r['healthy_fp'] * 100:<12.1f}"
            f"{r['separation']:<10.1f}"
        )
    print("\nevery variant has a distinct cache key, so no variant read another's features.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as fh:
            json.dump({"model": base.model.kind, "rows": rows}, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    return EXIT_OK


# --------------------------------------------------------------------------
# runs / serve
# --------------------------------------------------------------------------


def cmd_runs(args: argparse.Namespace) -> int:
    folders = list_runs(args.runs_root)
    if not folders:
        print(f"no run folders under {args.runs_root}")
        return EXIT_OK
    print(f"{'run_id':<28}{'status':<12}{'model':<14}{'config hash':<18}created")
    print("-" * 96)
    for folder in folders:
        d = folder.describe()
        print(
            f"{d['run_id']:<28}{d.get('status', '?'):<12}{d.get('model_kind', '?'):<14}"
            f"{d.get('config_hash', '?'):<18}{d.get('created_at', '?')}"
        )
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = _load_config(args)
    if args.run:
        import os

        os.environ["BEARINGWATCH_RUN_DIR"] = str(args.run)
    uvicorn.run(
        "bearingwatch.serve.app:app",
        host=args.host or cfg.serve.host,
        port=args.port or cfg.serve.port,
        reload=False,
    )
    return EXIT_OK


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bearingwatch",
        description=(
            "Run-to-failure bearing anomaly detection. train and infer are independent "
            "entry points communicating only through a run folder on disk."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, data: bool = True) -> None:
        p.add_argument("--config-dir", default="configs", help="directory of YAML config files")
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="dotted-path config override, e.g. --set model.kind=pca (repeatable)",
        )
        if data:
            p.add_argument("--data", help="data root (overrides data.root)")
            p.add_argument("--cache-dir", default=".cache/features", help="feature cache dir")
            p.add_argument("--no-cache", action="store_true", help="bypass the feature cache")
            p.add_argument(
                "--sample-files", type=int, default=5, help="files to format-check in preflight"
            )

    p = sub.add_parser("synth", help="generate synthetic IMS-format data")
    p.add_argument("--out", default="data/raw/synthetic")
    p.add_argument("--n-snapshots", type=int, default=320)
    p.add_argument("--n-channels", type=int, default=4)
    p.add_argument("--healthy-fraction", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("preflight", help="run the data gates and stop")
    common(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("train", help="fit on the healthy window, write a run folder")
    common(p)
    p.add_argument("--model", choices=["autoencoder", "pca", "kurtosis"])
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer", help="score a dataset using a run folder")
    common(p)
    p.add_argument("--run", help="run folder path (default: latest complete under --runs-root)")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", help="output directory (default: <run>/inference/<id>)")
    p.add_argument(
        "--fail-if-quiet",
        action="store_true",
        help="exit non-zero if the detector never fires (for scheduled smoke checks)",
    )
    p.add_argument(
        "--truth-degradation-index",
        type=int,
        help=(
            "index where degradation actually begins. Only known for synthetic data; "
            "the IMS sets carry no labelled fault time."
        ),
    )
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("compare", help="train all models on one split and print the table")
    common(p)
    p.add_argument("--models", nargs="*", choices=["autoencoder", "pca", "kurtosis"])
    p.add_argument("--out", help="write the comparison as JSON")
    p.add_argument(
        "--truth-degradation-index",
        type=int,
        help="known degradation start index (synthetic data only); adds true FP/detect columns",
    )
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("ablate", help="feature ablation over kurtosis and defect bands")
    common(p)
    p.add_argument("--model", choices=["autoencoder", "pca", "kurtosis"])
    p.add_argument("--out", help="write the ablation as JSON")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("runs", help="list run folders")
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("serve", help="start the FastAPI service")
    common(p, data=False)
    p.add_argument("--run", help="run folder to serve")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except (RunFolderError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
