# bearing-watch

Anomaly detection on run-to-failure bearing vibration data: ingest, feature
extraction, train on a declared healthy window, score, detect, serve, monitor.

I built this to work through a specific question — how do you tell a machine that
is failing from an instrument that changed? — and to get the surrounding
engineering right, because on this problem the surrounding engineering is where
the failures actually happen.

```bash
make demo   # from a clean clone. No downloads. ~5 minutes.
```

That generates synthetic data in the real dataset's on-disk format, runs the full
pipeline, prints the model comparison, and runs the drift demonstration.

---

## The problem

A bearing under constant load degrades over days. Somewhere in that record the
vibration signature changes from "healthy" to "failing", and the useful question
is how early you can say so with a threshold you would be willing to page someone
about.

**The wrong answer people usually give** is: train a model on the whole run, use
a rolling z-score to flag outliers, and report the lead time.

Three things are wrong with that, and each one produces a working-looking
detector rather than an error.

1. **Training on the whole run** teaches the model to reconstruct the degraded
   snapshots too, so the score stops rising exactly where it matters. Validation
   loss looks excellent.

2. **A rolling (self-referencing) z-score is mathematically blind to ramps.** For
   a monotone ramp the MAD scales with the ramp, and `max|z|` is pinned at
   ≈ 1.349 *independent of the ramp's magnitude*. A score that ramps by 2× and
   one that ramps by 1000× give the same answer, and neither clears any sensible
   threshold. Not "fires late" — never fires, at any magnitude. It still fires on
   spikes, so it passes every spike test written for it. Bearing degradation is a
   ramp. My companion project **[`tsad-harness`](#companion-project)** derives
   this result in general; this repo is where it gets paid for.

3. **A single lead-time number is a property of the config, not the machine.**
   For a ramp there is no moment when the signal changed. The apparent onset is
   wherever the ramp crosses a threshold set by whatever window you declared
   healthy, and it moves by hours when you change that window.

## What I measured

On synthetic data in the exact IMS format (all numbers regenerate with
`make demo`; the real-dataset caveat is below and in
[`docs/RESULTS.md`](docs/RESULTS.md)):

* **The detector fires during degradation and not before it.** 0% false
  positives on the declared healthy window and 0% before degradation actually
  begins, with 61.5% detection afterwards.
* **The autoencoder does not beat PCA.** They tie. PCA trains in milliseconds
  and has no seed, no learning rate and no early-stopping rule to get wrong.
* **A kurtosis threshold — one scalar, no training — gets 40.6% detection with
  no false positives.** Any pipeline that cannot beat that is not earning its
  complexity.
* **Neither kurtosis nor the defect-frequency band features earn their place**
  on this data. Dropping both leaves detection unchanged with *better*
  separation. This is the most domain-specific code in the repo and it is
  currently not paying for itself.
* **Lead time spans 20.7–38.8 hours**, an 18.2-hour spread, driven entirely by
  which baseline window defines "normal". Nothing physical changes across that
  range.
* **A drift monitor cannot tell a failing bearing from a changed sensor.**
  Rescaling the raw signal by 1.8× — an accelerometer swap — produces a *larger*
  drift signal (mean PSI 12.09) than the actual degradation (6.64).
* **A band-split discriminant can.** It separates them cleanly and recovers the
  injected gain to 0.0% error.

## What I concluded

The modelling was not the hard part and the deep model was not necessary. What
determined whether the pipeline was right or wrong was, in order: whether the
threshold referenced a declared stationary period or the series being scored;
whether the threshold was calibrated on data the model had actually seen; and
whether the artifacts could be traced back to the config that produced them.

Two bugs I introduced and found while building this both had the same shape — a
model that scored beautifully on one metric while being broken on another, with
nothing raising:

* Calibrating the threshold **in-sample** gave PCA a 100% detection rate and a
  **22.7% false-positive rate on genuinely healthy data**. It was not detecting a
  bearing; it was detecting "not my training set".
* PCA with components chosen by explained variance on a short healthy window
  **reconstructed its own training set almost exactly**, collapsing the residual
  reference to near zero.

Neither raised an error. Both looked like success.

---

## Dataset

The **NASA IMS bearing dataset**, from the Center for Intelligent Maintenance
Systems at the University of Cincinnati, distributed through the NASA Prognostics
Data Repository.

Four double-row bearings on one shaft, constant 2000 RPM, 6000 lb radial load,
run to failure. Each file is one snapshot of **20,480 points sampled at 20 kHz**
(one second of signal), recorded **every 10 minutes**. Files are ASCII,
tab-separated, one column per channel, no header. The filename is the timestamp,
`YYYY.MM.DD.HH.MM.SS`.

| Set | Files | Channels | Outcome |
|---|---|---|---|
| 1 | 2,156 | 8 (two accelerometers per bearing) | inner-race failure on bearing 3, roller-element failure on bearing 4 |
| 2 | 984 | 4 | outer-race failure on bearing 1 |
| 3 | 6,324 | 4 | outer-race failure on bearing 3 |

### Why IMS and not CWRU

CWRU is the more common bearing benchmark, and it is the wrong dataset for this
question. CWRU faults are **machined into the bearing before the run**, so every
record contains a fault that was always present and never developed. That makes
it a classification problem — which of these known fault types is this? — with
the fault present from the first sample.

IMS bearings **degrade over days** under constant conditions. That is what this
project is about: not identifying a fault that exists, but noticing one appear.

The healthy bearings in each set are the negative controls — same shaft, same
load, same speed, same duration, no fault. If a feature rises on every channel at
once, the rig or the instrumentation changed, not a bearing.

### The split, declared before any results

* **set 2 to develop** — features, model choice, thresholds
* **set 3 to validate** — do those choices transfer
* **set 1 as hold-out** — touched once, and whatever it says gets reported

Set 1 is the hold-out for a reason beyond being unseen: 8 channels instead of 4,
and two different failure modes instead of one. It tests generalisation across
channel layout and failure mode, not just across time.

This is written down here, in the config, and in `scripts/download_ims.md`, so
that revising it later would require editing three files and a git history.

### Synthetic data, and what it is not

The dataset is several gigabytes and is not included. Instead there is a
**synthetic generator** that writes files in the *exact* same on-disk format —
same row count, same channel count, same tab-separated ASCII, same timestamp
filenames — so the entire pipeline is verifiable today with zero downloads, and
the code being exercised is the real parser rather than a test double. If the
writer and the parser ever disagree, the test suite fails.

It models a stationary healthy phase, then an outer-race defect producing a
periodic impulse train that excites a structural resonance, with amplitude
growing roughly exponentially. It is **not a validated bearing model.**

**Every number in this README and in `docs/RESULTS.md` came from synthetic data
and must be regenerated on the real dataset before it is a claim.**
`docs/RESULTS.md` lists exactly what has to be redone and which findings are most
likely to reverse.

---

## Quickstart

```bash
make demo
```

Or step by step:

```bash
make venv                 # virtualenv + editable install
make synth                # generate synthetic data in IMS format
make preflight            # data gates: exists / density / format
make train                # -> runs/<run_id>/
make infer                # score, detect onset, report a lead-time RANGE
make compare              # autoencoder vs PCA vs kurtosis, one split
make ablate               # do the feature families earn their place
make drift-demo           # the gain-change punchline
make test                 # 364 tests
```

The CLI directly:

```bash
bearingwatch preflight --data data/raw/set2
bearingwatch train     --data data/raw/set2 --model pca
bearingwatch infer     --run runs/<run_id> --data data/raw/set2
bearingwatch serve     --run runs/<run_id>
```

Serving:

```bash
docker compose up          # API + MLflow + monitor
curl localhost:8000/ready
curl localhost:8000/model  # returns the exact feature order /score expects
```

---

## Results summary

Synthetic data, 320 snapshots, degradation begins at snapshot 128. Full detail in
[`docs/RESULTS.md`](docs/RESULTS.md).

| Model | Fired at | Detection rate | False positives | Separation (MAD) | Lead time (h) |
|---|---|---|---|---|---|
| Autoencoder | 202 | 61.5% | 0.0% | 11.1 | 20.7–38.8 |
| PCA | 205 | 59.4% | 0.0% | **12.6** | 40.7–40.8 |
| Kurtosis threshold | 242 | 40.6% | 0.0% | 1.4 | 12.5–12.7 |

Identical features, identical split, identical threshold and persistence rules.
Only the model varies. The autoencoder's 3-snapshot edge is 30 minutes on a
53-hour run.

I would ship PCA.

---

## What breaks in production

### A drift monitor cannot tell you what changed

This is the finding I would lead with.

Establish a reference distribution from the healthy window, replay the run, and
the drift monitor fires 34 hours before the end. That looks like a working
early-warning system.

Now rescale the raw signal by 1.8× — an accelerometer swap, a changed
charge-amplifier setting, a re-torqued mount. Nothing is wrong with the bearing.

| | Degrading bearing | Gain change only |
|---|---|---|
| Monitor fires | **yes** | **yes** |
| Drifted features | 95% | 100% |
| Mean PSI | 6.64 | **12.09** |

The sensor change produces a **larger** drift signal than the failing bearing.

This is not a tuning problem. At the level of "has this distribution moved" the
two events *are* the same event, and no threshold separates them. A drift alert
says "something changed" and stops there — which sends an engineer to inspect a
machine when a technician changed an instrument. Do that twice and the alerting
gets muted.

### Something that can: the band-split discriminant

A broadband gain *g* multiplies power by *g²*, which in log space adds the **same
constant to every band**. So decompose the log band-energy vector:

```
level = mean over bands        moves under both causes
shape = vector minus level     invariant under gain, moves under wear
```

Degradation is not broadband — it puts energy into the defect harmonics and the
resonance they excite — so it moves the shape. A gain change cannot.

| | Degrading bearing | Gain change only |
|---|---|---|
| Level z | 14.48 | 30.27 |
| **Shape z** | **37.01** | **0.00** |
| Implied gain | 1.32× | **1.800×** (injected: 1.800×) |
| Verdict | `mechanical_change` | `gain_change` |

Shape z is **exactly zero** under the gain change — the algebraic prediction, not
an approximation. The recovered gain (1.800× vs 1.800× injected, 0.0% error) is a
number a technician can check against the instrument record. And the bands
driving the mechanical verdict are the 3.0–4.2 kHz resonance bands, so it
identifies the physical mechanism rather than just the label.

**Where it does not work** — stated here rather than discovered later:

1. **Frequency-dependent instrumentation faults defeat it.** A loosening mount or
   damaged cable changes the response differently at different frequencies. That
   is a shape change and this calls it mechanical. Only broadband gain changes
   are handled.
2. **Late-stage degradation can read as gain.** Once a spall spreads the response
   becomes broadband, shape change flattens, level keeps climbing. Strongest
   early — fortunately when the answer is worth more.
3. **Clipping is not multiplicative.** An over-ranged ADC generates harmonics, so
   it reads as a shape change. Correctly not-a-gain-change, but not a bearing
   fault either, and this cannot say so.
4. **Simultaneous changes are not decomposed.** A sensor swap during real
   degradation reports "mechanical" and misses the instrumentation event.
5. **It needs the same bands as the reference.** Change
   `features.spectral.n_bands` and the reference is void.

What it buys is not certainty. It is the difference between "something changed"
and "the spectrum changed shape, which a gain change cannot do". The second is
worth waking someone up for.

### The other things that break

* **A model whose scores drift downward.** Usually double scaling, which lowers
  or inflates every score depending on the feature means and raises nothing
  either way. Guarded here — see below.
* **A stale feature cache.** The symptom is that results stop responding to
  parameter changes, which reads as "that parameter doesn't matter".
* **A partial data download.** Trains fine, scores fine, means nothing. The
  preflight density gate exists for this.
* **A scheduled job that resolves "the latest model" at runtime.** Behaviour
  changes with no deployment and nothing in git to show for it.

---

## The engineering conventions

These are the point of the project as much as the modelling. Each one addresses a
bug class that produces plausible numbers rather than an error. Full detail in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**1. Run-folder discipline.** Every training run writes model, scaler, threshold,
config snapshot, metrics and `manifest.json` into one self-describing directory
under `runs/<run_id>/`. The manifest records code version, config hash, data
fingerprint, timestamps and every output path, and is **written last** — so a
folder without one is a crashed run, and loading it is refused. Inference loads a
run folder by path, never loose files. Pairing a model from one run with a scaler
from another raises nothing and produces plausible numbers.

**2. Train / inference separation.** Two independent CLI entry points sharing no
process state, communicating only through a run folder. `infer` works in a
process that has never seen `train`. That is what makes each an orchestrated step
rather than half of a script.

**3. Feature-cache key correctness.** The key is a content hash over the *entire*
feature config, plus acquisition parameters, plus a data fingerprint. If a
parameter can move the output it is in the key, with no manual bump to forget.
34 tests assert that perturbing any single parameter moves the key, and that a
moved key forces a recompute rather than a stale hit.

**4. Double-scaling guard.** Feature tables carry a `scaled: bool` in their
parquet metadata; the scoring path asserts on it, and a *missing* flag is an
error rather than an assumed `False`. Measured here: double scaling changes
scores by ~7 orders of magnitude while raising nothing.

**5. Preflight gates.** Exists, at the density claimed, in the format expected —
and **fail loudly**. Every read asserts exactly 20,480 points and the expected
channel count. A short file is a corrupt file and aborts rather than warning; a
warning in a scheduled job is a log line nobody reads while the run still writes
a model.

---

## Limitations and negative results

Collected here rather than scattered, because they are the honest content.

* **Every number in this repo is from synthetic data.** The synthetic healthy
  phase is far more stationary than any real rig, which inflates every z-score by
  a large unknown factor. The `max|z|` of 3,269 is an artifact and should not be
  quoted.
* **The defect-frequency features do not earn their place** on this data.
  Dropping them and kurtosis leaves detection unchanged with better separation.
  That is the most domain-specific code in the repo. It may reverse on real data
  — the synthetic fault is a single clean frequency with no sidebands — but as
  measured, it does not pay for itself.
* **The autoencoder does not beat PCA.** Expected, and reported rather than
  buried. With 72 healthy snapshots in 67 dimensions there is no realistic
  prospect of learning manifold curvature.
* **The drift monitor produces false positives.** It fired 13 snapshots before
  degradation began, on data statistically identical to the reference. Left in
  rather than tuned away — raising the threshold until it disappeared would be
  fitting a monitor to an answer that exists only because the data is synthetic.
* **The detector fires 74 snapshots after degradation begins.** Early severity is
  genuinely below the noise; a detector firing at snapshot 128 would be reacting
  to something not yet present. Reported as a range for that reason.
* **The kurtosis baseline has a known blind spot.** Kurtosis is not monotone over
  a bearing's life: it rises with discrete impacts then falls back toward 3 as
  the defect spreads and contact becomes continuous. A late, badly worn bearing
  can read healthy on it.
* **The bearing geometry is unverified.** The shipped Rexnord ZA-2115 values
  reproduce the published defect frequencies, which checks the *formulas*, not
  the geometry. A wrong pitch diameter would shift everything consistently and
  this check would not catch it. The config says so at the point of use.
* **The data fingerprint is not a content hash.** It is count + span + sampled
  file stats. It catches "the directory changed", not a byte-level edit.
* **Docker, Kubernetes and Argo manifests are unvalidated against a live
  cluster.** They are structurally tested (49 tests) and YAML-validated, and the
  CI workflow builds the image and runs a container smoke test, but I have not
  run them on a real cluster.
* **Single-bearing scope.** No cross-machine transfer, no fleet-level
  calibration, no remaining-useful-life estimate. RUL in particular needs a
  failure-time label the IMS sets do not carry, and inventing one to regress
  against would be circular.

---

## Companion project

**[`tsad-harness`](https://github.com/johnhagedorncs/tsad-harness)** derives, in general, the result this project depends on: a
self-referencing MAD z-score has `max|z|` pinned at ≈ 1.349 on a monotone ramp,
independent of the ramp's magnitude, so it is mathematically blind to progressive
degradation at any scale. It shows the same holds for the windowed and streaming
variants.

That is the theory. **bearing-watch is the application**: bearing degradation
*is* a ramp, so a self-referencing detector is blind to exactly the thing it is
deployed to find. The fix — a declared stationary reference period — is
implemented in `src/bearingwatch/detect/reference_threshold.py`, and
`reference: self` remains available purely so the failure can be demonstrated by
running it rather than asserted in prose.

---

## Layout

```
configs/          data, features, model, detect, serve — one file per concern
src/bearingwatch/
  config.py       loading, validation, content hashing
  cli.py          train / infer / compare / ablate / preflight / serve
  data/           ims_parser, synthetic, preflight, cache
  features/       time_domain, spectral, defect_freqs, pipeline
  models/         autoencoder, pca_baseline, kurtosis_baseline, base
  detect/         reference_threshold, onset
  runs/           run_folder, manifest
  store/          artifact_store (local + s3)
  tracking/       mlflow_tracking (guarded)
  serve/          app, schemas, deps
  monitor/        drift, band_split
scripts/          make_synthetic_data, run_pipeline.sh, drift_demo, download_ims.md
deploy/argo/      WorkflowTemplates + CronWorkflow (suspended in base)
deploy/k8s/       kustomize base, enable-cron component, dev/prod overlays
tests/            364 tests
docs/             ARCHITECTURE.md, RESULTS.md
```

## Installing

```bash
pip install -e ".[dev]"                          # core + test tooling
pip install -e ".[dev,tracking,monitor,s3]"      # + mlflow, evidently, boto3
```

MLflow, Evidently and boto3 are all behind guarded imports; the pipeline, the
tests and the container run green without any of them. torch is needed only for
the autoencoder — the PCA and kurtosis baselines, and serving a PCA run folder,
work without it.

### Artifact storage on S3

Training writes a run folder to local disk; in a cluster the pod that serves is
not the pod that trained. `store/artifact_store.py` is the seam, configured by
URI:

```yaml
serve:
  artifact_store_uri: "s3://my-bucket/bearingwatch"   # or file://runs
```

Layout is `s3://<bucket>/<prefix>/<run_id>/<file>`, one object per run-folder
file. `manifest.json` is uploaded **last**, mirroring the on-disk rule, so
"manifest present means complete" holds in S3 too and a partial upload cannot be
mistaken for a finished run.

IAM: the workload needs `s3:GetObject` and `s3:ListBucket` on the prefix to serve,
plus `s3:PutObject` to train. The pod gets these through an assumed role
(IRSA — see the service-account annotation in `deploy/argo/rbac.yaml`), not a
static access key. A long-lived key in a secret cannot be rotated without a
deployment and survives in every backup of that namespace.

## License

MIT — see [LICENSE](LICENSE).

Uses the NASA IMS bearing dataset, which is not included and carries its own
terms; cite it as the NASA Prognostics Data Repository requires.
