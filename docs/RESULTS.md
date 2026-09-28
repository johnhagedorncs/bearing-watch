# Results

> **EVERY NUMBER ON THIS PAGE CAME FROM SYNTHETIC DATA.**
>
> The NASA IMS dataset is a multi-gigabyte download and is not included. The
> generator in `src/bearingwatch/data/synthetic.py` writes files in the exact
> IMS on-disk format so the pipeline is runnable immediately, and these numbers
> are what it produces. They demonstrate that the pipeline runs, that the
> detector fires during degradation and stays quiet while the bearing is
> healthy, and that the engineering properties this repo argues for are actually
> implemented and testable.
>
> They are **not results about bearings.** The synthetic healthy phase is more
> stationary than any real rig, which inflates every z-score reported here by a
> large and unknown factor. Regenerate all of this on the real dataset before
> treating any of it as a claim. See `scripts/download_ims.md`.

Reproduce with:

```bash
make demo
```

---

## Run configuration

| | |
|---|---|
| Data | synthetic, 320 snapshots, 4 channels, 20,480 points at 20 kHz |
| Cadence | 10 minutes; run spans 53.2 h |
| True degradation start | snapshot 128 (severity is exactly 0 before this) |
| Declared healthy window | snapshots 0–95 (`data.healthy_fraction = 0.30`) |
| Model fitted on | snapshots 0–71 |
| Threshold calibrated on | snapshots 72–95 (held out from fitting) |
| Features | 67 (8 time-domain + 40 log-spaced band energies + 13 defect-band + 6 context) |
| Detector | robust z ≥ 3.5 against the declared reference, 3-of-5 persistence |

The healthy window (30%) is deliberately shorter than the truly healthy phase
(40%). Snapshots 96–127 are still healthy but sit outside the reference, which
gives a real test of whether the detector fires on healthy data it has not seen.

---

## End-to-end detection (autoencoder)

| Measure | Value |
|---|---|
| Detector fired | yes, at snapshot 202 |
| False positives in the declared healthy window (0–95) | **0.0%** |
| False positives before true degradation (0–127) | **0.0%** |
| Detection rate after true degradation (128–319) | 61.5% |
| max &#124;z&#124; before degradation began | 1.43 |
| max &#124;z&#124; over the whole run | 3,269 |
| Onset minus true degradation start | +74 snapshots (12.3 h late) |
| Lead time to end of run | **20.7–38.8 h** (median 23.0, spread 18.2) |

The detector fires during degradation and not before it. That is the core claim
and it holds with zero false positives on both definitions of "healthy".

It fires **74 snapshots after** degradation actually starts, because severity
grows from zero with a cubic-ish profile and the early phase is genuinely below
the noise. A detector that fired at snapshot 128 on this data would be reacting
to something not yet present.

### Lead time is a range, not a number

20.7–38.8 h, a **spread of 18.2 hours** driven entirely by which baseline
fraction defines "normal":

| Baseline fraction | Reference snapshots | Onset | Lead (h) |
|---|---|---|---|
| 0.15 | 48 | 187 | 22.0 |
| 0.20 | 64 | 187 | 22.0 |
| 0.25 | 80 | 187 | 22.0 |
| 0.30 | 96 | 202 | 19.5 |

(0.40 and 0.50 are skipped: they reach past the declared healthy phase, so the
reference period would contain degraded data.)

Nothing physical changes between these rows. The bearing is the same bearing.
Reporting "the detector gave 22 hours of warning" would be reporting a property
of `configs/detect.yaml`, not of the machine.

---

## Model comparison

Identical features, identical split, identical threshold rule, identical
persistence rule. Only the model varies.

| Model | Fired at | Detection rate | False positives (declared) | False positives (true) | Separation (MAD) | Lead time (h) |
|---|---|---|---|---|---|---|
| Autoencoder | 202 | 61.5% | 0.0% | 0.0% | 11.1 | 20.7–38.8 |
| PCA | 205 | 59.4% | 0.0% | 0.0% | **12.6** | 40.7–40.8 |
| Kurtosis threshold | 242 | 40.6% | 0.0% | 0.0% | 1.4 | 12.5–12.7 |

*Detection rate and false positives are against the true degradation boundary
(snapshot 128). Separation is the median degraded score minus the median healthy
score, in healthy-window MADs — a scale-free comparison, since a PCA
reconstruction MSE and a kurtosis deviation are not comparable in raw units.*

### What this says

**The autoencoder does not beat PCA.** It fires 3 snapshots earlier (30 minutes
on a 53-hour run, which is noise) and PCA has slightly better separation. On any
reasonable reading they are tied.

That is the expected outcome and it is worth stating plainly. A linear
autoencoder with MSE loss spans the same subspace as PCA, so the deep model only
has something to offer when the healthy manifold is genuinely curved AND there is
enough healthy data to identify the curvature. Here the healthy window is 72
snapshots in 67 dimensions. There is no realistic prospect of learning curvature
from that, and the result reflects it.

PCA trains in milliseconds, has no seed, no learning rate, no early-stopping
rule, and no failure mode where a bad initialisation quietly produces a worse
model. **If this were a real deployment the right choice would be PCA**, and the
autoencoder's presence in the repo would be justified only as the thing that
proved PCA was sufficient.

**The kurtosis threshold is much worse but not useless.** 40.6% detection with no
false positives, from a single scalar requiring no training. It fires 114
snapshots after degradation begins because kurtosis only responds once the defect
produces discrete impacts. Any pipeline that cannot beat this is not earning its
complexity.

### Two findings that changed the table

Both were discovered while building this and both had the same shape: a model
that looked good on one metric while being broken on another.

**1. In-sample threshold calibration.** Calibrating the threshold on the same
snapshots the model was fitted on gave PCA a 100% detection rate — and a **22.7%
false-positive rate on genuinely healthy data**, firing 29 snapshots before
anything happened. A reconstruction model always fits its own training data
better than unseen data, so in-sample residuals are optimistically low and the
threshold derived from them is too tight. PCA was not detecting a bearing; it was
detecting "not my training set". Fixed by holding out 25% of the healthy window
from fitting and calibrating only on that. Measured optimism on this run:
held-out residuals are **1.87×** the in-sample ones.

**2. PCA rank versus sample count.** With `n_components` chosen by 95% explained
variance on a short healthy window, PCA retained nearly as many components as
there were snapshots, reconstructed its training set almost exactly, and
collapsed the residual reference to near zero. Fixed with
`model.pca.max_sample_ratio`, requiring at least 3 healthy snapshots per retained
component.

Neither of these raised an error. Both produced a model that looked like it was
working very well.

---

## Feature ablation

Does each feature family earn its place? Each variant gets its own feature cache
key, so an ablated run physically cannot read full-feature values.

**Autoencoder:**

| Variant | Features | Cache key | Fired at | Detection rate | Separation (MAD) |
|---|---|---|---|---|---|
| full | 67 | `d5fb6424c69f1914` | 202 | 52.7% | 11.1 |
| no kurtosis | 63 | `1fc00a2c777fd182` | 195 | 55.8% | 10.6 |
| no defect bands | 55 | `c8a49ce6a22db6ef` | 195 | 55.8% | 12.9 |
| neither | 51 | `01a6989c06464118` | 202 | 52.7% | **13.0** |

**PCA:**

| Variant | Features | Cache key | Fired at | Detection rate | Separation (MAD) |
|---|---|---|---|---|---|
| full | 67 | `d5fb6424c69f1914` | 205 | 50.9% | 12.6 |
| no kurtosis | 63 | `1fc00a2c777fd182` | 213 | 47.8% | **15.5** |
| no defect bands | 55 | `c8a49ce6a22db6ef` | 211 | 48.7% | 10.9 |
| neither | 51 | `01a6989c06464118` | 218 | 45.5% | 11.4 |

### Negative result: neither family earns its place on this data

Dropping **both** kurtosis and the defect bands leaves the autoencoder detecting
at the same snapshot with *better* separation (13.0 vs 11.1). The results are
essentially flat across all four variants; the differences are within the
variation you get from reseeding.

The honest conclusion on synthetic data: **the broadband RMS and band-energy rise
carries the entire signal, and the physics-derived features add nothing
measurable.** The defect-frequency machinery — the BPFO/BPFI/BSF/FTF formulas,
the geometry config, the narrow-band integration — is the most domain-specific
code in the repo and it is not currently paying for itself.

Two caveats before generalising:

1. The synthetic generator puts its impulse train at a **single** BPFO frequency
   with a clean resonance. A real spall produces sidebands, slip-induced
   smearing, and multiple interacting frequencies. Defect-band features may well
   earn their place on real data where the broadband rise is less dominant.
2. Defect bands are **normalised by total energy**, which is deliberate — it is
   what makes them gain-invariant and it is the basis of the band-split
   discriminant that *does* work (below). They may add little to detection while
   still being essential to diagnosis.

The way to settle this is to re-run the ablation on IMS set 2. That is a
one-command job once the data exists.

---

## The drift demonstration

Full output: `python scripts/drift_demo.py`. Summary of the four steps.

### Step 2 — the drift monitor fires before failure

Reference from the healthy window (snapshots 0–95), sliding 20-snapshot window.

| | |
|---|---|
| Drift monitor fired at | snapshot 115 |
| Lead time to end of run | 34.0 h |
| Drifted feature fraction | 35% |
| Mean PSI | 0.230 |

**It fired 13 snapshots before degradation actually began.** Those snapshots are
statistically identical to the reference period — the generator sets severity to
exactly zero there. This is a false positive from comparing a 20-snapshot window
against a 96-snapshot reference. It is reported rather than tuned away: the
threshold could be raised until it disappeared, but that is fitting a monitor to
an answer that exists only because the data is synthetic.

### Step 3 — the punchline

Re-featurise the **healthy** window with the raw waveform multiplied by 1.8×,
which is what an accelerometer swap or a changed amplifier setting does. Nothing
is wrong with the bearing.

| | Degrading bearing | Gain change only |
|---|---|---|
| Monitor fires | **yes** | **yes** |
| Drifted feature fraction | 95% | 100% |
| Mean PSI | 6.64 | **12.09** |
| Max PSI | 10.36 | 12.34 |

The sensor change produces a **larger** drift signal than the failing bearing.
A distribution-drift monitor reports these as the same event, because at the
level of "has this distribution moved" they *are* the same event. Escalating on
drift alone sends an engineer to inspect a machine when a technician changed an
instrument.

### Step 5 — the band-split discriminant

A broadband gain *g* multiplies power by *g²*, which in log space adds the same
constant to every band. So split the log band-energy vector into a **level**
(mean across bands) and a **shape** (the vector minus its level). The level moves
under both causes; the shape is invariant under gain and moves under wear.

| | Degrading bearing | Gain change only |
|---|---|---|
| Level z | 14.48 | 30.27 |
| **Shape z** | **37.01** | **0.00** |
| Implied gain | 1.32× | **1.800×** |
| Verdict | `mechanical_change` | `gain_change` |

Shape z is **exactly 0.00** under the gain change — the algebraic prediction, not
an approximation — while the degrading bearing gives 37.0. The recovered gain is
**1.800× against 1.800× injected, 0.0% error**, a number a technician can check
against the instrument record.

The bands driving the mechanical verdict are `ch0__band33_2985_3548Hz` (z=37.0)
and `ch0__band34_3548_4217Hz` (z=35.7) — the structural resonance the synthetic
impulses excite. The discriminant identifies the correct physical mechanism, not
just the correct label.

### Where the discriminant does not work

1. **Frequency-dependent instrumentation faults defeat it.** A loosening mount or
   a damaged cable changes the response differently at different frequencies.
   That is a shape change, and this reports it as mechanical. Only *broadband*
   gain changes are handled.
2. **Late-stage degradation can read as gain.** Once a spall spreads, the
   response becomes broadband, the shape change flattens, and the level keeps
   climbing. Strongest early — fortunately when the answer is worth more.
3. **Clipping is not multiplicative.** An over-ranged ADC generates harmonics,
   so it registers as a shape change. Correctly not-a-gain-change, but not a
   bearing fault either, and this cannot say so.
4. **Simultaneous changes are not decomposed.** A sensor swap during real
   degradation reports "mechanical" without noticing the instrumentation event.
5. **It requires the same bands as the reference.** Change
   `features.spectral.n_bands` and the reference is void.

---

## Bearing defect frequencies

Computed from the shipped Rexnord ZA-2115 geometry (16 rolling elements, 0.331 in
ball diameter, 2.815 in pitch diameter, 15.17° contact angle) at 2000 RPM:

| | Computed | Commonly published |
|---|---|---|
| BPFO | 236.40 Hz | ~236.4 Hz |
| BPFI | 296.93 Hz | ~296.9 Hz |
| BSF | 139.92 Hz | ~139.9 Hz |
| FTF | 14.78 Hz | ~14.8 Hz |

Asserted in `tests/test_features.py`. **These agree with published values, which
is a check on the formulas, not on the geometry.** The geometry itself still
needs verifying against the dataset documentation — a wrong pitch diameter would
shift every frequency consistently and this check would not catch it.

---

## Test suite

**364 tests, all passing, 92% statement coverage.**

Notable coverage:

* 34 tests that changing any single feature parameter moves the cache key, plus
  an end-to-end test that a changed key forces a recompute rather than a hit.
* 8 tests of the double-scaling guard, including the measured demonstration that
  double scaling changes scores by ~7 orders of magnitude while raising nothing.
* 18 tests of the reference threshold, including the ramp-blindness result
  verified across ramp slopes spanning 9 orders of magnitude.
* 49 structural tests of the deployment manifests.
* 14 end-to-end CLI tests running the real pipeline in a temporary directory.

---

## What has to be redone on the real dataset

Everything above. Specifically:

1. **The model comparison.** The autoencoder/PCA tie is the expected result, but
   the real healthy manifold may have structure the synthetic one lacks.
2. **The ablation.** The finding that defect-band features add nothing is the
   most likely to reverse. The synthetic fault is a single clean frequency.
3. **Every z-score.** The synthetic healthy phase is far more stationary than a
   real rig. A max &#124;z&#124; of 3,269 is an artifact of that and should not be
   quoted.
4. **Lead times.** The 20.7–38.8 h range comes from a 53-hour synthetic run. IMS
   set 2 spans about a week.
5. **The drift demonstration.** The gain-change punchline is algebraic and will
   hold. The *magnitudes* — PSI values, shape-z separation — will not transfer.

The declared split, fixed before any of this and unchanged since: **set 2 to
develop, set 3 to validate, set 1 as the hold-out touched once.**
