# Edge Vision for Retail: One Model, Many Stores, No One On Site

**The engineering behind running computer vision on unattended in-store accelerators.** It
covers four decisions:
- **Per-channel INT8 quantization,** so models keep their accuracy at 8 bits.
- **One generic model plus a tiny label-free calibration per store,** instead of a model per
  store.
- **Offline-first operation** that survives a long outage.
- **Staged updates with automatic rollback** for a fleet nobody can walk up to.

> **Honest framing.** This is not client code, and it benchmarks no real deployment or
> accelerator. The experiments use a small network on scikit-learn's
> bundled handwritten-digits images, so they run offline on a laptop CPU in about a minute.
> They show the *mechanisms*. Accuracy numbers from them are not retail-detection numbers.
> Every claim is labelled: **Measured** (produced by code here, on that stand-in data),
> **Tested** (behaviour verified by tests), **Derived** (arithmetic from stated assumptions),
> or **Designed** (not run here).

## The problem

A retailer wants shelf, queue and footfall analytics in hundreds of stores. Sending video to
the cloud is expensive and slow, so inference runs on a small accelerator in each store.
That creates four problems:

1. **The models must run at 8-bit precision to fit.** Naive 8-bit quantization can quietly
   cost accuracy.
2. **Every store's cameras see differently.** Lighting, white balance, sensor gain and dirty
   lenses all differ, so a model trained on one set of stores loses accuracy in the next.
   Training a model per store means hundreds of models to label, train, version and deploy.
3. **Store uplinks fail.** The analytics must keep working and catch up afterwards.
4. **Nobody is on site.** A bad update can't be fixed by hand.

## Design decisions

### 1. Per-channel INT8 quantization — *Measured*, [`quantize/ptq.py`](quantize/ptq.py)

At deployment, batch normalisation is folded into the convolution before it. Folding scales
each output channel by a different factor, so channels end up with very different weight
ranges. Quantization maps weights to 8-bit integers using a step size (the scale):
- **Per-tensor:** one step for the whole layer, set by the widest channel. Narrow channels
  get only a few distinct levels.
- **Per-channel:** one step per output channel, so every channel keeps its resolution.

On the stand-in model (540 test images):

| Precision | Per-tensor | Per-channel |
|---|---|---|
| FP32 (and FP32 with BN folded — identical) | 99.44% | 99.44% |
| INT8 weights, INT8 activations | 99.26% | **99.44%** |
| INT4 weights, INT8 activations | 97.59% | **99.26%** |

This toy network's channel ranges only differ about 2×, so the INT8 gap is a single image.
Production backbones, especially those with depthwise convolutions, often differ by 10–100×.
A controlled sweep on synthetic layers isolates that effect:

| Spread between channels (ratio of standard deviations) | 1× | 16× | 64× | 256× |
|---|---|---|---|---|
| Per-tensor: mean relative squared error per channel | 9.7e-5 | 3.9e-3 | 2.3e-2 | 1.7e-1 |
| Per-channel: mean relative squared error per channel | 5.1e-5 | 5.2e-5 | 4.8e-5 | 5.3e-5 |
| Weights rounded to zero, per-tensor | 1.5% | 6.8% | 13.7% | **30.6%** |
| Weights rounded to zero, per-channel | 1.1% | 0.8% | 1.1% | 1.1% |

Per-channel error stays flat. Per-tensor squared error grows about 1,800× across the sweep
(about 42× in RMS terms), and at a 256× spread it erases almost a third of the weights,
against about 1% with per-channel scales. That is where real accuracy losses come
from, and why per-channel scales are the default to ask an accelerator toolchain for.

### 2. One generic model plus per-store calibration — *Measured*, [`stores/adapt.py`](stores/adapt.py)

A network with batch normalisation (BN) keeps running statistics of the activations it saw
in training. Re-estimating those statistics on a few hundred **unlabelled** images from a new
store re-centres the network on that store's cameras. Nothing is trained and nothing is
labelled. The per-store artefact is those statistics, and nothing else.

Twelve simulated stores each get a different camera response: gain, offset, sensor noise
and blur. Each calibration uses 200 unlabelled images. Scoring uses held-out images with
independent sensor noise; labels are used only to score.

| Method | Mean accuracy | Worst store | Per-store artefact |
|---|---|---|---|
| Generic model, as is | 58.3% | 11.5% | — |
| **Generic + BN calibration** | **94.0%** | 61.5% | **224 numbers** (the BN statistics) |
| Generic + BN calibration, per-channel INT8 | 93.9% | 61.3% | per-channel scales and biases, plus activation ranges (below) |
| Store-specific model (fine-tuned on 200 *labelled* images) | 93.4% | 73.2% | 24,170 parameters |

What it shows:
- **Label-free calibration recovers almost all the accuracy a camera change costs.** On
  average it matches a store-specific model that needed labelled data, with an artefact
  about 100× smaller.
- **It survives INT8.** For the quantized model, the store's statistics are folded into the
  weights before quantization. Because per-channel scales absorb those per-channel factors,
  the 8-bit integer weights come out the same for every store. Only the per-channel scales,
  biases and activation ranges differ, so the runtime must accept those as per-store
  parameters without recompiling.
- **It is not magic.** On the hardest store, a labelled fine-tune is still about 12 points
  better. A real deployment would flag low-confidence stores for a small labelled top-up
  rather than assume calibration is enough everywhere.

The fine-tune baseline uses one seed and fixed settings. Calibration images are training
digits the model already saw clean, now seen through the store's camera.

### 3. Offline-first — *Tested*, [`edge/buffer.py`](edge/buffer.py)

Events are written to a local SQLite file first and forwarded when the link is up.
- **Nothing is lost while there is space.** An event is deleted only after the server
  acknowledges it.
- **Time order is preserved.** Forwarding stops at the first failure.
- **When the disk budget is reached, the least important and oldest events go first:**
  per-frame metadata, then per-minute counts, and alerts last.
- **Eviction is decided before anything is deleted.** An event that can't fit without
  displacing something more important is refused, and the buffer is left untouched.
- **The device ignores a repeated event ID.**
- **The buffer survives a restart** (the file is reopened in the tests).

*Designed, not tested here:*
- surviving a real power cut, which relies on SQLite's write-ahead log
- server-side de-duplication on the same event IDs, which makes retries idempotent end to
  end

**72 hours offline — *Derived*,** [`sizing/edge_sizing.py`](sizing/edge_sizing.py). One
store with 12 cameras produces about 7.9 MiB an hour, from these assumed rates:
- a 120-byte summary per camera per second
- 400-byte per-zone counts per camera per minute
- about 20 alerts an hour, at 1 KB each
- plus an estimated 64 bytes of overhead per stored row

Seventy-two hours therefore needs about 570 MiB; budget 1.1 GiB for a 2× margin.

### 4. Staged updates with local rollback — *Tested*, [`fleet/rollout.py`](fleet/rollout.py)

A release goes out in waves (1%, 10%, 50%, 100%), and each wave must pass a health gate
before the next starts. The gate compares each device with itself before the update:
- it is healthy (no crash loop)
- throughput hasn't dropped more than 10%
- the detection rate hasn't drifted more than 15% — an accuracy proxy that needs no labels

Each device keeps the previous version in a second slot (A/B). A rollback is a local switch
that needs no download, so it works even if that store's link is down.

The tests cover four cases:
- **A release that regresses on one hardware revision** is caught in the 1% canary wave and
  rolled back.
- **A regression the canary missed** (one device in the 10% wave) is caught there, and every
  device that took the release rolls back, including the canary.
- **A device that was offline** never receives a rolled-back release when it reconnects.
- **The same device** does catch up to a completed release.

In the code, a rollback is a switch between two slots. That it needs no network on a real
device is a property of the A/B design, not something tested here. By design, a single
failing device rolls back the whole release, which is conservative.

### 5. Four models on one device — *Derived*, [`sizing/edge_sizing.py`](sizing/edge_sizing.py)

Four INT8 models run on each device:
- a person detector
- a shelf classifier
- a queue/zone counter
- a re-identification embedder

The method is to add up each model's memory, holding weights twice during an A/B update, and
its compute per camera, then keep 30% headroom.

With the stated assumptions (8 GB and 40 sustained INT8 TOPS per device, 12 cameras), compute
is the binding constraint at 37% utilisation; memory is at 14%. One device carries **22
cameras** with all four models, or 31 with the detector alone. The inputs are illustrative,
not vendor specifications; replace them with measured numbers for a real device.

### 6. Toolchain and accelerator choice — *Designed*

The same pipeline applies to any INT8 inference accelerator, for example the Qualcomm Cloud
AI 100 family. The steps are:
1. Export the model to ONNX.
2. Apply per-channel post-training quantization with the vendor's quantization toolkit,
   using real store images for calibration.
3. Compile for the target with the vendor SDK.
4. Validate accuracy and throughput on the device before any fleet rollout.

Choosing between accelerators is a total-cost decision, not a peak-TOPS one. It weighs:
- cost per camera served (decision 5)
- power and cooling in a store back room
- how many model variants must be maintained (decision 2 reduces that to one)
- the fleet tooling for unattended updates

None of that is run here.

## What runs here

| Component | Status | Where |
|---|---|---|
| Stand-in CNN with batch normalisation, trained on bundled digits | **Measured** | [`model/`](model) |
| BN folding, per-tensor vs per-channel INT8/INT4, channel-range sweep | **Measured** | [`quantize/`](quantize), [`results/quantization.json`](results/quantization.json) |
| Per-store BN calibration vs generic vs store-specific models | **Measured** | [`stores/`](stores), [`results/stores.json`](results/stores.json) |
| Offline store-and-forward buffer | **Tested** | [`edge/`](edge) |
| Staged rollout with A/B rollback | **Tested** | [`fleet/`](fleet) |
| Device packing and 72-hour buffer sizing | **Derived** | [`sizing/`](sizing), [`results/sizing.json`](results/sizing.json) |
| Accelerator toolchain, on-device validation, total-cost comparison | **Designed** (not run) | — |

20 tests in total.

## Run it

```bash
pip install -r requirements.txt
./run_all.sh          # experiments, sizing and tests (~1 minute, CPU only)
```

## Limitations

- **The stand-in data is easy.** 8×8 handwritten digits, not retail images, and no detection
  task; real accuracy and the size of real quantization losses must be measured on real
  models and data.
- **Store differences are simulated** as camera responses (gain, offset, noise, blur). Real
  stores also differ in layout, products and occlusion, which BN calibration does not fix.
- **Device sizing uses illustrative inputs,** not vendor specifications or measurements.
- **The rollout gate's thresholds are defaults.** In production they come from the variance
  of each metric across healthy devices.
- **A device that catches up after an outage is not health-gated** before it switches
  versions. A real fleet would gate it too.

## Author

Saurabh Rai — AI infrastructure solution architecture.
GitHub: [saurabh9498](https://github.com/saurabh9498) · LinkedIn: [saurabh-rai-aipm](https://www.linkedin.com/in/saurabh-rai-aipm/)
