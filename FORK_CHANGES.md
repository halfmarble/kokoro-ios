# What this fork changes

Everything [halfmarble/kokoro-ios](https://github.com/halfmarble/kokoro-ios) carries on top of
[mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios), why it is here, and which release it
first shipped in. The README says what the fork *is*; this file tracks what is *in* it, and it
is updated in the same commit as the change it describes.

Nothing here is novel. Most entries have an upstream issue, or come from another public fork of
this package, and are credited where they do. The value is the set, applied together and tested
as a set — which no single upstream pull request gives you.

**Fork releases start at 2.0.0.** The inherited `1.0.x` tags are upstream's code and carry none
of this. Pin an exact version if you need one: there is no compatibility promise.

## Correctness

### The interpolation source index wrapped to the end of the input — since 2.0.3

`interpolate1d` clamped `xHigh` but neither end of `xLow`. Under `align_corners=false` the
source index is `x = (i + 0.5) * scale - 0.5`, so whenever `scale < 1` the leading output
samples have `x < 0`; `floor` gives -1, and **a negative fancy-index into an `MLXArray` wraps to
the last element rather than clamping to the first.**

`SineGen` upsamples phase by 300, so `x` stays negative for roughly the first 150 samples. Every
one of them borrowed the end of the input at exactly the point the phase cumsum begins, and a
cumsum carries that error through everything after it. Measured on `[0, 1, 2, 3, 1000]` upsampled
300x, the first output sample came back as **498.33** — 0.498 × 1000 — where it should be below 1.

Found by @antacosta on their fork. One deliberate difference: that fork keeps the unclamped
value for the fraction, which ramps from `input[0]` toward `input[1]` across the leading region.
PyTorch clamps the source index itself, holding `input[0]`, and these weights were validated
against that reference — so this matches PyTorch, and a test asserts the hold specifically. Both
remove the wraparound; only this reproduces the reference.

### Two audio guards carried from the upstream Python project — since 2.0.6

Both are confirmed in [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), the project this
package ports, as issues #815 and #803, fixed together in PR #814. Carried into this Swift port
by @antacosta and adopted here after checking that neither guard existed.

**The log-magnitude is clamped before `exp()`.** `convPost`'s raw output can reach magnitudes
near 1e11; `exp()` of that overflows to inf and propagates as NaN through the iSTFT. Short of
literal overflow, an unclamped large log-magnitude drives the waveform far outside audio range —
distortion rather than silence, which is the harder symptom to attribute.

**The bound matters more here than upstream, because these weights are F16.** float32 overflows
at ~3.4e38, so `exp()` survives a log-magnitude near 88; float16 overflows at 65504, which
`exp()` reaches at about 11.1 — roughly an eighth of the headroom. The test asserts that
`exp(bound)` stays under 65504 rather than restating the literal, so raising the bound past ~11
fails the suite instead of silently producing inf.

**`SineGen` is trimmed to a common length.** `_f02sine`'s downsample, cumsum and upsample round
trip is not strictly length-preserving and can return one `upsampleScale` hop more or fewer
samples than `uv`, which is computed from `f0` directly. The two are then multiplied, so a
mismatch either refuses to broadcast or silently misaligns harmonics from noise in time. The
tests sweep ten frame counts, because the mismatch appears only at some lengths and a single
size would pass with the bug present.

### A decoder graph is bounded by a duration budget — since 2.0.9

`generateAudio` sizes its prosody and decoder graphs from the **sum** of the predicted durations,
and nothing bounded that sum. `maxTokenCount` caps the phoneme count at 510, but each phoneme
expands to a predicted number of frames, so the two limits are far apart: 510 phonemes at a
typical 3 to 8 frames each spans roughly 38 to 102 seconds of audio.

`generateAudio` now takes `maximumDurationFrames`, defaulting to `Constants.maxDurationFrames`,
and throws `durationLimitExceeded` carrying both the total and the limit. The check runs as soon
as the durations reach the host, before anything is sized by them. A zero or negative budget
always throws rather than reading as "no limit"; `Int.max` is the opt-out.

Adapted from ahh1539's fork, with two deliberate departures. Their guard protects the one-hot
alignment matrix, which no longer exists here (see the gather change below), so the remaining
exposure is the decoder graph and the stated reason no longer applies. And their default is 700
frames, which throws on ordinary speech — measured with these weights at F16:

| words | frames | frames/word |
|---|---|---|
| 94 | 918 | 9.8 |
| 37 | 504 | 13.6 |
| 17 | 253 | 14.9 |
| 8 | 120 | 15.0 |

Frames per word **falls** as a sentence lengthens, so the worst case is the densest rate at a
realistic sentence ceiling rather than the longest text: 15 × 89 = 1,335. The default is 2400,
1.8x above that.

## Performance

Each of these removes work from the synthesis path. Where a number is quoted it was measured on
a device, and the measurement conditions are stated with it.

### Aligned frames are selected by gather, not by a one-hot matmul — since 2.0.4

`predictDurations` built a `[phonemes x frames]` one-hot alignment matrix and two call sites
multiplied by it. A one-hot matmul is a gather written the long way:
`matmul(X, onehot)[:, :, f]` is exactly `X[:, :, indices[f]]`. Removing it deletes three costs,
and the third is the one that matters:

- a `phonemes * frames` Float allocation on the host, per synthesis;
- two matmuls over that matrix, per synthesis;
- **an `.item()` call per frame** in the loop that filled it. Each is a GPU-to-CPU
  synchronisation, so a sentence of several hundred frames paid several hundred round trips
  before any audio existed.

Suggested by @antacosta's fork, which replaced the same two matmuls after diagnosing MLX's
matmul as returning values unrelated to the selected column, and blaming it for buzzy audio.
**That correctness failure did not reproduce here, and the tests say so rather than assuming**:
matmul and gather agree to within 1e-4 both at toy size and at production shape, and a second
assertion checks the gather against a hand-computed selection so the pair cannot agree by being
wrong in the same way. The change is taken for the cost above, not as a fix for audio this build
does not exhibit — and it removes the exposure either way.

### ALBERT's LayerNorm weights load as arrays — since 2.0.5

The three ALBERT initialisers built an MLXNN LayerNorm from fresh ones and zeros, then overwrote
it element by element — `hiddenSize * 2` subscript assignments per site, **4,608 in total**, each
its own graph op, paid every time the engine is constructed. `LayerNormInference` already exists
here to take the arrays directly, and wraps the same `MLXFast.layerNorm` that MLXNN's LayerNorm
calls, so the maths is unchanged.

Measured on an iPhone 16 Pro Max, three launches per arm with both paths in one build:
**KokoroTTS construction 345 ms → 291 ms.**

### Three host synchronisations removed — since 2.0.7

Each read back a value that was fully determined before any GPU work began.

`prepareInputTensors` built both masks on the GPU and read them back — `inputLengths.max().item()`
and `textMask.asArray(Bool.self)`. Each invocation carries one sequence padded to its own length,
so there are no padding positions: the condition is false everywhere and the attention mask is
its inverse. Both are now constructed directly, and the values are asserted against the old GPU
formula across seven lengths, so the substitution is proven rather than claimed.

`createAlignmentIndices` called `.item()` **once per phoneme** and handed that many
single-element arrays to `MLX.concatenated`. The durations are needed on the host either way, so
one bulk `asArray` costs one sync instead of N and the expansion is a Swift append.

`generateAudio` takes `predictTimestamps`, defaulting to true so existing callers are unaffected.
The predictor reads durations back to the host and cannot affect the samples — it only mutates
the token array — so a caller that discards the tokens can skip it. Only two host reads remain in
the path, and both are necessary: materialising the samples, and the single bulk duration read.

Adapted from @ahh1539's fork, which made the same three changes; the alignment part is reworked
because this port already selects frames by gather.

### An identity phase unwrap dropped, and the Hann window built once — since 2.0.8

`MLXSTFT.inverse` ran a full phase unwrap — several MLX ops and a cumsum over the whole phase
array, once per batch item per synthesis — that **provably returned its argument**. `unwrap`
zeroes its correction wherever the difference is under pi; `inverse` is called from exactly one
place, and that phase is `MLX.sin(...)`, so it lies in [-1, 1] and consecutive differences are at
most 2. The correction was zero everywhere.

`UnwrapIsIdentityTests` keeps the removed algorithm and asserts that over swept shapes and
frequencies, so the claim stays checkable after the code is gone. A control feeds it a phase with
jumps larger than pi and asserts it does **not** return its input — without that the identity
result would be vacuous, and it also names the condition under which the unwrap would have to
come back.

The Hann window depends only on `winLength`, so it is built once in `init` rather than recomputed
inside every `mlxIstft` call. Suggested by @ahh1539's fork; only these two changes are taken from
it, since the same commit also hoisted a weight-norm they later reverted.

## API additions

Both are additive. Existing callers are unchanged.

### `preloadG2P` — since 2.0.2

`setLanguage` is reached only from `generateAudio`, so the G2P engine — and with MisakiSwift the
gold and silver dictionaries it loads and grows — is built inside whichever synthesis runs first.
That call pays a one-time cost the others do not: measured on an iPhone, phonemization is **~130
ms on the first call and ~1 ms once the lexicon exists.**

An application that warms up in the background could not place that cost, because the only door
into `setLanguage` was a synthesis; when the warm-up is skipped or loses the race, the cost lands
on a user-facing call instead. `preloadG2P` is a public wrapper over the existing
`updateLanguageIfNeeded`, so there is no second code path to drift. It does not make the cost
smaller, and it builds only the language passed.

### G2P fallback statistics — since 2.0.9

`consumeFallbackStats()` on `G2PProcessor`, with a default returning zeros, plus
`lastG2PFallbackLookups` and `lastG2PFallbackHits` on `KokoroTTS`. These report zero until a G2P
memoizes its out-of-vocabulary lookups — halfmarble/MisakiSwift 2.2.1 does — so the wiring is one
override rather than a change here. Tests pin both sides, so they will not keep passing silently
once real numbers arrive.

## Profiling

### Per-stage synthesis timers, behind a flag — since 2.0.1

Ten stage timers. Five — Phonemize, BERT, Duration, Prosody, Decoder — were already declared
upstream as constants and never wired to anything; five are new, added so the parts account for
the whole.

**`profileStages` is off by default and forces `MLX.eval` at eight stage boundaries.** Without it
the numbers are not wrong so much as meaningless: MLX is lazy, so each timer measures graph
*construction* and the real work lands on whichever later line first demands values. On one
workload that made the decoder look like 4% of a synthesis; under eval it is 146 ms of 238 at
steady state, **61%**, the largest single stage. ALBERT is 18%, not the 2 to 6% the lazy numbers
suggested.

Two details that look like tidying and are not. **Materialise nests inside the total** —
`audio[0].asArray(Float.self)` is where deferred GPU work is forced, so it is 47 to 68% of a
synthesis, and stopping the total before timing it would exclude the single line most likely to
hold the missing time. And **a profiled total is not a driver's total**: forcing eval prevents
kernel fusion, so the two numbers measure different things and must not be quoted together. On
the workload used here they happened to agree, 244 ms lazy against 238 profiled, but that is a
coincidence of the input rather than a property of the flag.

A later change splits ALBERT's attention into three timers under the same flag, which answers a
question that otherwise gets argued: at 85 characters the part a fused attention kernel would
replace is 3.3% of the BERT stage and 0.45% of a synthesis, so the hand-written attention is not
worth replacing. Without the split, the whole BERT stage reads as 18% and looks like a target.

## Packaging and build

All of these are in **2.0.0**, and all are what it takes to ship this package in an iOS app.

- **iOS codesign rejected the resource bundle** whose top-level folder is literally named
  `Resources` — "bundle format unrecognized". Renamed to `KokoroData/`, with the one
  `Bundle.module` lookup updated to match. Upstream
  [#31](https://github.com/mlalma/kokoro-ios/issues/31).
- **mlx-swift pinned to 0.31.6.** 0.30.2 does not link against the iOS 26 simulator SDK —
  undefined `_MTLTensorDomain` and `_MTLIOErrorDomain`. Upstream
  [#30](https://github.com/mlalma/kokoro-ios/issues/30); see also
  [#28](https://github.com/mlalma/kokoro-ios/issues/28) on the exact pins.
- **The library product links statically.** Its explicit `type: .dynamic` embedded MLX and MLXNN
  as frameworks while packages such as MLXLLM linked the same modules statically — two MLX
  runtimes in one process, duplicate Objective-C class warnings at launch, and crashes that were
  hard to attribute to it. Upstream
  [#26](https://github.com/mlalma/kokoro-ios/issues/26).
- **`MLXFast` declared as a target dependency.** `BuildingBlocks/LayerNormInference.swift` imports
  it and calls `MLXFast.layerNorm(...)`, but the manifest never declared it. macOS `swift build`
  resolves it through transitive caching; an Xcode iOS device build fails at module resolution
  with "No such module 'MLXFast'".
- **The G2P dependency points at
  [halfmarble/MisakiSwift](https://github.com/halfmarble/MisakiSwift)**, which carries the
  matching resource rename and its own pronunciation fixes. That fork's releases also start at
  2.0.0, for the same reason.

Consuming this package: mlx-swift 0.31.5 and later ship a Linux-only `CudaBuild` plugin, so
command-line builds need `-skipPackagePluginValidation`, and Xcode asks once to trust it.

## What is deliberately not changed

**The espeak-ng G2P path stays commented out, as upstream leaves it.** It is GPL-3.0, and
enabling it would extend that licence to anything linking this package. MisakiSwift is the only
phonemizer here.

## Tests and CI

The suite runs `xcodebuild test` on a `macos-26` Apple Silicon runner, added in 2.0.1 — not
`swift test`, which cannot compile the Metal shaders mlx-swift needs and so aborts on a missing
`default.metallib` **while still exiting 0**. `macos-15` is too old: its default Xcode is Swift
6.1 and this package declares tools 6.2. `-skipPackagePluginValidation` is needed for the
`CudaBuild` plugin a fresh runner cannot approve.

Upstream's suite was a single test with an empty body, so a green run proved only that the
package compiles. That is worth having and it is not a test. Every change above ships with one,
and several ship with a **control** — an assertion that fails if the test's own premise stops
holding, so a fix cannot pass vacuously.

## Release index

| version | what it added |
|---|---|
| 2.0.0 | iOS packaging: bundle rename, mlx-swift pin, static linking, `MLXFast` declared, G2P repointed |
| 2.0.1 | Per-stage timers behind `profileStages`; CI |
| 2.0.2 | `preloadG2P` |
| 2.0.3 | Interpolation source index clamped at zero |
| 2.0.4 | Aligned frames selected by gather |
| 2.0.5 | ALBERT LayerNorm weights loaded as arrays |
| 2.0.6 | Log-magnitude clamp; `SineGen` length trim |
| 2.0.7 | Three host synchronisations removed; `predictTimestamps` |
| 2.0.8 | Identity phase unwrap dropped; Hann window built once |
| 2.0.9 | Decoder duration budget; G2P fallback statistics |

## Licence

MIT, same as upstream. The upstream copyright notice and licence travel unchanged in `LICENSE`,
and modified files carry a note saying what moved.
