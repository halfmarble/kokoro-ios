# Kokoro TTS for Swift — halfmarble fork

> **What this is.** A fork of [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios)
> carrying the fixes we needed to ship this package in an iOS app, kept here so others
> can use them. Upstream has been quiet since January 2026 and has thirteen open issues
> and pull requests; several cover the same ground independently. Nothing here is novel
> — it is these fixes applied together and tested as a set, which no single upstream PR
> gives you.
>
> - **iOS codesign rejected the resource bundle** whose top-level folder is literally
>   named `Resources`, failing with "bundle format unrecognized". Renamed to
>   `KokoroData/`, with the one `Bundle.module` lookup updated to match. Reported
>   upstream as [#31](https://github.com/mlalma/kokoro-ios/issues/31).
> - **mlx-swift 0.30.2 will not link against the iOS 26 simulator SDK** — undefined
>   `_MTLTensorDomain` and `_MTLIOErrorDomain`. Pinned to 0.31.6. Reported upstream as
>   [#30](https://github.com/mlalma/kokoro-ios/issues/30); see also
>   [#28](https://github.com/mlalma/kokoro-ios/issues/28) on the exact pins. Note that
>   mlx-swift 0.31.5 and later ship a Linux-only `CudaBuild` plugin, so command-line
>   builds need `-skipPackagePluginValidation` and Xcode asks once to trust it.
> - **Two MLX runtimes in one process.** The library product's explicit
>   `type: .dynamic` embedded MLX and MLXNN as frameworks while packages such as MLXLLM
>   linked the same modules statically — objc duplicate class warnings at launch, and
>   crashes that were hard to attribute to it. The product now links statically like its
>   dependencies. Reported upstream as
>   [#26](https://github.com/mlalma/kokoro-ios/issues/26).
> - **The G2P dependency points at [halfmarble/MisakiSwift](https://github.com/halfmarble/MisakiSwift)
>   2.0.0**, which carries the matching resource rename and a number-to-words fix:
>   `(20, "twenty")` was missing, so every 21-29 was spoken as its units digit alone —
>   "24" as "four", and 2024 as "24" because four-digit tokens route through
>   `toYear()`. That fork's own tags start at 2.0.0; its inherited 1.0.x tags are
>   upstream's code and contain none of this.
>
> **The espeak-ng G2P path stays commented out**, as upstream leaves it. It is GPL-3.0,
> and enabling it would extend that licence to anything linking this package.
> MisakiSwift is the only phonemizer here.
>
> **What this is not.** Not a hostile fork, and not a claim that upstream is wrong.
> Everything here has been reported upstream, and if upstream merges these we would
> rather you used upstream.
>
> **Maintenance.** halfmarble maintains this fork and intends to keep fixing and
> extending it, because we ship it in production software — bugs here reach real users,
> so they get fixed here first. Issues and pull requests are welcome. We make no
> release-cadence or backwards-compatibility promise; pin a commit if you need one.
>
> MIT, same as upstream. The upstream copyright notice and licence travel unchanged in
> `LICENSE`; modified files carry a note saying what moved.


✨ *New in 1.0.8:* Added timestamps for each token. Please check [Kokoro Test App](https://github.com/mlalma/KokoroTestApp) how to use them.

✨ *New in 1.0.5:* Voice styles are moved out of the library to the integrating application. Please check [Kokoro Test App](https://github.com/mlalma/KokoroTestApp) how to use them.

Kokoro is a high-quality TTS (text-to-speech) model, providing faster than real-time English audio generation.

*NOTE:* This is a SPM package of the TTS engine. For an application integrating Kokoro and showing how the neural speech synthesis works, please see [KokoroTestApp](https://github.com/mlalma/KokoroTestApp) project.

Kokoro TTS port is based on the great work done in [MLX-Audio project](https://github.com/Blaizzy/mlx-audio), where the model was ported from PyTorch to MLX Python. This project ports the MLX Python code to MLX Swift.

Currently the library generates audio ~3.3 times faster than real-time on the release build on iPhone 13 Pro after warm up / first run.

## Requirements

- iOS 18.0+
- macOS 15.0+
- (Other Apple platforms may work as well)

## Installation

Add KokoroSwift to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/mlalma/kokoro-ios.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "KokoroSwift", package: "kokoro-ios")
    ]
)
```

## Usage

```swift
import KokoroSwift

// Initialize the TTS engine
let modelPath = URL(fileURLWithPath: "path/to/your/model")
let tts = KokoroTTS(modelPath: modelPath, g2p: .misaki)

// Generate speech
let voiceEmbedding = ... // See KokoroTestApp on how to get a voice style as an `MLXArray`
let text = "Hello, this is a test of Kokoro TTS."
let audioBuffer = try tts.generateAudio(voice: voiceEmbedding, language: .enUS, text: text)

// audioBuffer now contains the synthesized speech
```

## G2P (Grapheme-to-Phoneme) Options

- `.misaki` - MisakiSwift, default G2P processor
- `.espeak` - eSpeakNG, an alternative G2P processor (commented out in current version)

## Model Files

You'll need to provide your own Kokoro TTS model file due to its large size as well as voice style. Please see example project [Kokoro Test App](https://github.com/mlalma/KokoroTestApp) how they can be included as a part of the application package.

## Dependencies

This package depends on:
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's MLX framework for Swift
- [MisakiSwift](https://github.com/mlalma/MisakiSwift) - G2P processor
- [MLXUtilsLibrary](https://github.com/mlalma/MLXUtilsLibrary) - Utility library

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.