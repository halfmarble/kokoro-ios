// swift-tools-version: 6.2
// MODIFIED BY HALFMARBLE LLC, 2026. Upstream: github.com/mlalma/kokoro-ios
// 1.0.11, MIT — that licence and its copyright notice travel with this copy in
// LICENSE, unchanged. This notice is courtesy, not obligation: MIT does not
// require it, and it is here because a reader deserves to know what moved.
//
// WHY THIS FORK EXISTS. Upstream's last commit was 2026-06-30 with 12 open
// PRs, including ones fixing the very problems patched below, so this is a
// long-lived fork rather than a staging area for patches about to land.
//
// Changes:
//   1. Resource directory renamed `Resources/` -> `KokoroData/`. iOS codesign
//      rejects a bundle whose top-level folder is literally `Resources/`
//      ("bundle format unrecognized") — upstream issue #31, open.
//   2. mlx-swift pin moved from exact 0.30.2 to exact 0.31.6. 0.30.2 fails to
//      LINK against the iOS 26 simulator SDK (undefined `_MTLTensorDomain` /
//      `_MTLIOErrorDomain`) — upstream issue #30, open. Note that mlx-swift
//      >= 0.31.5 ships a Linux-only `CudaBuild` plugin, so command-line builds
//      need `-skipPackagePluginValidation` (Xcode: one-time Trust & Enable).
//   3. `type: .dynamic` removed from the library product. Upstream's dynamic
//      product embedded MLX/MLXNN as frameworks while MLXLLM linked the same
//      modules statically — TWO MLX runtimes in one process, objc duplicate
//      class warnings at launch and crashes that were hard to attribute.
//      Upstream issue #26 reports this.
//   4. The MisakiSwift dependency points at halfmarble/MisakiSwift, which
//      carries the matching resource rename plus a G2P number fix. See below.
//
// NOT CHANGED, DELIBERATELY: the espeak-ng G2P path stays commented out, as
// upstream leaves it. espeak-ng is GPL-3.0 and enabling it would extend that
// licence's reach to everything linking this package. MisakiSwift is the only
// phonemizer here.
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "KokoroSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "KokoroSwift",
      targets: ["KokoroSwift"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    // .package(url: "https://github.com/mlalma/eSpeakNGSwift", from: "1.0.1"),
    // Halfmarble's fork of MisakiSwift, pinned by VERSION. Upstream's own tags
    // 1.0.0-1.0.6 are inherited by that fork and contain NONE of the changes
    // this package needs, so `from: "1.0.6"` would resolve to upstream's code
    // and silently drop the MisakiData/ rename, the static product, the
    // mlx-swift 0.31.6 pin and the (20, "twenty") number fix. 2.0.0 is the
    // fork's first own tag; the major bump reflects the rename and the change
    // in product linkage, both breaking against upstream 1.0.6.
    .package(url: "https://github.com/halfmarble/MisakiSwift.git", from: "2.0.0"),
    .package(url: "https://github.com/mlalma/MLXUtilsLibrary.git", exact: "0.0.6")
  ],
  targets: [
    .target(
      name: "KokoroSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        // BuildingBlocks/LayerNormInference.swift imports MLXFast (uses
        // MLXFast.layerNorm). Without this declaration, Xcode device builds
        // fail at module resolution; `swift build` on macOS may resolve via
        // transitive caching but iOS is strict.
        .product(name: "MLXFast", package: "mlx-swift"),
        // .product(name: "eSpeakNGLib", package: "eSpeakNGSwift"),
        .product(name: "MisakiSwift", package: "MisakiSwift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
      ],
      resources: [
       .copy("../../KokoroData/")
      ]
    ),
    .testTarget(
      name: "KokoroSwiftTests",
      dependencies: ["KokoroSwift"]
    ),
  ]
)
