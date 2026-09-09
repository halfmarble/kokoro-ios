import Testing
import MLX
import Foundation
@testable import KokoroSwift

// Two upstream-confirmed defects ported from Blaizzy/mlx-audio (#803 and #815,
// fixed together in PR #814), adopted here after @antacosta carried them into
// this Swift port.

// MARK: - #815, log-magnitude overflow

@Test func theLogMagnitudeClampKeepsExpInsideFloat16() {
  // THE POINT OF THIS TEST. These weights are F16. float32 overflows at
  // ~3.4e38 so exp() survives a log-magnitude near 88; float16 overflows at
  // 65504, which exp() reaches at about 11.1. A bound that is safe upstream
  // can therefore be unsafe here, and this asserts the margin instead of
  // trusting the number in the source.
  let bound = Generator.logMagnitudeClamp
  let peak = Foundation_exp(Double(bound))
  #expect(peak < 65504.0,
          "exp(\(bound)) = \(peak) exceeds float16's 65504 — the clamp does not protect an F16 build")
}

@Test func clampingBoundsAnOtherwiseOverflowingMagnitude() {
  // The mechanism, on the value the upstream issue actually reports.
  let hostile = MLXArray(converting: [1e11, -1e11, 3.0] as [Double])
  let clamped = MLX.clip(hostile, min: -Generator.logMagnitudeClamp, max: Generator.logMagnitudeClamp)
  let out = MLX.exp(clamped).asArray(Float.self)
  for v in out { #expect(v.isFinite, "exp produced \(v) after clamping") }
  #expect(out.max()! <= Foundation_expF(Generator.logMagnitudeClamp) + 1e-2)
}

@Test func withoutTheClampItReallyDoesOverflow() {
  // The control. If exp(1e11) were finite, the two tests above would be
  // guarding against nothing and would pass whatever the clamp did.
  let unclamped = MLX.exp(MLXArray(converting: [1e11] as [Double])).asArray(Float.self)
  #expect(!unclamped[0].isFinite,
          "exp(1e11) came back finite as \(unclamped[0]) — this defect cannot occur and the clamp is pointless")
}

// MARK: - #803, SineGen length mismatch

@Test func sineGenReturnsConsistentLengthsAcrossManyFrameCounts() {
  // _f02sine downsamples, cumsums and upsamples, and that round trip is not
  // strictly length-preserving — it can return one upsampleScale hop more or
  // fewer samples than uv, which is computed from f0 directly. The three
  // returned signals are multiplied together, so any disagreement either
  // refuses to broadcast or silently misaligns harmonics from noise in time.
  //
  // Sweeping the frame count is the point: the mismatch appears only at
  // certain lengths, so a single size would pass while the bug was present.
  let gen = SineGen(sampRate: 24000, upsampleScale: 300, harmonicNum: 8)
  for frames in [7, 8, 9, 16, 31, 32, 33, 64, 97, 128] {
    let f0 = MLXArray(converting: (0 ..< frames).map { 100.0 + Double($0 % 5) * 20.0 })
      .reshaped([1, frames, 1])
    let (result, uv, noise) = gen(f0)
    #expect(result.shape[1] == uv.shape[1],
            "frames=\(frames): result \(result.shape) vs uv \(uv.shape)")
    #expect(result.shape[1] == noise.shape[1],
            "frames=\(frames): result \(result.shape) vs noise \(noise.shape)")
    let flat = result.asArray(Float.self)
    #expect(flat.allSatisfy { $0.isFinite }, "frames=\(frames): non-finite sample in the output")
  }
}

private func Foundation_exp(_ x: Double) -> Double { Foundation.exp(x) }
private func Foundation_expF(_ x: Float) -> Float { Float(Foundation.exp(Double(x))) }
