import Testing
import MLX
import Foundation
@testable import KokoroSwift

// `MLXSTFT.inverse` used to run a phase unwrap that was the identity on every
// input it ever received. This file keeps the removed algorithm so the claim
// stays checkable, and names the condition under which it would stop holding.
//
// Removal suggested by @ahh1539's fork.

/// The algorithm exactly as it stood before removal.
private func removedUnwrap(_ p: MLXArray) -> MLXArray {
  let period: Float = 2.0 * .pi
  let discont: Float = period / 2.0
  let pDiff1 = p[0..., 0 ..< p.shape[1] - 1]
  let pDiff2 = p[0..., 1 ..< p.shape[1]]
  let pDiff = pDiff2 - pDiff1
  let intervalHigh: Float = period / 2.0
  let intervalLow: Float = -intervalHigh
  var pDiffMod = pDiff - intervalLow
  pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow
  let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)
  pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)
  var phCorrect = pDiffMod - pDiff
  phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)
  return MLX.concatenated([p[0..., 0 ..< 1], p[0..., 1...] + phCorrect.cumsum(axis: 1)], axis: 1)
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
  let x = a.asArray(Float.self), y = b.asArray(Float.self)
  var worst: Float = 0
  for i in 0 ..< min(x.count, y.count) { worst = max(worst, abs(x[i] - y[i])) }
  return worst
}

@Test func unwrapWasTheIdentityOnSinBoundedPhase() {
  // The phase reaching `inverse` is `MLX.sin(...)` from Generator, so it lies in
  // [-1, 1] and consecutive differences are at most 2, which is below pi. Sweep
  // shapes and frequencies so this is not one lucky array.
  for frames in [8, 33, 128] {
    for freq in [0.5, 3.0, 17.0] as [Double] {
      var vals: [Double] = []
      for i in 0 ..< (frames * 4) { vals.append(Foundation.sin(Double(i) * freq)) }
      let p = MLXArray(converting: vals).reshaped([4, frames])
      let out = removedUnwrap(p)
      #expect(maxAbsDiff(out, p) < 1e-5,
              "frames \(frames), freq \(freq): the unwrap changed sin-bounded input by \(maxAbsDiff(out, p))")
    }
  }
}

@Test func unwrapWasNOTTheIdentityOnUnboundedPhase() {
  // THE CONTROL, and the condition that would make the removal wrong. An
  // atan2-style phase wrapped into [-pi, pi] has jumps larger than pi, and there
  // the unwrap does real work. If this ever stops failing to match, the test
  // above proves nothing.
  let vals: [Double] = [0.0, 3.0, -3.0, 3.0, -3.0, 0.0, 3.0, -3.0]
  let p = MLXArray(converting: vals).reshaped([1, vals.count])
  let out = removedUnwrap(p)
  #expect(maxAbsDiff(out, p) > 1.0,
          "the unwrap left a phase with jumps larger than pi unchanged, so it does nothing at all and the identity test above is vacuous")
}
