import Testing
import MLX
@testable import KokoroSwift

// interpolate1d's linear branch clamped xHigh but not xLow. Under
// align_corners=false the source index is x = (i + 0.5)*scale - 0.5, so
// upsampling by a large factor leaves the leading output samples with x < 0 —
// floor gives -1, and a negative fancy-index into MLX wraps to the LAST
// element instead of clamping to the first. SineGen upsamples phase by 300, so
// this injects the final sample's value at the very point the phase cumsum
// begins, and a cumsum propagates it through everything after.
//
// Found by @antacosta on their fork of this package.

@Test func leadingSamplesDoNotWrapToTheEndOfTheInput() {
  // A ramp whose LAST value is wildly unlike its first. If the leading output
  // samples borrow from the end, they land near 1000 instead of near 0.
  let input = MLXArray(converting: [0.0, 1.0, 2.0, 3.0, 1000.0] as [Double])
    .reshaped([1, 1, 5])
  let out = interpolate1d(input: input, size: 5 * 300, mode: "linear", alignCorners: false)
  let first = out[0, 0, 0].item(Float.self)

  // Between input[0]=0 and input[1]=1, so at most 1. The wraparound produced a
  // ~0.5 blend with 1000, i.e. ~498.
  #expect(first <= 1.0 + 1e-3,
          "first output sample is \(first) — it borrowed from the end of the input")
}

@Test func theWholeLeadingRegionStaysInRange() {
  // x < 0 for i = 0..149 at scale 1/300, so the bug corrupts a BLOCK of
  // samples, not just one. Every one of them must lie between input[0] and
  // input[1].
  let input = MLXArray(converting: [0.0, 1.0, 2.0, 3.0, 1000.0] as [Double])
    .reshaped([1, 1, 5])
  let out = interpolate1d(input: input, size: 5 * 300, mode: "linear", alignCorners: false)
  var worst: Float = 0
  for i in 0 ..< 150 { worst = max(worst, out[0, 0, i].item(Float.self)) }
  #expect(worst <= 1.0 + 1e-3, "worst leading sample is \(worst), expected <= 1")
}

@Test func interiorInterpolationIsUnchanged() {
  // The control: the fix must not disturb the ordinary interior case, which
  // never had a negative index. Halfway between input[1]=10 and input[2]=20.
  let input = MLXArray(converting: [0.0, 10.0, 20.0, 30.0] as [Double]).reshaped([1, 1, 4])
  let out = interpolate1d(input: input, size: 8, mode: "linear", alignCorners: false)
  // i=3 -> x = (3+0.5)*0.5 - 0.5 = 1.25, so 0.75*input[1] + 0.25*input[2] = 12.5
  #expect(abs(out[0, 0, 3].item(Float.self) - 12.5) < 1e-3,
          "interior sample changed: \(out[0, 0, 3].item(Float.self))")
}

@Test func theLeadingRegionMatchesTheReferenceAndHoldsTheFirstSample() {
  // This is the assertion that distinguishes our fix from the one on the fork
  // where the bug was found. PyTorch's area_pixel_compute_source_index returns
  // 0 for a negative source index (non-cubic), so the samples before the first
  // input HOLD input[0]. The alternative — keeping the unclamped fraction —
  // ramps toward input[1] instead. Both are free of the wraparound; only this
  // one reproduces the reference the weights were validated against.
  let input = MLXArray(converting: [5.0, 100.0, 200.0] as [Double]).reshaped([1, 1, 3])
  let out = interpolate1d(input: input, size: 3 * 300, mode: "linear", alignCorners: false)
  #expect(abs(out[0, 0, 0].item(Float.self) - 5.0) < 1e-4,
          "leading sample is \(out[0, 0, 0].item(Float.self)), expected to hold input[0] = 5")
}
