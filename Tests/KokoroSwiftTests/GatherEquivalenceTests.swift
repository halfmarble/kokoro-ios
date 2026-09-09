import Testing
import MLX
@testable import KokoroSwift

// A one-hot matmul IS a gather. `createAlignmentTarget` builds a
// [phonemes x frames] matrix with exactly one 1.0 per column, so
// matmul(X, onehot)[:, :, f] == X[:, :, indices[f]] by construction.
//
// @antacosta reports that on their device MLX's matmul returns values with no
// relationship to any single selected column for this shape, and blames it for
// buzzy audio. This test asks that question directly on OUR machine: build both
// and compare. Either answer is useful — agreement means the gather is a free
// speed win, disagreement confirms a real MLX defect we are routing around.

@Test func aOneHotMatmulEqualsAGather() {
  let phonemes = 7
  let channels = 5
  let perPhoneme = [3, 1, 4, 2, 1, 5, 2]          // durations
  var indices: [Int32] = []
  for (p, d) in perPhoneme.enumerated() { indices += Array(repeating: Int32(p), count: d) }
  let frames = indices.count

  // X: [1, channels, phonemes], distinct values so any mis-selection shows.
  var xs = [Float](repeating: 0, count: channels * phonemes)
  for c in 0 ..< channels { for p in 0 ..< phonemes { xs[c * phonemes + p] = Float(c * 100 + p) } }
  let x = MLXArray(xs).reshaped([1, channels, phonemes])

  // The one-hot, built exactly as createAlignmentTarget does.
  var onehot = [Float](repeating: 0, count: phonemes * frames)
  for f in 0 ..< frames { onehot[Int(indices[f]) * frames + f] = 1.0 }
  let target = MLXArray(onehot).reshaped([phonemes, frames]).expandedDimensions(axis: 0)

  let viaMatmul = MLX.matmul(x, target)
  let viaGather = x.take(MLXArray(indices), axis: 2)

  #expect(viaMatmul.shape == viaGather.shape,
          "shape differs: \(viaMatmul.shape) vs \(viaGather.shape)")

  let m = viaMatmul.asArray(Float.self)
  let g = viaGather.asArray(Float.self)
  var worst: Float = 0
  for i in 0 ..< min(m.count, g.count) { worst = max(worst, abs(m[i] - g[i])) }
  #expect(worst < 1e-4, "matmul and gather disagree by \(worst) — MLX matmul is wrong here")

  // And independently: the gather must equal the hand-computed selection, so
  // this test cannot pass by both paths being wrong in the same way.
  var expected = [Float](repeating: 0, count: channels * frames)
  for c in 0 ..< channels { for f in 0 ..< frames { expected[c * frames + f] = Float(c * 100 + Int(indices[f])) } }
  var worstG: Float = 0
  for i in 0 ..< expected.count { worstG = max(worstG, abs(g[i] - expected[i])) }
  #expect(worstG < 1e-4, "gather does not match the hand-computed selection, off by \(worstG)")
}

@Test func aOneHotMatmulEqualsAGatherAtProductionScale() {
  // The small case above may miss a shape-dependent defect. Kokoro's real
  // shapes: textEncoding is [1, 512, phonemes] and a sentence runs to hundreds
  // of frames. This is the size the reported bug would actually occur at.
  let phonemes = 64
  let channels = 512
  var indices: [Int32] = []
  for p in 0 ..< phonemes { indices += Array(repeating: Int32(p), count: 9) }  // 576 frames
  let frames = indices.count

  var xs = [Float](repeating: 0, count: channels * phonemes)
  for c in 0 ..< channels {
    for p in 0 ..< phonemes { xs[c * phonemes + p] = Float((c * 31 + p * 7) % 1000) / 7.0 - 50 }
  }
  let x = MLXArray(xs).reshaped([1, channels, phonemes])

  var onehot = [Float](repeating: 0, count: phonemes * frames)
  for f in 0 ..< frames { onehot[Int(indices[f]) * frames + f] = 1.0 }
  let target = MLXArray(onehot).reshaped([phonemes, frames]).expandedDimensions(axis: 0)

  let m = MLX.matmul(x, target).asArray(Float.self)
  let g = x.take(MLXArray(indices), axis: 2).asArray(Float.self)
  var worst: Float = 0
  for i in 0 ..< min(m.count, g.count) { worst = max(worst, abs(m[i] - g[i])) }
  #expect(worst < 1e-3,
          "at [1,\(channels),\(phonemes)] x [\(phonemes),\(frames)] matmul and gather disagree by \(worst)")
}
