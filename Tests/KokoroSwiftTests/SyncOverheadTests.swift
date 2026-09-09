import Testing
import MLX
@testable import KokoroSwift

// Host synchronisations removed from the synthesis path, after @ahh1539's fork.
// Each of these used to read a value back from the GPU that was already
// determined on the host before any GPU work began.

// MARK: - the masks

@Test func maskValuesReproducesTheGpuFormulaItReplaced() {
  // THE POINT. The old code built textMask as `index + 1 > inputLength` on the
  // GPU, read it back with asArray(Bool.self), and inverted it into the
  // attention mask. This asserts the pure-Swift construction is identical, for
  // a range of lengths — so the substitution is proven, not assumed.
  for length in [1, 2, 3, 8, 17, 64, 129] {
    // The formula, exactly as it used to be computed.
    let inputLengths = MLXArray(length)
    var expectedText = MLXArray(0 ..< length)
    expectedText = expectedText + 1 .> inputLengths
    let expectedTextSwift = expectedText.asArray(Bool.self)
    let expectedAttention = expectedTextSwift.map { !$0 ? 1 : 0 }

    let got = KokoroTTS.maskValues(inputLength: length)
    #expect(got.text == expectedTextSwift,
            "length \(length): text mask differs from the GPU formula")
    #expect(got.attention == expectedAttention,
            "length \(length): attention mask differs from the GPU formula")
  }
}

// MARK: - the duration expansion

@Test func expandDurationsRepeatsEachPhonemeItsDurationTimes() {
  #expect(KokoroTTS.expandDurations([3, 1, 4]) == [0, 0, 0, 1, 2, 2, 2, 2])
  #expect(KokoroTTS.expandDurations([1]) == [0])
  #expect(KokoroTTS.expandDurations([]) == [])
}

@Test func expandDurationsSkipsZeroLengthPhonemesRatherThanEmittingThem() {
  // A zero duration must contribute no frames — not one, and not a crash.
  #expect(KokoroTTS.expandDurations([2, 0, 3]) == [0, 0, 2, 2, 2])
  #expect(KokoroTTS.expandDurations([0, 0]) == [])
}

@Test func expandDurationsTotalMatchesTheSumOfDurations() {
  // The invariant the decoder depends on: one frame per unit of duration.
  let durations: [Int32] = [5, 2, 9, 1, 4, 3]
  let expanded = KokoroTTS.expandDurations(durations)
  #expect(expanded.count == durations.reduce(0) { $0 + Int($1) })
  // and monotonic non-decreasing, since frames run in phoneme order
  #expect(zip(expanded, expanded.dropFirst()).allSatisfy { $0 <= $1 })
}

@Test func theExpansionTestsCanFail() {
  // Control: if expandDurations returned its input unchanged, every assertion
  // above would still need to be false. This pins that it does not.
  #expect(KokoroTTS.expandDurations([3]) != [3])
}
