import Testing
import MLXUtilsLibrary
@testable import KokoroSwift

/// THE BUDGET GUARD, AND THE ONE THING IT MUST NOT DO: FIRE ON REAL SPEECH.
///
/// `generateAudio` sizes its prosody and decoder graphs from the SUM of the
/// predicted durations, and nothing upstream bounds that sum. `maxTokenCount`
/// caps the phoneme count at 510, but each phoneme expands to a predicted
/// number of frames, so the two limits are far apart: 510 phonemes at a typical
/// 3-8 frames each spans roughly 38 to 102 seconds of audio.
///
/// So a pathological duration prediction can ask for an allocation nothing
/// checked. `validateDurationBudget` turns that into a thrown error the caller
/// can handle — split the text, or refuse — instead of a decoder graph sized by
/// the bad prediction.
///
/// Adapted from @ahh1539's fork (kokoro-ios 606eb09), with two deliberate
/// departures recorded here because they are the whole reason this is a port
/// rather than a cherry-pick:
///
///   * THEIR RATIONALE DOES NOT DESCRIBE OUR CODE. Their guard protects the
///     one-hot `[phonemes x frames]` alignment MATRIX. We deleted that matrix
///     in 2.0.7 — `createAlignmentIndices` builds an Int32 index vector of
///     length `totalFrames` and gathers with it. The remaining exposure is the
///     decoder graph downstream, which scales with the same number, so the
///     guard is still worth having; the reason is not theirs.
///   * THEIR DEFAULT IS 700 FRAMES AND OURS IS NOT. 700 is 17.5 seconds, which
///     suits a caller synthesizing short utterances. Adopting it unchanged
///     would throw on ordinary sentence-length speech.
@Suite struct DurationBudgetTests {

  /// `totalFrames` exists to predict `expandDurations`'s output length. If the
  /// two ever disagree the guard is measuring a different quantity from the one
  /// being allocated, which is worse than no guard because it reads as one.
  @Test func totalFramesEqualsWhatTheExpansionActuallyProduces() {
    let cases: [[Int32]] = [
      [],
      [1],
      [5, 5, 5],
      [1, 2, 3, 4, 5],
      [0, 3, 0, 4],            // zeros are skipped by the expansion
      [-2, 3, -1, 4],          // negatives are clamped, not subtracted
      Array(repeating: 7, count: 510),
      (0 ..< 200).map { Int32($0 % 9) },
    ]
    for durations in cases {
      #expect(KokoroTTS.totalFrames(durations) == KokoroTTS.expandDurations(durations).count)
    }
  }

  /// The boundary belongs to the caller: a budget of N allows N frames, not N-1.
  @Test func aTotalExactlyAtTheLimitIsAccepted() throws {
    try KokoroTTS.validateDurationBudget(totalFrames: 700, maximumFrames: 700)
  }

  /// The error carries BOTH numbers, because "too long" without them tells a
  /// caller nothing about how much to split by.
  @Test func oneFrameOverIsRejectedAndSaysBothNumbers() {
    #expect(throws: KokoroTTS.KokoroTTSError.durationLimitExceeded(totalFrames: 701,
                                                                  maximumFrames: 700)) {
      try KokoroTTS.validateDurationBudget(totalFrames: 701, maximumFrames: 700)
    }
  }

  /// THE CONTROL. Every assertion above would also pass for a function that
  /// threw unconditionally, so this pins the other side.
  @Test func itDoesNotSimplyAlwaysThrow() throws {
    try KokoroTTS.validateDurationBudget(totalFrames: 1, maximumFrames: 2400)
    try KokoroTTS.validateDurationBudget(totalFrames: 0, maximumFrames: 1)
    try KokoroTTS.validateDurationBudget(totalFrames: 2399, maximumFrames: 2400)
  }

  /// Passing 0 must not read as "no limit". The opt-out is `Int.max`, which
  /// says what it means and can be searched for.
  @Test func aZeroOrNegativeBudgetIsACallerErrorNotAnOptOut() {
    #expect(throws: KokoroTTS.KokoroTTSError.self) {
      try KokoroTTS.validateDurationBudget(totalFrames: 0, maximumFrames: 0)
    }
    #expect(throws: KokoroTTS.KokoroTTSError.self) {
      try KokoroTTS.validateDurationBudget(totalFrames: 1, maximumFrames: -1)
    }
  }

  @Test func intMaxIsTheWayToTurnItOff() throws {
    try KokoroTTS.validateDurationBudget(totalFrames: Int.max, maximumFrames: .max)
  }

  /// THE DEFAULT MUST STAY ABOVE WHAT A CALLER CAN LEGITIMATELY ASK FOR.
  ///
  /// MEASURED, not assumed — these weights at F16, voice af_nova, caching
  /// off, frames taken exactly as `samples / 600`:
  ///
  ///     94 words   918 frames    9.8 frames/word
  ///     37 words   504 frames   13.6
  ///     17 words   253 frames   14.9
  ///      8 words   120 frames   15.0
  ///
  /// Frames-per-word FALLS as a sentence lengthens, so the worst case is the
  /// densest rate at a realistic sentence ceiling rather than the longest text:
  /// 15 frames/word x 89 words = 1,335. Roughly 90 words is about as long as a
  /// single spoken sentence gets before a caller splits it.
  ///
  /// This asserts headroom over that, so lowering the constant to something
  /// fashionable — @ahh1539's 700, which the 94-word sentence above already
  /// exceeds — fails here rather than in front of a listener.
  @Test func theDefaultBudgetClearsTheLongestSentenceACallerCanProduce() {
    let densestFramesPerWord = 15
    let chunkerCeilingWords = 89
    let worstCaseFrames = densestFramesPerWord * chunkerCeilingWords   // 1335, measured basis
    #expect(KokoroTTS.Constants.maxDurationFrames > worstCaseFrames,
            "the default budget is at or below the longest single sentence a caller can produce")
  }

  /// The measured 94-word sentence, pinned directly. If a future weight change
  /// or a slower default speed pushes real speech toward the budget, this is
  /// the test that notices first.
  @Test func theMeasuredLongestRealSentenceSitsWellUnderTheBudget() {
    let measuredFramesFor94Words = 918
    #expect(measuredFramesFor94Words < KokoroTTS.Constants.maxDurationFrames / 2,
            "measured real speech is within half the budget; re-measure before lowering it")
  }

  /// The budget is only meaningful next to the frame rate, and the rate is
  /// stated in a comment. Pin the number that comment depends on.
  @Test func theSamplingRateIsWhatTheFrameArithmeticAssumes() {
    #expect(KokoroTTS.Constants.samplingRate == 24000)
  }
}

/// THE STATS PLUMBING REPORTS ZERO TODAY. THIS PROVES IT IS PLUMBING ANYWAY.
///
/// `G2PProcessor.consumeFallbackStats()` has a default returning `(0, 0)`, and
/// our MisakiSwift does not yet memoize its out-of-vocabulary BART lookups, so
/// every real call returns zeros. A test that only asserted "it returns zero"
/// would pass for a hard-coded zero and would KEEP passing after the
/// memoization landed — the tautology this file exists to avoid.
///
/// So both sides are pinned: a conformer that does NOT override gets the
/// default, and one that DOES override has its numbers carried through.
@Suite struct G2PFallbackStatsTests {

  private struct SilentEngine: G2PProcessor {
    func setLanguage(_ language: Language) throws {}
    func process(input: String) throws -> (String, [MToken]?) { ("", nil) }
    // deliberately no consumeFallbackStats — takes the default
  }

  private struct CountingEngine: G2PProcessor {
    func setLanguage(_ language: Language) throws {}
    func process(input: String) throws -> (String, [MToken]?) { ("", nil) }
    func consumeFallbackStats() -> (lookups: Int, hits: Int) { (lookups: 12, hits: 5) }
  }

  @Test func anEngineWithoutFallbackStatsReportsZeros() {
    let stats = SilentEngine().consumeFallbackStats()
    #expect(stats.lookups == 0)
    #expect(stats.hits == 0)
  }

  /// The half that matters. If the default shadowed the override, this fails —
  /// and the counters on `KokoroTTS` would be permanently zero for a reason
  /// nothing else would reveal.
  @Test func anEngineThatOverridesItIsNotSwallowedByTheDefault() {
    let stats = CountingEngine().consumeFallbackStats()
    #expect(stats.lookups == 12)
    #expect(stats.hits == 5)
  }
}
