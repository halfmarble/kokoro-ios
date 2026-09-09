import Testing
@testable import KokoroSwift

@Test func exampleTest() async throws {
}

/// `profileStages` forces `MLX.eval` at eight stage boundaries, which prevents
/// kernel fusion and makes synthesis slower. It is a diagnostic, so shipping it
/// enabled would slow every synthesis for every user while the numbers it
/// produces are not comparable to a normal run anyway. Default-off is the whole
/// safety property, and a default is exactly the kind of thing that flips during
/// a debugging session and does not flip back.
@Test func profileStagesIsOffByDefault() {
  #expect(KokoroTTS.profileStages == false)
}
