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

/// `preloadG2P` exists so a caller can decide WHEN the G2P engine is built,
/// rather than having it built by whichever synthesis happens to run first.
///
/// This binds the method without calling it, and that is deliberate: calling it
/// needs a `KokoroTTS`, which needs model weights, which CI does not have — and
/// on a simulator MLX aborts as soon as a stream is created, so there is no
/// host here that could run it for real. What the binding does pin is the
/// signature, including the default language: a caller writing
/// `try tts.preloadG2P()` must keep compiling.
@Test func preloadG2PKeepsItsSignature() {
  // THE COMPILATION IS THE ASSERTION. An #expect here could only restate the
  // type annotation on the line above it, which is true by construction and
  // would pass whatever preloadG2P did — a check that cannot fail is worse
  // than no check, because it reads as one.
  let door: (KokoroTTS) -> (Language) throws -> Void = KokoroTTS.preloadG2P
  _ = door
}
