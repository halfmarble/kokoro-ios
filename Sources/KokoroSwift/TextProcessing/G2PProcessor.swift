//
//  Kokoro-tts-lib
//
import Foundation
import MLXUtilsLibrary

/// Errors that can occur during G2P (grapheme-to-phoneme) processing.
enum G2PProcessorError : Error {
  /// The processor has not been initialized with a language.
  case processorNotInitialized
  /// The requested language is not supported by this processor.
  case unsupportedLanguage
}

/// Protocol defining the interface for grapheme-to-phoneme processors.
/// G2P processors convert written text (graphemes) into phonetic representations (phonemes)
/// for use in text-to-speech synthesis. Different implementations may use different
/// underlying engines or libraries.
protocol G2PProcessor {
  /// Configures the processor for a specific language.
  /// - Parameter language: The target language for phonemization.
  /// - Throws: `G2PProcessorError.unsupportedLanguage` if the language is not supported.
  func setLanguage(_ language: Language) throws
  
  /// Converts input text to phonetic representation.
  /// - Parameter input: The text string to be converted to phonemes.
  /// - Returns: A phonetic string representation of the input text and optionally arrays of tokens.
  /// - Throws: `G2PProcessorError.processorNotInitialized` if `setLanguage(_:)` has not been called.
  func process(input: String) throws -> (String, [MToken]?)

  /// Out-of-vocabulary fallback lookups and hits for the most recent `process`.
  ///
  /// Consuming: each call reports the counts since the previous call and resets
  /// them, so a caller reading once per `process` sees that call's figures
  /// rather than a running total.
  func consumeFallbackStats() -> (lookups: Int, hits: Int)
}

extension G2PProcessor {
  /// **THE DEFAULT IS ZEROS, AND MOST ENGINES WILL KEEP IT.** An engine with no
  /// OOV fallback, or one whose fallback is not memoized, has nothing to
  /// report. Returning zeros rather than making this a required method means
  /// adding the memoization later is a single override.
  func consumeFallbackStats() -> (lookups: Int, hits: Int) { (0, 0) }
}
