//
//  Kokoro-tts-lib
//
#if canImport(MisakiSwift)

import Foundation
import MisakiSwift
import MLXUtilsLibrary

/// A G2P processor that uses the MisakiSwift library for English phonemization.
/// Requires the MisakiSwift framework to be available at compile time.
final class MisakiG2PProcessor : G2PProcessor {
  /// The underlying MisakiSwift English G2P engine instance.
  /// This property is initialized when `setLanguage(_:)` is called and remains
  /// `nil` until the processor is properly configured.
  var misaki: EnglishG2P?
  
  /// Configures the processor for the specified language.
  /// - Parameter language: The target language for phonemization. Only `.enUS` and `.enGB` are supported.
  /// - Throws: `G2PProcessorError.unsupportedLanguage` if the language is not English (US or GB).
  func setLanguage(_ language: Language) throws {
    switch language {
    case .enUS:
      misaki = EnglishG2P(british: false)
    case .enGB:
      misaki = EnglishG2P(british: true)
    default:
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  /// Converts input text to phonetic representation.
  /// - Parameter input: The text string to be converted to phonemes.
  /// - Returns: A phonetic string representation of the input text and arrays of tokens.
  /// - Throws: `G2PProcessorError.processorNotInitialized` if `setLanguage(_:)` has not been called.
  func process(input: String) throws -> (String, [MToken]?) {
    guard let misaki else { throw G2PProcessorError.processorNotInitialized }
    return misaki.phonemize(text: input)
  }

  /// **RETURNS ZEROS UNTIL OUR MISAKI MEMOIZES ITS OOV LOOKUPS.** Deliberately
  /// left as the protocol default rather than reaching into `EnglishG2P`:
  /// halfmarble/MisakiSwift exposes no fallback statistics, so there is nothing
  /// to read and inventing a number here would be worse than reporting none.
  ///
  /// When it does, this becomes:
  ///
  ///     guard let misaki else { return (0, 0) }
  ///     let stats = misaki.consumeFallbackStats()
  ///     return (stats.lookups, stats.hits)
  ///
  /// which is how @ahh1539's fork reads it, against THEIR MisakiSwift fork at
  /// 1.0.12 — a different lineage from ours, so this is a port and not a pin
  /// bump.
}

#endif
