//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

/// Main class that encapsulates the complete Kokoro text-to-speech pipeline.
///
/// KokoroTTS converts text input into audio output by:
/// 1. Processing text through grapheme-to-phoneme (G2P) conversion
/// 2. Encoding the phonemes using BERT-based embeddings
/// 3. Predicting duration and prosody for natural speech
/// 4. Generating audio through a decoder network
///
/// Example usage:
/// ```swift
/// let tts = KokoroTTS(modelPath: modelURL, g2p: .misaki)
/// let audioData = try tts.generateAudio(voice: voiceEmbedding,
///                                       language: .english,
///                                       text: "Hello world",
///                                       speed: 1.0)
/// ```
public final class KokoroTTS {
  /// Errors from the TTS side
  public enum KokoroTTSError: Error {
    /// Thrown when input text exceeds maximum token count
    case tooManyTokens
  }
  
  /// BERT model for encoding phoneme sequences
  private let bert: CustomAlbert!
  
  /// Linear layer to project BERT embeddings
  private let bertEncoder: Linear!
  
  /// Encoder for duration prediction features
  private let durationEncoder: DurationEncoder!
  
  /// Bidirectional LSTM for duration prediction
  private let predictorLSTM: LSTM!
  
  /// Projection layer for final duration values
  private let durationProj: Linear!
  
  /// Predictor for prosodic features (F0, pitch)
  private let prosodyPredictor: ProsodyPredictor!
  
  /// Text encoder that processes phoneme sequences
  private let textEncoder: TextEncoder!
  
  /// Decoder that generates audio from encoded features
  private let decoder: Decoder!
  
  /// Grapheme-to-phoneme processor for text conversion
  private let g2pProcessor: G2PProcessor?
  
  /// Currently active language (cached to avoid reinitializing G2P)
  private var chosenLanguage: Language = .none
  
  /// Initializes the Kokoro TTS engine with model weights and G2P processor.
  /// - Parameters:
  ///   - modelPath: URL to the directory containing model weights
  ///   - g2p: Grapheme-to-phoneme processor type (default: Misaki)
  public init(modelPath: URL, g2p: G2P = .misaki) {
    // Load and sanitize model weights
    let sanitizedWeights = WeightLoader.loadWeights(modelPath: modelPath)
    let config = KokoroConfig.loadConfig()
    
    // Initialize BERT model for phoneme encoding
    bert = CustomAlbert(
      weights: sanitizedWeights,
      config: AlbertModelArgs(
        numHiddenLayers: config.plbert.numHiddenLayers,
        numAttentionHeads: config.plbert.numAttentionHeads,
        hiddenSize: config.plbert.hiddenSize,
        intermediateSize: config.plbert.intermediateSize,
        vocabSize: config.nToken
      )
    )
    
    // Initialize BERT output encoder
    bertEncoder = Linear(
      weight: sanitizedWeights["bert_encoder.weight"]!,
      bias: sanitizedWeights["bert_encoder.bias"]!
    )
    
    // Initialize duration prediction components
    durationEncoder = DurationEncoder(
      weights: sanitizedWeights,
      dModel: config.hiddenDim,
      styDim: config.styleDim,
      nlayers: config.nLayer
    )

    // Initialize bidirectional LSTM for duration prediction
    predictorLSTM = LSTM(
      inputSize: config.hiddenDim + config.styleDim,
      hiddenSize: config.hiddenDim / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    // Initialize duration projection layer
    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    // Initialize prosody predictor (F0, pitch, etc.)
    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: config.styleDim,
      dHid: config.hiddenDim
    )

    // Initialize text encoder
    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: config.hiddenDim,
      kernelSize: config.textEncoderKernelSize,
      depth: config.nLayer,
      nSymbols: config.nToken
    )

    // Initialize audio decoder
    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: config.hiddenDim,
      styleDim: config.styleDim,
      dimOut: config.nMels,
      resblockKernelSizes: config.istftNet.resblockKernelSizes,
      upsampleRates: config.istftNet.upsampleRates,
      upsampleInitialChannel: config.istftNet.upsampleInitialChannel,
      resblockDilationSizes: config.istftNet.resblockDilationSizes,
      upsampleKernelSizes: config.istftNet.upsampleKernelSizes,
      genIstftNFft: config.istftNet.genIstftNFFT,
      genIstftHopSize: config.istftNet.genIstftHopSize
    )

    // Initialize G2P processor for text-to-phoneme conversion
    g2pProcessor = try? G2PFactory.createG2PProcessor(engine: g2p)
  }

  /// Builds the G2P processor's language engine — and with it its lexicon —
  /// without synthesising anything.
  ///
  /// `setLanguage` is reached only from `generateAudio`, so the engine is
  /// constructed inside whichever synthesis runs first. With MisakiSwift that
  /// engine loads the gold and silver dictionaries and grows both, and the
  /// call carrying it pays a one-time cost the others do not: measured on an
  /// iPhone, phonemization takes ~130 ms on the first call against ~1 ms once
  /// the lexicon exists.
  ///
  /// A caller that wants to pay that cost up front currently cannot, because
  /// the only door into `setLanguage` is a synthesis. Where an application
  /// warms up in the background, the cost lands on whichever call wins the
  /// race — and on a run where the warm-up is skipped, that is a user-facing
  /// one. Calling this after `init` puts it in one predictable place.
  ///
  /// It does not make the cost smaller, and it builds only the language given:
  /// a later `generateAudio` in a different language still rebuilds.
  public func preloadG2P(language: Language = .enUS) throws {
    try updateLanguageIfNeeded(language)
  }
  
  /// Generates audio from text using the specified voice and parameters.
  ///
  /// This method performs the complete TTS pipeline:
  /// 1. Converts text to phonemes (G2P)
  /// 2. Tokenizes and encodes phonemes
  /// 3. Predicts duration and prosody
  /// 4. Generates audio waveform
  ///
  /// - Parameters:
  ///   - voice: Voice embedding array (contains speaker characteristics)
  ///   - language: Target language for pronunciation
  ///   - text: Input text to synthesize
  ///   - speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
  /// - Returns: Array of audio samples as Float values
  /// - Throws: `KokoroTTSError.tooManyTokens` if text is too long,
  ///           or `G2PProcessorError` if G2P processing fails
  ///   - predictTimestamps: whether to fill in per-token timestamps on the
  ///     returned tokens. Defaults to `true` so existing callers are unaffected.
  ///     Pass `false` when only the audio is wanted: the predictor reads
  ///     durations back to the host, and it cannot affect `samples` — it only
  ///     mutates the token array the caller is about to discard.
  public func generateAudio(
    voice: MLXArray,
    language: Language,
    text: String,
    speed: Float = 1.0,
    predictTimestamps: Bool = true
  ) throws -> ([Float], [MToken]?) {
    // Update language if it has changed
    try updateLanguageIfNeeded(language)

    // Start performance timing
    BenchmarkTimer.reset()
    BenchmarkTimer.startTimer(Constants.bm_TTS)

    // Step 1: Convert text to phonemes
    // PER-STAGE TIMING (halfmarble 2026-09-09). The bm_ constants below were
    // declared upstream and never used — only the TTSAudio total was wired, so
    // nobody could see WHERE a synthesis spent its time. Each stage is timed
    // with bm_TTS as its parent, so the report nests under the total.
    BenchmarkTimer.startTimer(Constants.bm_Phonemize, Constants.bm_TTS)
    let (phonemizedText, tokenArray) = try phonemizeText(text)
    BenchmarkTimer.stopTimer(Constants.bm_Phonemize)
    
    // Step 2: Tokenize and prepare input
    BenchmarkTimer.startTimer(Constants.bm_prepare, Constants.bm_TTS)
    let (paddedInputIds, attentionMask, inputLengths, textMask, inputIds) = try prepareInputTensors(phonemizedText)
    if Self.profileStages { MLX.eval(paddedInputIds, attentionMask, inputLengths, textMask) }
    BenchmarkTimer.stopTimer(Constants.bm_prepare)
    
    // Step 3: Extract style embeddings from voice
    BenchmarkTimer.startTimer(Constants.bm_style, Constants.bm_TTS)
    let (globalStyle, acousticStyle) = extractStyleEmbeddings(from: voice, tokenCount: inputIds.count)
    if Self.profileStages { MLX.eval(globalStyle, acousticStyle) }
    BenchmarkTimer.stopTimer(Constants.bm_style)
    
    // Step 4: Encode text with BERT and predict duration
    BenchmarkTimer.startTimer(Constants.bm_bert, Constants.bm_TTS)
    let durationFeatures = encodeBERTAndDuration(
      inputIds: paddedInputIds,
      attentionMask: attentionMask,
      inputLengths: inputLengths,
      textMask: textMask,
      style: globalStyle
    )
    
    if Self.profileStages { MLX.eval(durationFeatures) }
    BenchmarkTimer.stopTimer(Constants.bm_bert)

    // Step 5: Predict phoneme durations
    BenchmarkTimer.startTimer(Constants.bm_duration, Constants.bm_TTS)
    let (predictedDurations, alignmentIndices) = predictDurations(
      features: durationFeatures,
      speed: speed
    )
    
    if Self.profileStages { MLX.eval(predictedDurations, alignmentIndices) }
    BenchmarkTimer.stopTimer(Constants.bm_duration)

    // Step 6: Generate aligned encodings
    BenchmarkTimer.startTimer(Constants.bm_align, Constants.bm_TTS)
    let alignedEncoding = durationFeatures.transposed(0, 2, 1).take(alignmentIndices, axis: 2)
    if Self.profileStages { MLX.eval(alignedEncoding) }
    BenchmarkTimer.stopTimer(Constants.bm_align)
    
    // Step 7: Predict prosody (F0, pitch)
    BenchmarkTimer.startTimer(Constants.bm_prosody, Constants.bm_TTS)
    let (f0Prediction, nPrediction) = prosodyPredictor.F0NTrain(x: alignedEncoding, s: globalStyle)
    if Self.profileStages { MLX.eval(f0Prediction, nPrediction) }
    BenchmarkTimer.stopTimer(Constants.bm_prosody)
    
    // Step 8: Encode text for decoder
    BenchmarkTimer.startTimer(Constants.bm_textenc, Constants.bm_TTS)
    let textEncoding = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    let asrFeatures = textEncoding.take(alignmentIndices, axis: 2)
    if Self.profileStages { MLX.eval(textEncoding, asrFeatures) }
    BenchmarkTimer.stopTimer(Constants.bm_textenc)
    
    // Step 9: Generate audio
    BenchmarkTimer.startTimer(Constants.bm_decoder, Constants.bm_TTS)
    let audio = decoder(
      asr: asrFeatures,
      F0Curve: f0Prediction,
      N: nPrediction,
      s: acousticStyle
    )[0]
    
    if Self.profileStages { MLX.eval(audio) }
    BenchmarkTimer.stopTimer(Constants.bm_decoder)

    // Try to predict timestamp of each token if G2P processor returns tokens
    if predictTimestamps, let tokenArray {
      TimestampPredictor.preditTimestamps(tokens: tokenArray, predictionDuration: predictedDurations)
    }
    
    // THE LAZY TAIL, and it must be INSIDE the total. MLX builds a graph
    // above; this is where it is forced to produce numbers, so deferred GPU
    // work is charged here rather than to the stage that queued it. Timing it
    // after `stopTimer(bm_TTS)` would leave the total excluding the one line
    // most likely to hold the missing 400 ms — which is what the first version
    // of this patch did, caught by checking that every start nests in the
    // total before building.
    BenchmarkTimer.startTimer(Constants.bm_materialise, Constants.bm_TTS)
    let samples = audio[0].asArray(Float.self)
    BenchmarkTimer.stopTimer(Constants.bm_materialise)

    // Stop performance timing
    BenchmarkTimer.stopTimer(Constants.bm_TTS)

    return (samples, tokenArray)
  }
  
  /// Updates the G2P language if it differs from the current language.
  private func updateLanguageIfNeeded(_ language: Language) throws {
    guard chosenLanguage != language else { return }
    
    guard let g2pProcessor else {
      throw G2PProcessorError.processorNotInitialized
    }
    
    try g2pProcessor.setLanguage(language)
    chosenLanguage = language
  }
  
  /// Converts input text to phonemes using the G2P processor.
  private func phonemizeText(_ text: String) throws -> (String, [MToken]?) {
    let phonemizedOutput = try g2pProcessor?.process(input: text)
    guard let phonemizedOutput else {
      throw G2PProcessorError.processorNotInitialized
    }
    return phonemizedOutput
  }
  
  /// Prepares input tensors for the model from phonemized text.
  /// - Returns: Tuple containing:
  ///   - paddedInputIds: Tokenized and padded input sequence
  ///   - attentionMask: Mask for attention mechanism
  ///   - inputLengths: Length of input sequence
  ///   - textMask: Mask for text padding
  ///   - inputIds: Original token IDs before padding
  private func prepareInputTensors(_ phonemizedText: String) throws -> (MLXArray, MLXArray, MLXArray, MLXArray, [Int]) {
    // Tokenize phonemized text
    let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedText)
    
    // Check token count limit
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    // Add padding tokens at start and end
    let paddedInputIdsArray = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsArray).expandedDimensions(axes: [0])

    // BUILD BOTH MASKS DIRECTLY. The previous version computed them on the GPU
    // and read them back: `inputLengths.max().item()` was one host
    // synchronisation, and `textMask.asArray(Bool.self)` another — to derive
    // values that are fully determined before any GPU work starts.
    //
    // Each invocation carries ONE sequence, already padded to its own length, so
    // there are no padding positions: `index + 1 > inputLength` is false for
    // every index in 0..<inputLength, and the attention mask is its inverse.
    // `maskValues` computes exactly that and is asserted against the old
    // GPU-side formula in the tests, so this is a provable substitution rather
    // than a claim.
    let inputLength = paddedInputIdsArray.count
    let inputLengths = MLXArray(inputLength)

    let masks = Self.maskValues(inputLength: inputLength)
    let textMask = MLXArray(masks.text).reshaped([1, inputLength])
    let attentionMask = MLXArray(masks.attention).reshaped([1, inputLength])

    return (paddedInputIds, attentionMask, inputLengths, textMask, inputIds)
  }
  
  /// Extracts style embeddings from the voice array.
  /// - Parameters:
  ///   - voice: Voice embedding array
  ///   - tokenCount: Number of tokens in the input
  /// - Returns: Tuple of (globalStyle, acousticStyle)
  ///   - globalStyle: Style embedding for prosody/duration (indices 128+)
  ///   - acousticStyle: Style embedding for acoustic features (indices 0-127)
  private func extractStyleEmbeddings(from voice: MLXArray, tokenCount: Int) -> (MLXArray, MLXArray) {
    // Extract reference style from voice embedding
    let referenceStyle = voice[tokenCount - 1, 0 ... 1, 0...]
    
    // Split into global style (for prosody/duration) and acoustic style
    let globalStyle = referenceStyle[0 ... 1, 128...]
    let acousticStyle = referenceStyle[0 ... 1, 0 ... 127]
    
    return (globalStyle, acousticStyle)
  }
  
  /// Encodes text with BERT and generates duration prediction features.
  private func encodeBERTAndDuration(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    inputLengths: MLXArray,
    textMask: MLXArray,
    style: MLXArray
  ) -> MLXArray {
    // Pass through BERT model
    let (bertOutput, _) = bert(inputIds, attentionMask: attentionMask)
    
    // Project BERT output and transpose for duration encoder
    let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)
    
    // Generate duration features with style conditioning
    let durationFeatures = durationEncoder(
      bertEncoded,
      style: style,
      textLengths: inputLengths,
      m: textMask
    )
    
    return durationFeatures
  }
  
  /// Predicts phoneme durations and creates alignment target matrix.
  /// - Parameters:
  ///   - features: Duration prediction features from encoder
  ///   - batchSize: Size of the input batch
  ///   - speed: Speech speed multiplier
  /// - Returns: Predicted durations and alignment target matrix for duration expansion
  // `batchSize` is gone: it existed only to size the one-hot alignment matrix,
  // which no longer exists.
  private func predictDurations(features: MLXArray, speed: Float) -> (MLXArray, MLXArray) {
    // Pass through LSTM
    let (lstmOutput, _) = predictorLSTM(features)
    
    // Project to duration values
    let durationLogits = durationProj(lstmOutput)
    
    // Convert to actual durations (clamped to minimum of 1 frame)
    let durationSigmoid = MLX.sigmoid(durationLogits).sum(axis: -1) / speed
    let predictedDurations = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    
    // Per-frame phoneme indices. NOT a one-hot matrix any more — see
    // createAlignmentIndices.
    return (predictedDurations, createAlignmentIndices(durations: predictedDurations))
  }
  
  /// Per-frame phoneme indices: element `f` is the phoneme sounding at frame `f`,
  /// each phoneme repeated as many times as its predicted duration.
  ///
  /// THIS USED TO RETURN A ONE-HOT `[phonemes x frames]` MATRIX, and the callers
  /// multiplied by it. A one-hot matmul is a gather written the long way —
  /// `matmul(X, onehot)[:, :, f]` is exactly `X[:, :, indices[f]]` — so the
  /// matrix was built, uploaded and multiplied to express a selection.
  ///
  /// Removing it deletes three costs, the third being the one that matters:
  ///   * a `phonemes * frames` Float allocation on the host, per synthesis;
  ///   * a matmul over that matrix, per synthesis, twice;
  ///   * **a `.item()` call PER FRAME** in the loop that filled it. Each one is
  ///     a GPU->CPU synchronisation, so a sentence of several hundred frames
  ///     paid several hundred round trips before any audio existed.
  ///
  /// Suggested by @antacosta's fork, which replaced the same two matmuls with
  /// `.take(_:axis:)` after diagnosing them as returning values unrelated to the
  /// selected column. **That correctness failure did not reproduce here**: on
  /// this machine matmul and gather agree to 1e-4 both at toy size and at
  /// [1,512,64]x[64,576], asserted in GatherEquivalenceTests. This change is
  /// taken for the cost above, and it removes the exposure either way.
  private func createAlignmentIndices(durations: MLXArray) -> MLXArray {
    // ONE host read for the whole duration vector, then expand in Swift.
    //
    // This loop used to call `.item()` once PER PHONEME — a GPU->CPU
    // synchronisation each time — and then hand `MLX.concatenated` that many
    // single-element arrays to stitch back together. The durations are needed on
    // the host either way; reading them in one go costs one synchronisation
    // instead of N, and the expansion is a Swift array append.
    let swiftDurations = durations.asArray(Int32.self)
    return MLXArray(Self.expandDurations(swiftDurations))
  }

  /// Per-frame phoneme indices from a duration vector: phoneme `i` repeated
  /// `durations[i]` times. Pure Swift and `static` so the tests can check it
  /// exactly, with no GPU and no model.
  static func expandDurations(_ durations: [Int32]) -> [Int32] {
    var indices: [Int32] = []
    indices.reserveCapacity(durations.reduce(0) { $0 + Int(max(0, $1)) })
    for (phoneme, duration) in durations.enumerated() where duration > 0 {
      indices.append(contentsOf: repeatElement(Int32(phoneme), count: Int(duration)))
    }
    return indices
  }

  /// The two masks for a single, fully-valid sequence of `inputLength`
  /// positions. Extracted and `static` so the tests can assert it against the
  /// GPU-side formula it replaced.
  static func maskValues(inputLength: Int) -> (text: [Bool], attention: [Int]) {
    (text: [Bool](repeating: false, count: inputLength),
     attention: [Int](repeating: 1, count: inputLength))
  }
  
  /// FORCE EVALUATION AT EACH STAGE BOUNDARY, so a stage timer measures the
  /// work it queued rather than the cost of queueing it.
  ///
  /// OFF BY DEFAULT AND IT MUST STAY OFF IN NORMAL USE. MLX is lazy: it builds
  /// a graph and executes when something demands values. That is a performance
  /// FEATURE — it lets the framework fuse and reorder — and `eval` after every
  /// stage defeats it, serialising the pipeline into synchronous chunks. So
  /// this changes the thing it measures: total time under profiling is not the
  /// total a driver experiences, and the two must never be quoted together.
  ///
  /// It exists because without it the stage numbers are meaningless. Measured
  /// 2026-09-09 on the phone: `materialise` — the single line that first
  /// demands values — was 47% and 68% of a synthesis, because every stage above
  /// it was timing graph CONSTRUCTION. The per-stage figures that produced were
  /// reported, believed, and had to be retracted.
  public nonisolated(unsafe) static var profileStages = false

  /// Constants used throughout the TTS engine.
  public struct Constants {
    /// Maximum number of tokens allowed in input
    public static let maxTokenCount = 510
    
    /// Audio sampling rate in Hz
    public static let samplingRate = 24000
    
    // Benchmark timer identifiers
    public static let bm_TTS = "TTSAudio"
    static let bm_Phonemize = "Phonemize"
    static let bm_bert = "BERT"
    static let bm_duration = "Duration"
    static let bm_prosody = "Prosody"
    static let bm_decoder = "Decoder"
    // THE STEPS UPSTREAM NEVER NAMED (halfmarble 2026-09-09). With only the
    // five above, 400 ms of a 566 ms first synthesis fell outside every stage.
    // `bm_materialise` is the important one: MLX is LAZY, so a stage timer
    // measures GRAPH CONSTRUCTION, and the deferred GPU work lands wherever
    // something first forces evaluation — here, `asArray`.
    static let bm_prepare = "Prepare"
    static let bm_style = "Style"
    static let bm_align = "Align"
    static let bm_textenc = "TextEncode"
    static let bm_materialise = "Materialise"

    // ALBERT attention, split three ways. Only bm_attnCore is the part a fused
    // scaled-dot-product-attention kernel would replace; the projections
    // before it and the dense/LayerNorm after it stay regardless. Measured on
    // an iPhone with these timers: at 85 characters attnCore is 3.3% of the
    // BERT stage and 0.45% of a synthesis, which is why this library still
    // computes attention by hand.
    static let bm_attnProj = "AttnProj"
    static let bm_attnCore = "AttnCore"
    static let bm_attnOut  = "AttnOut"
  }
}
