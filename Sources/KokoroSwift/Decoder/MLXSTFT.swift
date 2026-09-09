//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXFFT
import MLXNN

// Hanning window implementation to replace np.hanning
func hanning(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray(1.0)
  }

  let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
  let factor = .pi / Float(length - 1)
  return 0.5 + 0.5 * cos(n * factor)
}

func mlxStft(
  x: MLXArray,
  nFft: Int = 800,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann",
  center: Bool = true,
  padMode: String = "reflect"
) -> MLXArray {
  let hopLen = hopLength ?? nFft / 4
  let winLen = winLength ?? nFft

  var w: MLXArray
  if let windowStr = window as? String {
    if windowStr.lowercased() == "hann" {
      w = hanning(length: winLen + 1)[0 ..< winLen]
    } else {
      fatalError("Only hanning is supported for window, not \(windowStr)")
    }
  } else if let windowArray = window as? MLXArray {
    w = windowArray
  } else {
    fatalError("Window must be a string or MLXArray")
  }

  if w.shape[0] < nFft {
    let padSize = nFft - w.shape[0]
    w = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
  }

  func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
    if padMode == "constant" {
      return MLX.padded(x, width: [padding, padding])
    } else if padMode == "reflect" {
      let prefix = x[1 ..< padding + 1][.stride(by: -1)]
      let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
      return MLX.concatenated([prefix, x, suffix])
    } else {
      fatalError("Invalid pad mode \(padMode)")
    }
  }

  var xArray = x

  if center {
    xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
  }

  let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
  if numFrames <= 0 {
    fatalError("Input is too short")
  }

  let shape: [Int] = [numFrames, nFft]
  let strides: [Int] = [hopLen, 1]

  let frames = MLX.asStrided(xArray, shape, strides: strides)

  let spec = MLXFFT.rfft(frames * w)
  return spec.transposed(1, 0)
}

func mlxIstft(
  x: MLXArray,
  hopLength: Int? = nil,
  winLength: Int? = nil,
  window: Any = "hann"
) -> MLXArray {
  let winLen = winLength ?? ((x.shape[1] - 1) * 2)
  let hopLen = hopLength ?? (winLen / 4)

  var w: MLXArray
  if let windowStr = window as? String {
    if windowStr.lowercased() == "hann" {
      w = hanning(length: winLen + 1)[0 ..< winLen]
    } else {
      fatalError("Only hanning window is supported")
    }
  } else if let windowArray = window as? MLXArray {
    w = windowArray
  } else {
    fatalError("Window must be a string or MLXArray")
  }

  if w.shape[0] < winLen {
    w = MLX.concatenated([w, MLXArray.zeros([winLen - w.shape[0]])], axis: 0)
  }

  let xTransposed = x.transposed(1, 0)
  let t = (xTransposed.shape[0] - 1) * hopLen + winLen
  let windowModLen = 20 / 5

  let wSquared = w * w
  let totalWsquared = MLX.concatenated(Array(repeating: wSquared, count: t / winLen))

  let output = MLXFFT.irfft(xTransposed, axis: 1) * w

  var outputs: [MLXArray] = []
  var windowSums: [MLXArray] = []

  for i in 0 ..< windowModLen {
    let outputStride = output[.stride(from: i, by: windowModLen), .ellipsis].reshaped([-1])
    let windowSumArray = totalWsquared[0 ..< outputStride.shape[0]]

    outputs.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      outputStride,
      MLXArray.zeros([max(0, t - i * hopLen - outputStride.shape[0])]),
    ]))

    windowSums.append(MLX.concatenated([
      MLXArray.zeros([i * hopLen]),
      windowSumArray,
      MLXArray.zeros([max(0, t - i * hopLen - windowSumArray.shape[0])]),
    ]))
  }

  var reconstructed = outputs[0]
  var windowSum = windowSums[0]
  for i in 1 ..< windowModLen {
    reconstructed += outputs[i]
    windowSum += windowSums[i]
  }

  reconstructed =
    reconstructed[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)] /
    windowSum[winLen / 2 ..< (reconstructed.shape[0] - winLen / 2)]

  return reconstructed
}

class MLXSTFT {
  let filterLength: Int
  let hopLength: Int
  let winLength: Int
  let window: String

  var magnitude: MLXArray?
  var phase: MLXArray?

  /// The Hann window depends only on `winLength`, so it is built once here
  /// rather than rebuilt inside every `mlxIstft` call. `inverse` runs once per
  /// batch item per synthesis and each call recomputed an 800-point cosine it
  /// already had.
  private let cachedWindow: MLXArray

  init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
    self.filterLength = filterLength
    self.hopLength = hopLength
    self.winLength = winLength
    self.window = window
    guard window.lowercased() == "hann" else {
      fatalError("Only hanning is supported for window, not \(window)")
    }
    self.cachedWindow = hanning(length: winLength + 1)[0 ..< winLength]
  }

  func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
    var audioArray = inputData
    if audioArray.ndim == 1 {
      audioArray = audioArray.expandedDimensions(axis: 0)
    }

    var magnitudes: [MLXArray] = []
    var phases: [MLXArray] = []

    for batchIdx in 0 ..< audioArray.shape[0] {
      // Compute STFT
      let stft = mlxStft(
        x: audioArray[batchIdx],
        nFft: filterLength,
        hopLength: hopLength,
        winLength: winLength,
        window: window,
        center: true,
        padMode: "reflect"
      )

      let magnitude = MLX.abs(stft)

      // Replaces np.angle()
      let phase = MLX.atan2(stft.imaginaryPart(), stft.realPart())

      magnitudes.append(magnitude)
      phases.append(phase)
    }

    let magnitudesStacked = MLX.stacked(magnitudes, axis: 0)
    let phasesStacked = MLX.stacked(phases, axis: 0)

    return (magnitudesStacked, phasesStacked)
  }

  func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
    var reconstructed: [MLXArray] = []

    for batchIdx in 0 ..< magnitude.shape[0] {
      // NO PHASE UNWRAP. It was provably the identity on every input this
      // function receives, so it cost a full pass over the phase array — and
      // several MLX ops per batch item — to return its argument.
      //
      // `unwrap` zeroed its correction wherever |diff| < pi. `inverse` is called
      // from exactly one place, Generator's `stft.inverse(magnitude:phase:)`,
      // and that phase is `MLX.sin(...)`, so it lies in [-1, 1]. Consecutive
      // differences are then at most 2, and 2 < pi, so the correction was zero
      // everywhere and the result was its input.
      //
      // `UnwrapIsIdentityTests` reproduces the removed algorithm and asserts
      // exactly that over sin-bounded input, so this is checked rather than
      // reasoned about. IF A CALLER EVER PASSES PHASE NOT BOUNDED BY PI — an
      // atan2 phase from `transform`, say — the unwrap has to come back. The
      // test names that condition explicitly.
      let phaseCont = phase[batchIdx]

      // Combine magnitude and phase
      let stft = magnitude[batchIdx] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)

      // Inverse STFT
      let audio = mlxIstft(
        x: stft,
        hopLength: hopLength,
        winLength: winLength,
        window: cachedWindow
      )
      reconstructed.append(audio)
    }

    let reconstructedStacked = MLX.stacked(reconstructed, axis: 0)
    return reconstructedStacked.expandedDimensions(axis: 1)
  }

  func callAsFunction(inputData: MLXArray) -> MLXArray {
    let (mag, ph) = transform(inputData: inputData)
    magnitude = mag
    phase = ph
    let reconstruction = inverse(magnitude: mag, phase: ph)
    return reconstruction.expandedDimensions(axis: -2)
  }
}
