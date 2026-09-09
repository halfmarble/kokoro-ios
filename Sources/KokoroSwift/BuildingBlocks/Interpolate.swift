//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

func interpolate(
  input: MLXArray,
  size: [Int]? = nil,
  scaleFactor: [Float]? = nil,
  mode: String = "nearest",
  alignCorners: Bool? = nil
) -> MLXArray {
  let ndim = input.ndim
  if ndim < 3 {
    fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
  }

  let spatialDims = ndim - 2

  // Handle size and scaleFactor
  if size != nil && scaleFactor != nil {
    fatalError("Only one of size or scaleFactor should be defined")
  } else if size == nil && scaleFactor == nil {
    fatalError("One of size or scaleFactor must be defined")
  }

  // Calculate output size from scale factor if needed
  var outputSize: [Int] = []
  if let scaleFactor = scaleFactor {
    let factors = scaleFactor.count == 1 ? Array(repeating: scaleFactor[0], count: spatialDims) : scaleFactor

    for i in 0 ..< spatialDims {
      // Use ceiling instead of floor to match PyTorch behavior
      let currSize = max(1, Int(ceil(Float(input.shape[i + 2]) * factors[i])))
      outputSize.append(currSize)
    }
  } else if let size = size {
    outputSize = size.count == 1 ? Array(repeating: size[0], count: spatialDims) : size
  }

  // Handle 1D case (N, C, W)
  if spatialDims == 1 {
    return interpolate1d(input: input, size: outputSize[0], mode: mode, alignCorners: alignCorners)
  } else {
    fatalError("Only 1D interpolation currently supported, got \(spatialDims)D")
  }
}

func interpolate1d(
  input: MLXArray,
  size: Int,
  mode: String = "linear",
  alignCorners: Bool? = nil
) -> MLXArray {
  let shape = input.shape
  let batchSize = shape[0]
  let channels = shape[1]
  let inWidth = shape[2]

  let outputSize = max(1, size)
  let inputWidth = max(1, inWidth)

  if mode == "nearest" {
    if outputSize == 1 {
      let indices = MLXArray(converting: [0]).asType(.int32)
      return input[0..., 0..., indices]
    } else {
      let scale = Float(inputWidth) / Float(outputSize)
      let indices = MLX.floor(MLXArray(0 ..< outputSize).asType(.float32) * scale).asType(.int32)
      let clippedIndices = MLX.clip(indices, min: 0, max: inputWidth - 1)
      return input[0..., 0..., clippedIndices]
    }
  }

  // Linear interpolation
  var x: MLXArray
  if alignCorners == true && outputSize > 1 {
    x = MLXArray(0 ..< outputSize).asType(.float32) * (Float(inputWidth - 1) / Float(outputSize - 1))
  } else {
    if outputSize == 1 {
      x = MLXArray(converting: [0.0]).asType(.float32)
    } else {
      x = MLXArray(0 ..< outputSize).asType(.float32) * (Float(inputWidth) / Float(outputSize))
      if alignCorners != true {
        x = x + 0.5 * (Float(inputWidth) / Float(outputSize)) - 0.5
      }
    }
  }

  if inputWidth == 1 {
    let outputShape = [batchSize, channels, outputSize]
    return MLX.broadcast(input, to: outputShape)
  }

  // CLAMP THE SOURCE INDEX AT ZERO BEFORE FLOORING IT. `xHigh` was always
  // clamped at the top; `xLow` was clamped at neither end, and that is a bug
  // rather than an asymmetry.
  //
  // Under align_corners=false the source index is x = (i + 0.5)*scale - 0.5, so
  // whenever scale < 1 the LEADING output samples have x < 0. `floor` gives -1,
  // and a negative fancy-index into an MLXArray does NOT clamp — it wraps to the
  // LAST element, NumPy style. SineGen upsamples phase by 300, so x stays
  // negative for the first ~150 samples and every one of them borrowed the end
  // of the input, right where the phase CUMSUM begins; a cumsum then carries
  // that error through everything after it.
  //
  // Measured on [0, 1, 2, 3, 1000] upsampled x300: the first output sample came
  // back as 498.33 — 0.498 * 1000 — where it should be under 1.
  //
  // Found by @antacosta, on their fork of this package.
  //
  // NOTE, deliberate difference from that fork: it keeps the unclamped value for
  // the fraction (xFrac = x - xLowRaw), which RAMPS from input[0] toward
  // input[1] across the leading region. PyTorch clamps the source index itself —
  // `area_pixel_compute_source_index` returns 0 when src_idx < 0 for non-cubic
  // modes — which HOLDS input[0] instead. Kokoro's weights were trained and
  // validated against that reference, so this matches PyTorch. Both remove the
  // wraparound; only this one reproduces the reference.
  let xClamped = MLX.maximum(x, MLXArray(Float(0)))
  let xLow = MLX.floor(xClamped).asType(.int32)
  let xHigh = MLX.minimum(xLow + 1, MLXArray(inputWidth - 1, dtype: .int32))
  let xFrac = xClamped - xLow.asType(.float32)

  let yLow = input[0..., 0..., xLow]
  let yHigh = input[0..., 0..., xHigh]

  let oneMinusXFrac = 1 - xFrac
  let output = yLow * oneMinusXFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0) +
    yHigh * xFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0)

  return output
}
