//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

class AlbertSelfAttention {
  let numAttentionHeads: Int
  let attentionHeadSize: Int
  let allHeadSize: Int

  let query: Linear
  let key: Linear
  let value: Linear
  let dense: Linear
  let layerNorm: LayerNormInference

  init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
    numAttentionHeads = config.numAttentionHeads
    attentionHeadSize = config.hiddenSize / config.numAttentionHeads
    allHeadSize = numAttentionHeads * attentionHeadSize

    query = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.query.weight"]!,
                   bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.query.bias"]!)
    key = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.key.weight"]!,
                 bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.key.bias"]!)
    value = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.value.weight"]!,
                   bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.value.bias"])
    dense = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.dense.weight"]!,
                   bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.dense.bias"]!)

    let layerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.LayerNorm.weight"]!
    let layerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).attention.LayerNorm.bias"]!

    guard layerNormWeights.count == config.hiddenSize, layerNormBiases.count == config.hiddenSize else {
      fatalError("Wrong shape for AlbertSelfAttention LayerNorm bias or weights!")
    }

    // WHOLE-ARRAY, NOT ELEMENT BY ELEMENT. This used to build an MLXNN
    // LayerNorm from fresh ones/zeros and then overwrite it one scalar at a
    // time — `count * 2` subscript assignments, each its own graph op, paid at
    // load. `LayerNormInference` already exists here to take the arrays
    // directly, and wraps the same MLXFast.layerNorm that MLXNN's LayerNorm
    // calls, so the maths is identical. Measured on an iPhone across three
    // launches per arm: KokoroTTS construction 345 ms -> 291 ms.
    layerNorm = LayerNormInference(weight: layerNormWeights, bias: layerNormBiases,
                                   eps: config.layerNormEps)
  }

  func transposeForScores(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    var newShape: [Int] = []

    for i in 0 ..< (shape.count - 1) {
      newShape.append(shape[i])
    }

    newShape.append(numAttentionHeads)
    newShape.append(attentionHeadSize)

    let reshaped = x.reshaped(newShape)
    return reshaped.transposed(0, 2, 1, 3)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    // THREE-WAY SPLIT, accumulated across all numHiddenLayers invocations
    // because BenchmarkTimer's delta is `+=`. The eval() calls are load-
    // bearing: without them these timers measure GRAPH CONSTRUCTION rather
    // than work, MLX being lazy. Everything here is behind profileStages,
    // which is off by default.
    let profiling = KokoroTTS.profileStages

    if profiling { BenchmarkTimer.startTimer(KokoroTTS.Constants.bm_attnProj,
                                             KokoroTTS.Constants.bm_bert) }
    let mixedQueryLayer = query(hiddenStates)
    let mixedKeyLayer = key(hiddenStates)
    let mixedValueLayer = value(hiddenStates)

    let queryLayer = transposeForScores(mixedQueryLayer)
    let keyLayer = transposeForScores(mixedKeyLayer)
    let valueLayer = transposeForScores(mixedValueLayer)
    if profiling {
      MLX.eval(queryLayer, keyLayer, valueLayer)
      BenchmarkTimer.stopTimer(KokoroTTS.Constants.bm_attnProj)
      BenchmarkTimer.startTimer(KokoroTTS.Constants.bm_attnCore,
                                KokoroTTS.Constants.bm_bert)
    }

    let keyLayerTransposed = keyLayer.transposed(0, 1, 3, 2)
    var attentionScores = MLX.matmul(queryLayer, keyLayerTransposed)
    attentionScores = attentionScores / sqrt(Float(attentionHeadSize))

    if let attentionMask = attentionMask {
      attentionScores = attentionScores + attentionMask
    }

    let attentionProbs = MLX.softmax(attentionScores, axis: -1)

    var contextLayer = MLX.matmul(attentionProbs, valueLayer)
    if profiling {
      MLX.eval(contextLayer)
      BenchmarkTimer.stopTimer(KokoroTTS.Constants.bm_attnCore)
      BenchmarkTimer.startTimer(KokoroTTS.Constants.bm_attnOut,
                                KokoroTTS.Constants.bm_bert)
    }
    contextLayer = contextLayer.transposed(0, 2, 1, 3)

    var newContextLayerShape: [Int] = []
    let shape = contextLayer.shape

    for i in 0 ..< (shape.count - 2) {
      newContextLayerShape.append(shape[i])
    }

    newContextLayerShape.append(allHeadSize)

    contextLayer = contextLayer.reshaped(newContextLayerShape)
    contextLayer = dense(contextLayer)
    contextLayer = layerNorm(contextLayer + hiddenStates)
    if profiling {
      MLX.eval(contextLayer)
      BenchmarkTimer.stopTimer(KokoroTTS.Constants.bm_attnOut)
    }

    return contextLayer
  }
}
