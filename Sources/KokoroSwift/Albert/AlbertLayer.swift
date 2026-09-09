//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayer {
  let attention: AlbertSelfAttention
  let fullLayerLayerNorm: LayerNormInference
  let ffn: Linear
  let ffnOutput: Linear

  init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
    attention = AlbertSelfAttention(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum)
    ffn = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.weight"]!,
                 bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.bias"]!)
    ffnOutput = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.weight"]!,
                       bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.bias"]!)
    let fullLayerLayerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.weight"]!
    let fullLayerLayerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.bias"]!

    guard fullLayerLayerNormWeights.count == config.hiddenSize, fullLayerLayerNormBiases.count == config.hiddenSize else {
      fatalError("Wrong shape for AlbertLayer FullLayerLayerNorm bias or weights!")
    }

    // WHOLE-ARRAY, NOT ELEMENT BY ELEMENT. This used to build an MLXNN
    // LayerNorm from fresh ones/zeros and then overwrite it one scalar at a
    // time — `count * 2` subscript assignments, each its own graph op, paid at
    // load. `LayerNormInference` already exists here to take the arrays
    // directly, and wraps the same MLXFast.layerNorm that MLXNN's LayerNorm
    // calls, so the maths is identical. Measured on an iPhone across three
    // launches per arm: KokoroTTS construction 345 ms -> 291 ms.
    fullLayerLayerNorm = LayerNormInference(weight: fullLayerLayerNormWeights,
                                           bias: fullLayerLayerNormBiases,
                                           eps: config.layerNormEps)
  }

  func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
    var ffnOutputArray = ffn(attentionOutput)
    ffnOutputArray = MLXNN.gelu(ffnOutputArray)
    ffnOutputArray = ffnOutput(ffnOutputArray)
    return ffnOutputArray
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
    let ffnOutput = ffChunk(attentionOutput)
    let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
    return output
  }
}
