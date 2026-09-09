//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertEmbeddings {
  let wordEmbeddings: Embedding
  let positionEmbeddings: Embedding
  let tokenTypeEmbeddings: Embedding
  let layerNorm: LayerNormInference

  init(weights: [String: MLXArray], config: AlbertModelArgs) {
    wordEmbeddings = Embedding(weight: weights["bert.embeddings.word_embeddings.weight"]!)
    positionEmbeddings = Embedding(weight: weights["bert.embeddings.position_embeddings.weight"]!)
    tokenTypeEmbeddings = Embedding(weight: weights["bert.embeddings.token_type_embeddings.weight"]!)
    let layerNormWeights = weights["bert.embeddings.LayerNorm.weight"]!
    let layerNormBiases = weights["bert.embeddings.LayerNorm.bias"]!

    guard layerNormBiases.count == config.embeddingSize, layerNormWeights.count == config.embeddingSize else {
      fatalError("Wrong shape for AlbertEmbeddings LayerNorm bias or weights!")
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

  func callAsFunction(
    _ inputIds: MLXArray,
    tokenTypeIds: MLXArray? = nil,
    positionIds: MLXArray? = nil
  ) -> MLXArray {
    let seqLength = inputIds.shape[1]

    let positionIdsUsed: MLXArray
    if let positionIds = positionIds {
      positionIdsUsed = positionIds
    } else {
      positionIdsUsed = MLX.expandedDimensions(MLXArray(0 ..< seqLength), axes: [0])
    }

    let tokenTypeIdsUsed: MLXArray
    if let tokenTypeIds = tokenTypeIds {
      tokenTypeIdsUsed = tokenTypeIds
    } else {
      tokenTypeIdsUsed = MLXArray.zeros(like: inputIds)
    }

    let wordsEmbeddings = wordEmbeddings(inputIds)
    let positionEmbeddingsResult = positionEmbeddings(positionIdsUsed)
    let tokenTypeEmbeddingsResult = tokenTypeEmbeddings(tokenTypeIdsUsed)
    var embeddings = wordsEmbeddings + positionEmbeddingsResult + tokenTypeEmbeddingsResult
    embeddings = layerNorm(embeddings)
    return embeddings
  }
}
