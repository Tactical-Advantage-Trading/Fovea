package fovea.transformer

import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.norm.{BatchNorm, Dropout, LayerNorm}
import ai.djl.nn.transformer.{PointwiseFeedForwardBlock, ScaledDotProductAttentionBlock}
import ai.djl.nn.{AbstractBlock, Block}
import ai.djl.training.ParameterStore
import fovea.utils.*
import fovea.utils.Common.{ParamsList, ShapeArray}

import java.util.Collections


enum Normalization:
  case Layer
  case Batch

// Replaces DJL TransformerEncoderBlock:
// its second residual adds the block input instead of the attention output, 
// so attention only reaches the block output through the feed-forward layer
class TransformerEncoderBlock(embeddingSize: Int,
                              headCount: Int, hiddenSize: Int, dropoutRate: Float,
                              activation: java.util.function.Function[NDList, NDList],
                              normalization: Normalization = Normalization.Layer,
                              preNorm: Boolean = false) extends AbstractBlock:

  require(embeddingSize % headCount == 0, s"Embedding size $embeddingSize must be divisible by head count $headCount")

  private val selfAttention: Block =
    val block = ScaledDotProductAttentionBlock.builder
      .optAttentionProbsDropoutProb(dropoutRate)
      .setEmbeddingSize(embeddingSize)
      .setHeadCount(headCount).build
    addChildBlock("selfAttention", block)

  private val attentionNorm: Block = addChildBlock("attentionNorm", makeNorm)

  private val feedForward: Block =
    val list = Collections.singletonList[Integer](hiddenSize)
    val pffBlock = PointwiseFeedForwardBlock(list, embeddingSize, activation)
    addChildBlock("outputBlock", pffBlock)

  private val outputNorm: Block = addChildBlock("outputNorm", makeNorm)

  private val attentionDropout: Dropout = Dropout.builder.optRate(dropoutRate).build

  private val feedForwardDropout: Dropout = Dropout.builder.optRate(dropoutRate).build

  private def makeNorm: Block = normalization match
    case Normalization.Layer => LayerNorm.builder.axis(2).build
    case Normalization.Batch => BatchNorm.builder.optAxis(2).build

  override def getOutputShapes(inputShapes: ShapeArray): ShapeArray = Array(inputShapes.head)

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit =
    selfAttention.initialize(manager, dataType, inputShapes*)
    for block <- List(attentionNorm, feedForward, outputNorm, attentionDropout, feedForwardDropout) do
      block.initialize(manager, dataType, inputShapes.head)

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: ParamsList): NDList =

    def run(block: Block, input: NDArray): NDArray = block.forward(parameterStore, input.asNDList, training).singletonOrThrow

    def attend(sequence: NDArray): NDArray =
      val attentionInputs = sequence.asNDList
      if inputs.size > 1 then attentionInputs.add(inputs get 1)
      val attended = selfAttention.forward(parameterStore, attentionInputs, training).singletonOrThrow
      run(attentionDropout, attended)

    def transform(sequence: NDArray): NDArray =
      val r0 = run(feedForward, sequence)
      run(feedForwardDropout, r0)

    if preNorm then
      val z = inputs.head add attend(run(attentionNorm, inputs.head))
      val z1 = z add transform(run(outputNorm, z))
      z1.asNDList
    else
      val z = run(attentionNorm, inputs.head add attend(inputs.head))
      val z1 = run(outputNorm, z add transform(z))
      z1.asNDList
