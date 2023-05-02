package fovea

import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.nn.*
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.nn.transformer.TransformerEncoderBlock
import ai.djl.training.ParameterStore
import fovea.encoding.PositionalEncoding
import fovea.utils.Common.{ParamsList, ShapeArray}


class TimeSeriesAttention(projection: Block, posEncoder: PositionalEncoding, nonLinearity: Block, headCount: Int, hiddenSize: Int,
                          encoderDropoutRate: Float, finalDropoutRate: Float, output: Block) extends AbstractBlock:

  private val finalInputShape = Shape(posEncoder.sequenceLen * posEncoder.encodingDim)

  val selfAttEncoder: TransformerEncoderBlock = TransformerEncoderBlock(posEncoder.encodingDim, headCount, hiddenSize, encoderDropoutRate, Activation.gelu)

  val flattener: Block = Blocks.batchFlattenBlock(finalInputShape.head)

  val finalDropout: Dropout = Dropout.builder.optRate(finalDropoutRate).build

  addChildBlock("projection", projection)
  addChildBlock("posEncoder", posEncoder)
  addChildBlock("selfAttEncoder", selfAttEncoder)
  addChildBlock("nonLinearity", nonLinearity)
  addChildBlock("flattener", flattener)
  addChildBlock("output", output)

  override def getOutputShapes(inputShapes: ShapeArray): ShapeArray =
    // Number of observations is not fixed (it can be a variable size minibatch or a single item when predicting)
    // But in any case the purpose is to consume the time series part and provide a classification/regression result
    for shape <- output.getOutputShapes(inputShapes) yield Shape(-1, shape.tail)

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit =

    projection.initialize(manager, dataType, inputShapes.head)

    val projectionOutputShapes = projection.getOutputShapes(inputShapes.toArray)

    posEncoder.initialize(manager, dataType, projectionOutputShapes.head)

    selfAttEncoder.initialize(manager, dataType, projectionOutputShapes.head)

    nonLinearity.initialize(manager, dataType, projectionOutputShapes.head)

    flattener.initialize(manager, dataType, projectionOutputShapes.head)

    finalDropout.initialize(manager, dataType, finalInputShape)

    output.initialize(manager, dataType, finalInputShape)

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: ParamsList): NDList =

    val inputsProjected = projection.forward(parameterStore, inputs, training)

    val positionEncoded = posEncoder.forward(parameterStore, inputsProjected, training)

    val attentionEncoded = selfAttEncoder.forward(parameterStore, positionEncoded, training)

    val attentionEncodedWithNonlinearity = nonLinearity.forward(parameterStore, attentionEncoded, training)

    val finalFlattened = flattener.forward(parameterStore, attentionEncodedWithNonlinearity, training)

    val finalDroppedOut = finalDropout.forward(parameterStore, finalFlattened, training)

    output.forward(parameterStore, finalDroppedOut, training)
