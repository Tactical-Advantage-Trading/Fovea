package fovea.encoding

import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.norm.Dropout
import ai.djl.nn.{AbstractBlock, Parameter}
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.UniformInitializer
import fovea.utils.*
import fovea.utils.Common.ParamsList

// Additive learnable positional encoding as seen in "A Transformer-based Framework for Multivariate Time Series Representation Learning" paper
class LearnablePositionalEncoding(val sequenceLen: Int, val encodingDim: Int, val dropoutRate: Float, manager: NDManager) extends PositionalEncoding:

  val encoding: Parameter = addParameter {
    val initializer = UniformInitializer(0.02)
    val shape = Shape(sequenceLen, encodingDim)
    
    Parameter.builder.setName("positionalEncoding")
      .setType(Parameter.Type.WEIGHT)
      .optInitializer(initializer)
      .optRequiresGrad(true)
      .optShape(shape)
      .build
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: ParamsList): NDList =
    val learned: NDArray = parameterStore.getValue(encoding, inputs.head.getDevice, training)
    posDropout.forward(parameterStore, inputs.head.add(learned).asNDList, training, params)

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit =
    posDropout.initialize(manager, dataType, inputShapes.head)
    encoding.initialize(manager, dataType)
