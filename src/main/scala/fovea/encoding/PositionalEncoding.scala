package fovea.encoding

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.AbstractBlock
import ai.djl.nn.norm.Dropout
import ai.djl.util.PairList
import fovea.utils.Common.ShapeArray


trait PositionalEncoding extends AbstractBlock:
  
  val dropoutRate: Float
  
  val sequenceLen: Int
  
  val encodingDim: Int

  val posDropout: Dropout = Dropout.builder.optRate(dropoutRate).build
  
  override def getOutputShapes(inputShapes: ShapeArray): ShapeArray = inputShapes
