package fovea.activation

import ai.djl.ndarray.NDList
import ai.djl.nn.{Block, LambdaBlock}
import fovea.utils.*


class Softmax(axis: Int) extends ActivationFunction:
  def activation(arrays: NDList): NDList = arrays.singletonOrThrow.softmax(axis).asNDList
  def block: Block = LambdaBlock(activation, "SOFTMAX")
