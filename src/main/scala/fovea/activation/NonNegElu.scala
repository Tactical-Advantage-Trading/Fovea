package fovea.activation

import ai.djl.ndarray.{NDArray, NDList}
import ai.djl.nn.{Activation, Block, LambdaBlock}
import fovea.utils.*


class NonNegElu(alpha: Float) extends ActivationFunction:
  // Adding 1 guarantees that ELU output will always be non-negative, this is sometimes desirable
  def activation(arrays: NDList): NDList = arrays.singletonOrThrow.getNDArrayInternal.elu(alpha).add(1).asNDList
  def block: Block = LambdaBlock(activation, "NONNEGELU")
