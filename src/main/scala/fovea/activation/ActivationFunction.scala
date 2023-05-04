package fovea.activation

import ai.djl.ndarray.NDList
import ai.djl.nn.Block


trait ActivationFunction:
  
  def activation(arrays: NDList): NDList
  
  def block: Block
