package fovea.distribution

import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.ndarray.types.Shape

// http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/

object Gumbel:
  // Used for sampling from discrete distributions
  def ofShape(low: Float, high: Float, shape: Shape, manager: NDManager): NDArray =
    manager.randomUniform(low, high, shape).log.negi.log.negi

  // Take source distribution values, mix them with values drawn from Gumbel, take argmax of that
  def argMaxSample(sourceDistribution: NDArray, manager: NDManager, axis: Int = 1): NDArray =
    val distribution = ofShape(low = 0, high = 1, sourceDistribution.getShape, manager)
    sourceDistribution.log.add(distribution).argMax(axis)
