package fovea.utils

import ai.djl.ndarray.NDArray
import ai.djl.translate.Transform


// Scales a whole dataset, not sensibly applicable to non-stationary series
class StandardTimeSeriesScaler(isSample: Boolean) extends Transform:
  private val dataAxis = Array(0)

  def transform(array: NDArray): NDArray =
    // We expect [observations, timesteps, features] format
    val Array(d1, d2, d3) = array.getShape.getShape
    val array1 = array.reshape(d1 * d2, d3)

    val mean = array1.mean(dataAxis)
    val divisor = if (isSample) d1 * d2 - 1 else d1 * d2
    val sd = array1.sub(mean).pow(2).sum(dataAxis).div(divisor).sqrt
    array.sub(mean).div(sd)
