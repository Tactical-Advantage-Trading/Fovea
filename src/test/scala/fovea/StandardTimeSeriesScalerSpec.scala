package fovea

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.scalatest.funsuite.AnyFunSuite
import fovea.utils.*


class StandardTimeSeriesScalerSpec extends AnyFunSuite {
  test("Standard time series scaler") {
    val scaler = utils.StandardTimeSeriesScaler(isSample = false)
    val manager = NDManager.newBaseManager
    val shape = Shape(2, 3, 3)

    val array = manager.create(Array[Float](1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9), shape)

    val ref = manager.create(Array[Float](-1.2247, -1.2247, -1.2247, 0.0,  0.0,  0.0, 1.2247,  1.2247,  1.2247,
      -1.2247, -1.2247, -1.2247, 0.0,  0.0,  0.0, 1.2247,  1.2247,  1.2247), shape)

    assert(scaler.transform(array).approxEquals(ref, 1e-4))
  }
}
