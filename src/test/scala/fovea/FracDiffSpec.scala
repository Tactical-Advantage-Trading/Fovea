package fovea

import fovea.features.FracDiff
import org.scalatest.funsuite.AnyFunSuite


class FracDiffSpec extends AnyFunSuite {
  test("Fractional differencing") {
    val seq = List(96D, 97D, 98D, 99D, 100D)
    val fd1 = FracDiff(weightDecayFactor = 0.4, threshold = 0.005, capacity = seq.size)
    val fd2 = FracDiff(weightDecayFactor = 1D, threshold = 0.005, capacity = seq.size)
    val fd3 = FracDiff(weightDecayFactor = 0D, threshold = 0.005, capacity = seq.size)

    assert(fd1.convolve[Double](seq, identity, (_, y) => y).last == 38.4384) // Fractional differencing
    assert(fd2.convolve(seq, identity, (_, y) => y).last == 1.0) // Integer differencing
    assert(fd3.convolve(seq, identity, (_, y) => y).last == 100) // No differencing
  }
}
