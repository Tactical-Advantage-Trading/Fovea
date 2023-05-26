package fovea

import fovea.features.EntropyRate
import fovea.utils.approxEquals
import org.scalatest.funsuite.AnyFunSuite


class EntropyRateSpec extends AnyFunSuite {
  test("Compute string entropy rate") {
    assert(EntropyRate.konto("11100001").approxEquals(0.968, 0.01))
    assert(EntropyRate.konto("01100001").approxEquals(0.843, 0.01))
    assert(EntropyRate.konto("QWELKYULKILKJHGFDSAZXCVBNM").approxEquals(2.204, 0.01))
    assert(EntropyRate.konto("00000000000000000000000000").approxEquals(0.388, 0.01))
  }

  test("Quantize to alphabet") {
    val series = List(0.01, 0.04, 0.079, 0.74, 0.91)
    assert(series.map(EntropyRate.quantize).mkString == "AAAHJ")
    assert(EntropyRate.konto(series.map(EntropyRate.quantize).mkString).approxEquals(0.646, 0.01))
  }

  test("Max possible entropy") {
    assert(EntropyRate.maxPossibleEntropy(EntropyRate.alphabet.length).approxEquals(3.321, 0.01))
  }
}
