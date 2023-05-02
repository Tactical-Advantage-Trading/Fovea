package fovea

import fovea.utils.Decimal
import fovea.utils.Decimal.given
import org.scalatest.funsuite.AnyFunSuite


class DecimalSpec extends AnyFunSuite {
  test("Rounding") {
    val d1: Decimal = 0.2 * 0.2
    assert(d1 === 0.04, "0.2 * 0.2 should be 0.04")
    val d2: Decimal = 2.1 - 0.2
    assert(d2 === 1.9, "2.1 - 0.2 should be 1.9")
    val d3: Decimal = 1L / 3.0
    assert(d3 === 0.3333333333, "1 / 3 should round exactly to 0.3333333333")
    val d4: Decimal = 0.2 / 1.456
    assert(d4 === 0.1373626374, "0.2 / 1.456 should round exactly to 0.1373626374")
  }

  test("Implicits and equality") {
    val a1: Decimal = 1
    val a2: Decimal = 12L
    val a3: Decimal = 0.552

    assert(a1 == 1)
    assert(a2 == 12L)
    assert(a3 == 0.552)
    assert(-a1 == -1)
  }
}
