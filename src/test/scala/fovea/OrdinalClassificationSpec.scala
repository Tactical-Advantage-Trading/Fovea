package fovea

import ai.djl.ndarray.{NDArray, NDManager}
import fovea.utils.*
import org.scalatest.funsuite.AnyFunSuite
import tech.tablesaw.plotly.Plot
import tech.tablesaw.plotly.components.Figure
import tech.tablesaw.plotly.traces.ScatterTrace


class OrdinalClassificationSpec extends AnyFunSuite {
  private val manager = NDManager.newBaseManager

  test("Ordinal encoding/decoding") {
    val data = Array[Float](1, 2, 0, 4)
    val ordinalLabels = OrdinalClassification.encode(manager.create(data))

    val ref = manager.create(Array(
      Array(1.0, 0.0, 0.0, 0.0),
      Array(1.0, 1.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0, 0.0),
      Array(1.0, 1.0, 1.0, 1.0)
    ))

    assert(ordinalLabels.approxEquals(ref, 1e-10))
    assert(OrdinalClassification.decode(ordinalLabels).toArray.sameElements(data))

    val hasBad = manager.create(Array(
      Array(1.0, 0.0, 0.0),
      Array(0.0, 1.0, 0.0),
      Array(0.0, 0.0, 0.0),
      Array(1.0, 1.0, 1.0)
    ))

    assert(OrdinalClassification.decode(hasBad).toArray.sameElements(Array(1, -1, 0, 3)))
  }
}
