package fovea

import fovea.utils.{Common, DeltaSeriesCollector, DeltaSeriesMaker, SeriesGrouper}
import org.scalatest.funsuite.AnyFunSuite


class DeltaSeriesMakerSpec extends AnyFunSuite {
  def makeFeature(backwardSnaps: Iterable[Snapshot] = Nil): Common.Features =
    Array(backwardSnaps.map(_.value).sum / backwardSnaps.size)

  case class Snapshot(value: Double)
  private val window = 4

  test("Empty series until collected enough data points") {
    var delta: DeltaSeriesMaker[Snapshot] = DeltaSeriesCollector[Snapshot](Vector.empty, makeFeature, window, historyMultiplier = 2)
    delta = delta.addItem(Snapshot(1D)).addItem(Snapshot(2D)).addItem(Snapshot(3D)).addItem(Snapshot(4D)).addItem(Snapshot(5D)).addItem(Snapshot(6D)).addItem(Snapshot(7D))
    assert(delta.processItems.series.isEmpty)
  }

  test("Accumulate and compute deltas") {
    val snapshots = List(
      Snapshot(1D), Snapshot(2D), Snapshot(3D), Snapshot(4D), Snapshot(5D),
      Snapshot(6D), Snapshot(7D), Snapshot(8D), Snapshot(9D), Snapshot(10D)
    )

    val check = snapshots.sliding(window, 1).map(makeFeature).toList.map(_.toList)

    var delta: DeltaSeriesMaker[Snapshot] = DeltaSeriesCollector[Snapshot](Vector.empty, makeFeature, window, historyMultiplier = 2)
    delta = snapshots.foldLeft(delta)(_ addItem _).processItems
    assert(delta.series.map(_.toList) == check.takeRight(window))

    val snapshots1 = snapshots :+ Snapshot(11D) :+ Snapshot(12D)
    val check1 = snapshots1.sliding(window, 1).map(makeFeature).toList.map(_.toList)

    delta = delta.addItem(Snapshot(11D)).addItem(Snapshot(12D))
    assert(delta.processItems.series.map(_.toList) == check1.takeRight(window))
    assert(delta.addItem(Snapshot(11D)).processItems.series.map(_.toList) == Vector(List(8.5), List(9.5), List(10.5), List(11.0)))
  }

  test("Condense data points") {
    val snapshots = (0 to 30 by 2).map(_.toDouble).map(Snapshot.apply).toList
    val merger: Seq[Snapshot] => Snapshot = shots => Snapshot(shots.map(_.value).sum / shots.size)
    val maker = DeltaSeriesCollector[Snapshot](Vector.empty, makeFeature, window, historyMultiplier = 2)
    var grouper = SeriesGrouper[Snapshot, Snapshot](maker, Vector.empty, merger, groupSize = 2)

    grouper = snapshots.foldLeft(grouper)((grouper1, item) => grouper1.addItem(item).process)
    val check = snapshots.grouped(2).map(merger).sliding(window, 1).map(makeFeature).toList.map(_.toList)
    assert(grouper.maker.series.map(_.toList) == check.takeRight(window))
    assert(snapshots.size == 16)

    grouper = grouper.addItem(Snapshot(32)).process.addItem(Snapshot(34)).process
    val check1 = (snapshots :+ Snapshot(32) :+ Snapshot(34)).grouped(2).map(merger).sliding(window, 1).map(makeFeature).toList.map(_.toList)
    assert(grouper.maker.series.map(_.toList) == check1.takeRight(window))

    grouper = grouper.addItem(Snapshot(36)).addItem(Snapshot(38)).addItem(Snapshot(40)).addItem(Snapshot(42)).addItem(Snapshot(44)).process
    val tail = Nil :+ Snapshot(32) :+ Snapshot(34) :+ Snapshot(36) :+ Snapshot(38) :+ Snapshot(40) :+ Snapshot(42)
    val check2 = (snapshots ++ tail).grouped(2).map(merger).sliding(window, 1).map(makeFeature).toList.map(_.toList)
    assert(grouper.maker.series.map(_.toList) == check2.takeRight(window))
    assert(grouper.accumulator == Vector(Snapshot(44)))
  }
}
