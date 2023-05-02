package fovea

import fovea.features.Statistics
import fovea.utils.*
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random


class StatisticsSpec extends AnyFunSuite {
  case class Wrapper(dataPoint: Double)

  private val data = (1 to 100).toList.map(_.toDouble).map(Wrapper.apply)

  test("Remove outliers") {
    val check = (11 to 90).toList.map(_.toDouble).map(Wrapper.apply)
    assert(Statistics.removeOutliers[Wrapper, Double](_.dataPoint)(Random.shuffle(data), lowerPct = 0.1, upperPct = 0.1) == check)
  }

  test("Moments of mean, median, Pearson") {
    val mean = Statistics.meanBy[Wrapper, Double](_.dataPoint)(data)
    assert(Statistics.varianceBy[Wrapper, Double](_.dataPoint)(data, mean) == 833.25)
    assert(Statistics.stdDevBy[Wrapper, Double](_.dataPoint)(data, mean).approxEquals(28.86, 0.01))
    assert(Statistics.medianBy[Wrapper, Double](_.dataPoint)(data, whenEmpty = 0D, skew = 0.5) == 51D)
    assert(Statistics.medianBy[Wrapper, Double](_.dataPoint)(data, whenEmpty = 0D, skew = 0.8) == 81D)
    assert(Statistics.pearsonBy(data, mean)(_.dataPoint)(_.dataPoint).approxEquals(1.0, 0.0001))
    assert(mean == 50.5)
  }

  test("Incremental average") {
    val meanBy = Statistics.meanBy[Wrapper, Double](_.dataPoint)
    val inc1 = Statistics.emptyIncrementalAverage
    val inc2 = Statistics.incrementalAverageFrom[Wrapper](_.dataPoint)(data)
    assert(data.foldLeft(inc1)(_ addNewDataPoint _.dataPoint).current == meanBy(data))
    assert(inc2.current == meanBy(data))
  }

  test("Running sum") {
    val points1 = List.fill(10)(1)
    assert(Statistics.emptyRunningSum(period = 1).addPoints(points1)(identity).current == 1D)
    assert(Statistics.emptyRunningSum(period = 5).addPoints(points1)(identity).current == 5D)
    assert(Statistics.emptyRunningSum(period = 10).addPoints(points1)(identity).current == 10D)

    val points2 = List(100, 1, 1, 2, 1)
    assert(Statistics.emptyRunningSum(period = 100).addPoints(points2)(identity).current == 105D)
    assert(Statistics.emptyRunningSum(period = 1).addPoints(points2)(identity).current == 1D)
    assert(Statistics.emptyRunningSum(period = 4).addPoints(points2)(identity).current == 5D)
  }
}
