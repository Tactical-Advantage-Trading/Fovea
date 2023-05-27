package fovea.features

import scala.annotation.tailrec

// Advances In Financial Machine Learning: Fractional differencing chapter

case class FracDiff(weightDecayFactor: Double, threshold: Double, capacity: Int):
  val weights: Vector[Double] = weights(Vector(-weightDecayFactor), 2).take(capacity - 1).reverse :+ 1D

  @tailrec
  final def weights(accumulator: Vector[Double], index: Int): Vector[Double] =
    val nextWeight = accumulator.last * (index - 1 - weightDecayFactor) / index
    if nextWeight < -threshold then weights(accumulator :+ nextWeight, index + 1)
    else accumulator

  def dotProduct(sequence: Seq[Double] = Nil): Double =
    sequence.lazyZip(weights).map(_ * _).sum

  def convolve[T](sequence: Seq[T], extract: T => Double, update: (T, Double) => T): Seq[T] =
    sequence.sliding(weights.size, 1).toVector.map { items =>
      val values = items.map(extract)
      val result = dotProduct(values)
      update(items.last, result)
    }