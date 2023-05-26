package fovea.features


object Statistics:
  def removeOutliers[T, N](extractor: T => N)(items: Seq[T], lowerPct: Double, upperPct: Double)(
    using n: Numeric[N]
  ): Seq[T] =
    val size = items.size
    val lower = (size * lowerPct).toInt
    val upper = (size * upperPct).toInt
    items.sortBy(extractor).drop(lower).dropRight(upper)

  def meanBy[T, N](extractor: T => N)(items: Seq[T] = Nil)(
    using n: Numeric[N]
  ): Double =
    items.foldLeft(0D) {
      case (total, item) =>
        val extracted = extractor(item)
        n.toDouble(extracted) + total
    } / items.size

  def varianceBy[T, N](extractor: T => N)(items: Seq[T], mean: Double)(
    using n: Numeric[N]
  ): Double =
    items.foldLeft(0D) {
      case (total, item) =>
        val extracted = extractor(item)
        val distance = n.toDouble(extracted) - mean
        math.pow(distance, 2) + total
    } / items.size.toDouble

  def stdDevBy[T, N](extractor: T => N)(items: Seq[T] = Nil, mean: Double)(
    using n: Numeric[N]
  ): Double =
    val variance = varianceBy(extractor)(items, mean)
    math.sqrt(variance)

  def medianBy[T, N](extractor: T => N)(items: Seq[T], whenEmpty: N, skew: Double)(
    using n: Numeric[N]
  ): N =
    val toDrop = (items.size * skew).toInt
    val rest = items.sortBy(extractor).drop(toDrop)
    if (items.nonEmpty) extractor(rest.head) else whenEmpty

  def zScoresBy[T, N](extractor: T => N)(items: Seq[T], mean: Double)(
    using n: Numeric[N]
  ): Seq[Double] =
    val computedStdDevBy = stdDevBy(extractor)(items, mean)

    items.map { item =>
      val extracted = extractor(item)
      val distance = n.toDouble(extracted) - mean
      distance / computedStdDevBy
    }

  def pearsonBy[A, B, C](items: Seq[A], mean: Double)(x: A => B)(y: A => C)(
    using n: Numeric[B], q: Numeric[C]
  ): Double =
    val xScores = zScoresBy(x)(items, mean)
    val yScores = zScoresBy(y)(items, mean)

    xScores.zip(yScores).foldLeft(0D) { case (total, items1) =>
      val (scoredXItem, scoredYItem) = items1
      scoredXItem * scoredYItem + total
    } / items.size.toDouble

  // An efficient average which does not need to keep data point history

  def emptyIncrementalAverage: IncrementalAverage =
    IncrementalAverage(current = 0D, increments = 1)

  def incrementalAverageFrom[T](extractor: T => Double)(items: Seq[T] = Nil): IncrementalAverage =
    val start = IncrementalAverage(current = extractor(items.head), increments = 2)
    items.tail.map(extractor).foldLeft(start)(_ addNewDataPoint _)

  case class IncrementalAverage(current: Double, increments: Long):
    def addNewDataPoint(dataPoint: Double): IncrementalAverage =
      val update = current + (dataPoint - current) / increments
      IncrementalAverage(update, increments + 1)

  // An efficient running sum that does not need to recompute history on new data

  def emptyRunningSum(period: Int): RunningSum =
    RunningSum(sum = 0D, sumSquared = 0D, Vector.empty, leftover = period)

  case class RunningSum(sum: Double, sumSquared: Double, history: Vector[Double], leftover: Int):
    def addPoints[T](items: Seq[T] = Nil)(extractor: T => Double): RunningSum =
      items.map(extractor).foldLeft(this)(_ addNewDataPoint _)

    def addNewDataPoint(dataPoint: Double): RunningSum =
      if leftover > 0 then RunningSum(sum + dataPoint, sumSquared + math.pow(dataPoint, 2), history :+ dataPoint, leftover = leftover - 1)
      else RunningSum(sum - history.head + dataPoint, sumSquared - math.pow(history.head, 2) + math.pow(dataPoint, 2), history.tail :+ dataPoint, leftover = 0)

    lazy val mean: Double = sum / history.size

    lazy val variance: Double = sumSquared / history.size - math.pow(mean, 2)