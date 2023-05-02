package fovea.utils


trait DeltaSeriesMaker[T]:
  def processItems: DeltaSeriesMaker[T]
  def addItem(item: T): DeltaSeriesMaker[T]
  val series: Vector[Common.Features]
  val unprocessed: Vector[T]

// When predicting we typically get raw data from a streaming source and we need to collect some history before we can process it
case class DeltaSeriesCollector[T](unprocessed: Vector[T], convert: Seq[T] => Common.Features, size: Int, historyMultiplier: Int) extends DeltaSeriesMaker[T]:
  override def processItems: DeltaSeriesMaker[T] = if unprocessed.size < size * historyMultiplier then this else DeltaSeriesProducer[T](unprocessed.take(size), unprocessed.drop(size), Vector.empty, convert, size).processItems
  override def addItem(item: T): DeltaSeriesMaker[T] = copy(unprocessed = unprocessed :+ item)
  override val series: Vector[Common.Features] = Vector.empty

// The idea is to compute only recently added items, not the whole history window on each added datapoint because feature computation may be slow and prediction is expected ASAP
case class DeltaSeriesProducer[T](processed: Vector[T], unprocessed: Vector[T], series: Vector[Common.Features], convert: Seq[T] => Common.Features, size: Int) extends DeltaSeriesMaker[T]:
  require(processed.size == size, "Can't initiate with non-full processed vector, use DeltaSeriesCollector when there is not enough data yet")
  override def processItems: DeltaSeriesMaker[T] = if unprocessed.isEmpty then this else doProcess
  override def addItem(item: T): DeltaSeriesMaker[T] = copy(unprocessed = unprocessed :+ item)

  def doProcess: DeltaSeriesMaker[T] =
    val combined = processed ++ unprocessed
    val range = (combined.size - unprocessed.size + 1).to(combined.size)
    val series1 = series ++ range.map(_ - size).zip(range).map(combined.slice).map(convert)
    copy(series = series1.takeRight(size), processed = combined.takeRight(size), unprocessed = Vector.empty)

// This should be used if we want to group and somehow precompute a bunch of incoming data points before proceeding
case class SeriesGrouper[T, V](maker: DeltaSeriesMaker[V], accumulator: Vector[T], convert: Seq[T] => V, groupSize: Int):
  private def updatedMaker: DeltaSeriesMaker[V] = (convert andThen maker.addItem)(accumulator take groupSize).processItems
  def process: SeriesGrouper[T, V] = if accumulator.size >= groupSize then copy(updatedMaker, accumulator drop groupSize).process else this
  def addItem(item: T): SeriesGrouper[T, V] = copy(accumulator = accumulator :+ item)
