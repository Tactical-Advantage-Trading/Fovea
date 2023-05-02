package fovea.utils


class SequentialTimestampSplitter[T](ops: SequentialTimestampFileOps[T], takeItems: Int):
  // Assumes all files are numerically named and file name defines an ordering

  type FeatureList = List[T]

  def getAllFeatures: FeatureList =
    ops.getSortedFeatureChunks(takeItems)
      .reduce(_ ++ _)

  def split(features: FeatureList, splits: Double*): List[FeatureList] =
    val chunks = (0D +: splits :+ 1D).map(_ * features.size).map(_.toInt).toList
    chunks.zip(chunks.tail).map(features.slice)
