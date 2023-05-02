package fovea.utils

import java.io.File
import java.nio.file.{Files, Path}
import scala.collection.parallel.CollectionConverters._


abstract class SequentialTimestampFileOps[T](val saveDir: File):
  // Assumes all files are numerically named and file name defines an ordering
  require(saveDir.isDirectory, "saveDir must be a directory, not a file")

  def getSortedFiles: List[File] =
    saveDir.listFiles.toList.sortBy(_.getName.toLong)

  def getSortedFeatureChunks(takeItems: Int): List[DataList] =
    getSortedFiles.takeRight(takeItems).par
      .map(Path of _.getAbsolutePath)
      .map(Files.readAllBytes)
      .map(decode)
      .toList

  def persist(data: fovea.utils.Common.Bytes, fromMsecs: Long): Unit =
    Files.write(Path.of(saveDir.getAbsolutePath + "/" + fromMsecs), data)

  def encode(data: DataList): fovea.utils.Common.Bytes
  def decode(raw: Common.Bytes): DataList
  type DataList = List[T]
