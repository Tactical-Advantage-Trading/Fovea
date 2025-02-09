package fovea.utils

import scala.annotation.tailrec


// Downloads data by time chunks, skips existing chunks, skips empty download results up to reserve retries
abstract class SequentialTimestampDownloader[T](ops: SequentialTimestampFileOps[T], stepMsecs: Long):
  def downloadBatch(fromMsecs: Long, toMsecs: Long): List[T]
  def log(message: String): Unit

  def skipExistingDownload(fromMsecs: Long, reserve: Int): Unit =
    skipExisting(fromMsecs, reserve, reserve)

  @tailrec
  private def downloadInternal(fromMsecs: Long, remains: Int, reserve: Int): Unit =
    val nextTimeSpan = fromMsecs + stepMsecs
    log(s"$fromMsecs - $nextTimeSpan")

    downloadBatch(fromMsecs, nextTimeSpan) match
      case Nil if remains <= 0 =>
        log("Download complete")

      case Nil =>
        log(s"Skipped empty result, remains=$remains")
        skipExisting(nextTimeSpan, remains - 1, reserve)

      case batch =>
        log(s"Downloaded ${batch.size} items")
        ops.persist(ops.encode(batch), fromMsecs)
        downloadInternal(nextTimeSpan, reserve, reserve)

  @tailrec
  private def skipExisting(fromMsecs: Long, remains: Int, reserve: Int): Unit =
    log(s"Downloading $fromMsecs unless it exists already, $remains")
    val file = new java.io.File(ops.saveDir.getAbsolutePath + "/" + fromMsecs)
    if file.exists then skipExisting(fromMsecs + stepMsecs, remains, reserve)
    else downloadInternal(fromMsecs, remains, reserve)