package fovea.utils

import java.io.File


abstract class SequentialTimestampDownloader[T](ops: SequentialTimestampFileOps[T], stepMsecs: Long):
  // Used for chunked fetching of raw data from a possibly remote source
  
  def downloadBatch(fromMsecs: Long, toMsecs: Long): List[T]

  def log(message: String): Unit

  def skipExistingDownload(fromMsecs: Long, remains: Int): Unit =
    log(s"Downloading $fromMsecs unless it exists already, $remains")
    val file = new File(ops.saveDir.getAbsolutePath + "/" + fromMsecs)
    if (file.exists) skipExistingDownload(fromMsecs + stepMsecs, remains)
    else download(fromMsecs, remains)

  def download(fromMsecs: Long, remains: Int): Unit =
    val nextTimeSpan = fromMsecs + stepMsecs
    log(s"$fromMsecs - $nextTimeSpan")

    downloadBatch(fromMsecs, nextTimeSpan) match
      case Nil if remains <= 0 =>
        log("Download complete")

      case Nil =>
        log(s"Skipped empty result, remains=$remains")
        skipExistingDownload(nextTimeSpan, remains - 1)

      case batch =>
        log(s"Downloaded ${batch.size} items")
        ops.persist(ops.encode(batch), fromMsecs)
        download(nextTimeSpan, remains)
