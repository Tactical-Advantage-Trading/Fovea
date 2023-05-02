package fovea.utils

import java.io.File


abstract class SequentialTimestampDownloader[T](ops: SequentialTimestampFileOps[T], stepMsecs: Long):
  // Used for chunked fetching of raw data from a possibly remote source
  
  def downloadBatch(fromMsecs: Long, toMsecs: Long): List[T]

  def log(message: String): Unit

  def skipExistingDownload(fromMsecs: Long): Unit = 
    log(s"Downloading $fromMsecs unless it exists already")
    val file = new File(ops.saveDir.getAbsolutePath + "/" + fromMsecs)
    if file.exists then skipExistingDownload(fromMsecs + stepMsecs) else download(fromMsecs)

  def download(fromMsecs: Long): Unit = 
    val nextTimeSpan = fromMsecs + stepMsecs
    log(s"Downloading $fromMsecs - $nextTimeSpan")
    val batch = downloadBatch(fromMsecs, nextTimeSpan)
    if batch.isEmpty then log("Download complete") else {
      ops.persist(ops.encode(batch), fromMsecs)
      log(s"Downloaded ${batch.size} items")
      download(nextTimeSpan)
    }
