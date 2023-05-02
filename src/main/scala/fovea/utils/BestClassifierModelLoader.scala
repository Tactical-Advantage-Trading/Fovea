package fovea.utils

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.nn.AbstractBlock

import java.io.File
import scala.util.Using


trait ClassifierEvaluation {
  def merge(that: ClassifierEvaluation): ClassifierEvaluation
  val score: Double
  val dir: File
}

// It expects a parent directory with a number of subdirectories, each must contain a single model file (see BestSaveModelTrainingListener)
abstract class BestClassifierModelLoader(parentDir: File, classifier: AbstractBlock, modelName: String, features: NDArray, labels: NDArray, chunkSize: Int, device: Device):

  def this(parentDir: String, classifier: AbstractBlock, modelName: String, features: NDArray, labels: NDArray, chunkSize: Int, device: Device) =
    this(new File(parentDir), classifier, modelName, features, labels, chunkSize, device)

  require(parentDir.exists && parentDir.isDirectory, "Parent dir must exist and be a directory")
  require(features.getShape.head == labels.getShape.head, "Features and labels must have the same length")

  def doEvaluate(networkOutput: NDArray, labelsChunk: NDArray, dir: File): ClassifierEvaluation

  def extractWithSubmanager(items: NDArray, span: String): NDArray =
    val subManager = items.getManager.newSubManager
    val extractedChunk = items.get(span)
    extractedChunk.attach(subManager)
    extractedChunk

  def evaluateModel(subdirectory: File): ClassifierEvaluation =
    val chunks = math.ceil(features.getShape.head / chunkSize.toDouble).toInt
    val helper = PredictionHelper(subdirectory.getAbsolutePath, classifier, modelName, device)

    val results = for
      idx <- 0 until chunks
      start = idx * chunkSize
      span = s"$start:${start + chunkSize}"
      labelsChunk = extractWithSubmanager(labels, span)
      featuresChunk = extractWithSubmanager(features, span)
      prediction = helper.predictor.predict(featuresChunk.asNDList)
      eval = doEvaluate(prediction.head, labelsChunk, subdirectory)
      _ = featuresChunk.getManager.close
      _ = labelsChunk.getManager.close
    yield eval

    helper.freeUsedResources
    results.reduce(_ merge _)

  def bestModelDir: ClassifierEvaluation =
    parentDir.listFiles.toList
      .filter(_.isDirectory)
      .map(evaluateModel)
      .maxBy(_.score)
    