package fovea.utils

import ai.djl.engine.Engine
import ai.djl.inference.Predictor
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.{AbstractBlock, Parameter}
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.evaluator.AbstractAccuracy
import ai.djl.training.initializer.{Initializer, XavierInitializer}
import ai.djl.training.listener.{SaveModelTrainingListener, TrainingListener}
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.training.{Trainer, TrainingResult}
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList
import ai.djl.{Device, Model}
import org.apache.commons.io.FileUtils
import org.slf4j.Logger

import java.io.{File, FileInputStream, FileOutputStream}
import java.nio.file.{Path, Paths}
import java.util.stream.Collectors
import scala.util.{Random, Using}


extension(array: NDArray)
  def asNDList: NDList = NDList(array)
  
  def doubleValueArray: Array[Double] = array.toArray.map(_.doubleValue)

  def approxEquals(secondArray: NDArray, epsilon: Double): Boolean =
    val (res1, res2) = (array sub secondArray, secondArray sub array)
    res1.get(res1 gte epsilon).isEmpty && res2.get(res2 gte epsilon).isEmpty


extension(value: Double)
  def approxEquals(other: Double, epsilon: Double): Boolean =
    val absoluteDifference = math.abs(other - value)
    absoluteDifference <= epsilon


object ParallelBlockUtils:
  type ListOfNDLists = java.util.List[NDList]
  
  def flattenSingletonHeads(listOfNDLists: ListOfNDLists): NDList =
    val heads = listOfNDLists.stream.map(_.singletonOrThrow).collect(Collectors.toList)
    NDList(heads)


object Common:
  type Bytes = Array[Byte]
  type Features = Array[Double]
  type ShapeArray = Array[Shape]
  type ParamsList = PairList[String, Object]

  // Represents a single observation and followup value
  case class SingleObservation(features: Features, value: Double)
  // Represents a series of feature observations and a followup value
  case class SerialObservations(features: Array[Features], value: Double)

  def featuresToNDArray(features: List[SerialObservations], targetShape: Shape, manager: NDManager, logger: Logger): (NDArray, NDArray) =
    val (splitFeatures, splitValues) = Random.shuffle(features).map(container => container.features.flatten -> container.value).unzip
    val finalFeatures = manager.create(splitFeatures.toArray).reshape(targetShape).toType(DataType.FLOAT32, false)
    val finalValues = manager.create(splitValues.toArray).toType(DataType.FLOAT32, false)
    logger.info(splitValues.groupBy(identity).view.mapValues(_.size).toMap.toString)
    (finalFeatures, finalValues)

  def logBase(x: Double, base: Int): Double =
    math.log10(x) / math.log10(base)

  def saveNDList(list: NDList, file: File): Unit =
    val output: FileOutputStream = new FileOutputStream(file)
    output.write(list.encode)
    output.close

  def loadNDList(manager: NDManager, file: File): NDList =
    val input: FileInputStream = new FileInputStream(file)
    val bytes = ai.djl.util.Utils.toByteArray(input)
    val result = NDList.decode(manager, bytes)
    input.close
    result

  def unpackNDList(list: NDList, parts: Int): List[NDArray] =
    // Storage format for train/test/validation splits
    (0 until parts).toList.map(list.get)

  def cleanupInferiorModels(bestDir: File): File =
    // Removes all inferior models and renames the best one
    bestDir.getParentFile.listFiles.toList.filterNot(bestDir.==).foreach(FileUtils.deleteDirectory)
    val newDir = new File(bestDir.getParentFile.getPath + "/" + System.currentTimeMillis + "/")
    bestDir.renameTo(newDir)
    newDir

  def loadModel(parentDir: String): File =
    val parentDirFile = new File(parentDir)
    // DJL expects a directory which contains a model file
    // Loads a single best model which was left after removing all inferior ones
    require(parentDirFile.listFiles.length == 1, "Parent dir must contain a single model dir")
    parentDirFile.listFiles.filter(_.isDirectory).head

  // Hides some training boilerplate
  // Taining and test NDArrays must be attached to a submanager
  def blockingTrainClassifier(trainFeatures: NDArray, trainLabels: NDArray, testFeatures: NDArray, testLabels: NDArray,
                              modelClassifierBlock: AbstractBlock, inputShape: Shape, miniBatchSize: Int, shuffle: Boolean,
                              listener: BestSaveModelTrainingListener, lossFunction: Loss, accuracy: AbstractAccuracy,
                              trainEpochs: Int, initializer: Initializer, modelName: String): Unit =

    val cfg =
      val learnRate = Tracker.fixed(0.004)
      val gpu = Engine.getInstance.getDevices(12)
      ai.djl.training.DefaultTrainingConfig(lossFunction)
        .optOptimizer(Optimizer.adamW.optLearningRateTracker(learnRate).build)
        .addTrainingListeners(TrainingListener.Defaults.basic *)
        .optInitializer(initializer, Parameter.Type.WEIGHT)
        .addTrainingListeners(listener)
        .addEvaluator(accuracy)
        .optDevices(gpu)

    val trainDataSet =
      (new ArrayDataset.Builder)
        .setSampling(miniBatchSize, shuffle)
        .setData(trainFeatures)
        .optLabels(trainLabels)
        .build

    val testDataSet =
      (new ArrayDataset.Builder)
        .setSampling(miniBatchSize, shuffle)
        .setData(testFeatures)
        .optLabels(testLabels)
        .build

    val model = Model.newInstance(modelName)
    model.setBlock(modelClassifierBlock)
    val trainer = model.newTrainer(cfg)
    trainer.initialize(inputShape)

    ai.djl.training.EasyTrain.fit(trainer, trainEpochs, trainDataSet, testDataSet)
    List(trainer, model).foreach(_.close)
