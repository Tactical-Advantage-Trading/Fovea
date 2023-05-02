package fovea

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.Activation
import ai.djl.nn.core.Linear
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import com.github.TKnudsen.ComplexDataObject.model.io.arff.ARFFInstancesIO
import fovea.utils.*
import org.scalatest.Ignore
import org.scalatest.funsuite.AnyFunSuite


@Ignore
class TimeSeriesAttentionSpec extends AnyFunSuite {
  private val manager = NDManager.newBaseManager

  test("EEG face detection") {
    val miniBatchSize: Int = 64
    val sequenceLength: Int = 62
    val featureDimention: Int = 144

    val dir = System.getProperty("user.dir")
    // Download dataset from http://www.timeseriesclassification.com/description.php?Dataset=FaceDetection, put ARFF files into "resources/FaceDetection" folder
    val (trainFeatures, trainLabels) = getFaceDetectionData("FaceDetection/FaceDetection_TRAIN.arff", datasetSize = 5890, sequenceLength, featureDimention)
    val (testFeatures, testLabels) = getFaceDetectionData("FaceDetection/FaceDetection_TEST.arff", datasetSize = 3524, sequenceLength, featureDimention)

    val accuracy = new Accuracy
    val listener = new BestSaveModelTrainingListener(dir, checkpoint = 2, saveAfter = Int.MaxValue, accuracy.getName):
      override def onUpdate(epoch: Int, accuracy: Float, trainLoss: Float, testLoss: Float): Unit =
        println(s"$epoch: accuracy=$accuracy, trainLoss=$trainLoss, testLoss=$testLoss")

    val output = Linear.builder.optBias(true).setUnits(2).build
    val projection = Linear.builder.optBias(true).setUnits(128).build
    val posEncoder = encoding.LearnablePositionalEncoding(sequenceLength, encodingDim = 128, dropoutRate = 0.3, manager)
    val classifier = TimeSeriesAttention(projection, posEncoder, Activation.geluBlock, headCount = 8, hiddenSize = 128, encoderDropoutRate = 1, finalDropoutRate = 0.9, output)

    Common.blockingTrainClassifier(trainFeatures, trainLabels, testFeatures, testLabels,
      classifier, inputShape = Shape(miniBatchSize, sequenceLength, featureDimention), miniBatchSize, shuffle = false,
      listener, Loss.softmaxCrossEntropyLoss, accuracy, trainEpochs = 100, new XavierInitializer, modelName = "EEGSeriesFaceDetection")
  }

  private def getFaceDetectionData(name: String, datasetSize: Int, sequenceLength: Int, featureDimention: Int) =
    val path = getClass.getClassLoader.getResource(name)
    val arff = ARFFInstancesIO.loadARFF(path.getPath)

    val series = for
      indexInstance <- 0.until(arff.numInstances).toArray
      rel = arff.get(indexInstance).attribute(0).relation(indexInstance)
      indexSeriesOfFeatures <- 0.until(rel.numInstances).toArray
      features = rel.instance(indexSeriesOfFeatures)
      idx <- 0.until(features.numValues).toArray
    yield features.value(idx)

    val labels = for
      indexInstance <- 0.until(arff.numInstances).toArray
    yield arff.get(indexInstance).value(1)

    val invertedShape = Shape(datasetSize, featureDimention, sequenceLength)
    val ndFeatures = manager.create(series, invertedShape).toType(DataType.FLOAT32, false)
    val ndLabels = manager.create(labels).toType(DataType.FLOAT32, false)
    // Transposing because of specifics of ARFF encoding format
    // This dataset is already normalized (mean=0, sd=1)
    (ndFeatures.transpose(0, 2, 1), ndLabels)
}
