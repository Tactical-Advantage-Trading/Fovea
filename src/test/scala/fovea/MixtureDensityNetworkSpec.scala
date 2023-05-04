package fovea

import ai.djl.engine.Engine
import ai.djl.inference.Predictor
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.core.Linear
import ai.djl.nn.{Activation, SequentialBlock}
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.{UniformInitializer, XavierInitializer}
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import ai.djl.{Device, Model}
import fovea.distribution.Gumbel
import fovea.loss.GaussianMDN
import fovea.utils.*
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.{Ignore, color}
import tech.tablesaw.plotly.Plot
import tech.tablesaw.plotly.api.{Histogram, ScatterPlot}
import tech.tablesaw.plotly.components.{Figure, Layout}
import tech.tablesaw.plotly.traces.{HistogramTrace, ScatterTrace}


@Ignore
class MixtureDensityNetworkSpec extends AnyFunSuite {
  private val manager = NDManager.newBaseManager
  private val datasetSize = 1000
  private val hiddenNeurons = 20

  private val datasetShape = Shape(datasetSize, 1)
  private val modelName = "InverseSinusoidal"

  private val accuracy = new Accuracy
  private val dir = System.getProperty("user.dir")
  private val listener = new BestSaveModelTrainingListener(dir, checkpoint = 2, saveAfter = 8000, accuracy.getName):
    override def onUpdate(epoch: Int, accuracy: Float, trainLoss: Float, testLoss: Float): Unit =
      if epoch % 100 == 0 then println(s"epoch=$epoch, accuracy=$accuracy, loss=$testLoss")

  private val baseBlock = (new SequentialBlock)
    .add(Linear.builder.setUnits(hiddenNeurons).build)
    .add(Activation.sigmoidBlock)

  // f(x) = 7sin(0.75x) + randomNoise
  private def generateData(low: Float, high: Float) =
    val xData = manager.randomUniform(low, high, datasetShape)
    val randomNoiseData = manager.randomNormal(datasetShape, DataType.FLOAT32)
    val yData = xData.mul(0.75).sin.mul(7).add(randomNoiseData)
    (xData, yData)

  test("Approximate an inverted noisy rising sinusoidal with MDN") {
    val net = baseBlock add GaussianMDN.outputBlock(fovea.activation.NonNegElu(alpha = 1), numGaussians = 5)

    val (xData, yData) = generateData(-11, 11)

    // Here we invert x/y data on purpose such that every input has multiple correct outputs
    Common.blockingTrainClassifier(yData, xData, yData, xData, net, inputShape = datasetShape, miniBatchSize = 250,
      shuffle = true, listener, new GaussianMDN, accuracy, trainEpochs = listener.saveAfter, new XavierInitializer, modelName)

    val helper = fovea.utils.PredictionHelper(s"$dir/${listener.saveAfter}/", net, modelName, Device.gpu)
    val List(coefs, mus, sigmas) = fovea.utils.Common.unpackNDList(helper.predictor.predict(xData.asNDList), 3)

    // Sampling from mixture of Gaussian distributions
    val sampledCoefs = Gumbel.argMaxSample(coefs, manager)
    val index = NDIndex("{}, {}", manager.arange(datasetSize), sampledCoefs)

    // First we get coefficient samples using Gumbel, then we use those to obtain Gaussians
    val sampled = sigmas.get(index).add(mus get index)
    val trace1 = ScatterTrace.builder(yData.doubleValueArray, xData.doubleValueArray).build
    val trace2 = ScatterTrace.builder(xData.doubleValueArray, sampled.doubleValueArray).build
    Plot show Figure(trace1, trace2)
  }

  test("MDN loss") {
    val y = manager.create(Array(10)).reshape(-1, 1)
    val mu = manager.create(Array(4, 5, 6)).reshape(-1, 3)
    val sigma = manager.create(Array(1, 2, 3)).reshape(-1, 3)
    val coef = manager.create(Array(0.2, 0.5, 0.3)).reshape(-1, 3)
    val res1 = (new GaussianMDN).evaluate(NDList(y), NDList(coef, mu, sigma))
    val res2 = fovea.loss.GaussianMDN.crossEntropyLoss(Array(1), coef, mu, sigma, y)
    assert(res1.toDoubleArray.head.approxEquals(3.8736, 0.0001))
    assert(res2.toDoubleArray.head.approxEquals(3.8736, 0.0001))
  }
}
