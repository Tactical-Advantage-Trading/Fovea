package fovea.utils

import ai.djl.{Device, Model}
import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.nn.AbstractBlock
import ai.djl.translate.NoopTranslator

import java.nio.file.Path


class PredictionHelper(dir: String, classifier: AbstractBlock, modelName: String, device: Device):
  lazy val predictor: Predictor[NDList, NDList] = model.newPredictor(new NoopTranslator, device)
  val model: Model = Model.newInstance(modelName)
  
  model.setBlock(classifier)
  model.load(Path of dir)

  def freeUsedResources: Unit =
    predictor.close
    model.close
    