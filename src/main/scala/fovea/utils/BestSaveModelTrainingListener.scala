package fovea.utils

import ai.djl.training.{Trainer, TrainingResult}
import ai.djl.training.listener.SaveModelTrainingListener

import java.io.File
import java.nio.file.Paths


// DJL saves a number of snapshots in a directory and then loads the one with greatest epoch suffix
// the greatest one may not be the best one so we save each snapshot into a separate dir so it will be the only file in that dir
abstract class BestSaveModelTrainingListener(dir: File, checkpoint: Int, val saveAfter: Int, accuracyName: String) extends SaveModelTrainingListener(dir.getAbsolutePath, null, checkpoint):

  def this(dir: String, checkpoint: Int, saveAfter: Int, accuracyName: String) = this(new File(dir), checkpoint, saveAfter, accuracyName)

  def onUpdate(epoch: Int, accuracy: Float, trailLoss: Float, validationLoss: Float): Unit

  override def onEpoch(mt: Trainer): Unit =
    val result: TrainingResult = mt.getTrainingResult
    val accuracy = result.getValidateEvaluation(accuracyName)
    onUpdate(result.getEpoch, accuracy, result.getTrainLoss, result.getValidateLoss)
    super.onEpoch(mt)

  override def saveModel(mt: Trainer): Unit =
    Option(mt.getTrainingResult.getEpoch).collect { 
      case trainingEpoch if trainingEpoch >= saveAfter =>
        val path = Paths.get(dir.getAbsolutePath + "/" + trainingEpoch + "/")
        mt.getModel.setProperty("Epoch", trainingEpoch.toString)
        mt.getModel.save(path, mt.getModel.getName)
    }