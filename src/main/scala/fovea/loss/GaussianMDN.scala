package fovea.loss

import ai.djl.ndarray.{NDArray, NDList}
import ai.djl.nn.core.Linear
import ai.djl.nn.{Block, ParallelBlock, SequentialBlock}
import ai.djl.training.loss.Loss

// https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
// https://pdfs.semanticscholar.org/d4ca/18249446328c86d9da295a21c679aea1ed77.pdf

object GaussianMDN:
  val normalizeFactor: Double = 1 / math.sqrt(math.Pi * 2)

  def gaussian(mu: NDArray, sigma: NDArray, y: NDArray): NDArray =
    val expPart = y.sub(mu).div(sigma).square.mul(-0.5).exp
    expPart.div(sigma).mul(normalizeFactor)

  def crossEntropyLoss(dim: Array[Int], coef: NDArray, mu: NDArray, sigma: NDArray, y: NDArray): NDArray =
    // A custom objective function for Gaussian mixture density networks, approach is similar to cross-entropy loss
    // dim should be Array(1) if we use minibatches of data or Array(0) if we do not use a minibatches
    gaussian(mu, sigma, y).mul(coef).sum(dim, true).log.neg.mean

  def outputBlock(sigmaActivation: fovea.activation.ActivationFunction, numGaussians: Int): Block =
    // Coefficients must sum to 1 to make it a proper PDF and sigmas (SDs) for Gaussian can't be negative
    val coefficients = Linear.builder.setUnits(numGaussians)
    val sigmas = Linear.builder.setUnits(numGaussians)
    val means = Linear.builder.setUnits(numGaussians)

    val coefsBlock = (new SequentialBlock).add(coefficients.build).add(fovea.activation.Softmax(axis = -1).block)
    val sigmasBlock = (new SequentialBlock).add(sigmas.build).add(sigmaActivation.block)
    val slots = java.util.Arrays.asList[Block](coefsBlock, means.build, sigmasBlock)
    ParallelBlock(fovea.utils.ParallelBlockUtils.flattenSingletonHeads, slots)


class GaussianMDN extends Loss("GaussianMDNLoss"):
  override def evaluate(label: NDList, prediction: NDList): NDArray =
    GaussianMDN.crossEntropyLoss(dim = Array(1), coef = prediction.get(0),
      mu = prediction.get(1), sigma = prediction.get(2), y = label.singletonOrThrow)
