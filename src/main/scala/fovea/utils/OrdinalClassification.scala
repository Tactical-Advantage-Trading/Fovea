package fovea.utils

import ai.djl.ndarray.NDArray

// https://arxiv.org/pdf/0704.1028.pdf

object OrdinalClassification:
  def encode(array: NDArray): NDArray =
    // 2 -> [1, 1]; 1 -> [1, 0]; 0 -> [0, 0]
    val maxLabel: Int = array.max.getFloat(/* no index */).toInt
    require(array.min.eq(0D).getBoolean(/* no index */), "Classification label must start from 0 and go upwards")
    val res = for label <- array.toArray yield Array.fill(label.intValue)(1F) ++ Array.fill(maxLabel - label.intValue)(0F)
    array.getManager.create(res)

  def decode(array: NDArray): NDArray =
    // [1, 1] -> 2; [1, 0] -> 1; [0, 0] -> 0
    // Bad encoding like [0, 1] decoded as -1
    val labelDimention = array.getShape.tail.toInt
    array.getManager.create(array.toArray.map(_.floatValue).grouped(labelDimention).map {
      case ordEncoding if ordEncoding.head < 1F && ordEncoding.tail.exists(_ > 0F) => -1
      case ordEncoding => ordEncoding.takeWhile(_ > 0F).length
    }.toArray)
