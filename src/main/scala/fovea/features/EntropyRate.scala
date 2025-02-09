package fovea.features

// Advances In Financial Machine Learning: Entropy features chapter

object EntropyRate:
  type Message = Seq[Char]
  final val alphabet: Message = Vector('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')

  def longestMatchLength(msg: Message, startIdx: Int, window: Int): LazyList[Int] =
    require(startIdx >= window, s"startIdx=$startIdx must be >= than window=$window")
    require(msg.size >= startIdx + window, s"Message length must be >= startIdx + window")

    for
      length <- LazyList.from(window until 0 by -1)
      canonical = msg.slice(startIdx, startIdx + length)
      candidateIdx <- LazyList.from(startIdx - window until startIdx)
      candidate = msg.slice(candidateIdx, candidateIdx + length)
      if canonical == candidate
    yield canonical.size + 1

  def konto(msg: Message): Double =
    require(msg.nonEmpty, "Message can not be empty")
    val points = (1 until msg.size / 2 + 1).toList

    points.map { index =>
      val length = longestMatchLength(msg, index, index).headOption
      fovea.utils.Common.logBase(index + 1, base = 2) / length.getOrElse(1)
    }.sum / points.size

  def quantize(value: Double): Char =
    require(value >= 0D && value <= 1D)
    val index = (value * 10).floor.toInt
    alphabet(index)

  def maxPossibleEntropy(alphabetSize: Int): Double =
    fovea.utils.Common.logBase(alphabetSize, base = 2)
