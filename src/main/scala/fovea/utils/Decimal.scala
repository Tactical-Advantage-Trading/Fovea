package fovea.utils

import scala.annotation.targetName


// A rounding Double without undetermined behavior at the end of the number
class Decimal(private val dbl: Double) extends AnyVal with Ordered[Decimal]:

  @targetName("div")
  def /(that: Decimal): Decimal = Decimal(this.dbl / that.dbl)

  @targetName("add")
  def +(that: Decimal): Decimal = Decimal(this.dbl + that.dbl)
  
  @targetName("sub")
  def -(that: Decimal): Decimal = Decimal(this.dbl - that.dbl)

  @targetName("mul")
  def *(that: Decimal): Decimal = Decimal(this.dbl * that.dbl)

  def unary_- = Decimal(-dbl)
  
  def toInt: Int = dbl.toInt

  def toLong: Long = dbl.toLong

  def toDouble: Double = dbl

  override def compare(that: Decimal): Int = java.lang.Double.compare(dbl, that.dbl)
  
  @targetName("eqInt")
  def ==(that: Int): Boolean = this.dbl == that.toDouble

  @targetName("eqLong")
  def ==(that: Long): Boolean = this.dbl == that.toDouble

  @targetName("eqDecimal")
  def ==(that: Decimal): Boolean = this.dbl == that.dbl

  override def toString: String =
    if dbl == dbl.toInt then dbl.toInt.toString
    else dbl.toString

  def canEqual(other: Any): Boolean =
    other.isInstanceOf[Decimal]

  override def equals(other: Any): Boolean = other match
    case that: Decimal => (that canEqual this) && dbl == that.dbl
    case that: Long => dbl == that.toDouble
    case that: Int => dbl == that.toDouble
    case that: Double => dbl == that
    case _ => false


object Decimal:
  val DEFAULT_SCALE_FACTOR: Double = 1e10

  def apply(value: Double)(implicit factor: Double = DEFAULT_SCALE_FACTOR): Decimal =
    if value > Long.MaxValue / factor || value < -Long.MaxValue / factor then new Decimal(dbl = value)
    else new Decimal(dbl = (if value < 0 then value * factor - 0.5 else value * factor + 0.5).toLong / factor)

  given Conversion[Double, Decimal] = Decimal.apply
  given Conversion[Long, Decimal] = Decimal.apply
  given Conversion[Int, Decimal] = Decimal.apply

  given CanEqual[Decimal, Double] = CanEqual.derived
  given CanEqual[Decimal, Long] = CanEqual.derived
  given CanEqual[Decimal, Int] = CanEqual.derived
