package fovea.features

// https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

object CyclicTime:
  val MSECS_IN_DAY: Int = 24 * 3600 * 1000
  val DAYS_IN_WEEK: Int = 7

  def timeOfDaySinCos(stampMsec: Long): (Double, Double) = 
    val millisecondsSinceMidnight: Long = stampMsec % MSECS_IN_DAY
    val sin = math.sin(millisecondsSinceMidnight * 2 * math.Pi / MSECS_IN_DAY)
    val cos = math.cos(millisecondsSinceMidnight * 2 * math.Pi / MSECS_IN_DAY)
    (sin, cos)

  def dayOfWeekSinCos(stampMsec: Long): (Double, Double) = 
    val dayOfWeek: Int = new org.joda.time.DateTime(stampMsec).dayOfWeek.get
    val sin = math.sin(dayOfWeek * 2 * math.Pi / DAYS_IN_WEEK)
    val cos = math.cos(dayOfWeek * 2 * math.Pi / DAYS_IN_WEEK)
    (sin, cos)
