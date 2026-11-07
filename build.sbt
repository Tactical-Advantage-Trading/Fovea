ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.8"

lazy val root = project in file(".") settings (name := "Fovea")

libraryDependencies += "org.scalatest" % "scalatest_3" % "3.2.20"

libraryDependencies += "com.github.tknudsen" % "complex-data-object" % "0.2.13"

libraryDependencies += "commons-io" % "commons-io" % "20030203.000550"

libraryDependencies += "joda-time" % "joda-time" % "2.15.1"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "2.0.20"

libraryDependencies += "org.scala-lang.modules" % "scala-parallel-collections_3" % "1.3.0"

// DNN

libraryDependencies += "ai.djl" % "api" % "0.40.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-engine" % "0.40.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-jni" % "2.7.1-0.40.0"

libraryDependencies += "tech.tablesaw" % "tablesaw-jsplot" % "0.44.4"
