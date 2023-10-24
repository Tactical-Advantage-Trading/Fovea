ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.2.1"

lazy val root = project in file(".") settings (name := "Fovea")

libraryDependencies += "org.scalatest" % "scalatest_3" % "3.2.15"

libraryDependencies += "com.github.tknudsen" % "complex-data-object" % "0.2.13"

libraryDependencies += "commons-io" % "commons-io" % "20030203.000550"

libraryDependencies += "joda-time" % "joda-time" % "2.12.2"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "2.0.5"

libraryDependencies += "org.scala-lang.modules" % "scala-parallel-collections_3" % "1.0.4"

// DNN

libraryDependencies += "ai.djl" % "api" % "0.21.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-engine" % "0.21.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-jni" % "1.13.1-0.21.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-native-cu117" % "1.13.1"

libraryDependencies += "tech.tablesaw" % "tablesaw-jsplot" % "0.43.1"
