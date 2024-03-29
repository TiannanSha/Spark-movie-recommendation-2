name := "m2_322517"
version := "1.0"
maintainer := "tiannan.sha@epfl.ch"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0" % Test
libraryDependencies += "org.rogach" %% "scallop" % "4.0.2"
libraryDependencies += "org.json4s" %% "json4s-jackson" % "3.6.10"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0"

scalaVersion in ThisBuild := "2.12.13"
enablePlugins(JavaAppPackaging)

