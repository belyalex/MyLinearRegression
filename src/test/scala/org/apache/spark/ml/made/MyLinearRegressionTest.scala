package org.apache.spark.ml.made

import java.io.File

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.scalatest.flatspec._
import org.scalatest.matchers._

class MyLinearRegressionTest extends AnyFlatSpec with should.Matchers {
  val spark: SparkSession = SparkSession.builder
    .appName("MyLinearRegressionTest Application")
    .master("local[4]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  val sqlc: SQLContext = spark.sqlContext


  def read_csv(fName: String): DataFrame = {
    val schema: StructType = new StructType()
      .add("Frequency", DoubleType)
      .add("Angle", DoubleType)
      .add("Length", DoubleType)
      .add("Velocity", DoubleType)
      .add("Thickness", DoubleType)
      .add("Noise", DoubleType)

    val df = sqlc.read.format("csv")
      .option("header", "true")
      .schema(schema)
      .load(fName)

    val assembler = new VectorAssembler()
      .setInputCols(Array("Frequency", "Angle", "Length", "Velocity", "Thickness"))
      .setOutputCol("features")

    assembler
      .transform(df)
      .drop("Frequency", "Angle", "Length", "Velocity", "Thickness")

  }

  val df_train: DataFrame = read_csv("train.csv")
  val df_test: DataFrame = read_csv("test.csv")

  def check_model(rm: MyLinearRegressionModel, df_train: DataFrame, df_test: DataFrame): Unit = {
    println(s"Коэффициенты: ${rm.coefficients}")

    val dfp_train = rm.transform(df_train)
    val dfp_test = rm.transform(df_test)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("Noise")
      .setPredictionCol("Pred")
      .setMetricName("r2")

    val r2_train = evaluator.evaluate(dfp_train)
    val r2_test = evaluator.evaluate(dfp_test)
    println(s"Валидация R2 score train: $r2_train, test: $r2_test")
    r2_train should be(0.52 +- 0.005)
    r2_test should be(0.49 +- 0.005)
  }

  "Model" should "work" in {
    val lr = new MyLinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("Noise")
      .setPredictionCol("Pred")

    val rm = lr.fit(df_train)

    check_model(rm, df_train, df_test)
  }

  "Model" should "work after save-load" in {
    val pipeline = new Pipeline().setStages(Array(
      new MyLinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("Noise")
        .setPredictionCol("Pred")
    ))

    val model = pipeline.fit(df_train)
    val modelFolder = new File("model")
    modelFolder.mkdir
    model.write.overwrite().save(modelFolder.getAbsolutePath)

    val reRead = PipelineModel.load(modelFolder.getAbsolutePath)

    check_model(reRead.stages(0).asInstanceOf[MyLinearRegressionModel], df_train, df_test)
  }


}
