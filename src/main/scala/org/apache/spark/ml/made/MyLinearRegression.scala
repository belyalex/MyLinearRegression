package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.functions.col

private trait MyLinearRegressionParams extends PredictorParams


class MyLinearRegression private[made](override val uid: String)
  extends Regressor[Vector, MyLinearRegression, MyLinearRegressionModel] {
  def this() = this(Identifiable.randomUID("myLinReg"))

  override def copy(extra: ParamMap): MyLinearRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): MyLinearRegressionModel = {
    val m = getDenseMatrixFromDS(dataset.select(col($(featuresCol))))
    val y = getDenseVectorFromDS(dataset.select(col($(labelCol))))

    val M = DenseMatrix.horzcat(DenseMatrix.ones[Double](m.rows, 1), m)

    val coefficients = inv(M.t * M) * M.t * y

    copyValues(new MyLinearRegressionModel(coefficients)).setParent(this)
  }

  def getDenseMatrixFromDS(ds: Dataset[_]): breeze.linalg.DenseMatrix[Double] = {
    val featuresTrain = ds.columns
    val rows = ds.count().toInt

    val newFeatureArray: Array[Double] = featuresTrain
      .indices
      .flatMap(i => ds
        .select(featuresTrain(i))
        .collect())
      .map(r => r.toSeq.toArray).toArray.flatten.flatMap(_.asInstanceOf[org.apache.spark.ml.linalg.DenseVector].values)

    val newCols = newFeatureArray.length / rows
    val denseMat = new breeze.linalg.DenseMatrix[Double](newCols, rows, newFeatureArray).t
    denseMat
  }

  def getDenseVectorFromDS(featuresDS: Dataset[_]): breeze.linalg.DenseVector[Double] = {
    val featuresTrain = featuresDS.columns
    val cols = featuresDS.columns.length

    cols match {
      case i if i > 1 => throw new IllegalArgumentException
      case _ =>
        def addArray(acc: Array[Array[Double]], cur: Array[Double]): Array[Array[Double]] = {
          acc :+ cur
        }

        val newFeatureArray: Array[Double] = featuresTrain
          .indices
          .flatMap(i => featuresDS
            .select(featuresTrain(i))
            .collect())
          .map(r => r.toSeq.toArray.map(e => e.asInstanceOf[Double])).toArray.flatten

        val denseVec = new breeze.linalg.DenseVector[Double](newFeatureArray)
        denseVec
    }
  }
}

class MyLinearRegressionModel private[made](override val uid: String, val coefficients: breeze.linalg.Vector[Double])
  extends RegressionModel[Vector, MyLinearRegressionModel] with MLWritable {
  def this(coefficients: breeze.linalg.Vector[Double]) = this(Identifiable.randomUID("myLinRegModel"), coefficients)

  override def copy(extra: ParamMap): MyLinearRegressionModel = copyValues(new MyLinearRegressionModel(coefficients))

  override def predict(features: Vector): Double = {
    val x = DenseVector.vertcat(DenseVector(1.0), features.asBreeze.toDenseVector)
    x dot coefficients
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val coeffs = Tuple1(Vectors.fromBreeze(coefficients))

      sqlContext.createDataFrame(Seq(coeffs)).write.parquet(path + "/coefficients")
    }
  }
}


object MyLinearRegressionModel extends MLReadable[MyLinearRegressionModel] {
  override def read: MLReader[MyLinearRegressionModel] = new MLReader[MyLinearRegressionModel] {
    override def load(path: String): MyLinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val coeffs = sqlContext.read.parquet(path + "/coefficients")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val coefficients = coeffs.select(coeffs("_1").as[Vector]).first().asBreeze

      val model = new MyLinearRegressionModel(coefficients)
      metadata.getAndSetParams(model)
      model
    }
  }
}
