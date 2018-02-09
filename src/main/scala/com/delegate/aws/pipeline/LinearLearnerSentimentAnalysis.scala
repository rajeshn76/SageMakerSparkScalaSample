package com.delegate.aws.pipeline

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.algorithms.LinearLearnerBinaryClassifier
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession

object LinearLearnerSentimentAnalysis extends App {
  val conf = ConfigFactory.load()

  val spark = SparkSession.builder.master("local[*]").getOrCreate
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", conf.getString("aws.s3.accessKey"))
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", conf.getString("aws.s3.secretAccessKey"))

  val inputData = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(conf.getString("aws.s3.inputFile"))

  val reviewData = inputData.selectExpr("review", "cast(sentiment as double) label")

  val tokenizer = new Tokenizer().setInputCol("review").setOutputCol("words")
  val wordsData = tokenizer.transform(reviewData)

  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features").setNumFeatures(conf.getInt("aws.sageMaker.featureDim"))
  val featurizedData = hashingTF.transform(wordsData)

  val Array(trainingData, testData) = featurizedData
    .select("review", "features", "label")
    .randomSplit(Array[Double](0.7, 0.3), 1)
  testData.show

  val estimator = new LinearLearnerBinaryClassifier(
    sagemakerRole = IAMRole(conf.getString("aws.roleArn")),
    trainingInstanceType = "ml.p2.xlarge",
    trainingInstanceCount = 1,
    endpointInstanceType = "ml.c4.xlarge",
    endpointInitialInstanceCount = 1
  ).setBinaryClassifierModelSelectionCriteria("f1").setFeatureDim(conf.getInt("aws.sageMaker.featureDim"))

  val pipeline = new Pipeline().setStages(Array(estimator))

  //train
  val model = pipeline.fit(trainingData)

  val transformedData = model.transform(testData)
  transformedData.show
}
