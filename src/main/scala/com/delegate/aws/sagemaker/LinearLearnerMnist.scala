package com.delegate.aws.sagemaker

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.algorithms.LinearLearnerBinaryClassifier
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession

object LinearLearnerMnist extends App {
  val conf = ConfigFactory.load()

  val spark = SparkSession.builder.master("local[*]").getOrCreate
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", conf.getString("aws.s3.accessKey"))
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", conf.getString("aws.s3.secretAccessKey"))

  val trainingData = spark.read.format("libsvm")
    .option("numFeatures", "784")
    .load("s3n://sagemaker-sample-data-us-east-1/spark/mnist/train")

  val testData = spark.read.format("libsvm")
    .option("numFeatures", "784")
    .load("s3n://sagemaker-sample-data-us-east-1/spark/mnist/test")

  val estimator = new LinearLearnerBinaryClassifier(
    sagemakerRole = IAMRole(conf.getString("aws.roleArn")),
    trainingInstanceType = "ml.p2.xlarge",
    trainingInstanceCount = 1,
    endpointInstanceType = "ml.c4.xlarge",
    endpointInitialInstanceCount = 1
  ).setBinaryClassifierModelSelectionCriteria("f1").setFeatureDim(784)

  val model = estimator.fit(trainingData)

  val transformedData = model.transform(testData)
  transformedData.show
}
