package com.delegate.aws.sagemaker

import com.amazonaws.services.sagemaker.sparksdk.IAMRole
import com.amazonaws.services.sagemaker.sparksdk.algorithms.LinearLearnerBinaryClassifier
import org.apache.spark.sql.SparkSession

object LinearLearnerMnist extends App {

  val spark = SparkSession.builder.master("local[*]").getOrCreate
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "AKIAI3RBG42G5OOCMPOQ")
  spark.sparkContext.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "NkSYwQL0XkDhh73e5bZF1HX1fiIfZU5yIFTeQvK+")

  val trainingData = spark.read.format("libsvm")
    .option("numFeatures", "784")
    .load("s3n://sagemaker-sample-data-us-east-1/spark/mnist/train")

  val testData = spark.read.format("libsvm")
    .option("numFeatures", "784")
    .load("s3n://sagemaker-sample-data-us-east-1/spark/mnist/test")

  val roleArn = "arn:aws:iam::130291900959:role/service-role/AmazonSageMaker-ExecutionRole-20180125T083005"

  val estimator = new LinearLearnerBinaryClassifier(
    sagemakerRole = IAMRole(roleArn),
    trainingInstanceType = "ml.p2.xlarge",
    trainingInstanceCount = 1,
    endpointInstanceType = "ml.c4.xlarge",
    endpointInitialInstanceCount = 1
  ).setBinaryClassifierModelSelectionCriteria("f1").setFeatureDim(784)

  val model = estimator.fit(trainingData)

  val transformedData = model.transform(testData)
  transformedData.show
}
