package com.delegate.aws.client

import com.amazonaws.services.sagemaker.sparksdk.SageMakerModel
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LinearLearnerBinaryClassifierProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession

object SentimentAnalysisInference extends App {
  private val review = "i love the check deposit feature this actually made me open a chase checking account i used to " +
    "just have their credit card the only downside is the  limit but i really dont have many checks that hit that limit " +
    "everyone who is whining about their paychecks being too big to go through mine is too thank you please enter the " +
    "st century and get direct deposit the person to person quick pay is actually not quick at all it works well for " +
    "situations in which you will have repeated transactions with the same person but not just paying a random friend " +
    "once in a blue moon this is due to the fact that if you dont have a chase bank account they make the person you " +
    "are sending or receiving money tofrom go through a drawn out account setup which involves verifying  small deposits " +
    "made to your bank account this makes it impractical for one off use  however once you complete the initial account " +
    "setup its actually pretty easy there after so if you pay or get paid from someone weekly or monthly its definitely " +
    "worth your time to set it up and better than sending a check overall this app and chase is far ahead of the services " +
    "provided at any other bank great job"

  val conf = ConfigFactory.load()

  val spark = SparkSession.builder.master("local[*]").getOrCreate

  val model = SageMakerModel.fromEndpoint(
    endpointName = conf.getString("aws.sageMaker.endpoint"),
    requestRowSerializer = new ProtobufRequestRowSerializer(),
    responseRowDeserializer = new LinearLearnerBinaryClassifierProtobufResponseRowDeserializer()
  )

  val reviewData = spark.createDataFrame(Seq(Tuple1(review))).toDF("review")

  val tokenizer = new Tokenizer().setInputCol("review").setOutputCol("words")
  val wordsData = tokenizer.transform(reviewData)
  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features").setNumFeatures(conf.getInt("aws.sageMaker.featureDim"))
  val testData = hashingTF.transform(wordsData)

  val prediction = model.transform(testData)
  prediction.selectExpr("review", "score", "predicted_label prediction").show
}
