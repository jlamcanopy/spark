/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * An example app for running item similarity computation on MovieLens format
 * sparse data (http://grouplens.org/datasets/movielens/) through column based
 * similarity calculation and compare with row based similarity calculation and
 * ALS + row based similarity calculation flow. For running row and column based
 * similarity on raw features, we are using implicit matrix factorization.
 *
 *
 * A synthetic dataset in MovieLens format can be found at `data/mllib/sample_movielens_data.txt`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable

object TextTFIDF {
  case class Rating(user: Int, item: Int, tfidf: Double)

  case class Params(
      input: String = null,
      delim: String = "::",
      topk: Int = 50,
      threshold: Double = 1e-4
                     ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("MovieLensSimilarity") {
      head("MovieLensSimilarity: an example app for similarity flows on MovieLens data.")
      opt[Int]("topk")
        .text("topk for ALS validation")
        .action((x, c) => c.copy(topk = x))
      opt[Double]("threshold")
        .text("threshold for dimsum sampling and kernel sparsity")
        .action((x, c) => c.copy(threshold = x))
      opt[String]("delim")
        .text("use delimiter, default ::")
        .action((x, c) => c.copy(delim = x))
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/run-example mllib.MovieLensSimilarity \
          |  --rank 25 --numIterations 20 --alpha 0.01 --topk 25\
          |  data/mllib/sample_movielens_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf =
      new SparkConf()
        .setAppName(s"Scalability Test MovieLensSimilarity with $params")
        .registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))

    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)
    val sqlContext = new SQLContext(sc)

    val movies = sqlContext.read.parquet(params.input)
    val featureRdd = movies.map( row => Rating(row.getInt(0)-1, row.getInt(1)-1, row.getDouble(2)))

    val numRatings = featureRdd.count()
    val numUsers = featureRdd.map(_.user).distinct().count()
    val numMovies = featureRdd.map(_.item).distinct().count()


    val products : DataFrame = ???
    val tokenizer = new Tokenizer().setInputCol("description").setOutputCol("words")
    val wordsData = tokenizer.transform(products)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(wordsData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)


    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

    val itemFeatures = featureRdd.map { feature =>
      MatrixEntry(feature.user, feature.item, feature.tfidf)
    }
    val itemMatrix = new CoordinateMatrix(itemFeatures).toIndexedRowMatrix()

    val rowSimilaritiesApprox = itemMatrix.rowSimilarities(topk= params.topk, threshold = params.threshold)

    sc.stop()
  }

  def transform(sqlContext: SQLContext, delim: String): Unit = {
    import sqlContext.implicits._
    for (i <- 1 to 10) {
      val df = sqlContext.sparkContext.textFile("/movies_rating.csv").map{line =>
        val fields = line.split(",")
        (fields(0).toInt + (138493 * i), fields(1).toInt, fields(2).toDouble)
      }.toDF("user_id", "movie_id", "rating")
      df.write.parquet(s"/movie_dataset/${i}/")
    }

  }
}
