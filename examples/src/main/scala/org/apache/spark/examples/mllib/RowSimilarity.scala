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

package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable

object RowSimilarity {
  case class CustomEntry (nRows: Long, nCols: Long, matrixEntry: MatrixEntry)

  case class Params(
      input: String = null,
      output: String = null,
      topk: Int = 50,
      threshold: Double = 1e-4
                     ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Product Similarity") {
      head("Indigo Row Similarity")
      opt[Int]("topk")
        .text("top k similar products")
        .action((x, c) => c.copy(topk = x))
      opt[Double]("threshold")
        .text("threshold for kernel sparsity")
        .action((x, c) => c.copy(threshold = x))
      opt[String]("output")
        .required()
        .text("output paths for recommendation")
        .action((x, c) => c.copy(output = x))
      arg[String]("<input>")
        .required()
        .text("input paths to a product dataset")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/run-example mllib.RowSimilarity \
          |  --topk 100 s3a://sample_product_data
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
        .setAppName(s"Row Similarity Recommendation for Indigo with $params")
        .registerKryoClasses(Array(classOf[mutable.BitSet], classOf[CustomEntry]))

    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)
    val sqlContext = new SQLContext(sc)

    val featureRdd = sqlContext.read.parquet(params.input)

    val numProducts = featureRdd.select("uid").distinct.count
    val numFeatures = featureRdd.select("features").take(1).map{row => row.getAs[Vector]("features").size}
    println(s"Got ${numFeatures(0)} features per product from $numProducts products.")

    val itemFeatures = featureRdd.map { feature =>
      IndexedRow(feature.getAs[Long]("uid"),feature.getAs[Vector]("features"))
    }.repartition(sc.defaultParallelism)

    val itemMatrix = new IndexedRowMatrix(itemFeatures, numProducts, numFeatures(0))

    val rowSimilaritiesApprox = itemMatrix.rowSimilarities(topk= params.topk, threshold = params.threshold)

    import sqlContext.implicits._
    rowSimilaritiesApprox.entries.toDF.write.parquet(params.output)
    sc.stop()
  }
}
