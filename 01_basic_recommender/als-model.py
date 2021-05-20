#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit model-test.py <any arguments you wish to add>
'''
#hadoop fs -mkdir ./checkpoint

import sys
import getpass
import pandas as pd
import os
import numpy as np
import itertools

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics


def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #read data
    train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet').repartition(32, ['user_id', 'track_id'])
    val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet').repartition("user_id", "track_id")
    test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet').repartition("user_id", "track_id")

    #index strings as integers
    indexer_uid = StringIndexer(inputCol="user_id", outputCol="user_id_int", handleInvalid= 'keep').fit(train)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_int", handleInvalid= 'keep').fit(train)

    indexed_train = indexer_uid.transform(train)
    indexed_train = indexer_track.transform(indexed_train)
    indexed_train.write.parquet('hdfs:/user/bjb433/indexed_train.parquet')
    indexed_train = indexed_train.select("user_id_int", "track_id_int", "count")

    indexed_val = indexer_uid.transform(val)
    indexed_val = indexer_track.transform(indexed_val)
    indexed_val = indexed_val.select("user_id_int", "track_id_int", "count")

    indexed_test = indexer_uid.transform(test)
    indexed_test = indexer_track.transform(indexed_test)
    indexed_test = indexed_test.select("user_id_int", "track_id_int", "count")
    
    #set hyperparamters
    rank = 100
    alpha = 20
    lambd = 0
    maxIter = 20

    hyper_params = ("Rank" + str(rank) + "\nAlpha: " + str(alpha) + "\nLambda: " + str(lambd) + "\nMax Iterations: " + str(maxIter))
    
    #create and fit model
    als = ALS(maxIter = maxIter, regParam = lambd, userCol="user_id_int", itemCol="track_id_int", ratingCol="count",
              coldStartStrategy="drop", rank = rank, alpha = alpha, implicitPrefs = True, nonnegative = False,
              checkpointInterval= 2)

    model = als.fit(indexed_train)

    #Calculate mAP for training set
    users = indexed_val.select("user_id_int").distinct()
    true_set = indexed_val.groupBy("user_id_int").agg(collect_set("track_id_int").alias("ground_truth"))

    recUsers = model.recommendForUserSubset(users, 500).select("user_id_int", col("recommendations.track_id_int"))
    final_df = true_set.join(recUsers, true_set.user_id_int == recUsers.user_id_int, how = 'inner').select(true_set.user_id_int, true_set.ground_truth, recUsers.track_id_int)

    rdd_val = final_df.select('track_id_int', 'ground_truth').rdd.map(tuple)

    metrics_val = RankingMetrics(rdd_val)
    mAP_val = metrics_val.meanAveragePrecision  

    val_results = ("Validation mAP: " + str(mAP_val))

    #Calculate mAP for test set
    users_test = indexed_test.select("user_id_int").distinct()
    true_set_test = indexed_test.groupBy("user_id_int").agg(collect_set("track_id_int").alias("ground_truth"))

    recUsers_test = model.recommendForUserSubset(users_test, 500).select("user_id_int", col("recommendations.track_id_int"))
    final_df_test = true_set_test.join(recUsers_test, true_set_test.user_id_int == recUsers_test.user_id_int, how = 'inner').select(true_set_test.user_id_int, true_set_test.ground_truth, recUsers_test.track_id_int)
    
    rdd_test = final_df_test.select('track_id_int', 'ground_truth').rdd.map(tuple)
    
    metrics_test = RankingMetrics(rdd_test)
    mAP_test = metrics_test.meanAveragePrecision

    test_results = ("Test mAP: " + str(mAP_test))

    #save and print results
    f = open("results.txt", "x")
    f.write(hyper_params + '\n' + val_results + '\n' + test_results)

    model.save("trained-model")
    print(hyper_params + '\n' + val_results + '\n' + test_results)


if __name__ == "__main__":

    spark = SparkSession.builder.config("spark.executor.memory", "16g").config("spark.blacklist.enabled", False).appName('run_model').getOrCreate()
    
    spark.sparkContext.setCheckpointDir('./checkpoint')

    main(spark)