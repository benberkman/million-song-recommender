#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Import command line arguments and helper functions(if necessary)
import sys
import getpass
import pandas as pd
import os
import numpy as np
import itertools

import pyspark.sql.functions as F

from pyspark.sql.functions import sum as _sum
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, array, lit
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
    train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet').repartition(32, ['user_id', 'track_id'])
    val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet').repartition("user_id", "track_id")
    test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet').repartition("user_id", "track_id")

    indexer_uid = StringIndexer(inputCol="user_id", outputCol="user_id_int", handleInvalid= 'keep').fit(train)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_int", handleInvalid= 'keep').fit(train)

    indexed_train = indexer_uid.transform(train)
    indexed_train = indexer_track.transform(indexed_train)
    indexed_train = indexed_train.select("user_id_int", "track_id_int", "count")

    indexed_val = indexer_uid.transform(val)
    indexed_val = indexer_track.transform(indexed_val)
    indexed_val = indexed_val.select("user_id_int", "track_id_int", "count")

    indexed_test = indexer_uid.transform(test)
    indexed_test = indexer_track.transform(indexed_test)
    indexed_test = indexed_test.select("user_id_int", "track_id_int", "count").persist()
    
    top500 = indexed_train.groupby("track_id_int").agg(_sum("count").alias('sum')).orderBy(col('sum').desc()).select("track_id_int").limit(500)
    
    global_rec = list(top500.toPandas()["track_id_int"])
        
    true_set = indexed_test.groupBy("user_id_int").agg(collect_set("track_id_int").alias("ground_truth")).select("ground_truth")
    
    true_set = true_set.withColumn("global", array([lit(x) for x in global_rec]))
    
    rdd2 = true_set.select("global","ground_truth").rdd.map(tuple)
    metrics = RankingMetrics(rdd2)
    print("MAP: ", metrics.meanAveragePrecision)
    
        

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config("spark.executor.memory", "16g").config("spark.blacklist.enabled", False).appName('run_model').getOrCreate()
    
    spark.sparkContext.setCheckpointDir('./checkpoint')

    main(spark)