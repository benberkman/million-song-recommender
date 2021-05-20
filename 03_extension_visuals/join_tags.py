#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
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

    #index strings as integers
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_int", handleInvalid= 'keep').fit(train)
    indexed_train = indexer_track.transform(train)

    indexed_train = indexed_train.groupBy('track_id', 'track_id_int').sum('count')

    genres_df = spark.read.option("delimiter", "\t").csv("genres.txt").withColumnRenamed("_c0","track_id").withColumnRenamed("_c1","genre")
    genres_df = genres_df.withColumnRenamed("_1","track_id").withColumnRenamed("_2","genre")

    full = indexed_train.join(genres_df, indexed_train.track_id == genres_df.track_id, "inner").select(genres_df.track_id, indexed_train.track_id_int, genres_df.genre)

    full.write.parquet('hdfs:/user/bjb433/with_genres.parquet')


if __name__ == "__main__":

    spark = SparkSession.builder.config("spark.executor.memory", "16g").config("spark.blacklist.enabled", False).appName('run_model').getOrCreate()
    
    spark.sparkContext.setCheckpointDir('./checkpoint')

    main(spark)