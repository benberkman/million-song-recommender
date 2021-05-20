#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit model-test.py <any arguments you wish to add>
'''
#hadoop fs -mkdir ./checkpoint

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit model-test.py <any arguments you wish to add>
'''
#hadoop fs -mkdir ./checkpoint


# Import command line arguments and helper functions(if necessary)
import sys
import getpass
import pandas as pd
import os
import numpy as np
import itertools

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import rank

from pyspark.sql.functions import sum as _sum
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, array, lit, explode
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.evaluation import RankingEvaluator
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
    indexed_test = indexed_test.select("user_id_int", "track_id_int", "count")
    
    top1000 = indexed_train.groupby("track_id_int").agg(_sum("count").alias('sum')).orderBy(col('sum').desc()).select("track_id_int", 'sum').limit(1000)
    
    #Caluclate MAP for Validation set
    val_users = indexed_val.select("user_id_int").distinct()
    
    df = val_users.crossJoin(top1000)
    
    indexed_train_2 = indexed_train.select("user_id_int", "track_id_int")
        
    top_df = df.join(indexed_train_2, (df.user_id_int == indexed_train_2.user_id_int) & (df.track_id_int == indexed_train_2.track_id_int), how = "left_anti")
        
    w = Window.partitionBy('user_id_int').orderBy(col('sum').desc())
    
    top_500 = top_df.select('*', rank().over(w).alias('rank')).filter(col('rank') <= 500)
    
    top_500 = top_500.withColumn('sorted', F.collect_list('track_id_int').over(w)).groupBy('user_id_int').agg(F.max('sorted').alias('top_recs'))
    
    true_set = indexed_val.groupBy("user_id_int").agg(collect_set("track_id_int").alias("ground_truth"))
    
    final_df = top_500.join(true_set, on = "user_id_int", how = "inner").select("top_recs", "ground_truth")
    
    rdd2 = final_df.rdd.map(tuple)
    
    metrics_val = RankingMetrics(rdd2)
    
    print("Validation MAP: ", metrics_val.meanAveragePrecision)
    
    #Calculate MAP for Test set
    test_users = indexed_test.select("user_id_int").distinct()
    
    df_test = test_users.crossJoin(top1000)
    
    top_df_test = df_test.join(indexed_train_2, (df_test.user_id_int == indexed_train_2.user_id_int) & (df_test.track_id_int == indexed_train_2.track_id_int), how = "left_anti")
    
    top_500_test = top_df_test.select('*', rank().over(w).alias('rank')).filter(col('rank') <= 500)
    
    top_500_test = top_500_test.withColumn('sorted', F.collect_list('track_id_int').over(w)).groupBy('user_id_int').agg(F.max('sorted').alias('top_recs'))
    
    true_set_test = indexed_test.groupBy("user_id_int").agg(collect_set("track_id_int").alias("ground_truth"))
    
    final_df_test = top_500_test.join(true_set_test, on = "user_id_int", how = "inner").select("top_recs", "ground_truth")
    
    rdd_test = final_df_test.rdd.map(tuple)
    
    metrics_test = RankingMetrics(rdd_test)
    
    print("Test MAP: ", metrics_test.meanAveragePrecision)
    
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config("spark.executor.memory", "16g").config("spark.blacklist.enabled", False).config("spark.sql.crossJoin.enabled", "true").appName('run_model').getOrCreate()
    

    spark.sparkContext.setCheckpointDir('./checkpoint')

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)