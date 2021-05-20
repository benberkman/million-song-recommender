#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit subsample-data.py <any arguments you wish to add>
'''

# Import command line arguments and helper functions(if necessary)
import sys
import getpass
import pandas as pd
import random

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #read in train, val, test    
    cf_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet") #shape: (49824519, 4)
    cf_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet") #shape: (135938, 4)
    cf_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet") #shape: (1368430, 4)

    #subset train to only include IDs from validation
    
    cf_train_sample = cf_train.join(cf_val, ["user_id"], "inner")
    cf_train_sample = cf_train_sample.select(cf_train_sample.columns[:3])
   
    #downsample data
    #for train, collect distinct user_id's and make set from user list
    user_train_distinct = cf_train.select('user_id').distinct().collect()
    train_user = set(row['user_id'] for row in user_train_distinct)
    
    #for val, collect distinct user_id's and make set from user list
    user_val_distinct = cf_val.select('user_id').distinct().collect()
    val_user = set(row['user_id'] for row in user_val_distinct)
    
    # Align train and val users
    prev_user = list(train_user - val_user)
   
    # Sample randomly on 10%
    samp = int(0.1 * len(prev_user))
    prev_user_sample = random.sample(prev_user, samp)
    cf_train_down_10 = cf_train.where(cf_train.user_id.isin(prev_user_sample + list(val_user)))

    #sample train data frame
    cf_train_sample = cf_train.sample(0.01, 123)
    cf_train_10 = cf_train.sample(0.1, 123)
    cf_train_25 = cf_train.sample(0.25, 123)
    cf_train_50 = cf_train.sample(0.50, 123)
    cf_train_sample_small = cf_train.sample(0.001, 123)
    cf_train_sample_smallest = cf_train.sample(0.0001, 123)
    cf_val_sample = cf_val.sample(.01, seed = 44) #shape: (1317, 4)
    cf_test_sample = cf_test.sample(.01, seed = 44) #shape: (13720, 4)

    #write to parquet
    cf_train_sample.write.mode('overwrite').parquet("hdfs:/user/bjb433/data-samples/cf_train_sample.parquet")
    cf_train_sample_small.write.mode('overwrite').parquet("hdfs:/user/bjb433/data-samples/cf_train_sample_small.parquet")
    cf_train_sample_smallest.write.mode('overwrite').parquet("hdfs:/user/bjb433/data-samples/cf_train_sample_smallest.parquet")
    cf_train_down_10.write.mode('overwrite').parquet("hdfs:/user/dt2229/data-samples/cf_train_down_10.parquet")
    cf_train_25.write.mode('overwrite').parquet("hdfs:/user/bjb433/data-samples/cf_train_25.parquet")
    cf_train_50.write.mode('overwrite').parquet("hdfs:/user/bjb433/data-samples/cf_train_50.parquet")

    cf_val_sample.write.mode('ignore').parquet("hdfs:/user/bjb433/data-samples/cf_val_sample.parquet")
    cf_test_sample.write.mode('ignore').parquet("hdfs:/user/bjb433/data-samples/cf_test_sample.parquet")



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config("spark.executor.memory", "15g").appName('subset_data').getOrCreate()

    main(spark)