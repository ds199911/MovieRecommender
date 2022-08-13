import sys
from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RankingEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pandas as pd
import getpass
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import lit

def main(spark, netID):
    file = 'large' 
    val_test = 'val'
    # Obtain the ground truth and predictions
    val = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/{val_test}_{file}.csv', schema = 'index INT, userId INT, movieId DOUBLE, rating FLOAT, timestamp INT')
    popular_100 = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/{file}_{val_test}_result.csv', schema = 'index INT, movieId DOUBLE')
    print('see popular_100 results')
    popular_100.show()

    # Aggregate predictions into a cell of list of items, and show()
    popular_100 = popular_100.agg(F.collect_set('movieId'))
    # Add index column, entry = 0. For later table join usage
    popular_100 = popular_100.withColumn('index', lit(0))
    # popular_100.show()
    
    # Aggregate ground truth, into the format of two columns, user and collection of movies respectively, for later comparison
    val_pd = val.groupby('userId').agg(F.collect_set('movieId').alias('label')).orderBy('userId')
    # val_pd.printSchema()

    # Add an extra index column, entry = 0, for later table join usage
    val_pd = val_pd.withColumn('index', lit(0))
    # val_pd.show()

    # Join table by index column, to do broadcasting
    res = val_pd.join(popular_100, val_pd.index == popular_100.index, 'inner')
    # res.show()

    # Select related columns, and fed into evaluation
    res_rdd = res.select('label','collect_set(movieId)').rdd
    metrics = RankingMetrics(res_rdd)

    # Print results
    k = 15
    
    print('Prcision at', k,'for',file, val_test, metrics.precisionAt(k))
    print('Mean Average Precision at',k,'for',file,val_test, metrics.meanAveragePrecisionAt(k))
    
    k2 = 25
    print('Prcision at', k2,'for',file, val_test, metrics.precisionAt(k2))
    print('Mean Average Precision at',k2,'for',file,val_test, metrics.meanAveragePrecisionAt(k2))
    
    k3 = 100
    print('Prcision at', k3,'for',file, val_test, metrics.precisionAt(k3))
    print('Mean Average Precision at',k3,'for',file,val_test, metrics.meanAveragePrecisionAt(k3))


if __name__ == "__main__":

    spark = SparkSession.builder.appName('baseline_evaluation').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    # Call our main routine
    main(spark, netID)
