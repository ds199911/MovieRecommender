from pyspark.sql import SparkSession
import getpass

from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *

from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import os
import itertools as it



def main(spark, netID):

    #test a random split with original
    # ratings = spark.read.csv(f'hdfs:/user/{netID}/ratings.csv', schema = 'userId INT, movieId INT, rating FLOAT, timestamp INT')
    # ratings.createOrReplaceTempView('ratings')

    # ratings = spark.createDataFrame(ratings)
    # (training, test) = ratings.randomSplit([0.8, 0.2])
    
    schema = StructType([
    StructField('index', IntegerType(), False),
    StructField("user", IntegerType(), False),
    StructField("item", IntegerType(), False),
    StructField("rating", DoubleType(), False),
    StructField("timestamp", IntegerType(), False),
    ])
    flag = 'val'

    # training = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/train_small.csv', schema = schema)
    # test = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/test_small.csv', schema = schema)
    # val = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/val_small.csv', schema = schema)

    training = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/train_large.csv', schema = schema)
    val = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/val_large.csv', schema = schema)
    test = spark.read.options(header='true').csv(f'hdfs:/user/{netID}/test_large.csv', schema = schema)
    val_user = val.select('user').distinct().orderBy('user')
    test_user = test.select('user').distinct().orderBy('user')

    # val.show()

    true_label_val = val.groupby('user').agg(F.collect_set('item').alias('true_label')).orderBy('user')
    true_label_test = test.groupby('user').agg(F.collect_set('item').alias('true_label')).orderBy('user')
    # true_label.show()

    ranks  = [250]
    regParams = [0.01]
    alphas = [1]
    max_its = [30]
    params = it.product(max_its, ranks, regParams, alphas)
    res = []
    for i in params:
        als = ALS(maxIter=i[0], rank = i[1], regParam=i[2], alpha = i[3], coldStartStrategy="drop")
        model = als.fit(training)

        if flag == 'val':
            user = val_user
            true_label = true_label_val
            data = val
        else:
            user = test_user
            true_label = true_label_test
            data = test


        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=model.transform(data)
        rmse=evaluator.evaluate(predictions)


        rec = model.recommendForUserSubset(user, 100)
        rec_label = rec.select('user','recommendations.item').orderBy('user')
        # rec_label.show()

        rec_true_combined = rec_label.join(true_label, rec_label.user == true_label.user, 'inner').select(rec_label.item, true_label.true_label)
        # rec_true_combined.show()


        result = rec_true_combined.rdd
        metrics = RankingMetrics(result)
        
        # pk15 = metrics.precisionAt(15)
        # pk25 = metrics.precisionAt(25)
        pk100 = metrics.precisionAt(100)

        # map15 = metrics.meanAveragePrecisionAt(15)
        # map25 = metrics.meanAveragePrecisionAt(25)

        map100 = metrics.meanAveragePrecisionAt(100)
        ndcg100 = metrics.ndcgAt(100)

        # print(f'Below is the result when max_iteration = {i[0]}, rank = {i[1]}, regParams = {i[2]}, alpha = {i[3]}')
        # print(f'PrecisionAtk w/ k = 15, 25, 100 respectively is {pk15} {pk25} {pk100}')
        # print(f'MAP w/ k = 15, 25, 100 respectively is {map15} {map25} {map100}')
        partial_res = [[i[0],i[1],i[2],i[3]],pk100,ndcg100,map100,rmse]
        res.append(partial_res)

        #Save Model to hdfs 
        pwd_path = f'hdfs:/user/{netID}/'
        model_path = os.path.join(pwd_path, 'final-project-group_40/models/ALS_{}_{}_{}_{}'.format(model.rank, als.getRegParam(),als.getMaxIter(),flag))
        model.save(model_path)
    
    print(res)




    # to move hdfs model to local run:
    #     hadoop fs -get hdfs:/user/netID/final-project-group_40/models/
    

    # Generate top 10 movie recommendations for each user : Needs to Change to RDD


    # userRecs = model.recommendForAllUsers(100)
    # userRecs.show(20, truncate = False)
    # temp = userRecs.toPandas().sort_values(by = ['user'])
    # print(temp)





    # scoresandlabels = []
    # for i in range(len(val)):
    #     items_pred = []
    #     items_true = val[val['user'] == i+1]['item'].unique()
    #     print(i)
    #     for j in range(len(temp.iloc[i,1])):
    #         item = temp.iloc[i,1][j][0]
    #         items_pred.append(item)
    #     scoresandlabels.append((items_pred, items_true))
    # print(scoresandlabels[0:10])

    

    
    # userRecs.write.csv('/scratch/czw206/final-project-group_40/data/10_recommend.csv')
    
    #evaluae with mean average precision
    # predictions = model.transform(test)
    # predictions.show()



    # evaluator = RankingEvaluator(metricName="meanAveragePrecision", labelCol="rating",predictionCol="prediction")
    # evaluator.setPredictionCol("prediction")
    

    # MAP = evaluator.evaluate(predictions)
    # print("menAvePrecision= " + str(MAP))
    
    '''
    pyspark.sql.utils.IllegalArgumentException: requirement failed: 
    Column prediction must be of type equal to one of the following types: 
    [array<double>, array<double>] but was actually of type float.
    '''

if __name__ == "__main__":

    # Create the spark session obje
    spark = SparkSession.builder.appName('ALS').getOrCreate()

    checkpoint_path = 'hdfs:/user/ct1942/final-project-group_40/models/checkpoint'
    spark.sparkContext.setCheckpointDir(checkpoint_path)

    # Get user netID from the command line
    netID = getpass.getuser()
    # Call our main routine
    main(spark, netID)
