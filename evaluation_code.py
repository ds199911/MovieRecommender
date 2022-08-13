from pyspark.ml.evaluation import RankingEvaluator

predictions = ####matrix of score and labels
evaluator = RankingEvaluator(metricName="meanAveragePrecision", labelCol="rating",predictionCol="prediction")

map = evaluator.evaluate(predictions)
