#Coded by - Wajira Abeysinghe
#this file use the output of FeatureExtractor.py

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df = spark.read.csv("file:////output_tfid.csv", inferSchema=True,sep = "\t", header = True, encoding = "ISO-8859-1")

#rename label and features - if there are label and features column after extracting the features using NLTK
df = df.withColumnRenamed("label", "label_feature")
df = df.withColumnRenamed("labels", "label")
df = df.withColumnRenamed("features", "features_f")


# df.take(10)
# df.select([c for c in df.columns if c in ['label']]).show()

# df.select(df.columns[:2]).take(100)

ignore = ['_c0', 'label']
assembler = VectorAssembler(
    inputCols=[x for x in df.columns if x not in ignore],
    outputCol='features')
preppedDataDF = assembler.transform(df)


lrModel = LogisticRegression().fit(preppedDataDF)
dataset = preppedDataDF.select(["label", "features"])
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
predictions = lrModel.transform(testData)


# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)
confusionTable = predictions.groupBy("label", "prediction").count()
acc = evaluator.evaluate(predictions)


#0.5556633115372371


################Cross validation###################


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)


dataset = preppedDataDF.select(["label", "features"])

(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed=100)
# Run cross validations
cvModel = cv.fit(trainingData)

# Use test set to measure the accuracy of our model on new data
predictions = cvModel.transform(testData)
evaluator.evaluate(predictions)

evaluator.getMetricName()

############################# Random Forest ##################################
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline

data = preppedDataDF.select(["label", "features"])


labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=7, featureSubsetStrategy="auto",impurity='gini')
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=100)
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)

#0.579129574679

################################################################## GBT####################

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data = preppedDataDF.select(["label", "features"])
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
(trainingData, testData) = data.randomSplit([0.7, 0.3])
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction", "indexedLabel", "features").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)
