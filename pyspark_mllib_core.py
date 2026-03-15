from pyspark.sql.functions import year, col, current_date, count, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --- 1. Scalable Feature Engineering ---
# Generic table name 
df = spark.table("sample_healthcare_dataset")

# Age calculation using PySpark Year functions
df = df.withColumn("age", year(current_date()) - year(col('dob')))

# Handling missing values
df = df.fillna({'initialcarccode': 'Unknown'})

# --- 2. PySpark ML Pipeline ---
categorical_cols = ['insuranceName', 'initialcarccode']
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in categorical_cols]

# Numerical Vectorization
numeric_assembler = VectorAssembler(inputCols=['age', 'totalCharges'], outputCol='numerical_features')
scaler = StandardScaler(inputCol='numerical_features', outputCol='scaled_numeric_cols')

# Final Assembly
final_features = ['insuranceName_index', 'initialcarccode_index', 'scaled_numeric_cols']
final_assembler = VectorAssembler(inputCols=final_features, outputCol="features")

# Target Encoding
label_idx = StringIndexer(inputCol='appealOutcome', outputCol='appealOutcomeEncoded')

pipeline = Pipeline(stages=indexers + [numeric_assembler, scaler, final_assembler, label_idx])
df_transformed = pipeline.fit(df).transform(df)

# --- 3. Distributed Training ---
# 70% train, 15% validation, 15% test
train_df, val_df, test_df = df_transformed.randomSplit([0.7, 0.15, 0.15], seed=42)

# Native PySpark RandomForest for Big Data scale
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='appealOutcomeEncoded',
    numTrees=20,
    maxDepth=5,
    maxBins=128
)

rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)

# --- 4. Evaluation ---
evaluator = MulticlassClassificationEvaluator(
    labelCol='appealOutcomeEncoded',
    predictionCol='prediction',
    metricName='accuracy'
)
print(f"Test Accuracy: {evaluator.evaluate(predictions):.3f}")
