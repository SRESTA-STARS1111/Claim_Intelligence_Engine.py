# --- dataset ---
# Using the generic table 
dataset = spark.table("sample dataset")
dataset.display()

# --- age ---
from pyspark.sql import functions as F

# Calculating age by finding difference between current date and dob, convert days to years, cast converts float to int
df = dataset.withColumn("age", (F.datediff(F.current_date(), F.col("dob")) / 365.25).cast("int"))

# displays 10 of them
display(df.select("dob", "age").limit(10))

# --- Converting to pandas ---
pdf = df.toPandas() 

import pandas as pd
import numpy as np
from datetime import datetime

# ensure dob is datetime
pdf["dob"] = pd.to_datetime(pdf["dob"], errors="coerce")

# create age in whole years
pdf["age"] = (datetime.now().year - pdf["dob"].dt.year).fillna(0).astype(int)

# --- Cleanup encoded columns ---
for col in categorical_cols:
    if col + "_freq" in df.columns:
        df = df.drop(col + "_freq")

# --- PySpark ML Pipeline ---
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql import functions as F

# Fill nulls in categorical columns
categorical_cols = ["insuranceName", "initialcarccode"]
for col in categorical_cols:
    df = df.fillna({col: "unknown"})

# Frequency encoding for categorical columns
for col in categorical_cols:
    freq_df = df.groupBy(col).agg((F.count("*") / df.count()).alias(col + "_freq"))
    df = df.join(freq_df.select(col, col + "_freq"), on=col, how="left")

# Label encoding for target
label_indexer = StringIndexer(inputCol="appealOutcome", outputCol="label")

# Assemble numeric columns and scale
numeric_cols = ["totalCharges", "age"]
assembler_num = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")

# Assemble all features (scaled numeric + freq-encoded categorical)
feature_cols = [col + "_freq" for col in categorical_cols] + ["scaled_numeric_features"]
final_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# Build pipeline
pipeline = Pipeline(stages=[label_indexer, assembler_num, scaler, final_assembler])
model = pipeline.fit(df)
df_transformed = model.transform(df)

df_transformed.select("features", "label").show(5)

# --- Scikit-Learn Preprocessing ---
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Columns
categorical_freq_cols = ["insuranceName", "initialcarccode"]
numeric_cols = ["totalCharges", "age"] 
target_col = "appealOutcome"

for col in categorical_freq_cols:
    freq_map = pdf[col].value_counts(normalize=True).to_dict()
    pdf[col + "_freq"] = pdf[col].map(freq_map)

scaler = StandardScaler() 
scaled_nums = scaler.fit_transform(pdf[numeric_cols])
scaled_df = pd.DataFrame(scaled_nums, columns=numeric_cols, index=pdf.index)

le = LabelEncoder() 
pdf["label"] = le.fit_transform(pdf[target_col])

# Check nulls and replace with 0
cols_to_check = ["insuranceName_freq", "initialcarccode_freq", "totalCharges", "age"]
pdf[cols_to_check] = pdf[cols_to_check].fillna(0)

freq_cols = [col + "_freq" for col in categorical_freq_cols]
final_df = pd.concat([pdf[freq_cols], scaled_df, pdf[["label"]]], axis=1)

# --- train_test_split ---
from sklearn.model_selection import train_test_split

X = final_df.drop(columns=["label"])
y = final_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- SMOTE for Imbalance ---
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, sampling_strategy={0: 650, 2: 650})
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# --- RandomForest Training ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=2,
    max_features="sqrt", random_state=42, n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# --- Hyperparameter Tuning (GridSearch) ---
from sklearn.model_selection import GridSearchCV

rf_grid = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(rf_grid, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=1, verbose=2)
grid.fit(X_train, y_train)

# --- Gradient Boosting ---
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, min_samples_leaf=10, max_features=2, random_state=42
)
gb.fit(X_train, y_train)

# --- CatBoost with Balanced Weights ---
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

cat = CatBoostClassifier(
    loss_function="MultiClass", iterations=1000, learning_rate=0.03,
    depth=7, class_weights=class_weights, l2_leaf_reg=5,
    random_seed=42, early_stopping_rounds=50, eval_metric="TotalF1", verbose=100
)
cat.fit(X_train, y_train, eval_set=(X_test, y_test))

# --- SHAP Analysis ---
import shap

explainer = shap.TreeExplainer(cat)
shap_values = explainer.shap_values(X_test)

# --- Final Results Mapping ---
class_labels = {0: 'Denied', 1: 'Document Request', 2: 'Paid'}
results = []
y_pred_final = cat.predict(X_test).astype(int).ravel()
y_proba_final = cat.predict_proba(X_test)

for i in range(len(y_pred_final)):
    result = {
        'Predicted Outcome': class_labels[y_pred_final[i]]
    }
    for cls, label in class_labels.items():
        result[f'Prob: {label}'] = f"{y_proba_final[i][cls]:.3f}"
    results.append(result)

results_df = pd.DataFrame(results)
