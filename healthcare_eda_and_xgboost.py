import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, split, floor, datediff, current_date
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import mord

# --- 1. EXPLORATORY ANALYSIS ---
dataset = spark.table("sample_healthcare_dataset")

# Converting a sample to Pandas for statistical profiling
df_pandas = dataset.toPandas()
print(df_pandas['appealOutcome'].value_counts()) # Checking class imbalance
print(df_pandas.groupby('appealOutcome')['totalCharges'].median()) # Business logic check

# --- 2. FEATURE ENGINEERING (CARC PREFIX) ---
df = dataset.withColumn('carc_prefix', 
                   when(col('initialcarccode').isNull() | (col('initialcarccode') == '-') | (col('initialcarccode') == ''), 'Unknown')
                   .otherwise(split(col('initialcarccode'), '-').getItem(0)))

mapping_expr = when(col('carc_prefix') == 'CO', 'Contractual Obligations') \
    .when(col('carc_prefix') == 'PR', 'Patient Responsibility') \
    .when(col('carc_prefix') == 'PI', 'Payer Initiated Adjustments') \
    .when(col('carc_prefix') == 'OA', 'Other Adjustments') \
    .otherwise('Unknown')

df = df.withColumn('carc_category', mapping_expr)

# --- 3. ENCODING & PIPELINE ---
# Encoding Target
df = df.withColumn('appealOutcome_encoded',
    when(col('appealOutcome') == 'Denied', 0)
    .when(col('appealOutcome') == 'Document Request', 1)
    .when(col('appealOutcome') == 'Paid', 2))

# Frequency Encoding for Insurance
total_count = df.count()
freq_df = df.groupBy('insuranceName').agg((F.count('insuranceName') / total_count).alias('insurance_freq_enc'))
df = df.join(freq_df, on='insuranceName', how='left')

# Convert DOB to Age
df = df.withColumn("age", floor(datediff(current_date(), col("dob")) / 365.25))

# --- 4. PREPARING MODEL DATA ---
df_pd = df.select('totalCharges', 'age', 'appealOutcome_encoded', 'insurance_freq_enc').toPandas()
df_pd.fillna(0, inplace=True)

X = df_pd.drop("appealOutcome_encoded", axis=1)
y = df_pd["appealOutcome_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. SMOTE & XGBOOST CLASSIFIER ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42)
xgb_clf.fit(X_train_res, y_train_res)

# --- 6. VISUAL EVALUATION ---
y_pred = xgb_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
class_names = ['Denied', 'Docs Requested', 'Paid']

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap')
plt.show()

print(classification_report(y_test, y_pred))
