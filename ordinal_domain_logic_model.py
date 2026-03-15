# --- dataset ---
# Using the generic table name for NDA safety
dataset = spark.table("sample_healthcare_dataset")
dataset.display()

# --- SQL Analysis (Converted to PySpark SQL) ---
spark.sql("select distinct initialcarccode from sample_healthcare_dataset").show()
spark.sql("select distinct insurancename from sample_healthcare_dataset").show()

# --- prefix categorising ---
from pyspark.sql.functions import col, when, split, isnan

# Define a new column 'carc_prefix' by extracting substring before first '-'
df = dataset.withColumn('carc_prefix', 
                   when(col('initialcarccode').isNull() | (col('initialcarccode') == '-') | (col('initialcarccode') == ''), 'Unknown')
                   .otherwise(split(col('initialcarccode'), '-').getItem(0)))

# Count distinct prefixes
df.groupBy('carc_prefix').count().show()

# --- mapping initial claim status ---
from pyspark.sql.functions import when, split, col

# Extract prefix from initialcarccode
df = df.withColumn(
    'carc_prefix',
    when(
        (col('initialcarccode').isNull()) | (col('initialcarccode') == '-') | (col('initialcarccode') == ''),
        'Unknown'
    ).otherwise(split(col('initialcarccode'), '-').getItem(0))
)

# Define a mapping dictionary for PySpark
mapping_expr = when(col('carc_prefix') == 'CO', 'Contractual Obligations') \
    .when(col('carc_prefix') == 'PR', 'Patient Responsibility') \
    .when(col('carc_prefix') == 'PI', 'Payer Initiated Adjustments') \
    .when(col('carc_prefix') == 'OA', 'Other Adjustments') \
    .when(col('carc_prefix') == 'CR', 'Correction and Reversal') \
    .otherwise('Unknown')

df = df.withColumn('carc_category', mapping_expr)

# --- appealOutcome encoding ---
from pyspark.sql.functions import when, col

df = df.withColumn(
    'appealOutcome_encoded',
    when(col('appealOutcome') == 'Denied', 0)
    .when(col('appealOutcome') == 'Document Request', 1)
    .when(col('appealOutcome') == 'Paid', 2)
)

# --- documenttype encoding ---
from pyspark.sql import functions as F

# Calculate frequency of each category
freq_df = df.groupBy('documentType').count()
total_count = df.count()

# Add frequency column as fraction of total
freq_df = freq_df.withColumn('documentType_freq', F.col('count') / total_count).select('documentType', 'documentType_freq')

# Join frequency back to original dataframe
df = df.join(freq_df, on='documentType', how='left')

# --- insurance name encoding ---
# Calculate frequency of each insuranceName
freq_df_ins = df.groupBy('insuranceName').agg(
    (F.count('insuranceName') / df.count()).alias('insurance_freq_enc')
)

# Join frequency back to original df
df = df.join(freq_df_ins, on='insuranceName', how='left')

# --- initial claim status : encoding ---
categories = ['Contractual Obligations', 'Patient Responsibility', 'Payer Initiated Adjustments', 'Other Adjustments', 'Correction and Reversal', 'Unknown']

for cat in categories:
    safe_cat = cat.replace(" ", "_")
    df = df.withColumn(f"carc_category_{safe_cat}", when(col("carc_category") == cat, 1).otherwise(0))

# --- converting dob to age ---
from pyspark.sql.functions import col, current_date, datediff, floor

# Convert dob to age (in years)
df_before = df.withColumn("age", floor(datediff(current_date(), col("dob")) / 365.25))

# --- final dataframe ---
# Columns to drop
drop_cols = [
    'initialcarccode_freq', 'firstName', 'lastName', 'gender',
    'carc_category', 'carc_prefix', 'appealOutcome', 'facilityState',
    'pcn', 'facilityId', 'documentType', 'initialcarccode',
    'insurancename', 'dob', 'insurance_freq_enc'
]

# Drop unwanted columns
df_clean = df_before.drop(*drop_cols)

# Columns to keep
keep_cols = [
    'totalCharges',
    'age',
    'appealOutcome_encoded',
    'documentType_freq', 
    'carc_category_Contractual_Obligations',
    'carc_category_Patient_Responsibility',
    'carc_category_Payer_Initiated_Adjustments',
    'carc_category_Other_Adjustments'
]

df_model = df_clean.select(*keep_cols)

# --- Convert to Pandas & Train/Test Split ---
df_pd = df_model.toPandas()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_pd.drop("appealOutcome_encoded", axis=1)
y = df_pd["appealOutcome_encoded"]

# Fill missing values before scaling
X["documentType_freq"] = X["documentType_freq"].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Logistic Regression Pipeline ---
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42))
])

pipeline.fit(X_train, y_train)

# --- SMOTE Integration ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

imb_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs'))
])

imb_pipeline.fit(X_train, y_train)

# --- Ordinal Regression (mord) ---
import mord
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate and fit the model
model_ord = mord.LogisticAT(alpha=1.0)
model_ord.fit(X_train_scaled, y_train)

# Evaluate
y_pred_ord = model_ord.predict(X_test_scaled)
