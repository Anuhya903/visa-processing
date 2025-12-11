import pandas as pd
import numpy as np
from pathlib import Path

RAW_FILE = Path("data/raw/Combined_LCA_Disclosure_Data_FY2024.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

#2.Load the dataset
print("Reading file:", RAW_FILE)
df = pd.read_csv(RAW_FILE, low_memory=False)
print("Original shape:", df.shape)

# 3.Standardize column names
df.columns = df.columns.str.upper().str.strip()

# 4.To detect important columns
application_col = None
decision_col = None
status_col = None

for col in df.columns:
    if "CASE_SUBMITTED" in col or "RECEIVED" in col:
        application_col = col
    if "DECISION" in col or "CERTIFICATION_DATE" in col:
        decision_col = col
    if "CASE_STATUS" in col or "STATUS" in col:
        status_col = col

print("Application column:", application_col)
print("Decision column:", decision_col)
print("Status column:", status_col)

if application_col is None or decision_col is None:
    print("Columns in file:", list(df.columns))
    raise ValueError("Could not find application or decision date columns.")

# 5.Convert date columns
df[application_col] = pd.to_datetime(df[application_col], errors="coerce")
df[decision_col] = pd.to_datetime(df[decision_col], errors="coerce")

#To remove rows where dates are invalid
df = df.dropna(subset=[application_col, decision_col])

# 6.Create processing time in days
df["PROCESSING_DAYS"] = (df[decision_col] - df[application_col]).dt.days

#Keep only realistic values: 0 to 10 years
df = df[(df["PROCESSING_DAYS"] >= 0) & (df["PROCESSING_DAYS"] <= 3650)]
print("After creating PROCESSING_DAYS:", df.shape)

# 7.Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include="object").columns

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isnull().any():
        if not df[c].mode().empty:
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna("Unknown")

# 8.Select useful columns
keep_cols = [
    "CASE_STATUS",
    "EMPLOYER_NAME",
    "JOB_TITLE",
    "SOC_NAME",
    "WAGE_RATE_OF_PAY",
    "FULL_TIME_POSITION",
    "WORKSITE_STATE",
    "VISA_CLASS",
    "PROCESSING_DAYS"
]

keep_cols = [c for c in keep_cols if c in df.columns]
print("Keeping columns:", keep_cols)

df_clean = df[keep_cols].copy()

# 9.Reduce many categories for employer and job title
def reduce_cardinality(series, top_n=20):
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other="Other")

for col in ["EMPLOYER_NAME", "JOB_TITLE"]:
    if col in df_clean.columns:
        df_clean[col] = reduce_cardinality(df_clean[col], top_n=20)

# 10.One-hot encode categorical columns
cat_cols = df_clean.select_dtypes(include="object").columns
print("One-hot encoding:", list(cat_cols))
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

# 11.Save processed data
clean_path = PROCESSED_DIR / "visa_clean.csv"
enc_path = PROCESSED_DIR / "visa_clean_encoded.csv"

df_clean.to_csv(clean_path, index=False)
df_encoded.to_csv(enc_path, index=False)

print("Saved cleaned data to:", clean_path)
print("Saved encoded data to:", enc_path)
print("Done with Milestone 1 preprocessing.")
