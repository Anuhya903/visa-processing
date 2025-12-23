import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
INPUT_FILE = Path("data/processed/visa_clean.csv")
OUTPUT_FILE = Path("data/processed/visa_features_engineered.csv")
# Load dataset
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print("Dataset shape:", df.shape)
print(df.head())
print("\nBasic statistics:")
print(df["PROCESSING_DAYS"].describe())
# Distribution of Processing Time
plt.figure(figsize=(8, 5))
sns.histplot(df["PROCESSING_DAYS"], bins=50, kde=True)
plt.title("Distribution of Visa Processing Time (Days)")
plt.xlabel("Processing Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
plt.close()
# Processing Time by Visa Class
plt.figure(figsize=(10, 5))
sns.boxplot(x="VISA_CLASS", y="PROCESSING_DAYS", data=df)
plt.xticks(rotation=45)
plt.title("Processing Time by Visa Class")
plt.tight_layout()
plt.show()
plt.close()
# Processing Time by Worksite State (Top 10)
top_states = df["WORKSITE_STATE"].value_counts().head(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(
    x="WORKSITE_STATE",
    y="PROCESSING_DAYS",
    data=df[df["WORKSITE_STATE"].isin(top_states)]
)
plt.xticks(rotation=45)
plt.title("Processing Time by Worksite State (Top 10)")
plt.tight_layout()
plt.show()
plt.close()
# Correlation Analysis
df_corr = df[["PROCESSING_DAYS"]].copy()
df_corr["PROCESSING_DAYS_LOG"] = np.log1p(df["PROCESSING_DAYS"])
plt.figure(figsize=(5, 4))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
plt.close()
# Feature Engineering
# Log-transformed processing days
df["PROCESSING_DAYS_LOG"] = np.log1p(df["PROCESSING_DAYS"])
# Average processing time per state
state_avg = df.groupby("WORKSITE_STATE")["PROCESSING_DAYS"].mean()
df["STATE_AVG_PROCESSING_DAYS"] = df["WORKSITE_STATE"].map(state_avg)
# Average processing time per visa class
visa_avg = df.groupby("VISA_CLASS")["PROCESSING_DAYS"].mean()
df["VISA_CLASS_AVG_PROCESSING_DAYS"] = df["VISA_CLASS"].map(visa_avg)
# Fill any new missing values
df.fillna(df.median(numeric_only=True), inplace=True)
# Save Feature Engineered Dataset
df.to_csv(OUTPUT_FILE, index=False)
print("\nFeature-engineered dataset saved at:", OUTPUT_FILE)
print("Final dataset shape:", df.shape)
print("Milestone 2 (EDA & Feature Engineering) completed successfully.")
