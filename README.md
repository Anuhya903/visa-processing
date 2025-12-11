# AI Enabled Visa Status Prediction & Processing Time Estimator

## Milestone 1 – Data Collection & Preprocessing

### Dataset
- **Source:** Kaggle – Combined_LCA_Disclosure_Data_FY2024
- **Raw file location:** `data/raw/Combined_LCA_Disclosure_Data_FY2024.csv`

### Steps Performed

1. **Data Loading**
   - Loaded the raw CSV file using `pandas`.

2. **Standardization of Columns**
   - Converted all column names to uppercase and stripped extra spaces.

3. **Date Columns**
   - Detected:
     - `RECEIVED_DATE` → application date  
     - `DECISION_DATE` → decision date
   - Converted them to `datetime` format.

4. **Target Label Creation**
   - Created new column:
     - `PROCESSING_DAYS = DECISION_DATE - RECEIVED_DATE` (in days)
   - Removed rows with invalid dates.
   - Kept only realistic processing times (0–3650 days).

5. **Handling Missing Values**
   - Numerical columns → filled with median.
   - Categorical columns → filled with mode (most frequent value) or `"Unknown"`.

6. **Feature Selection**
   - Selected main features:
     - `CASE_STATUS`
     - `EMPLOYER_NAME`
     - `JOB_TITLE`
     - `FULL_TIME_POSITION`
     - `WORKSITE_STATE`
     - `VISA_CLASS`
     - `PROCESSING_DAYS` (target)

7. **Categorical Encoding**
   - Reduced too many categories for `EMPLOYER_NAME` and `JOB_TITLE` to top 20 + `"Other"`.
   - Applied one-hot encoding to categorical columns.

### Outputs

Processed files saved in `data/processed/`:
- `visa_clean.csv` – cleaned selected columns
- `visa_clean_encoded.csv` – cleaned + one-hot encoded, ready for modeling

### How to Run

```bash
python src/milestone1_preprocessing.py
