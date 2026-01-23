# Predictive Modeling
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load feature-engineered dataset
DATA_PATH = Path("data/processed/visa_features_engineered.csv")
print("Loading dataset")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
# Define target and features
TARGET = "PROCESSING_DAYS"
# Use only numeric features
X = df.drop(columns=[TARGET])
X = X.select_dtypes(include=[np.number])
y = df[TARGET]
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
# Model 1: Linear Regression (Baseline)
print("\nTraining Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression MAE:", mae_lr)
print("Linear Regression RMSE:", rmse_lr)
# Model 2: Random Forest Regressor
print("\nTraining Random Forest Regressor")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest MAE:", mae_rf)
print("Random Forest RMSE:", rmse_rf)
# Model Comparison
print("\nModel Comparison")
print(f"Linear Regression  -> MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
print(f"Random Forest      -> MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
if rmse_rf < rmse_lr:
    print("\n Random Forest performs better and is selected as the final model.")
else:
    print("\n Linear Regression performs better and is selected as the final model.")
print("\nPredictive Modelingcompleted successfully.")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
print("Random Forest model saved successfully.")
