
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset/land_acquisition_dataset.csv")

# Separate features and target
X = df.drop(columns=[
    "project_id",
    "delay_occurred",
    "delay_days",
    "actual_duration_days"
])

y = df["delay_occurred"]

# Convert categorical values
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

# XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_predictions)

# Compare results
print("\nMODEL COMPARISON")
print("-----------------------")
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print("XGBoost Accuracy:", round(xgb_accuracy * 100, 2), "%")

print("\nXGBOOST CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    xgb_predictions,
    zero_division=0
))
