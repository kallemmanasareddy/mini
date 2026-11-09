
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

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

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

# Save model and feature columns
joblib.dump(model, "land_acquisition_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model saved successfully!")
print("Files created:")
print("1. land_acquisition_model.pkl")
print("2. model_columns.pkl")
