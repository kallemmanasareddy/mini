
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/land_acquisition_dataset.csv")

# Prepare input and target
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

# Get details of a new project
new_project = pd.DataFrame([{
    "project_type": "Highway",
    "district": "Mysuru",
    "land_area_acres": 250,
    "number_of_parcels": 80,
    "affected_households": 100,
    "ownership_dispute_percentage": 45,
    "litigation_cases": 12,
    "land_records_verified_percentage": 60,
    "compensation_readiness_percentage": 50,
    "environmental_clearance_pending": 1,
    "forest_land_percentage": 10,
    "public_objections_count": 20,
    "utility_shifting_required": 1,
    "planned_duration_days": 365
}])

# Match training columns
new_project = pd.get_dummies(new_project, drop_first=True)
new_project = new_project.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict delay probability
probability = model.predict_proba(new_project)[0][1]

# Assign risk category
if probability >= 0.70:
    risk = "HIGH"
elif probability >= 0.40:
    risk = "MEDIUM"
else:
    risk = "LOW"

# Display result
print("\nLAND ACQUISITION DELAY PREDICTION")
print("--------------------------------")
print("Delay Probability:", round(probability * 100, 2), "%")
print("Risk Category:", risk)
