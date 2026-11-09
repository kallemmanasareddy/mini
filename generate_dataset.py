
import os
import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of synthetic projects
n = 1000

# Generate project information
df = pd.DataFrame({
    "project_id": range(1, n + 1),
    "project_type": np.random.choice(
        ["Highway", "Railway", "Industrial", "Dam", "Airport"], n
    ),
    "district": np.random.choice(
        ["Mysuru", "Mandya", "Bengaluru", "Hassan", "Tumakuru"], n
    ),
    "land_area_acres": np.random.randint(10, 1000, n),
    "number_of_parcels": np.random.randint(5, 200, n),
    "affected_households": np.random.randint(0, 300, n),
    "ownership_dispute_percentage": np.random.randint(0, 101, n),
    "litigation_cases": np.random.randint(0, 30, n),
    "land_records_verified_percentage": np.random.randint(20, 101, n),
    "compensation_readiness_percentage": np.random.randint(10, 101, n),
    "environmental_clearance_pending": np.random.randint(0, 2, n),
    "forest_land_percentage": np.random.randint(0, 51, n),
    "public_objections_count": np.random.randint(0, 50, n),
    "utility_shifting_required": np.random.randint(0, 2, n),
    "planned_duration_days": np.random.randint(90, 730, n)
})

# Create a delay risk tendency from project conditions
risk = (
    df["ownership_dispute_percentage"] * 0.025
    + df["litigation_cases"] * 0.08
    + (100 - df["land_records_verified_percentage"]) * 0.025
    + (100 - df["compensation_readiness_percentage"]) * 0.025
    + df["environmental_clearance_pending"] * 1.2
    + df["forest_land_percentage"] * 0.025
    + df["public_objections_count"] * 0.035
    + df["utility_shifting_required"] * 0.5
)

# Convert risk tendency into probability
probability = 1 / (1 + np.exp(-(risk - 6)))

# Generate delay outcome
df["delay_occurred"] = np.random.binomial(1, probability)

# Generate delay duration
df["delay_days"] = np.where(
    df["delay_occurred"] == 1,
    np.random.randint(30, 500, n),
    0
)

# Generate actual project duration
df["actual_duration_days"] = (
    df["planned_duration_days"] + df["delay_days"]
)

# Save dataset in the mini folder
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/land_acquisition_dataset.csv", index=False)

print("Dataset created successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print(df.head())
