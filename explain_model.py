import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# Convert categorical data
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.TreeExplainer(model)

# Explain a sample of test records
sample = X_test.iloc[:50]
shap_values = explainer.shap_values(sample)

# Plot feature importance
print("Generating SHAP explanation...")

shap.summary_plot(
    shap_values[:, :, 1],
    sample,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()

print("SHAP explanation completed!")
print("Saved as shap_summary.png")
