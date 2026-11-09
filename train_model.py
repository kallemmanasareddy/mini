
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("dataset/land_acquisition_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate input features and target
X = df.drop(columns=[
    "project_id",
    "delay_occurred",
    "delay_days",
    "actual_duration_days"
])

y = df["delay_occurred"]

# 3. Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 6. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", accuracy * 100, "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
