
import pandas as pd

# Load the generated dataset
df = pd.read_csv("dataset/land_acquisition_dataset.csv")

# Display first 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names and data types
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check delay distribution
print("\nDELAY DISTRIBUTION:")
print(df["delay_occurred"].value_counts())

# Check basic statistics
print("\nSTATISTICAL SUMMARY:")
print(df.describe())
