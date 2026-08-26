from ucimlrepo import fetch_ucirepo

# Fetch dataset
student_performance = fetch_ucirepo(id=320)

# Get features and target data
features_data = student_performance.data.features
target_data = student_performance.data.targets

# Select features
X = features_data[
    [
        "studytime",
        "failures",
        "absences"
    ]
].copy()

# G1 and G2 are present in target_data
X["G1"] = target_data["G1"]
X["G2"] = target_data["G2"]

# Final grade is our target
y = target_data["G3"]

print("Selected Features:")
print(X.head())

print("\nTarget (G3):")
print(y.head())

print("\nDataset Shape:")
print(X.shape)