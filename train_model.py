from ucimlrepo import fetch_ucirepo

# Fetch Student Performance dataset
student_performance = fetch_ucirepo(id=320)

# Features and target
X = student_performance.data.features
y = student_performance.data.targets

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nDataset Shape:")
print(X.shape)


print("\nAll Features:")
print(X.columns.tolist())

print("\nMissing Values:")
print(X.isnull().sum())

print("\nTarget Information:")
print(y.describe())