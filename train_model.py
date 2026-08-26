from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

student_performance = fetch_ucirepo(id=320)

features_data = student_performance.data.features
target_data = student_performance.data.targets

X = features_data[
    [
        "studytime",
        "failures",
        "absences"
    ]
].copy()

X["G1"] = target_data["G1"]
X["G2"] = target_data["G2"]

y = target_data["G3"]

print("Selected Features:")
print(X.head())

print("\nTarget (G3):")
print(y.head())

print("\nDataset Shape:")
print(X.shape)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)