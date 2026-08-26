from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# STEP 1: Fetch Dataset
# ==========================================

student_performance = fetch_ucirepo(id=320)

features_data = student_performance.data.features
target_data = student_performance.data.targets


# ==========================================
# STEP 2: Select Features
# ==========================================

X = features_data[
    [
        "studytime",
        "failures",
        "absences"
    ]
].copy()

# G1 and G2 are available in target_data
X["G1"] = target_data["G1"]
X["G2"] = target_data["G2"]

# Final grade G3 is our target
y = target_data["G3"]


# ==========================================
# STEP 3: Display Dataset Information
# ==========================================

print("Selected Features:")
print(X.head())

print("\nTarget (G3):")
print(y.head())

print("\nDataset Shape:")
print(X.shape)


# ==========================================
# STEP 4: Train-Test Split
# ==========================================

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


# ==========================================
# STEP 5: Create Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# STEP 6: Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")