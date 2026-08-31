from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib

# =====================================================
# STEP 1 : Load Dataset
# =====================================================

student_performance = fetch_ucirepo(id=320)

features_data = student_performance.data.features
target_data = student_performance.data.targets

# =====================================================
# STEP 2 : Feature Selection
# =====================================================

X = features_data[["studytime", "failures", "absences"]].copy()

# Add G1 and G2
X["G1"] = target_data["G1"]
X["G2"] = target_data["G2"]

# Target
y = target_data["G3"]

print("========== DATASET ==========")
print(X.head())
print("\nTarget (G3):")
print(y.head())
print("\nDataset Shape:", X.shape)

# =====================================================
# STEP 3 : Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# =====================================================
# STEP 4 : Create Random Forest Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# =====================================================
# STEP 5 : Train Model
# =====================================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# =====================================================
# STEP 6 : Prediction
# =====================================================

y_pred = model.predict(X_test)

print("\n========== FIRST 10 PREDICTIONS ==========")
for actual, pred in zip(y_test.iloc[:10], y_pred[:10]):
    print(f"Actual: {actual:2} | Predicted: {pred:.2f}")

# =====================================================
# STEP 7 : Model Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE       : {mae:.2f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R2 Score  : {r2:.2f}")

# =====================================================
# STEP 8 : Sample Student Prediction
# =====================================================

sample_student = pd.DataFrame({
    "studytime": [3],
    "failures": [0],
    "absences": [4],
    "G1": [13],
    "G2": [14]
})

predicted_grade = model.predict(sample_student)

print("\n========== SAMPLE STUDENT ==========")
print(sample_student)
print(f"\nPredicted Final Grade (G3): {predicted_grade[0]:.2f}")
model_path = "model/performance_model.pkl"
joblib.dump(model, model_path)