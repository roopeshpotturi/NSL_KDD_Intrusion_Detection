
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 1: PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "SMOTENC_Balanced_LightGBM.joblib"
)

DATA_PATH = (
    ROOT / "data" / "processed" / "prepared_data.joblib"
)

TEST_PATH = (
    ROOT / "data" / "raw" / "KDDTest+.TXT"
)

OUTPUT_DIR = (
    ROOT / "results" / "final_project_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 2: NSL-KDD FEATURE NAMES
# ============================================================

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


# ============================================================
# STEP 3: LOAD TRAINED MODEL
# ============================================================

print("\n========================================")
print("   MULTI-CLASS INTRUSION DETECTION")
print("========================================")

print("\nLoading trained LightGBM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

print("Expected input features:", model.n_features_in_)

print(
    "Model classes:",
    model.named_steps["classifier"].classes_
)


# ============================================================
# STEP 4: LOAD PREPROCESSING INFORMATION
# ============================================================

print("\nLoading preprocessing information...")

prepared_data = joblib.load(DATA_PATH)

scaler = prepared_data["scaler"]

numerical_features = list(
    prepared_data["numerical_features"]
)

categorical_features = list(
    prepared_data["categorical_features"]
)

print("Scaler loaded successfully!")


# ============================================================
# STEP 5: LOAD NSL-KDD TEST DATA
# ============================================================

print("\nLoading NSL-KDD test dataset...")

data = pd.read_csv(
    TEST_PATH,
    header=None
)

if data.shape[1] != 43:
    raise ValueError(
        f"Expected 43 columns, found {data.shape[1]}"
    )

data.columns = columns + [
    "attack",
    "difficulty"
]

print("Test dataset loaded successfully!")

print("Dataset shape:", data.shape)


# ============================================================
# STEP 6: PREPARE INPUT FEATURES
# ============================================================

print("\nPreparing input features...")

X = data[columns].copy()

# Apply the same numerical scaling used during training
X[numerical_features] = scaler.transform(
    X[numerical_features]
)

# Convert categorical columns to strings
for col in categorical_features:
    X[col] = X[col].astype(str)

# Maintain the exact feature order expected by the model
X = X[list(model.feature_names_in_)]

print("Feature preparation completed!")

print("Input shape:", X.shape)


# ============================================================
# STEP 7: LOAD ACTUAL TEST LABELS
# ============================================================

print("\nLoading actual test labels...")

y_actual = pd.Series(
    prepared_data["y_test"]
).reset_index(drop=True)

if len(y_actual) != len(X):
    raise ValueError(
        "Test feature and label counts do not match!"
    )

print("Actual labels loaded successfully!")


# ============================================================
# STEP 8: PREDICT NETWORK TRAFFIC
# ============================================================

print("\nPredicting network traffic...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = probabilities.max(axis=1) * 100

print("Predictions completed successfully!")


# ============================================================
# STEP 9: CREATE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({

    "Actual Category": y_actual,

    "Predicted Category": predictions,

    "Confidence (%)": confidence.round(2)

})

print("\n========================================")
print("          SAMPLE PREDICTIONS")
print("========================================")

print(
    results.head(15).to_string(index=False)
)


# ============================================================
# STEP 10: EVALUATE MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_actual,
    predictions
)

print("\n========================================")
print("          MODEL PERFORMANCE")
print("========================================")

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

labels = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_actual,
        predictions,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# STEP 11: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_actual,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:\n")

print(cm_df)


# ============================================================
# STEP 12: SAVE PREDICTION RESULTS
# ============================================================

output_file = (
    OUTPUT_DIR / "prediction_demo_results.csv"
)

results.to_csv(
    output_file,
    index=False
)

cm_file = (
    OUTPUT_DIR / "prediction_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

print("\n========================================")
print("          FILES SAVED")
print("========================================")

print("\nPrediction results:")
print(output_file)

print("\nConfusion matrix:")
print(cm_file)

print("\nProject prediction execution completed!")
