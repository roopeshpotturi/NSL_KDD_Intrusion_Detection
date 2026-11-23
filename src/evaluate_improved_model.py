
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 1: PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "results" / "improvement_experiments"
    / "best_class_weighted_lightgbm.joblib"
)

TEST_PATH = ROOT / "data" / "raw" / "KDDTest+.TXT"

OUTPUT_DIR = ROOT / "results" / "improvement_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 2: FEATURE NAMES
# ============================================================

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# ============================================================
# STEP 3: ATTACK MAPPING
# ============================================================

attack_mapping = {
    "normal": "Normal",

    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",

    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "named": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "worm": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",

    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R"
}

labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ============================================================
# STEP 4: LOAD MODEL
# ============================================================

print("\n========== LOADING IMPROVED MODEL ==========")

model = joblib.load(MODEL_PATH)

print("Improved model loaded successfully!")

# ============================================================
# STEP 5: LOAD OFFICIAL TEST DATA
# ============================================================

print("\n========== LOADING TEST DATA ==========")

data = pd.read_csv(
    TEST_PATH,
    header=None
)

if data.shape[1] != 43:
    raise ValueError("Unexpected test dataset column count.")

data.columns = columns + ["attack", "difficulty"]

X_test = data[columns].copy()

for col in ["protocol_type", "service", "flag"]:
    X_test[col] = X_test[col].astype(str)

# Convert numeric features safely
numeric_features = [
    col for col in columns
    if col not in ["protocol_type", "service", "flag"]
]

X_test[numeric_features] = X_test[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

attack_names = (
    data["attack"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.rstrip(".")
)

y_test = attack_names.map(attack_mapping)

if y_test.isna().any():
    raise ValueError(
        f"Unknown attack labels: "
        f"{attack_names[y_test.isna()].unique()}"
    )

print("Test records:", len(X_test))

# ============================================================
# STEP 6: PREDICTION
# ============================================================

print("\n========== MAKING PREDICTIONS ==========")

predictions = model.predict(X_test)

print("Predictions completed!")

# ============================================================
# STEP 7: EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, predictions)

macro_precision = precision_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

print("\n========== IMPROVED MODEL PERFORMANCE ==========")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    )
)

# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========== CONFUSION MATRIX ==========")

print(cm_df)

# ============================================================
# STEP 9: SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": predictions
})

results.to_csv(
    OUTPUT_DIR / "improved_test_predictions.csv",
    index=False
)

cm_df.to_csv(
    OUTPUT_DIR / "improved_test_confusion_matrix.csv"
)

metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1
}])

metrics.to_csv(
    OUTPUT_DIR / "improved_test_metrics.csv",
    index=False
)

print("\nAll evaluation files saved successfully!")
print("Output folder:", OUTPUT_DIR)
