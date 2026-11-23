
import time
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from lightgbm import LGBMClassifier


# ============================================================
# STEP 1: PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.TXT"

OUTPUT_DIR = ROOT / "results" / "improvement_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

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
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

df = pd.read_csv(
    TRAIN_PATH,
    names=columns,
    header=None
)

print("Dataset shape:", df.shape)


# ============================================================
# STEP 3: ATTACK CATEGORY MAPPING
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

df["attack_category"] = (
    df["label"].astype(str).str.strip().str.lower()
    .map(attack_mapping)
)

if df["attack_category"].isna().any():
    unknown = df.loc[
        df["attack_category"].isna(), "label"
    ].unique()

    raise ValueError(f"Unknown attack labels: {unknown}")


# ============================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["label", "difficulty", "attack_category"]
).copy()

y = df["attack_category"].copy()

numerical_features = [
    col for col in X.columns
    if col not in ["protocol_type", "service", "flag"]
]

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

X[numerical_features] = X[numerical_features].apply(
    pd.to_numeric,
    errors="coerce"
)

X[numerical_features] = X[numerical_features].replace(
    [float("inf"), float("-inf")],
    float("nan")
)

for col in categorical_features:
    X[col] = X[col].astype(str)


# ============================================================
# STEP 5: STRATIFIED TRAINING AND VALIDATION SPLIT
# ============================================================

print("\n========== CREATING VALIDATION SPLIT ==========")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nValidation distribution:")
print(y_val.value_counts())


# ============================================================
# STEP 6: PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# STEP 7: BASELINE LIGHTGBM MODEL
# ============================================================

print("\n========== TRAINING BASELINE LIGHTGBM ==========")

baseline_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=1,
        verbosity=-1
    ))
])

start = time.time()

baseline_model.fit(X_train, y_train)

training_time = time.time() - start

baseline_predictions = baseline_model.predict(X_val)


# ============================================================
# STEP 8: EVALUATE BASELINE
# ============================================================

def evaluate_model(name, actual, predicted, duration):

    accuracy = accuracy_score(actual, predicted)

    macro_precision = precision_score(
        actual, predicted,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        actual, predicted,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual, predicted,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        actual, predicted,
        average="weighted",
        zero_division=0
    )

    print("\n========================================")
    print(name)
    print("========================================")

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision:{macro_precision:.4f}")
    print(f"Macro Recall:   {macro_recall:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Training time:  {duration:.2f} seconds")

    print("\nClassification Report:\n")

    print(
        classification_report(
            actual,
            predicted,
            labels=["Normal", "DoS", "Probe", "R2L", "U2R"],
            zero_division=0
        )
    )

    return {
        "Experiment": name,
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Training_Time": duration
    }


baseline_results = evaluate_model(
    "Baseline_LightGBM",
    y_val,
    baseline_predictions,
    training_time
)


# ============================================================
# STEP 9: SAVE BASELINE RESULTS
# ============================================================

results_df = pd.DataFrame([baseline_results])

results_df.to_csv(
    OUTPUT_DIR / "validation_baseline_results.csv",
    index=False
)

joblib.dump(
    baseline_model,
    OUTPUT_DIR / "validation_baseline_model.joblib"
)

print("\nBaseline experiment saved successfully!")

print("Output folder:", OUTPUT_DIR)

# ============================================================
# STEP 10: LIGHTGBM HYPERPARAMETER TUNING
# ============================================================

print("\n========== LIGHTGBM HYPERPARAMETER TUNING ==========")

from sklearn.base import clone

experiments = [
    {
        "name": "Tuned_LGBM_1",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1
    },
    {
        "name": "Tuned_LGBM_2",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 10
    },
    {
        "name": "Tuned_LGBM_3",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 12
    }
]

tuning_results = [baseline_results]

best_macro_f1 = baseline_results["Macro_F1"]
best_model = baseline_model
best_model_name = "Baseline_LightGBM"

for config in experiments:

    print("\nTraining:", config["name"])

    classifier = LGBMClassifier(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )

    candidate_model = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", classifier)
    ])

    start = time.time()

    candidate_model.fit(X_train, y_train)

    duration = time.time() - start

    predictions = candidate_model.predict(X_val)

    candidate_results = evaluate_model(
        config["name"],
        y_val,
        predictions,
        duration
    )

    tuning_results.append(candidate_results)

    # Select using validation Macro F1
    if candidate_results["Macro_F1"] > best_macro_f1:

        best_macro_f1 = candidate_results["Macro_F1"]
        best_model = candidate_model
        best_model_name = config["name"]


# ============================================================
# STEP 11: COMPARE ALL EXPERIMENTS
# ============================================================

comparison_df = pd.DataFrame(tuning_results)

comparison_df = comparison_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== HYPERPARAMETER COMPARISON ==========")

print(
    comparison_df.to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_DIR / "lightgbm_tuning_comparison.csv",
    index=False
)


# ============================================================
# STEP 12: SAVE BEST VALIDATION MODEL
# ============================================================

best_model_path = (
    OUTPUT_DIR / "best_tuned_lightgbm_validation.joblib"
)

joblib.dump(best_model, best_model_path)

print("\n========== BEST VALIDATION MODEL ==========")

print("Best model:", best_model_name)

print("Best Macro F1:", round(best_macro_f1, 4))

print("Saved model:", best_model_path)

print("\nHyperparameter tuning completed!")

# ============================================================
# STEP 13: CONTROLLED CLASS-WEIGHT EXPERIMENTS
# ============================================================

print("\n========== CLASS WEIGHT OPTIMIZATION ==========")

# Use the best tuned configuration
# Tuned_LGBM_3:
# n_estimators=200
# learning_rate=0.05
# num_leaves=63
# max_depth=12

weight_experiments = {
    "Tuned_Unweighted": {
        "Normal": 1.0,
        "DoS": 1.0,
        "Probe": 1.0,
        "R2L": 1.0,
        "U2R": 1.0
    },

    "Mild_Weighting": {
        "Normal": 1.0,
        "DoS": 1.0,
        "Probe": 1.0,
        "R2L": 1.5,
        "U2R": 2.0
    },

    "Moderate_Weighting": {
        "Normal": 1.0,
        "DoS": 1.0,
        "Probe": 1.0,
        "R2L": 2.0,
        "U2R": 3.0
    },

    "Higher_Weighting": {
        "Normal": 1.0,
        "DoS": 1.0,
        "Probe": 1.0,
        "R2L": 3.0,
        "U2R": 5.0
    }
}

weight_results = []

best_weighted_f1 = -1
best_weighted_model = None
best_weighted_name = None

for name, weights in weight_experiments.items():

    print("\nTraining:", name)
    print("Class weights:", weights)

    classifier = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=12,
        class_weight=weights,
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )

    candidate_model = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", classifier)
    ])

    start = time.time()

    candidate_model.fit(X_train, y_train)

    duration = time.time() - start

    predictions = candidate_model.predict(X_val)

    result = evaluate_model(
        name,
        y_val,
        predictions,
        duration
    )

    weight_results.append(result)

    # Select by Macro F1
    if result["Macro_F1"] > best_weighted_f1:

        best_weighted_f1 = result["Macro_F1"]
        best_weighted_model = candidate_model
        best_weighted_name = name


# ============================================================
# STEP 14: SAVE WEIGHT COMPARISON
# ============================================================

weight_comparison = pd.DataFrame(weight_results)

weight_comparison = weight_comparison.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== CLASS WEIGHT COMPARISON ==========")

print(weight_comparison.to_string(index=False))

weight_comparison.to_csv(
    OUTPUT_DIR / "class_weight_comparison.csv",
    index=False
)


# ============================================================
# STEP 15: SAVE BEST WEIGHTED MODEL
# ============================================================

joblib.dump(
    best_weighted_model,
    OUTPUT_DIR / "best_class_weighted_lightgbm.joblib"
)

print("\n========== BEST CLASS-WEIGHTED MODEL ==========")

print("Best experiment:", best_weighted_name)

print("Best validation Macro F1:", best_weighted_f1)

print("\nClass-weight experiments completed!")
    