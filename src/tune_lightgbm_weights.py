
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from lightgbm import LGBMClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "prepared_data.joblib"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results", "weight_tuning"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD ORIGINAL TRAINING DATA
# ==========================================

data = joblib.load(DATA_PATH)

X = data["X_train_original"].copy()
y = data["y_train_original"].copy()

numerical_features = data["numerical_features"]
categorical_features = data["categorical_features"]

print("Original training data:", X.shape)
print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 3. ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

classes = label_encoder.classes_

print("\nClass mapping:")

for index, label in enumerate(classes):
    print(label, "=", index)


# ==========================================
# 4. TRAIN-VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)
# ==========================================
# CHECK DUPLICATE RECORD LEAKAGE
# ==========================================

train_hashes = pd.util.hash_pandas_object(
    X_train, index=False
)

val_hashes = pd.util.hash_pandas_object(
    X_val, index=False
)

train_hash_set = set(train_hashes)

overlap = val_hashes.isin(train_hash_set)

print("\nDuplicate leakage check:")
print("Validation records:", len(X_val))
print("Records also present in training:", overlap.sum())
print(
    "Duplicate overlap percentage:",
    round(overlap.mean() * 100, 2), "%"
)

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# ==========================================
# 5. PREPROCESSING
# ==========================================

# Fit scaler only on the training split

scaler = StandardScaler()

X_train.loc[:, numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

X_val.loc[:, numerical_features] = scaler.transform(
    X_val[numerical_features]
)


# Fit categorical encoder only on training split

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)

print("\nEncoded training:", X_train_encoded.shape)
print("Encoded validation:", X_val_encoded.shape)


# ==========================================
# 6. CALCULATE BASE CLASS WEIGHTS
# ==========================================

class_ids = np.unique(y_train)

base_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train
)

base_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, base_weights)
}

print("\nBase class weights:")

for class_id, weight in base_weight_dict.items():
    print(classes[class_id], ":", round(weight, 3))


# ==========================================
# 7. DEFINE WEIGHT EXPERIMENTS
# ==========================================

# Multipliers apply to the balanced weights of
# R2L and U2R only.

weight_settings = {
    "Balanced_Baseline": (1, 1),
    "R2L_2x_U2R_2x": (2, 2),
    "R2L_5x_U2R_5x": (5, 5),
    "R2L_10x_U2R_10x": (10, 10),
    "R2L_5x_U2R_10x": (5, 10)
}


# ==========================================
# 8. TRAIN AND VALIDATE
# ==========================================

results = []

for experiment_name, (r2l_multiplier, u2r_multiplier) in weight_settings.items():

    weights = base_weight_dict.copy()

    r2l_id = list(classes).index("R2L")
    u2r_id = list(classes).index("U2R")

    weights[r2l_id] *= r2l_multiplier
    weights[u2r_id] *= u2r_multiplier

    print("\n" + "=" * 60)
    print("Experiment:", experiment_name)
    print("=" * 60)

    print("R2L weight:", round(weights[r2l_id], 3))
    print("U2R weight:", round(weights[u2r_id], 3))

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        class_weight=weights,
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )

    model.fit(X_train_encoded, y_train)

    predictions = model.predict(X_val_encoded)

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_val,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(
        RESULTS_DIR,
        experiment_name + "_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    results.append({
        "Experiment": experiment_name,
        "R2L_Weight_Multiplier": r2l_multiplier,
        "U2R_Weight_Multiplier": u2r_multiplier,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })


# ==========================================
# 9. SAVE VALIDATION COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    RESULTS_DIR,
    "lightgbm_weight_tuning_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("FINAL VALIDATION COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="Macro_F1",
        ascending=False
    ).to_string(index=False)
)

print("\nResults saved to:")
print(results_path)

print("\nAll weight tuning experiments completed!")
