
import os
import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from lightgbm import LGBMClassifier


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "prepared_data.joblib"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results", "weighted_experiments"
)

MODELS_DIR = os.path.join(
    BASE_DIR, "models", "weighted_models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD ORIGINAL DATA
# ==========================================

data = joblib.load(DATA_PATH)

X_train = data["X_train_original"].copy()
y_train = data["y_train_original"].copy()

X_test = data["X_test"].copy()
y_test = data["y_test"].copy()

numerical_features = data["numerical_features"]
categorical_features = data["categorical_features"]

print("Original training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nOriginal training class distribution:")
print(y_train.value_counts())


# ==========================================
# 3. SCALE NUMERICAL FEATURES
# ==========================================

scaler = data["scaler"]

X_train.loc[:, numerical_features] = scaler.transform(
    X_train[numerical_features]
)


# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

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
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)


# ==========================================
# 5. ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# ==========================================
# 6. DEFINE CLASS-WEIGHTED MODELS
# ==========================================

models = {
    "Weighted_RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
        max_features="sqrt"
    ),

    "Weighted_LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    model.fit(X_train_encoded, y_train_encoded)

    predictions = model.predict(X_test_encoded)

    accuracy = accuracy_score(y_test_encoded, predictions)

    precision = precision_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    report = classification_report(
        y_test_encoded,
        predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(
        RESULTS_DIR, name + "_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder
        },
        os.path.join(MODELS_DIR, name + ".joblib")
    )

    results.append({
        "Experiment": name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })


# ==========================================
# 8. SAVE COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    RESULTS_DIR, "weighted_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("CLASS-WEIGHTED MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nResults saved to:", results_path)
print("\nBoth class-weighted experiments completed!")
