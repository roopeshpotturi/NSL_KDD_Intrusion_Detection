
import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from lightgbm import LGBMClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "prepared_data.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "results", "selected_features.joblib"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results", "selected_feature_experiments"
)

MODELS_DIR = os.path.join(
    BASE_DIR, "models", "selected_feature_models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA AND SELECTED FEATURES
# ==========================================

data = joblib.load(DATA_PATH)

selected_features = joblib.load(FEATURE_PATH)

X_train_original = data["X_train_original"].copy()
y_train_original = data["y_train_original"].copy()

X_train_balanced = data["X_train_balanced"].copy()
y_train_balanced = data["y_train_balanced"].copy()

X_test = data["X_test"].copy()
y_test = data["y_test"].copy()

numerical_features = data["numerical_features"]
categorical_features = data["categorical_features"]

selected_numerical = [
    feature for feature in selected_features
    if feature in numerical_features
]

selected_categorical = [
    feature for feature in selected_features
    if feature in categorical_features
]

print("Selected features:", selected_features)
print("Number of selected features:", len(selected_features))


# ==========================================
# 3. PREPARE ORIGINAL TRAINING DATA
# ==========================================

scaler = data["scaler"]

X_train_original.loc[:, numerical_features] = scaler.transform(
    X_train_original[numerical_features]
)


# Keep only the selected 13 features

X_train_original = X_train_original[selected_features]
X_train_balanced = X_train_balanced[selected_features]
X_test = X_test[selected_features]


print("\nDataset shapes after feature selection:")
print("Original training:", X_train_original.shape)
print("Balanced training:", X_train_balanced.shape)
print("Testing:", X_test.shape)


# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", selected_numerical),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            selected_categorical
        )
    ],
    remainder="drop"
)

# Fit only on original training data
X_train_original_encoded = preprocessor.fit_transform(
    X_train_original
)

# Apply the same encoding to other datasets
X_train_balanced_encoded = preprocessor.transform(
    X_train_balanced
)

X_test_encoded = preprocessor.transform(X_test)


print("\nEncoded dataset shapes:")
print("Original:", X_train_original_encoded.shape)
print("Balanced:", X_train_balanced_encoded.shape)
print("Testing:", X_test_encoded.shape)


# ==========================================
# 5. ENCODE TARGET LABELS
# ==========================================

label_encoder = LabelEncoder()

y_train_original_encoded = label_encoder.fit_transform(
    y_train_original
)

y_train_balanced_encoded = label_encoder.transform(
    y_train_balanced
)

y_test_encoded = label_encoder.transform(y_test)


# ==========================================
# 6. DEFINE EXPERIMENTS
# ==========================================

experiments = [
    {
        "name": "Selected_Original_RandomForest",
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
            max_features="sqrt"
        ),
        "X_train": X_train_original_encoded,
        "y_train": y_train_original_encoded
    },
    {
        "name": "Selected_Original_LightGBM",
        "model": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=1,
            verbosity=-1
        ),
        "X_train": X_train_original_encoded,
        "y_train": y_train_original_encoded
    },
    {
        "name": "Selected_Balanced_RandomForest",
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
            max_features="sqrt"
        ),
        "X_train": X_train_balanced_encoded,
        "y_train": y_train_balanced_encoded
    },
    {
        "name": "Selected_Balanced_LightGBM",
        "model": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=1,
            verbosity=-1
        ),
        "X_train": X_train_balanced_encoded,
        "y_train": y_train_balanced_encoded
    }
]


# ==========================================
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

all_results = []

for experiment in experiments:

    name = experiment["name"]
    model = experiment["model"]

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    model.fit(
        experiment["X_train"],
        experiment["y_train"]
    )

    predictions = model.predict(X_test_encoded)

    # Calculate evaluation metrics

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

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision:{precision:.4f}")
    print(f"Macro Recall:   {recall:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")

    # Classification report

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

    # Confusion matrix

    cm = confusion_matrix(
        y_test_encoded,
        predictions,
        labels=list(range(len(label_encoder.classes_)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(name)
    plt.tight_layout()

    cm_path = os.path.join(
        RESULTS_DIR, name + "_confusion_matrix.png"
    )

    plt.savefig(cm_path)
    plt.close(fig)

    # Save trained model

    model_path = os.path.join(
        MODELS_DIR, name + ".joblib"
    )

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "selected_features": selected_features
        },
        model_path
    )

    # Store results

    all_results.append({
        "Experiment": name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })


# ==========================================
# 8. SAVE COMPARISON RESULTS
# ==========================================

results_df = pd.DataFrame(all_results)

results_path = os.path.join(
    RESULTS_DIR, "selected_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 55)
print("FINAL SELECTED FEATURE MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_path)

print("\nAll four selected-feature experiments completed!")
