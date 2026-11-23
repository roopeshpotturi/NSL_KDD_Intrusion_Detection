
# ============================================================
# PROJECT: MULTI-CLASS INTRUSION DETECTION
# MODEL TRAINING AND COMPARATIVE EVALUATION
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib

# Use a non-interactive backend for saving plots
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from time import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# STEP 1: LOAD PREPARED DATA
# ============================================================

print("\n========== STEP 1: LOAD PREPARED DATA ==========")

BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / "data" / "processed" / "prepared_data.joblib"

data = joblib.load(data_path)

X_train_original = data["X_train_original"].copy()
y_train_original = data["y_train_original"].copy()

X_train_balanced = data["X_train_balanced"].copy()
y_train_balanced = data["y_train_balanced"].copy()

X_test = data["X_test"].copy()
y_test = data["y_test"].copy()

scaler = data["scaler"]

numerical_features = data["numerical_features"]
categorical_features = data["categorical_features"]

print("Original training:", X_train_original.shape)
print("Balanced training:", X_train_balanced.shape)
print("Testing:", X_test.shape)


# ============================================================
# STEP 2: PREPARE CONSISTENT INPUT FEATURES
# ============================================================

print("\n========== STEP 2: FEATURE PREPARATION ==========")

# Original training data contains unscaled numerical values.
# Scale it using the scaler fitted on the original training set.

X_train_original.loc[:, numerical_features] = scaler.transform(
    X_train_original[numerical_features]
)

# Balanced training and testing numerical features are
# already scaled in main.py.

# Convert categorical columns to strings for encoding.
# This also handles pandas category data types.

for col in categorical_features:
    X_train_original[col] = X_train_original[col].astype(str)
    X_train_balanced[col] = X_train_balanced[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print("Feature preparation completed.")


# ============================================================
# STEP 3: CREATE PREPROCESSING PIPELINE
# ============================================================

print("\n========== STEP 3: ONE-HOT ENCODING ==========")

def create_preprocessor():

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    return transformer


# ============================================================
# STEP 4: DEFINE MACHINE LEARNING CLASSIFIERS
# ============================================================

print("\n========== STEP 4: DEFINE CLASSIFIERS ==========")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
        max_features="sqrt"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )
}


# ============================================================
# STEP 5: CREATE RESULTS FOLDER
# ============================================================

results_dir = BASE_DIR / "results"
models_dir = BASE_DIR / "models"

results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

results = []
all_reports = {}


# ============================================================
# STEP 6: TRAIN AND EVALUATE FOUR EXPERIMENTS
# ============================================================

experiments = [
    ("Original", X_train_original, y_train_original),
    ("SMOTENC_Balanced", X_train_balanced, y_train_balanced)
]

for experiment_name, X_train, y_train in experiments:

    for model_name, classifier in models.items():

        print("\n" + "=" * 55)
        print("EXPERIMENT:", experiment_name)
        print("CLASSIFIER:", model_name)
        print("=" * 55)

        # Create a fresh preprocessing pipeline
        preprocessor = create_preprocessor()

        # Build complete pipeline
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ])

        # Train model
        print("\nTraining started...")
        start_time = time()

        pipeline.fit(X_train, y_train)

        training_time = time() - start_time

        print("Training completed!")
        print("Training time:", round(training_time, 2), "seconds")

        # Predict on the same untouched test dataset
        print("\nTesting model...")

        y_pred = pipeline.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        print("\nMODEL PERFORMANCE")
        print("Accuracy:", round(accuracy, 4))
        print("Macro Precision:", round(precision, 4))
        print("Macro Recall:", round(recall, 4))
        print("Macro F1-score:", round(f1, 4))
        print("Weighted F1-score:", round(weighted_f1, 4))

        # Detailed classification report
        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        print("\nCLASSIFICATION REPORT:")
        print(report)

        # Save classification report
        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report_dict).transpose()

        report_filename = (
            experiment_name + "_" +
            model_name.replace(" ", "_") +
            "_report.csv"
        )

        report_df.to_csv(results_dir / report_filename)

        # Save confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["Normal", "DoS", "Probe", "R2L", "U2R"]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "DoS", "Probe", "R2L", "U2R"]
        )

        display.plot(cmap="Blues", xticks_rotation=45)

        plt.title(experiment_name + " - " + model_name)
        plt.tight_layout()

        cm_filename = (
            experiment_name + "_" +
            model_name.replace(" ", "_") +
            "_confusion_matrix.png"
        )

        plt.savefig(results_dir / cm_filename, dpi=300)
        plt.close()

        # Save trained model
        model_filename = (
            experiment_name + "_" +
            model_name.replace(" ", "_") +
            ".joblib"
        )

        joblib.dump(
            pipeline,
            models_dir / model_filename
        )

        # Store metrics
        results.append({
            "Experiment": experiment_name,
            "Classifier": model_name,
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Weighted F1": weighted_f1,
            "Training Time (seconds)": training_time
        })

        all_reports[
            experiment_name + "_" + model_name
        ] = report_dict


# ============================================================
# STEP 7: COMPARE ALL CLASSIFIERS
# ============================================================

print("\n========== FINAL MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv(
    results_dir / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# ============================================================
# STEP 8: FINAL SUMMARY
# ============================================================

print("\n========== PROJECT TRAINING COMPLETED ==========")

print("\nResults folder:", results_dir)
print("Models folder:", models_dir)

print("\nGenerated files:")
print("- model_comparison.csv")
print("- Four classification reports")
print("- Four confusion matrices")
print("- Four trained model files")

print("\nALL FOUR EXPERIMENTS COMPLETED!")
