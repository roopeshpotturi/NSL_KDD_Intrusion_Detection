
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_DIR = os.path.join(
    RESULTS_DIR, "final_project_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD EXPERIMENT RESULTS
# ==========================================

all_results = []


def add_results(file_path, experiment_group, accuracy_col,
                precision_col, recall_col, f1_col, weighted_col):

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    df = pd.read_csv(file_path)

    for _, row in df.iterrows():

        all_results.append({
            "Experiment": row["Experiment"],
            "Group": experiment_group,
            "Accuracy": row[accuracy_col],
            "Macro_Precision": row[precision_col],
            "Macro_Recall": row[recall_col],
            "Macro_F1": row[f1_col],
            "Weighted_F1": row[weighted_col]
        })


# ==========================================
# 3. BASELINE AND BALANCED MODELS
# ==========================================

add_results(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    "Baseline and Balanced",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1"
)


# ==========================================
# 4. SELECTED FEATURE MODELS
# ==========================================

add_results(
    os.path.join(
        RESULTS_DIR,
        "selected_feature_experiments",
        "selected_model_comparison.csv"
    ),
    "XGBoost Selected Features",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1"
)


# ==========================================
# 5. CLASS-WEIGHTED MODELS
# ==========================================

add_results(
    os.path.join(
        RESULTS_DIR,
        "weighted_experiments",
        "weighted_model_comparison.csv"
    ),
    "Class Weighted",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1"
)


# ==========================================
# 6. VALIDATE RESULTS
# ==========================================

if not all_results:
    raise ValueError("No experiment results were found.")

final_df = pd.DataFrame(all_results)

final_df = final_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

final_df.insert(
    0,
    "Rank",
    range(1, len(final_df) + 1)
)


# ==========================================
# 7. SAVE FINAL COMPARISON
# ==========================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "complete_model_comparison.csv"
)

final_df.to_csv(comparison_path, index=False)

print("\n" + "=" * 65)
print("COMPLETE MODEL COMPARISON")
print("=" * 65)

print(final_df.to_string(index=False))

print("\nBest model based on Macro F1:")
print(final_df.iloc[0].to_string())


# ==========================================
# 8. ACCURACY COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(12, 7))

plt.bar(
    final_df["Experiment"],
    final_df["Accuracy"]
)

plt.title("Accuracy Comparison of All Experiments")
plt.xlabel("Experiments")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "accuracy_comparison.png"),
    dpi=300
)

plt.close()


# ==========================================
# 9. MACRO F1 COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(12, 7))

plt.bar(
    final_df["Experiment"],
    final_df["Macro_F1"]
)

plt.title("Macro F1 Comparison of All Experiments")
plt.xlabel("Experiments")
plt.ylabel("Macro F1 Score")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "macro_f1_comparison.png"),
    dpi=300
)

plt.close()


# ==========================================
# 10. SAVE VALIDATION RESULTS SEPARATELY
# ==========================================

validation_path = os.path.join(
    RESULTS_DIR,
    "weight_tuning",
    "lightgbm_weight_tuning_comparison.csv"
)

if os.path.exists(validation_path):

    validation_df = pd.read_csv(validation_path)

    validation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_weight_tuning_results.csv"
        ),
        index=False
    )


print("\n" + "=" * 65)
print("FINAL RESULTS GENERATED SUCCESSFULLY")
print("=" * 65)

print("\nSaved files:")
print("1. complete_model_comparison.csv")
print("2. accuracy_comparison.png")
print("3. macro_f1_comparison.png")
print("4. validation_weight_tuning_results.csv")

print("\nOutput folder:")
print(OUTPUT_DIR)
