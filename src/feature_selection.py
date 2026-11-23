from sklearn.preprocessing import LabelEncoder
import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==========================================
# 1. LOAD PREPARED DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(
    BASE_DIR, "data", "processed", "prepared_data.joblib"
)

data = joblib.load(data_path)

X_train = data["X_train_original"].copy()
y_train = data["y_train_original"].copy()

# Encode text labels into integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print("\nClass mapping:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{label} = {index}")

numerical_features = data["numerical_features"]
categorical_features = data["categorical_features"]

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)


# ==========================================
# 2. SCALE NUMERICAL FEATURES
# ==========================================

scaler = data["scaler"]

X_train.loc[:, numerical_features] = scaler.transform(
    X_train[numerical_features]
)


# ==========================================
# 3. ENCODE CATEGORICAL FEATURES
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

feature_names = preprocessor.get_feature_names_out()

print("Encoded training shape:", X_train_encoded.shape)


# ==========================================
# 4. TRAIN XGBOOST CLASSIFIER
# ==========================================

print("\nTraining XGBoost for feature ranking...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=1
)

model.fit(X_train_encoded, y_train_encoded)


# ==========================================
# 5. CALCULATE FEATURE IMPORTANCE
# ==========================================

importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "importance": importance_values
})


# ==========================================
# 6. GROUP ONE-HOT FEATURES
# ==========================================

def get_original_feature(encoded_name):

    name = encoded_name.split("__", 1)[-1]

    for feature in categorical_features:
        if name.startswith(feature + "_"):
            return feature

    return name


importance_df["original_feature"] = (
    importance_df["encoded_feature"].apply(get_original_feature)
)

ranking_df = (
    importance_df.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

ranking_df.columns = ["feature", "importance"]

ranking_df["rank"] = range(1, len(ranking_df) + 1)


# ==========================================
# 7. DISPLAY FEATURE RANKING
# ==========================================

print("\nXGBoost Feature Ranking:")
print(ranking_df.to_string(index=False))


# ==========================================
# 8. SAVE FEATURE RANKING
# ==========================================

results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

ranking_path = os.path.join(
    results_dir, "xgboost_feature_ranking.csv"
)

ranking_df.to_csv(ranking_path, index=False)

print("\nFeature ranking saved to:")
print(ranking_path)


# ==========================================
# 9. SELECT TOP 13 FEATURES
# ==========================================

selected_features = ranking_df.head(13)["feature"].tolist()

print("\nSelected Top 13 Features:")
for feature in selected_features:
    print(feature)


selected_path = os.path.join(
    results_dir, "selected_features.joblib"
)

joblib.dump(selected_features, selected_path)

print("\nSelected features saved successfully.")
