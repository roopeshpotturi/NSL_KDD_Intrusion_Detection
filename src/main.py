
# ============================================================
# M.TECH DATA SCIENCE PROJECT
# MULTI-CLASS INTRUSION DETECTION USING NSL-KDD
# SMOTE-BASED CLASS BALANCING
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTENC


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n========== STEP 1: DATASET LOADING ==========")

BASE_DIR = Path(__file__).resolve().parents[1]

train_path = BASE_DIR / "data" / "raw" / "KDDTrain+.TXT"
test_path = BASE_DIR / "data" / "raw" / "KDDTest+.TXT"

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

train_df = pd.read_csv(train_path, names=columns, header=None)
test_df = pd.read_csv(test_path, names=columns, header=None)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)


# ============================================================
# STEP 2: ATTACK CATEGORY MAPPING
# ============================================================

print("\n========== STEP 2: ATTACK CLASSIFICATION ==========")

attack_mapping = {

    # DoS attacks
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

    # Probe attacks
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # R2L attacks
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

    # U2R attacks
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",

    # Normal traffic
    "normal": "Normal"
}

train_df["attack_category"] = train_df["label"].map(attack_mapping)
test_df["attack_category"] = test_df["label"].map(attack_mapping)

# Check unmapped labels
unmapped_train = train_df.loc[
    train_df["attack_category"].isna(), "label"
].unique()

unmapped_test = test_df.loc[
    test_df["attack_category"].isna(), "label"
].unique()

print("Unmapped training labels:", unmapped_train)
print("Unmapped testing labels:", unmapped_test)

if len(unmapped_train) > 0 or len(unmapped_test) > 0:
    raise ValueError("Unmapped attack labels found!")

print("\nTraining class distribution:")
print(train_df["attack_category"].value_counts())

print("\nTesting class distribution:")
print(test_df["attack_category"].value_counts())


# ============================================================
# STEP 3: SEPARATE INPUT FEATURES AND TARGET
# ============================================================

print("\n========== STEP 3: FEATURE SEPARATION ==========")

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

numerical_features = [
    col for col in columns
    if col not in [
        "protocol_type",
        "service",
        "flag",
        "label",
        "difficulty"
    ]
]

X_train = train_df[numerical_features + categorical_features].copy()
X_test = test_df[numerical_features + categorical_features].copy()

y_train = train_df["attack_category"].copy()
y_test = test_df["attack_category"].copy()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# STEP 4: HANDLE MISSING VALUES
# ============================================================

print("\n========== STEP 4: MISSING VALUE HANDLING ==========")

# Convert numerical columns into numeric values
X_train[numerical_features] = X_train[
    numerical_features
].apply(pd.to_numeric, errors="coerce")

X_test[numerical_features] = X_test[
    numerical_features
].apply(pd.to_numeric, errors="coerce")

# Replace infinity with missing values
X_train[numerical_features] = X_train[
    numerical_features
].replace([np.inf, -np.inf], np.nan)

X_test[numerical_features] = X_test[
    numerical_features
].replace([np.inf, -np.inf], np.nan)

# Numerical imputation
num_imputer = SimpleImputer(strategy="median")

X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numerical_features]),
    columns=numerical_features
)

X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[numerical_features]),
    columns=numerical_features
)

# Categorical imputation
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_train[categorical_features]),
    columns=categorical_features
)

X_test_cat = pd.DataFrame(
    cat_imputer.transform(X_test[categorical_features]),
    columns=categorical_features
)

print("Missing values handled successfully.")


# ============================================================
# STEP 5: PREPARE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

print("\n========== STEP 5: FEATURE PREPARATION ==========")

# Keep numerical features unscaled for SMOTENC
X_train_prepared = pd.concat(
    [
        X_train_num.reset_index(drop=True),
        X_train_cat.reset_index(drop=True)
    ],
    axis=1
)

X_test_prepared = pd.concat(
    [
        X_test_num.reset_index(drop=True),
        X_test_cat.reset_index(drop=True)
    ],
    axis=1
)

# Mark categorical columns
for col in categorical_features:
    X_train_prepared[col] = X_train_prepared[col].astype("category")

    X_test_prepared[col] = pd.Categorical(
        X_test_prepared[col],
        categories=X_train_prepared[col].cat.categories
    )

print("Training shape:", X_train_prepared.shape)
print("Testing shape:", X_test_prepared.shape)


# ============================================================
# STEP 6: SMOTENC FOR THREE ATTACK CATEGORIES
# ============================================================

print("\n========== STEP 6: SMOTENC BALANCING ==========")

print("\nOriginal class distribution:")
print(y_train.value_counts())

# Balance DoS, Probe and R2L using SMOTENC.
# U2R is excluded because it has only 52 records.

smotenc = SMOTENC(
    categorical_features=categorical_features,
    sampling_strategy={
        "DoS": 67343,
        "Probe": 67343,
        "R2L": 67343
    },
    random_state=42,
    k_neighbors=5
)

X_after_smote, y_after_smote = smotenc.fit_resample(
    X_train_prepared,
    y_train.reset_index(drop=True)
)

print("\nAfter SMOTENC:")
print(pd.Series(y_after_smote).value_counts())


# ============================================================
# STEP 7: RANDOM OVERSAMPLING FOR U2R
# ============================================================

print("\n========== STEP 7: U2R OVERSAMPLING ==========")

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(
    sampling_strategy={"U2R": 67343},
    random_state=42
)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_after_smote,
    y_after_smote
)

print("\nFinal balanced class distribution:")
print(pd.Series(y_train_balanced).value_counts())

print("\nBalanced training shape:", X_train_balanced.shape)


# ============================================================
# STEP 8: SCALE NUMERICAL FEATURES
# ============================================================

print("\n========== STEP 8: NUMERICAL SCALING ==========")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only on original training numerical data
scaler.fit(X_train_num)

# Transform balanced training numerical features
X_train_balanced.loc[:, numerical_features] = scaler.transform(
    X_train_balanced[numerical_features]
)

# Transform untouched testing numerical features
X_test_prepared.loc[:, numerical_features] = scaler.transform(
    X_test_prepared[numerical_features]
)

print("Numerical scaling completed.")


# ============================================================
# STEP 9: FINAL VALIDATION
# ============================================================

print("\n========== STEP 9: FINAL VALIDATION ==========")

print("Training shape:", X_train_balanced.shape)
print("Testing shape:", X_test_prepared.shape)

print("\nTraining missing values:",
      X_train_balanced.isnull().sum().sum())

print("Testing missing values:",
      X_test_prepared.isnull().sum().sum())

print("\nTraining numerical values finite:",
      np.isfinite(
          X_train_balanced[numerical_features].to_numpy()
      ).all())

print("Testing numerical values finite:",
      np.isfinite(
          X_test_prepared[numerical_features].to_numpy()
      ).all())

print("\nPROJECT DATA PREPARATION COMPLETED SUCCESSFULLY!")



# ============================================================
# STEP 10: SAVE PREPARED DATASETS
# ============================================================

import joblib

print("\n========== SAVING PREPARED DATASETS ==========")

processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

project_data = {
    "X_train_original": X_train_prepared,
    "y_train_original": y_train.reset_index(drop=True),
    "X_train_balanced": X_train_balanced,
    "y_train_balanced": y_train_balanced,
    "X_test": X_test_prepared,
    "y_test": y_test.reset_index(drop=True),
    "scaler": scaler,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features
}

save_path = processed_dir / "prepared_data.joblib"

joblib.dump(project_data, save_path)

print("Prepared datasets saved successfully!")
print("Saved location:", save_path)
