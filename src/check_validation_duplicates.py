
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "data" / "raw" / "KDDTrain+.TXT"

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

df = pd.read_csv(path, names=columns, header=None)

mapping = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS",
    "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L",
    "imap": "R2L", "multihop": "R2L", "phf": "R2L",
    "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "named": "R2L",
    "sendmail": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L",
    "xlock": "R2L", "xsnoop": "R2L",
    "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "perl": "U2R", "rootkit": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"
}

df["target"] = (
    df["label"].str.strip().str.lower().map(mapping)
)

features = df.drop(
    columns=["label", "difficulty", "target"]
)

X_train, X_val, y_train, y_val = train_test_split(
    features,
    df["target"],
    test_size=0.20,
    random_state=42,
    stratify=df["target"]
)

# Find exact feature records appearing in both splits.
train_keys = pd.MultiIndex.from_frame(
    X_train.reset_index(drop=True)
)

val_keys = pd.MultiIndex.from_frame(
    X_val.reset_index(drop=True)
)

overlap = val_keys.isin(train_keys)

print("\n========== DUPLICATE CHECK ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print(
    "Validation records also found in training:",
    int(overlap.sum())
)

print(
    "Percentage of validation records overlapping:",
    round(overlap.mean() * 100, 4), "%"
)

# Check whether duplicated feature records have conflicting labels.
train_lookup = {}

for key, label in zip(train_keys, y_train):
    train_lookup.setdefault(key, set()).add(label)

conflicts = []

for key, label in zip(val_keys, y_val):
    if key in train_lookup and label not in train_lookup[key]:
        conflicts.append((key, label, train_lookup[key]))

print(
    "Overlapping records with conflicting labels:",
    len(conflicts)
)

print("\nDuplicate check completed.")
