import json
import pandas as pd
import numpy as np
import joblib
import sys
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# CONFIG
DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/labeled_dataset.json"
MODEL_OUT = "model/iforest_model_tuned.pkl"
SCALER_OUT = "model/scaler_tuned.pkl"

ML_FEATURES = [
    "rssi_mean",
    "rssi_std",
    "packets_per_second",
    "beacon_timing_jitter",
    "beacon_timing_irregularity",
    "beacon_count",
    "seq_number_irregularity",
    "seq_number_backwards",
    "ssid_bssid_count",
    "simultaneous_same_ssid_same_channel",
    "disappearance_count",
    "uptime_inconsistency",
    "encryption_numeric",
    "locally_administered_mac",
    "ie_order_changed",
    "ie_count_mean",
    "ie_count_variance",
    "vht_capable"
]

LOG_FEATURES = [
    "beacon_timing_jitter",
    "beacon_timing_irregularity",
    "beacon_count",
    "seq_number_irregularity",
]

# -----------------------------
# Load dataset
# -----------------------------
print(f"[*] Loading labeled dataset from {DATASET_FILE}")

with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

rows = []
labels = []

for w in dataset.get("windows", []):
    features = w.get("features", {})
    label = w.get("label", None)
    if label is None:
        continue
    rows.append(features)
    labels.append(label)

df = pd.DataFrame(rows)
y = np.array(labels).astype(int)

print(f"[+] Loaded {len(df)} windows")
print(f"[+] Normal samples: {(y == 0).sum()}, Anomalies: {(y == 1).sum()}")

# -----------------------------
# Feature checks
# -----------------------------
missing = [f for f in ML_FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

# Log transform
for col in LOG_FEATURES:
    df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

X = df[ML_FEATURES].apply(pd.to_numeric, errors="coerce").values

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("NaN or Inf detected in feature matrix")

X_train = X[y == 0]   
X_test = X           

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Hyperparameter grid
# -----------------------------
param_grid = {
    "n_estimators": [200, 300, 500, 700],
    "max_samples": [128, 256, 512, 1024],
    "max_features": [0.5, 0.3, 0.7, 1.0,]
}

print("\n[*] Starting hyperparameter tuning...\n") 

results = []

for n_estimators, max_samples, max_features in product(
    param_grid["n_estimators"],
    param_grid["max_samples"],
    param_grid["max_features"],
):
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    # higher = more anomalous
    scores = -model.decision_function(X_test_scaled)

    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)

    results.append((roc, ap, n_estimators, max_samples, max_features))

    print(
        f"n_estimators={n_estimators}, "
        f"max_samples={max_samples}, "
        f"max_features={max_features} "
        f"=> ROC-AUC={roc:.4f}, PR-AUC={ap:.4f}"
    )

# -----------------------------
# Select best config
# -----------------------------
results.sort(reverse=True, key=lambda x: (x[0], x[1]))
best = results[0]

print("\n================ BEST CONFIG ================")
print(
    f"ROC-AUC={best[0]:.4f}, PR-AUC={best[1]:.4f}\n"
    f"n_estimators={best[2]}\n"
    f"max_samples={best[3]}\n"
    f"max_features={best[4]}"
)

# -----------------------------
# Train final tuned model
# -----------------------------
best_model = IsolationForest(
    n_estimators=best[2],
    max_samples=best[3],
    max_features=best[4],
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train_scaled)

joblib.dump(best_model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)

print(f"\n[+] Tuned model saved to:  {MODEL_OUT}")
print(f"[+] Scaler saved to:      {SCALER_OUT}")