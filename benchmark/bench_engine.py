import json
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================

DATASET_FILE = "data/et_labeled_dataset_for_ppt.json"
MODEL_PATH = "model/iforest_model.pkl"
SCALER_PATH = "model/scaler.pkl"
THRESHOLD_PATH = "model/threshold.txt"

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

# =========================
# LOAD MODEL
# =========================

print("[*] Loading model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

print(f"[*] Threshold: {threshold:.6f}")

# =========================
# LOAD DATASET
# =========================

print("[*] Loading dataset...")
with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

windows = dataset["windows"]

X = []
y_true = []

for entry in windows:
    features = entry["features"]

    # Apply same log transform as training
    for col in LOG_FEATURES:
        if col in features:
            features[col] = np.log1p(features[col])

    X.append([features.get(f, 0) for f in ML_FEATURES])
    y_true.append(entry["label"])

X = np.array(X)
y_true = np.array(y_true)

print(f"[*] Total samples: {len(y_true)}")

# =========================
# BATCH INFERENCE
# =========================

X_scaled = scaler.transform(X)
scores = model.decision_function(X_scaled)
y_pred_model = (scores < threshold).astype(int)

auc = roc_auc_score(y_true, -scores)

# =========================
# MODEL METRICS
# =========================

acc_model = accuracy_score(y_true, y_pred_model)
prec_model = precision_score(y_true, y_pred_model, zero_division=0)
rec_model = recall_score(y_true, y_pred_model, zero_division=0)
f1_model = f1_score(y_true, y_pred_model, zero_division=0)

# =========================
# MODEL + HEURISTICS
# =========================

y_pred_engine = []

for i, entry in enumerate(windows):
    features = entry["features"]
    score = scores[i]

    is_threat = False

    if score < threshold:
        # Only apply heuristics if model already suspicious
        is_threat = True

        # Optional: strengthen decision
        if features.get("vendor_mismatch", 0) == 1:
            is_threat = True

        if features.get("encryption_downgrade", 0) == 1:
            is_threat = True

    y_pred_engine.append(1 if is_threat else 0)

# =========================
# ENGINE METRICS
# =========================

acc_engine = accuracy_score(y_true, y_pred_engine)
prec_engine = precision_score(y_true, y_pred_engine, zero_division=0)
rec_engine = recall_score(y_true, y_pred_engine, zero_division=0)
f1_engine = f1_score(y_true, y_pred_engine, zero_division=0)

# =========================
# PRINT RESULTS
# =========================

print("\n==============================")
print("MODEL ONLY PERFORMANCE")
print("==============================")
print(f"AUC:        {auc:.4f}")
print(f"Accuracy:   {acc_model:.4f}")
print(f"Precision:  {prec_model:.4f}")
print(f"Recall:     {rec_model:.4f}")
print(f"F1 Score:   {f1_model:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_model))

print("\n==============================")
print("MODEL + HEURISTIC ENGINE")
print("==============================")
print(f"Accuracy:   {acc_engine:.4f}")
print(f"Precision:  {prec_engine:.4f}")
print(f"Recall:     {rec_engine:.4f}")
print(f"F1 Score:   {f1_engine:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_engine))

print("\n==============================")
print("COMPARISON SUMMARY")
print("==============================")
print(f"{'Metric':<12} {'Model':<10} {'Engine':<10}")
print(f"{'Accuracy':<12} {acc_model:<10.4f} {acc_engine:<10.4f}")
print(f"{'Precision':<12} {prec_model:<10.4f} {prec_engine:<10.4f}")
print(f"{'Recall':<12} {rec_model:<10.4f} {rec_engine:<10.4f}")
print(f"{'F1':<12} {f1_model:<10.4f} {f1_engine:<10.4f}")