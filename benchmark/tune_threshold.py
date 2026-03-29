import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)

# =========================
# CONFIG
# =========================

DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/et_labeled_dataset.json"
MODEL_PATH = "model/iforest_model_tuned.pkl"
SCALER_PATH = "model/scaler_tuned.pkl"

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

print("[*] Loading dataset...")
with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

rows = []
labels = []

for entry in dataset["windows"]:
    rows.append(entry["features"])
    labels.append(entry["label"])

df = pd.DataFrame(rows)
y_true = np.array(labels).astype(int)

# =========================
# PREPROCESS (MATCH TRAINING)
# =========================

for col in LOG_FEATURES:
    df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

X = df[ML_FEATURES].apply(pd.to_numeric, errors="coerce").values

if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("NaN or Inf detected in features")

X_scaled = scaler.transform(X)

# =========================
# SCORING
# =========================

print("[*] Scoring...")
anomaly_scores = -model.decision_function(X_scaled)  

# =========================
# ROC CURVE
# =========================

fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
auc = roc_auc_score(y_true, anomaly_scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

print(f"\nROC-AUC: {auc:.4f}")

# =========================
# FIND BEST THRESHOLD (F1)
# =========================

thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 300)

best_f1 = 0
best_threshold = thresholds[0]

for t in thresholds:
    y_pred = (anomaly_scores > t).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest Threshold: {best_threshold:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")

# =========================
# FINAL EVALUATION
# =========================

y_pred = (anomaly_scores > best_threshold).astype(int)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

print(f"\nAccuracy: {accuracy*100:.2f}%")

# =========================
# CONFUSION MATRIX PLOT
# =========================

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Normal", "ET"])
plt.yticks([0, 1], ["Normal", "ET"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================

precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# =========================
# SCORE DISTRIBUTION
# =========================

plt.figure()
plt.hist(anomaly_scores[y_true == 0], bins=50, alpha=0.6, label="Normal")
plt.hist(anomaly_scores[y_true == 1], bins=50, alpha=0.6, label="Evil Twin")
plt.axvline(best_threshold, linestyle="--", label=f"Threshold ({best_threshold:.4f})")
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score (higher = more anomalous)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# =========================
# F1 vs THRESHOLD
# =========================

f1_values = []
for t in thresholds:
    y_pred_tmp = (anomaly_scores > t).astype(int)
    f1_values.append(f1_score(y_true, y_pred_tmp))

plt.figure()
plt.plot(thresholds, f1_values)
plt.axvline(best_threshold, linestyle="--", label=f"Best Threshold ({best_threshold:.4f})")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold")
plt.legend()
plt.grid()
plt.show()

# =========================
# SAVE BEST THRESHOLD
# =========================

os.makedirs("model", exist_ok=True)
threshold_path = "model/threshold.txt"
with open(threshold_path, "w") as f:
    f.write(str(best_threshold))

print(f"\n[+] Best threshold saved to: {threshold_path}")