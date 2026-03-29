import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import time

MODEL_PATH = "model/iforest_model.pkl"
SCALER_PATH = "model/scaler.pkl"
DATASET_PATH = "data/et_labeled_dataset.json"

THRESHOLD = -0.0775
ANOMALY_WINDOW = 5
ANOMALY_REQUIRED = 3

ML_FEATURES = [
    "rssi_mean",
    "rssi_std",
    "packets_per_second",
    "beacon_timing_jitter",
    "beacon_timing_irregularity",
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

print("[*] Loading model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("[*] Loading dataset...")
with open(DATASET_PATH, "r") as f:
    data = json.load(f)

rows = []
labels = []

for entry in data["windows"]:
    rows.append(entry["features"])
    labels.append(entry["label"])

df = pd.DataFrame(rows)
X = df[ML_FEATURES].values
y_true = np.array(labels)

print("[*] Scaling...")
X_scaled = scaler.transform(X)

print("[*] Scoring...")
scores = model.decision_function(X_scaled)

# Convert scores to anomaly flags
anomaly_flags = (scores < THRESHOLD).astype(int)

# Temporal smoothing
smoothed_flags = []
buffer = []

for flag in anomaly_flags:
    buffer.append(flag)
    if len(buffer) > ANOMALY_WINDOW:
        buffer.pop(0)

    if sum(buffer) >= ANOMALY_REQUIRED:
        smoothed_flags.append(1)
    else:
        smoothed_flags.append(0)

y_pred = np.array(smoothed_flags)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

# Detection latency
attack_indices = np.where(y_true == 1)[0]
detected_indices = np.where(y_pred == 1)[0]

if len(attack_indices) > 0 and len(detected_indices) > 0:
    first_attack = attack_indices[0]
    detection_after_attack = detected_indices[detected_indices >= first_attack]

    if len(detection_after_attack) > 0:
        latency_windows = detection_after_attack[0] - first_attack
        print(f"\nDetection latency (windows): {latency_windows}")
    else:
        print("\nAttack not detected.")
