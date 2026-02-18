import json 
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt


# CONFIG

DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/overlap_test_dataset.json"
MODEL_OUTPUT = "../model/iforest_model.pkl"
SCALER_OUTPUT = "../model/scaler.pkl"

ML_FEATURES = [
    # Signal behavior
    "rssi_mean",
    "rssi_std",

    # Timing behavior
    "packets_per_second",
    "beacon_timing_jitter",
    "beacon_timing_irregularity",

    # Sequence behavior
    "seq_number_irregularity",
    "seq_number_backwards",

    # Topology behavior
    "ssid_bssid_count",
    "simultaneous_same_ssid_same_channel",
    "disappearance_count",
    "uptime_inconsistency",

    # Protocol behavior
    "encryption_numeric",
    "locally_administered_mac",
    "ie_order_changed",
    "ie_count_mean",
    "ie_count_variance",
    "vht_capable"
]

print(f"[*] Loading dataset from {DATASET_FILE}")

with open (DATASET_FILE, "r") as f:
	dataset = json.load(f)
	
rows = []

for round_data in dataset.get("extraction_rounds", []):
	for ap in round_data.get("access_points", []):
		features = ap.get("features", {})
		rows.append(features)
		
if not rows:
	print("[!] No feature data found.")
	sys.exit(1)
	
df = pd.DataFrame(rows)

print(f"[+] Loaded {len(df)} feature vectors")
""" print("[DEBUG] DataFrame shape:", df.shape)
print("[DEBUG] Columns:", df.columns.tolist())

print("[DEBUG] Checking selected features...") """
# SELECT FEATURES

print("[*] Selecting ML features")

missing = [f for f in ML_FEATURES if f not in df.columns]
if missing:
	print(f"[!] Missing features in dataset: {missing}")
	sys.exit(1)

X = df[ML_FEATURES].apply(pd.to_numeric, errors='coerce').values

if np.isnan(X).any() or np.isinf(X).any():
	print("[!] NaN or Inf detected in feature matrix")
	sys.exit(1)


# SCALE FEATURES

print("[*] Scaling features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TRAIN IFOREST

print("[*] Training model")

model = IsolationForest(
	n_estimators=300,
	contamination="auto",    # auto kept as dataset used is clean
	random_state=42,
	n_jobs=-1
)

model.fit(X_scaled)

print('[+] Training complete')

# EVALUATE SCORE DISTRIBUTION

scores = model.decision_function(X_scaled)

print("\n=========== SCORE STATISTICS ===========")
print(f"Mean score: {np.mean(scores):.4f}")
print(f"Std score:  {np.std(scores):.4f}")
print(f"Min score:  {np.min(scores):.4f}")
print(f"Max score:  {np.max(scores):.4f}")

print("\nSuggested anomaly threshold (5th percentile):")
threshold = np.percentile(scores, 5)
print(f"{threshold:.4f}")



os.makedirs("../model", exist_ok=True)

joblib.dump(model, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)

print(f"\n[+] Model saved to:  {MODEL_OUTPUT}")
print(f"[+] Scaler saved to: {SCALER_OUTPUT}")


""" plt.hist(scores, bins=50)
plt.title("Isolation Forest Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show() """