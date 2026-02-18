import json
import pandas as pd
import numpy as np
import sys

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/overlap_test_dataset.json"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print(f"[*] Loading dataset from: {DATASET_FILE}")

with open(DATASET_FILE, "r") as f:
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
print(f"[+] Total features: {len(df.columns)}")

# --------------------------------------------------
# BASIC STATISTICS
# --------------------------------------------------
print("\n================ BASIC STATISTICS ================")
print(df.describe().T)

# --------------------------------------------------
# CHECK FOR MISSING VALUES
# --------------------------------------------------
print("\n================ MISSING VALUES ==================")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

# --------------------------------------------------
# CHECK ZERO VARIANCE FEATURES
# --------------------------------------------------
print("\n================ ZERO VARIANCE FEATURES ==========")
zero_var = df.loc[:, df.std() == 0]
if len(zero_var.columns) > 0:
    print("Features with zero variance:")
    print(zero_var.columns.tolist())
else:
    print("No zero-variance features.")

# --------------------------------------------------
# CHECK LOW VARIANCE FEATURES
# --------------------------------------------------
print("\n================ LOW VARIANCE FEATURES ===========")
low_var = df.loc[:, df.std() < 1e-3]
if len(low_var.columns) > 0:
    print("Features with near-zero variance:")
    print(low_var.columns.tolist())
else:
    print("No near-zero variance features.")

# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------
print("\n================ HIGH CORRELATION FEATURES =======")
corr_matrix = df.corr(numeric_only=True)
high_corr_pairs = []

threshold = 0.95

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append(
                (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            )

if high_corr_pairs:
    print("Highly correlated feature pairs (>0.95):")
    for f1, f2, corr in high_corr_pairs:
        print(f"{f1}  <-->  {f2}   (corr={corr:.3f})")
else:
    print("No highly correlated features.")

# --------------------------------------------------
# FEATURE RANGE CHECK
# --------------------------------------------------
print("\n================ FEATURE RANGES ==================")
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}")

print("\n[✓] Dataset inspection complete.")
