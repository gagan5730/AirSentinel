import json
import joblib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/et_labeled_dataset.json"
MODEL_PATH = "model/iforest_model_tuned.pkl"
SCALER_PATH = "model/scaler_tuned.pkl"

# Use the SAME parameters as your live detection engine
ALERT_THRESHOLD = 0.3  # From AirSentinelEngine.__init__
WARMUP_LENIENCY = 0.25
WARMUP_EVALS = 3
CONFIRMATION_REQUIRED = 2

ML_FEATURES = [
    "rssi_mean", "rssi_std", "packets_per_second",
    "beacon_timing_jitter", "beacon_timing_irregularity",
    "beacon_count", "seq_number_irregularity", "seq_number_backwards",
    "ssid_bssid_count", "simultaneous_same_ssid_same_channel",
    "disappearance_count", "uptime_inconsistency",
    "encryption_numeric", "locally_administered_mac",
    "ie_order_changed", "ie_count_mean", "ie_count_variance", "vht_capable"
]

LOG_FEATURES = [
    "beacon_timing_jitter",
    "beacon_timing_irregularity",
    "beacon_count",
    "seq_number_irregularity",
]

# =========================
# LOAD MODEL & DATA
# =========================
print("="*70)
print("🛡️  AirSentinel Detection Engine - Benchmark")
print("="*70)
print()

print("[*] Loading model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"  ✓ Model: {type(model).__name__}")

print("[*] Loading dataset...")
with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

# =========================
# ORGANIZE DATA
# =========================
print("[*] Processing dataset...")

all_windows = dataset["windows"]
print(f"  ✓ Total windows: {len(all_windows)}")

# Group by SSID for multi-AP context
ssid_to_windows = defaultdict(list)
bssid_to_windows = defaultdict(list)

for window in all_windows:
    bssid = window.get("bssid", "UNKNOWN")
    # Extract SSID from features or use a placeholder
    features = window.get("features", {})
    # If SSID is not in the window, we'll need to infer it from context
    # For now, we'll use BSSID as a proxy for unique networks
    ssid = window.get("ssid", bssid)  # Fallback to BSSID if SSID missing
    
    window['ssid'] = ssid
    window['bssid'] = bssid
    
    ssid_to_windows[ssid].append(window)
    bssid_to_windows[bssid].append(window)

print(f"  ✓ Unique BSSIDs: {len(bssid_to_windows)}")
print(f"  ✓ Unique SSIDs: {len(ssid_to_windows)}")

# Analyze multi-AP SSIDs
multi_ap_ssids = {ssid: windows for ssid, windows in ssid_to_windows.items() if len(set(w['bssid'] for w in windows)) > 1}
print(f"  ✓ SSIDs with multiple BSSIDs: {len(multi_ap_ssids)}")

# =========================
# DETECTION SIMULATION
# =========================
print()
print("[*] Running detection engine simulation...")
print()

# State tracking (like live engine)
ap_eval_counts = defaultdict(int)
consecutive_anomalies = defaultdict(int)
ap_trust_scores = defaultdict(float)

y_true = []
y_pred = []
detection_details = []

for window_idx, window in enumerate(all_windows):
    features_dict = window['features']
    label = window['label']
    bssid = window['bssid']
    ssid = window['ssid']
    
    y_true.append(label)
    
    # ==========================================
    # STEP 1: ML MODEL INFERENCE
    # ==========================================
    df_row = pd.DataFrame([features_dict])
    
    # Apply log transform (SAME as live engine)
    for col in LOG_FEATURES:
        if col in df_row.columns:
            df_row[col] = np.log1p(np.clip(df_row[col], a_min=0, a_max=None))
    
    # Extract feature vector
    X = df_row[ML_FEATURES].apply(pd.to_numeric, errors="coerce").values
    X_scaled = scaler.transform(X)
    
    # Get ML predictions
    score = -model.decision_function(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]
    
    # ==========================================
    # STEP 2: WARM-UP LENIENCY (Zero-Trust)
    # ==========================================
    eval_count = ap_eval_counts[bssid] + 1
    ap_eval_counts[bssid] = eval_count
    
    confidence_factor = min(1.0, eval_count / WARMUP_EVALS)
    warmup_bonus = WARMUP_LENIENCY * (1.0 - confidence_factor)
    effective_threshold = ALERT_THRESHOLD + warmup_bonus
    
    # ==========================================
    # STEP 3: INITIAL SUSPICION ASSESSMENT
    # ==========================================
    is_suspicious = False
    reasons = []
    
    # ML score check with effective threshold
    if score > effective_threshold:
        is_suspicious = True
        reasons.append(f"ML_score:{score:.3f}")
    
    if prediction == -1 and score > effective_threshold:
        reasons.append("ML_anomaly")
    
    # Software MAC check
    software_mac = features_dict.get('locally_administered_mac', 0)
    if software_mac == 1:
        is_suspicious = True
        reasons.append("Software_MAC")
    
    # Signal stability
    stability = features_dict.get('signal_stability', 1.0)
    if stability < 0.5:
        reasons.append(f"Unstable:{stability:.2f}")
    
    # ==========================================
    # STEP 4: EVIL TWIN DETECTION LOGIC
    # ==========================================
    ssid_count = features_dict.get('ssid_bssid_count', 1)
    seq_ooo = features_dict.get('seq_out_of_order_rate', 0)
    seq_vol = features_dict.get('seq_volatility', 0)
    
    # Perfect Clone Detection (EXACT thresholds from live engine)
    is_perfect_clone = (
        (seq_ooo > 0.15 or seq_vol > 50) or 
        (score > (ALERT_THRESHOLD + 0.3) and stability < 0.4)
    )
    
    # Anti-hotspot filter
    if ssid_count <= 1 and software_mac == 1:
        is_perfect_clone = False
    
    is_threat = False
    threat_level = 'NONE'
    
    # CASE 1: Perfect Clone (BSSID Spoofing)
    if is_perfect_clone and ssid_count <= 1:
        is_threat = True
        threat_level = 'HIGH'
        reasons.insert(0, "PERFECT_CLONE")
        if seq_ooo > 0.15:
            reasons.append(f"SeqOOO:{seq_ooo:.2%}")
        if stability < 0.4:
            reasons.append(f"SigInstable:{stability:.2f}")
    
    # CASE 2: Evil Twin (Multi-AP analysis)
    elif ssid_count > 1:
        # Get all BSSIDs for this SSID
        same_ssid_windows = ssid_to_windows[ssid]
        same_ssid_bssids = set(w['bssid'] for w in same_ssid_windows)
        
        # Analyze peer APs
        has_authentic_peer = False
        peer_vendors = set()
        current_vendor = features_dict.get('vendor', 'Unknown')
        
        for peer_bssid in same_ssid_bssids:
            if peer_bssid != bssid:
                # Get latest features for this peer
                peer_windows = bssid_to_windows[peer_bssid]
                if peer_windows:
                    peer_features = peer_windows[-1]['features']
                    peer_software_mac = peer_features.get('locally_administered_mac', 0)
                    peer_vendor = peer_features.get('vendor', 'Unknown')
                    
                    if peer_software_mac == 0:
                        has_authentic_peer = True
                    
                    peer_vendors.add(peer_vendor)
        
        # MAIN RULE: Suspicious AP + Authentic peer exists
        if is_suspicious and has_authentic_peer:
            is_threat = True
            threat_level = 'HIGH'
            reasons.insert(0, "EVIL_TWIN")
        
        # CASE 3: Vendor mismatch (hardware MAC only)
        elif not software_mac:
            all_vendors = peer_vendors | {current_vendor}
            if len(all_vendors) > 1 and current_vendor == 'Unknown':
                is_threat = True
                threat_level = 'HIGH'
                reasons.append("Unknown_vendor")
        
        # FALSE POSITIVE MITIGATION (Enterprise WiFi)
        if is_threat and not software_mac:
            all_vendors = peer_vendors | {current_vendor}
            enterprise_vendors = ['Cisco', 'Aruba', 'Ubiquiti', 'Ruckus']
            if len(all_vendors) == 1 and list(all_vendors)[0] in enterprise_vendors:
                is_threat = False
                threat_level = 'NONE'
                reasons = []
    
    # ==========================================
    # STEP 5: TRUST SCORING (Zero-Trust)
    # ==========================================
    current_trust = ap_trust_scores[bssid]
    if is_threat:
        new_trust = -1.0  # Immediate floor
    elif is_suspicious:
        new_trust = max(-1.0, current_trust - 0.10)  # Slow erosion
    else:
        new_trust = min(1.0, current_trust + 0.05)  # Slow accumulation
    
    ap_trust_scores[bssid] = new_trust
    
    # ==========================================
    # STEP 6: CONSECUTIVE ANOMALY CONFIRMATION
    # ==========================================
    if is_threat:
        consec = consecutive_anomalies[bssid] + 1
        consecutive_anomalies[bssid] = consec
        
        # Only alert after consecutive confirmations
        if consec >= CONFIRMATION_REQUIRED:
            final_prediction = 1
            detection_status = f"THREAT_CONFIRMED({consec})"
        else:
            final_prediction = 0  # Suppress until confirmed
            detection_status = f"SUSPICIOUS({consec}/{CONFIRMATION_REQUIRED})"
    elif is_suspicious:
        consec = consecutive_anomalies[bssid] + 1
        consecutive_anomalies[bssid] = consec
        final_prediction = 0
        detection_status = f"SUSPICIOUS({consec})"
    else:
        consecutive_anomalies[bssid] = 0  # Reset on clean eval
        final_prediction = 0
        detection_status = "MONITORING"
    
    y_pred.append(final_prediction)
    
    # Store details for analysis
    detection_details.append({
        'bssid': bssid,
        'ssid': ssid,
        'label': label,
        'prediction': final_prediction,
        'ml_score': score,
        'threat_level': threat_level,
        'status': detection_status,
        'trust_score': new_trust,
        'reasons': reasons,
        'ssid_count': ssid_count,
        'software_mac': software_mac,
        'eval_count': eval_count
    })
    
    # Progress indicator
    if (window_idx + 1) % 500 == 0:
        print(f"  Processed {window_idx + 1}/{len(all_windows)} windows...")

print(f"  ✓ Completed {len(all_windows)} evaluations")
print()

# =========================
# EVALUATION
# =========================
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("="*70)
print("DETECTION ENGINE BENCHMARK RESULTS")
print("="*70)
print()

cm = confusion_matrix(y_true, y_pred)
print("=== Confusion Matrix ===")
print(cm)
print()

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Evil Twin']))

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
print("=== Detailed Metrics ===")
print(f"True Negatives:  {tn:5d} (Correct normal classifications)")
print(f"False Positives: {fp:5d} (Normal flagged as threat)")
print(f"False Negatives: {fn:5d} (Threat missed)")
print(f"True Positives:  {tp:5d} (Correct threat detections)")
print()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print()

# =========================
# DETAILED ANALYSIS
# =========================
print("="*70)
print("DETECTION ANALYSIS")
print("="*70)
print()

# Analyze false positives
fps = [d for d in detection_details if d['label'] == 0 and d['prediction'] == 1]
fns = [d for d in detection_details if d['label'] == 1 and d['prediction'] == 0]

if fps:
    print(f"=== False Positives Analysis ({len(fps)} cases) ===")
    reason_counts = defaultdict(int)
    for fp in fps[:10]:  # Show first 10
        print(f"  BSSID: {fp['bssid']}")
        print(f"    ML Score: {fp['ml_score']:.3f}, Trust: {fp['trust_score']:.2f}")
        print(f"    Reasons: {', '.join(fp['reasons'])}")
        print(f"    SSID Count: {fp['ssid_count']}, Software MAC: {fp['software_mac']}")
        print()
        for reason in fp['reasons']:
            reason_counts[reason] += 1
    
    print("  Top FP Reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {reason}: {count} occurrences")
    print()

if fns:
    print(f"=== False Negatives Analysis ({len(fns)} cases) ===")
    for fn in fns[:10]:  # Show first 10
        print(f"  BSSID: {fn['bssid']}")
        print(f"    ML Score: {fn['ml_score']:.3f}, Status: {fn['status']}")
        print(f"    Trust: {fn['trust_score']:.2f}")
        print(f"    Reasons: {', '.join(fn['reasons']) if fn['reasons'] else 'None'}")
        print(f"    SSID Count: {fn['ssid_count']}, Software MAC: {fn['software_mac']}")
        print()

# Show some true positives for validation
tps = [d for d in detection_details if d['label'] == 1 and d['prediction'] == 1]
if tps:
    print(f"=== True Positives Sample ({len(tps)} total) ===")
    for tp in tps[:5]:  # Show first 5
        print(f"  BSSID: {tp['bssid']}")
        print(f"    Threat: {tp['threat_level']}, Status: {tp['status']}")
        print(f"    Reasons: {', '.join(tp['reasons'])}")
        print()

print("="*70)
print("SYSTEM CONFIGURATION USED")
print("="*70)
print(f"Alert Threshold:        {ALERT_THRESHOLD}")
print(f"Warm-up Leniency:       {WARMUP_LENIENCY}")
print(f"Warm-up Evaluations:    {WARMUP_EVALS}")
print(f"Confirmation Required:  {CONFIRMATION_REQUIRED}")
print()
print("Detection Logic:")
print("  ✓ ML-based anomaly scoring")
print("  ✓ Zero-trust warm-up leniency")
print("  ✓ Multi-AP context (authentic peer detection)")
print("  ✓ Perfect clone detection (BSSID spoofing)")
print("  ✓ Consecutive anomaly confirmation")
print("  ✓ Trust score tracking")
print("  ✓ Enterprise WiFi false positive mitigation")
print("="*70)