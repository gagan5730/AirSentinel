#!/usr/bin/env python3
"""
AirSentinel Detection Engine
Real-time evil twin detection with trained model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.all import sniff
import joblib
import numpy as np
import json
import requests
from datetime import datetime
from collections import defaultdict, deque
import argparse
from data_collection.channel_hopper import ChannelHopper
from data_collection.extract import FeatureExtractor
from utils.notifications import TelegramNotifier
from data_collection.capture import extract_ap_features

TRUST_UNVERIFIED = 'UNVERIFIED'   
TRUST_MONITORING = 'MONITORING'   
TRUST_SUSPICIOUS = 'SUSPICIOUS'   
TRUST_THREAT     = 'THREAT'       

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


class AirSentinelEngine:
    def __init__(self, model_path, scaler_path=None, min_packets=10, alert_threshold=0.08, telegram_token=None, telegram_chat_id=None):
        print("="*70)
        print("🛡️  AirSentinel Detection Engine v1.0")
        print("="*70)
        print()
        
        print("[*] Loading trained model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        self.extractor = FeatureExtractor(use_packet_time=False)
        self.feature_names = ML_FEATURES

        print("  → Using default feature names (18 features)")
       
        print(f"  ✓ Model loaded: {type(self.model).__name__}")
        print(f"  ✓ Features: {len(self.feature_names)}")
        print()
        
        self.min_packets = min_packets
        self.alert_threshold = alert_threshold
        
        self.trust_states = {}          
        self.ap_trust_scores = {}       
        self.ap_eval_counts = {}        
        self.WARMUP_EVALS    = 3        
        self.WARMUP_LENIENCY = 0.25     
        self.consecutive_anomalies  = {} 
        self.CONFIRMATION_REQUIRED  = 2 #how many anomalies till alert

        # General observation tracking
        
        self.ap_info = defaultdict(dict)
        self.ssid_bssid_map = defaultdict(set)
        self.checked_aps = set()
        self.alerts = []
        self.last_alert_time = {}
        
        # Stats
        self.total_packets = 0
        self.start_time = datetime.now()
        
        # Notifications
        self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        self.IS_NOTIF_ON = False
        config_path = 'data/config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    config = json.load(file)
                    val = config.get("IS_NOTIF_ON", "False")
                    self.IS_NOTIF_ON = val.lower() == "true" if isinstance(val, str) else bool(val)
            except Exception as e:
                print(f"  [!] Error loading config.json: {e}")
        else:
            print(f"  [!] Config file not found at {config_path}, using defaults.")
        
        print("[*] Configuration:")
        print(f"  Min packets for detection: {min_packets}")
        print(f"  Alert threshold: {alert_threshold}")
        print(f"  Notfications: {self.IS_NOTIF_ON}")
        
        # Clear previous alerts on startup
        self._clear_previous_alerts()
        
        # Dashboard API
        self.dashboard_base_url = "http://localhost:5000/api"
        print(f"  Dashboard API: {self.dashboard_base_url}")
        print()

    def _set_trust_state(self, bssid, state):
        old = self.trust_states.get(bssid, TRUST_UNVERIFIED)
        if old == state:
            return
        self.trust_states[bssid] = state
        self._report_trust_state_to_dashboard(bssid, state)
        print(f"[ZT] {bssid}  {old} \u2192 {state}")

    # ── Packet Observation ─────────────────────────────────────────────────

    def observe_packet(self, packet):
        """Process captured packet."""
        
        self.total_packets += 1
        
        try:
            packet_features = extract_ap_features(packet)
        except Exception:
            return
        
        if not packet_features or 'bssid' not in packet_features:
            return
        
        bssid = packet_features['bssid'].upper()
        ssid = packet_features.get('ssid', '')
        
        self.extractor.observe_packet(packet_features)
        self.ssid_bssid_map[ssid].add(bssid)

        # ── Zero-Trust: First-Seen Handling ────────────────────────────────
        if bssid not in self.ap_info:
            self.ap_info[bssid] = {
                'ssid': ssid,
                'first_seen': datetime.now(),
                'vendor': packet_features.get('vendor', 'Unknown'),
                'encryption': packet_features.get('encryption_type', 'Unknown'),
            }
            self.trust_states[bssid] = TRUST_UNVERIFIED
            print(f"[ZT] New AP: {bssid} ({ssid}) — UNVERIFIED")

            self._report_network_to_dashboard(bssid, packet_features)
        
        # ── Threat / Trust Evaluation ──────────────────────────────────────
        packet_count = len(self.extractor.ap_observations[bssid])
        if packet_count >= self.min_packets:
            if bssid not in self.checked_aps:
                # Promote to MONITORING before the first full check
                if self.trust_states.get(bssid) == TRUST_UNVERIFIED:
                    self._set_trust_state(bssid, TRUST_MONITORING)
                self.check_threat(bssid)
                self.checked_aps.add(bssid)
            elif packet_count % 50 == 0:
                self.check_threat(bssid)
    
    def check_threat(self, bssid):
        """
        Zero-Trust threat evaluation pipeline 
          1. Low-confidence baseline  — dampen scoring during warm-up
          2. Generate & compare anomaly score
          3. Incremental trust score   — slow accumulation / fast drop
          4. Anomaly confirmation      — require consecutive hits before alerting
        """
        
        features = self.extractor.extract_features(bssid, window_seconds=120)
        # Extract features
        if not features:
            return
        
        ssid = features.get('ssid', 'Unknown')

        # ── Step 1: Low-Confidence Baseline ───────────────────────────────────
        eval_count = self.ap_eval_counts.get(bssid, 0) + 1
        self.ap_eval_counts[bssid] = eval_count

        confidence_factor = min(1.0, eval_count / self.WARMUP_EVALS)  # 0.33 → 0.67 → 1.0
        warmup_bonus = self.WARMUP_LENIENCY * (1.0 - confidence_factor)  # 0.25 → 0.08 → 0
        effective_threshold = self.alert_threshold + warmup_bonus

        if eval_count <= self.WARMUP_EVALS:
            print(f"[ZT] {bssid} warm-up eval {eval_count}/{self.WARMUP_EVALS} "
                  f"(confidence={confidence_factor:.0%}, eff.threshold={effective_threshold:.3f})")

        # ── Step 2: Generate & evaluate anomaly score ────────────────────────
        is_threat = False
        threat_level = 'NONE'

        for col in LOG_FEATURES:
            if col in features:
                features[col] = np.log1p(max(features[col], 0))

        feature_vector = [features.get(fname, 0) for fname in self.feature_names]
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        score = -self.model.decision_function(X_scaled)[0]

        is_suspicious = False
        reasons = []

        # Use effective_threshold (warm-up leniency applied)
        if score > effective_threshold:
            is_suspicious = True
            reasons.append(f"High threat score ({score:.3f} > {effective_threshold:.3f})"
                           + (f" [warm-up {eval_count}/{self.WARMUP_EVALS}]" if eval_count <= self.WARMUP_EVALS else ""))

        if prediction == -1 and score > effective_threshold:
            reasons.append(f"Anomalous behavior detected by model (score: {score:.3f})")

        if is_suspicious and score > (self.alert_threshold + 0.05):
            is_threat = True
            threat_level = 'HIGH'
            reasons.insert(0, f"ML-based anomaly detected (score: {score:.3f})")
            
        # Software MAC check
        current_has_software_mac = features.get('locally_administered_mac', 0) == 1
        if current_has_software_mac:
            is_suspicious = True
            reasons.append("Software MAC detected (typical of hotspots/software APs)")
            
        # Signal Stability check (supplementary reason)
        stability = features.get('signal_stability', 1.0)
        if stability < 0.5:
            reasons.append(f"Highly unstable signal (stability: {stability:.2f})")
            
        # 2. Check for SSID collision (The "Evil Twin" condition)
        same_ssid_bssids = list(self.ssid_bssid_map[ssid])
        ssid_bssid_count = len(same_ssid_bssids)
        
        # Perfect Clone Detection (Same BSSID Spoofing)
        # Using Sequence Number analysis and signal stability
        seq_ooo = features.get('seq_out_of_order_rate', 0)
        seq_vol = features.get('seq_volatility', 0)
        
        #perfect clone
        is_perfect_clone = (seq_ooo > 0.15 or seq_vol > 50) or (score > (self.alert_threshold + 0.3) and stability < 0.4)
        #anti horspot
        if ssid_bssid_count <= 1 and current_has_software_mac:
            is_perfect_clone = False

                   
        if is_perfect_clone and ssid_bssid_count <= 1:
            is_threat = True
            threat_level = 'HIGH'
            reasons.insert(0, f"CRITICAL: 'Perfect Clone' (BSSID Spoofing) detected on SSID '{ssid}'")
            if seq_ooo > 0.15:
                reasons.append(f"Sequence number anomaly: {seq_ooo:.1%} out-of-order frames detected (Hardware mismatch)")
            if stability < 0.4:
                reasons.append(f"Severe signal instability: {stability:.2f} (Likely multiple transmitters on BSSID {bssid})")
            
            self._set_trust_state(bssid, TRUST_THREAT)
            self.alert(bssid, ssid, threat_level, reasons, score, features)
            return
            
        # 3. Analyze the group of APs sharing this SSID
        
        
        bssid_analysis = []
        for other_bssid in same_ssid_bssids:
            other_buffer = self.extractor.ap_observations.get(other_bssid)

            # Check buffer exists and has enough data
            if not other_buffer or len(other_buffer) < 3:
                continue

            # Get latest observation (most recent packet)
            latest_obs = other_buffer.buffer[-1]

            other_info = {
                'bssid': other_bssid,
                'vendor': self.extractor.bssid_info[other_bssid].get('vendor', 'Unknown'),
                'locally_admin': latest_obs.get('locally_administered_mac', 0),
                'first_seen': self.extractor.bssid_info[other_bssid]['first_seen'],
                'is_current': other_bssid == bssid
            }

            bssid_analysis.append(other_info)
            
        if len(bssid_analysis) < 2:
            # Not enough data on peers yet — stay MONITORING
            return
            
        # Check for presence of an authentic (hardware) peer
        other_bssids = [b for b in bssid_analysis if not b['is_current']]
        has_authentic_peer = any(b['locally_admin'] == 0 for b in other_bssids)
        
        # MAIN RULE: Current AP is suspicious AND an authentic peer exists
        if is_suspicious and has_authentic_peer:
            is_threat = True
            threat_level = 'HIGH'
            reasons.insert(0, f"Evil Twin detected: Suspicious AP shadowing an authentic hardware AP ('{ssid}')")
                
        # CASE 3: Vendor Mismatch among hardware MACs (Secondary check)
        elif not current_has_software_mac:
            unique_vendors = set(b['vendor'] for b in bssid_analysis)
            if len(unique_vendors) > 1 and 'Unknown' in unique_vendors:
                current_vendor = features.get('vendor', 'Unknown')
                if current_vendor == 'Unknown':
                    is_threat = True
                    threat_level = 'HIGH'
                    reasons.append(f"Unknown AP vendor among known providers for SSID '{ssid}'")
                
        # CASE 4: Age difference check (only if already somewhat suspicious)
        # if not is_threat and is_suspicious and len(bssid_analysis) > 1:
        #     sorted_by_age = sorted(bssid_analysis, key=lambda x: x['first_seen'])
        #     if sorted_by_age[-1]['bssid'] == bssid:
        #         age_diff = (sorted_by_age[-1]['first_seen'] - sorted_by_age[0]['first_seen']).total_seconds()
        #         if age_diff > 45:
        #             # Only flag if there's at least one other AP that appeared much earlier
        #             is_threat = True
        #             threat_level = 'MEDIUM'
        #             reasons.append(f"AP appeared significantly later ({age_diff:.0f}s) than others with same SSID")

        # FALSE POSITIVE MITIGATION (Enterprise WiFi)
        if is_threat and not current_has_software_mac:
            unique_vendors = set(b['vendor'] for b in bssid_analysis)
            if len(unique_vendors) == 1 and list(unique_vendors)[0] in ['Cisco', 'Aruba', 'Ubiquiti', 'Ruckus']:
                is_threat = False

        # ── Step 3: Incremental Trust Score ────────────────────────────────
        current_score = self.ap_trust_scores.get(bssid, 0.0)
        if is_threat:
            new_score = -1.0                              # confirmed threat: immediate floor
        elif is_suspicious:
            new_score = max(-1.0, current_score - 0.10)  # slow erosion
        else:
            new_score = min(1.0, current_score + 0.05)   # slow accumulation on clean evals
        self.ap_trust_scores[bssid] = round(new_score, 3)

        # ── Step 4: Anomaly Confirmation ────────────────────────────────────
        # Require CONFIRMATION_REQUIRED consecutive anomalous evals before alerting.
        # This suppresses one-off false positives ("Log & Suppress alert" in flowchart).
        if is_threat:
            consec = self.consecutive_anomalies.get(bssid, 0) + 1
            self.consecutive_anomalies[bssid] = consec

            if consec >= self.CONFIRMATION_REQUIRED:
                self._set_trust_state(bssid, TRUST_THREAT)
                self.alert(bssid, ssid, threat_level, reasons, score, features)
            else:
                self._set_trust_state(bssid, TRUST_SUSPICIOUS)
                print(f"[ZT] {bssid} anomaly #{consec} — awaiting confirmation "
                      f"({self.CONFIRMATION_REQUIRED - consec} more eval(s) needed) "
                      f"[trust_score={new_score:.2f}]")
        elif is_suspicious:
            consec = self.consecutive_anomalies.get(bssid, 0) + 1
            self.consecutive_anomalies[bssid] = consec
            self._set_trust_state(bssid, TRUST_SUSPICIOUS)
        else:
            # Clean eval — reset consecutive counter so next anomaly starts fresh
            self.consecutive_anomalies[bssid] = 0
            self._set_trust_state(bssid, TRUST_MONITORING)
        
    
    def alert(self, bssid, ssid, level, reasons, score, features):
        """Send alert"""
        
        # Rate limiting (max 1 alert per AP per minute)
        now = datetime.now()
        if bssid in self.last_alert_time:
            if (now - self.last_alert_time[bssid]).seconds < 60:
                return
        
        self.last_alert_time[bssid] = now
        
        # Create alert
        alert = {
            'timestamp': now,
            'bssid': bssid,
            'ssid': ssid,
            'level': level,
            'score': score,
            'reasons': reasons,
            'features': features
        }
        self.alerts.append(alert)
        
        # Display alert
        emoji = {'HIGH': '🚨', 'MEDIUM': '⚠️', 'LOW': '🟡'}
        
        print()
        print("="*70)
        print(f"{emoji.get(level, '⚠️')} THREAT DETECTED - {level} PRIORITY")
        print("="*70)
        print(f"Time:   {now.strftime('%H:%M:%S')}")
        print(f"SSID:   {ssid}")
        print(f"BSSID:  {bssid}")
        print(f"Vendor: {features.get('vendor', 'Unknown')}")
        print(f"Score:  {score:.3f}")
        print()
        print("Reasons:")
        for i, reason in enumerate(reasons, 1):
            print(f"  {i}. {reason}")
        print()
        print("Details:")
        print(f"  Signal Stability: {features.get('signal_stability', 0):.2f}")
        print(f"  RSSI Std Dev:     {features.get('rssi_std', 0):.1f} dBm")
        print(f"  Seq OOO Rate:     {features.get('seq_out_of_order_rate', 0):.1%}")
        print(f"  Software MAC:     {'Yes ⚠️' if features.get('locally_administered_mac') else 'No'}")
        
        if level == 'HIGH':
            print()
            print("⚠️  RECOMMENDED ACTION:")
            print("  → DO NOT CONNECT to this network")
            print("  → Investigate immediately")
            print("  → Possible evil twin attack")
        
        print("="*70)
        print()
        
        # Log to file
        self._log_alert(alert)
        if self.IS_NOTIF_ON:
            self.notifier.send_alert(alert)
            
        # Send to Dashboard
        self._report_to_dashboard(alert)
    
    def _report_network_to_dashboard(self, bssid, packet_features):
        """Send newly-discovered AP info to dashboard, including its trust state."""
        try:
            trust = self.trust_states.get(bssid, TRUST_UNVERIFIED)
            payload = {
                'ssid': packet_features.get('ssid', 'Unknown') or 'Unknown',
                'mac': bssid,
                'status': trust,           # e.g. UNVERIFIED, MONITORING, TRUSTED …
                'signal': int(packet_features.get('rssi', -100)) if packet_features.get('rssi') else -100,
                'channel': int(packet_features.get('channel', 1)),
                'vendor': packet_features.get('vendor', 'Unknown'),
                'trust_state': trust,
            }
            requests.post(f"{self.dashboard_base_url}/networks", json=payload, timeout=1)
        except:
            pass

    def _report_trust_state_to_dashboard(self, bssid, state):
        """Push a trust-state change to the dashboard for a specific BSSID."""
        try:
            payload = {'mac': bssid, 'trust_state': state, 'status': state}
            requests.patch(f"{self.dashboard_base_url}/networks/{bssid}", json=payload, timeout=1)
        except:
            pass

    def _report_to_dashboard(self, alert):
        """Send threat to dashboard API"""
        try:
            payload = {
                'ssid': alert['ssid'],
                'mac': alert['bssid'],
                'legitimateMac': 'Unknown',  # Default if not available
                'signal': int(alert['features'].get('rssi_mean', 0)),
                'channel': int(alert['features'].get('channel', 1)),
                'encryption': alert['features'].get('encryption', 'Open'),
                'severity': alert['level'].capitalize(),
                'clientCount': 0  # Default or extract from features if available
            }
            
            requests.post(f"{self.dashboard_base_url}/threats", json=payload, timeout=2)
        except Exception as e:
            pass
    
 
    def _clear_previous_alerts(self):
        """Reset alerts.json file on startup."""
        log_file = "data/alerts.json"
        try:
            os.makedirs('data', exist_ok=True)
            with open(log_file, 'w') as f:
                json.dump([], f)
            print("  ✓ History cleared: alerts.json")
        except Exception as e:
            print(f"  [!] Failed to clear history: {e}")

    def _encode_encryption(self, enc_type):
        """Encode encryption as number"""
        mapping = {
            'Open': 0,
            'WEP': 1,
            'WPA': 2,
            'WPA2': 3,
            'WPA3': 3,
        }
        return mapping.get(enc_type, 0)
    
    def _log_alert(self, alert):
        """Log alert to file"""
        alert_copy = alert.copy()
        alert_copy['timestamp'] = alert['timestamp'].isoformat()
        
        # Exclude features from the JSON output as requested
        if 'features' in alert_copy:
            del alert_copy['features']
            
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        log_file = "data/alerts.json"
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(alert_copy)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def print_status(self):
        """Print current status including zero-trust state counts."""
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        unverified = sum(1 for s in self.trust_states.values() if s in (TRUST_UNVERIFIED, TRUST_MONITORING))
        suspicious = sum(1 for s in self.trust_states.values() if s == TRUST_SUSPICIOUS)
        threats    = sum(1 for s in self.trust_states.values() if s == TRUST_THREAT)
        
        print(f"\r[Status] Pkts: {self.total_packets} | "
              f"APs: {len(self.extractor.ap_observations)} | "
              f"Unverified: {unverified} | Suspicious: {suspicious} | Threats: {threats} | "
              f"Alerts: {len(self.alerts)} | Time: {elapsed:.0f}s",
              end='', flush=True)
    
    def print_summary(self):
        """Print session summary including zero-trust state breakdown."""
        
        print("\n")
        print("="*70)
        print("SESSION SUMMARY")
        print("="*70)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"Duration:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Packets:        {self.total_packets}")
        print(f"APs observed:   {len(self.extractor.ap_observations)}")
        print(f"Threats found:  {len(self.alerts)}")
        print()
        
        # Zero-Trust breakdown
        state_counts = defaultdict(int)
        for s in self.trust_states.values():
            state_counts[s] += 1
        
        print("AP Breakdown:")
        print(f"MONITORING:  {state_counts[TRUST_MONITORING]}")
        print(f"UNVERIFIED:  {state_counts[TRUST_UNVERIFIED]}")
        print(f"SUSPICIOUS:  {state_counts[TRUST_SUSPICIOUS]}")
        print(f"THREAT:      {state_counts[TRUST_THREAT]}")
        print()
        
        if self.alerts:
            print("Threat Summary:")
            high   = sum(1 for a in self.alerts if a['level'] == 'HIGH')
            medium = sum(1 for a in self.alerts if a['level'] == 'MEDIUM')
            low    = sum(1 for a in self.alerts if a['level'] == 'LOW')
            
            print(f"  HIGH:   {high}")
            print(f"  MEDIUM: {medium}")
            print(f"  LOW:    {low}")
            print()
            
            print("Recent Threats:")
            for a in self.alerts[-5:]:
                time_str = a['timestamp'].strftime('%H:%M:%S')
                print(f"  [{time_str}] {a['level']:6s} - {a['ssid']}")
        
        print("="*70)
    
    def start(self, interface='wlan0mon', duration=None,
          channels=None, dwell_time=1.0):

        print(f"Interface: {interface}")

        if channels:
            print(f"Channel hopping enabled: {channels}")
            hopper = ChannelHopper(interface, channels, dwell_time)
            hopper.start()
        else:
            hopper = None

        print()
        print("Starting detection...")
        print()

        try:
            packet_count = [0]

            def packet_handler(pkt):
                # Optional: tag packet with current channel
                if hopper:
                    pkt.current_channel = hopper.get_current_channel()

                self.observe_packet(pkt)

                packet_count[0] += 1
                if packet_count[0] % 100 == 0:
                    self.print_status()

            if duration:
                sniff(iface=interface,
                    prn=packet_handler,
                    timeout=duration,
                    store=False)
            else:
                sniff(iface=interface,
                    prn=packet_handler,
                    store=False)

        except KeyboardInterrupt:
            print("\nStopping detection...")

        finally:
            if hopper:
                hopper.stop()

        self.print_summary()