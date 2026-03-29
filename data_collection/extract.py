from scapy.sendrecv import sniff
import numpy as np
import pandas as pd
from .capture import extract_ap_features
from collections import defaultdict, deque
from datetime import datetime, timedelta
from scapy.all import rdpcap
from .json_output import create_output_data, save_to_json, print_save_summary

class RollingBuffer:
	def __init__(self, max_age_seconds):
		self.max_age = timedelta(seconds=(max_age_seconds-1))
		self.buffer = deque()
		
	def append(self, item):
		# item should have 'timestamp' field
		
		self.buffer.append(item)
		self._expire_old(item['timestamp'])
		
	def _expire_old(self, current_time):
		cutoff = current_time - self.max_age
		while self.buffer and self.buffer[0]['timestamp'] < cutoff:
			self.buffer.popleft()
			
	def get_window(self, window_seconds=None):
		"""
		Returns items within last window seconds, if it is None, returns the entire buffer
		"""
		
		if window_seconds is None:
			return list(self.buffer)
			
		cutoff = self.buffer[-1]['timestamp'] - timedelta(seconds=window_seconds)
		return [item for item in self.buffer if item['timestamp'] >= cutoff]
		
	def __len__(self):
		return len(self.buffer)
		
	def empty(self):
		return len(self.buffer) == 0

class FeatureExtractor:
    def __init__(self, use_packet_time=False):
        """
        Args:
            use_packet_time: If True, use packet timestamps (for PCAP files)
                           If False, use system time (for live capture)
        """
        self.ap_observations = defaultdict(
        	lambda: RollingBuffer(max_age_seconds=300)  #  Rolling Window size 
        )
        
        self.ssid_bssid_map = defaultdict(set)
        self.bssid_info = defaultdict(dict)
        self.use_packet_time = use_packet_time
        
        # For PCAP mode: track capture time range
        self.first_packet_time = None
        self.last_packet_time = None
    
    def observe_packet(self, packet_features, packet_timestamp=None):
        """
        Args:
            packet_features: dict with AP features
            packet_timestamp: float (unix timestamp from packet.time) - only for PCAP mode
        """
        bssid = packet_features['bssid']
        ssid = packet_features['ssid']

        # Determine timestamp to use
        if self.use_packet_time and packet_timestamp is not None:
            timestamp = datetime.fromtimestamp(packet_timestamp)
            
            # Track capture time range
            if self.first_packet_time is None:
                self.first_packet_time = timestamp
            self.last_packet_time = timestamp
        else:
            timestamp = packet_features['timestamp']

        observation = {
            'timestamp': timestamp,  # FIXED: Use correct timestamp
            'rssi': packet_features['rssi'],
            'channel': packet_features['channel'],
            'beacon_interval': packet_features['beacon_interval'],
            'encryption': packet_features['encryption_type'],
            'vendor': packet_features['vendor'],
            'sequence_number': packet_features.get('sequence_number', 0),
            'capability_raw': packet_features.get('capability_raw', 0),
            'supported_rates': packet_features.get('supported_rates', []),
            'max_rate': packet_features.get('max_rate', 0),
            'ie_order': packet_features.get('ie_order', ()),
            'ie_count': packet_features.get('ie_count', 0),
            'ht_capable': packet_features.get('ht_capable', False),
            'vht_capable': packet_features.get('vht_capable', False),
            'locally_administered_mac': packet_features.get('locally_administered_mac', False),
        }
        
        self.ap_observations[bssid].append(observation)
        self.ssid_bssid_map[ssid].add(bssid)

        if bssid not in self.bssid_info:
            self.bssid_info[bssid] = {
                'ssid': ssid,
                'first_seen': timestamp,  # FIXED: Use correct timestamp
                'last_seen': timestamp,
                'vendor': packet_features['vendor'],
                'mac_oui': packet_features.get('mac_oui', ''),
                'initial_encryption': packet_features['encryption_type'],
                'initial_channel': packet_features['channel'],
                'initial_capabilities': packet_features.get('capability_raw', 0),
                'initial_ie_order': packet_features.get('ie_order', ()),
                'disappearance_count': 0,
                'last_disappearance': None,
            }
        else:
            prev_last_seen = self.bssid_info[bssid]['last_seen']
            current_time = timestamp
            gap = (current_time - prev_last_seen).total_seconds()
            if gap > 60:
                self.bssid_info[bssid]['disappearance_count'] += 1
                self.bssid_info[bssid]['last_disappearance'] = prev_last_seen
            self.bssid_info[bssid]['last_seen'] = current_time
            self.bssid_info[bssid]['vendor'] = packet_features['vendor']
    
    def extract_features(self, bssid, window_seconds=None, reference_time=None):
        """
        Args:
            bssid: MAC address
            window_seconds: Time window for recent observations
                          If None, uses all observations
            reference_time: Time to calculate time_since_first_seen from
                          If None, uses appropriate default based on mode
        """
        buffer = self.ap_observations[bssid]
        if buffer.empty():
            return None
        
        recent_obs = buffer.get_window(window_seconds)
        
        if not recent_obs or len(recent_obs) < 5:
            return None
        
        # Determine reference time for time_since_first_seen
        if reference_time is None:
            if self.use_packet_time:
                # PCAP mode: use last packet time
                reference_time = self.last_packet_time or recent_obs[-1]['timestamp']
            else:
                # Live mode: use current system time
                reference_time = datetime.now()
        
        rssi_values = [obs['rssi'] for obs in recent_obs if obs['rssi']]
        beacon_intervals = [obs['beacon_interval'] for obs in recent_obs]
        channels = [obs['channel'] for obs in recent_obs if obs['channel']]
        
        features = {}

        # ===== SIGNAL FEATURES =====
        if rssi_values:
            features['rssi_mean'] = np.mean(rssi_values)
            features['rssi_std'] = np.std(rssi_values)
            features['rssi_min'] = np.min(rssi_values)
            features['rssi_max'] = np.max(rssi_values)
            features['rssi_range'] = features['rssi_max'] - features['rssi_min']
            rssi_changes = np.diff(rssi_values)
            features['rssi_sudden_change_max'] = np.max(np.abs(rssi_changes)) if len(rssi_changes) > 0 else 0
            features['rssi_sudden_change_mean'] = np.mean(np.abs(rssi_changes)) if len(rssi_changes) > 0 else 0
            features['rssi_too_strong'] = int(features['rssi_mean'] > -30)
            features['signal_stability'] = 1 - (features['rssi_std'] / max(abs(features['rssi_mean']), 1))
        else:
            features['rssi_mean'] = -100
            features['rssi_std'] = 0
            features['rssi_min'] = -100
            features['rssi_max'] = -100
            features['rssi_range'] = 0
            features['rssi_sudden_change_max'] = 0
            features['rssi_sudden_change_mean'] = 0
            features['rssi_too_strong'] = 0
            features['signal_stability'] = 0
        
        # ===== TEMPORAL FEATURES =====
        features['beacon_interval_mean'] = np.mean(beacon_intervals)
        features['beacon_interval_std'] = np.std(beacon_intervals)
        
        # FIXED: observation_duration based on actual packet timestamps
        features['observation_duration'] = (
            recent_obs[-1]['timestamp'] - recent_obs[0]['timestamp']
        ).total_seconds()
        
        features['packets_per_second'] = len(recent_obs) / max(features['observation_duration'], 1)
        
        # FIXED: time_since_first_seen using correct reference time
        first_seen = self.bssid_info[bssid]['first_seen']
        features['time_since_first_seen'] = (
            reference_time - first_seen
        ).total_seconds()
        
        timestamps = [obs['timestamp'] for obs in recent_obs]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() * 1000 
                      for i in range(len(timestamps)-1)]
        features['beacon_timing_jitter'] = np.std(time_diffs) if len(time_diffs) > 0 else 0
        features['beacon_timing_irregularity'] = np.max(np.abs(np.diff(time_diffs))) if len(time_diffs) > 1 else 0
        
        features['disappearance_count'] = self.bssid_info[bssid]['disappearance_count']
        features['uptime_inconsistency'] = int(features['disappearance_count'] > 0)
        
        # ===== TOPOLOGY FEATURES =====
        ssid = self.bssid_info[bssid]['ssid']
        features['ssid_bssid_count'] = len(self.ssid_bssid_map[ssid])
        features['channel_changes'] = len(set(channels)) - 1
        
        same_ssid_channels = set()
        same_ssid_same_channel_count = 0
        current_channel = channels[-1] if channels else None
        for other_bssid in self.ssid_bssid_map[ssid]:
            if other_bssid != bssid:
                other_buffer = self.ap_observations[other_bssid]
                if not other_buffer.empty():
                    other_channel = other_buffer.buffer[-1]['channel']
                    same_ssid_channels.add(other_channel)
                    if other_channel == current_channel:
                        same_ssid_same_channel_count += 1
        
        features['same_ssid_different_channels'] = len(same_ssid_channels)
        features['simultaneous_same_ssid_same_channel'] = same_ssid_same_channel_count
        
        # ===== SECURITY FEATURES =====
        encryption_map = {'Open': 0, 'WEP': 1, 'WPA': 2, 'WPA2/WPA3': 3}
        encryption_types = [obs['encryption'] for obs in recent_obs]
        current_encryption = encryption_types[-1]
        initial_encryption = self.bssid_info[bssid]['initial_encryption']
        
        features['encryption_numeric'] = encryption_map.get(current_encryption, 0)
        features['encryption_changed'] = int(len(set(encryption_types)) > 1)
        
        current_enc_level = encryption_map.get(current_encryption, 0)
        initial_enc_level = encryption_map.get(initial_encryption, 0)
        features['encryption_downgrade'] = int(current_enc_level < initial_enc_level)
        
        features['unexpected_open_network'] = int(
            current_encryption == 'Open' and features['ssid_bssid_count'] > 1
        )
        
        # ===== VENDOR FEATURES =====
        vendor = self.bssid_info[bssid]['vendor']
        common_vendors = ['Cisco', 'Aruba', 'Ubiquiti', 'Ruckus', 'TP-Link', 'D-Link', 'Netgear']
        features['vendor_is_common'] = int(vendor in common_vendors)
        
        features['locally_administered_mac'] = int(recent_obs[-1].get('locally_administered_mac', False))
        
        same_ssid_vendors = set()
        for other_bssid in self.ssid_bssid_map[ssid]:
            same_ssid_vendors.add(self.bssid_info[other_bssid]['vendor'])
        
        features['vendor_mismatch'] = int(len(same_ssid_vendors) > 1 and vendor not in ['Unknown'])
        
        # ===== PROTOCOL FEATURES =====
        capability_values = [obs['capability_raw'] for obs in recent_obs]
        initial_capability = self.bssid_info[bssid]['initial_capabilities']
        
        features['capability_changed'] = int(len(set(capability_values)) > 1)
        features['capability_differs_from_initial'] = int(
            capability_values[-1] != initial_capability if capability_values else False
        )
        
        seq_numbers = [obs.get('sequence_number') for obs in recent_obs if obs.get('sequence_number') is not None]

        if len(seq_numbers) >= 2:
            MAX_SEQ = 4096

            raw_diffs = np.diff(seq_numbers)
            has_backward = np.any(raw_diffs < 0)

            seq_diffs = []
            for i in range(len(seq_numbers) - 1):
                diff = seq_numbers[i+1] - seq_numbers[i]

                if diff < -2000:
                    diff += MAX_SEQ

                seq_diffs.append(diff)

            seq_diffs = np.array(seq_diffs)

            std_diff = np.std(seq_diffs)

            features['seq_number_irregularity'] = std_diff
            features['seq_number_backwards'] = int(has_backward)
            features['seq_out_of_order_rate'] = float(
                np.sum(seq_diffs < 0) / len(seq_diffs)
            ) if len(seq_diffs) > 0 else 0.0
            features['seq_volatility'] = float(std_diff)

        else:
            features['seq_number_irregularity'] = 0.0
            features['seq_number_backwards'] = 0
            features['seq_out_of_order_rate'] = 0.0
            features['seq_volatility'] = 0.0
        
        ie_orders = [obs['ie_order'] for obs in recent_obs]
        initial_ie_order = self.bssid_info[bssid]['initial_ie_order']
        
        features['ie_order_changed'] = int(len(set(ie_orders)) > 1)
        features['ie_order_differs_from_initial'] = int(
            ie_orders[-1] != initial_ie_order if ie_orders and initial_ie_order else False
        )
        
        ie_counts = [obs['ie_count'] for obs in recent_obs]
        features['ie_count_mean'] = np.mean(ie_counts)
        features['ie_count_variance'] = np.var(ie_counts)
        
        max_rates = [obs['max_rate'] for obs in recent_obs if obs['max_rate'] > 0]
        if max_rates:
            features['max_data_rate'] = np.max(max_rates)
            features['max_rate_changed'] = int(len(set(max_rates)) > 1)
        else:
            features['max_data_rate'] = 0
            features['max_rate_changed'] = 0
        
        ht_capable = [obs['ht_capable'] for obs in recent_obs]
        vht_capable = [obs['vht_capable'] for obs in recent_obs]
        features['ht_capable'] = int(any(ht_capable))
        features['vht_capable'] = int(any(vht_capable))
        features['ht_capability_changed'] = int(len(set(ht_capable)) > 1)
        features['vht_capability_changed'] = int(len(set(vht_capable)) > 1)
        
        features['beacon_count'] = len(recent_obs)
        features['channel_stability'] = 1 - (features['channel_changes'] / max(len(recent_obs), 1))
        
        # Convert numpy types to standard python types for JSON serialization
        for key, value in features.items():
            if hasattr(value, 'item'):
                features[key] = value.item()
            elif isinstance(value, np.ndarray):
                features[key] = value.tolist()

        return features
    
    def print_summary(self):
        """Print capture summary (useful for PCAP mode)"""
        if self.use_packet_time and self.first_packet_time and self.last_packet_time:
            duration = (self.last_packet_time - self.first_packet_time).total_seconds()
            print(f"\n{'='*70}")
            print(f"PCAP CAPTURE SUMMARY")
            print(f"{'='*70}")
            print(f"First packet: {self.first_packet_time}")
            print(f"Last packet:  {self.last_packet_time}")
            print(f"Duration:     {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"Total APs:    {len(self.ap_observations)}")
            print(f"Total SSIDs:  {len(self.ssid_bssid_map)}")
            print(f"{'='*70}\n")

"""
# =====================================================================
# USAGE EXAMPLES
# =====================================================================

def process_pcap(pcap_file, output_dir="../data"):
    # Process a PCAP file
    print(f"[*] Processing PCAP: {pcap_file}")
    
    # IMPORTANT: Set use_packet_time=True for PCAP files
    extractor = FeatureExtractor(use_packet_time=True)
    
    captured_packets = rdpcap(pcap_file)
    print(f"[*] Total packets: {len(captured_packets)}")
    
    beacon_count = 0
    for packet in captured_packets:
        # Extract packet timestamp
        packet_time = float(packet.time)  # Unix timestamp
        
        packet_features = extract_ap_features(packet)
        if packet_features and 'bssid' in packet_features:
            # Pass packet_time to observe_packet
            extractor.observe_packet(packet_features, packet_timestamp=packet_time)
            beacon_count += 1
            
            if beacon_count % 100 == 0:
                print(f"[*] Processed {beacon_count} beacon frames...", end='\r')
    
    print(f"\n[+] Processed {beacon_count} beacon frames")
    
    # Print capture summary
    extractor.print_summary()
    
    # Extract features for all APs
    feature_vectors = []
    bssids = []
    
    for bssid in extractor.ap_observations.keys():
        # Don't use window_seconds for PCAP - use all data
        features = extractor.extract_features(bssid, window_seconds=None)
        if features:
            feature_vectors.append(features)
            bssids.append(bssid)
    
    # Calculate capture duration
    capture_duration = None
    if extractor.use_packet_time and extractor.first_packet_time and extractor.last_packet_time:
        capture_duration = (extractor.last_packet_time - extractor.first_packet_time).total_seconds()
    
    # Create output data using json_output module
    output_data = create_output_data(
        bssids=bssids,
        feature_vectors=feature_vectors,
        bssid_info=extractor.bssid_info,
        capture_duration=capture_duration
    )
    
    # Save to file
    filepath = save_to_json(output_data, output_dir, mode='pcap')
    
    # Print summary
    print_save_summary(output_data, filepath)
    
    # Show suspicious APs
   # check_suspicious_aps(feature_vectors)
    
    return output_data


def process_live(interface="mon1", count=200, output_dir="../data"):
    # Process live capture
    print(f"[*] Starting live capture on {interface}")
    
    # IMPORTANT: Set use_packet_time=False for live capture
    extractor = FeatureExtractor(use_packet_time=False)
    
    captured_packets = sniff(count=count, iface=interface)
    print(f"[+] Captured {len(captured_packets)} packets")
    
    for packet in captured_packets:
        packet_features = extract_ap_features(packet)
        if packet_features and 'bssid' in packet_features:
            # Don't pass packet_timestamp for live capture
            extractor.observe_packet(packet_features)
    
    # Extract features
    feature_vectors = []
    bssids = []
    
    for bssid in extractor.ap_observations.keys():
        # Use window for live capture
        features = extractor.extract_features(bssid, window_seconds=300)
        if features:
            feature_vectors.append(features)
            bssids.append(bssid)
    
    # Create output data using json_output module
    output_data = create_output_data(
        bssids=bssids,
        feature_vectors=feature_vectors,
        bssid_info=extractor.bssid_info
    )
    
    # Save to file
    filepath = save_to_json(output_data, output_dir, mode='live')
    
    # Print summary
    print_save_summary(output_data, filepath)
    
    return output_data


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # PCAP mode
        pcap_file = sys.argv[1]
        process_pcap(pcap_file)
    else:
        # Live mode
        process_live()
"""
