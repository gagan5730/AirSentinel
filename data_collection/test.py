import time
import os
import json
from datetime import datetime
from scapy.all import sniff
from capture import extract_ap_features
from extract import FeatureExtractor
from channel_hopper import ChannelHopper
from json_output import create_output_data, print_save_summary

interface = "wlan1"

WINDOWS = [30, 120, 300]
EXTRACT_INTERVAL = 20
MIN_PACKETS = 5

DATASET_FILE = "../data/ni.json"


if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w") as f:
        json.dump(
            {
                "metadata": {
                    "interface": interface,
                    "start_time": datetime.now().isoformat(),
                    "windows": WINDOWS,
                    "extract_interval": EXTRACT_INTERVAL
                },
                "extraction_rounds": []
            },
            f,
            indent=2
        )


hopper = ChannelHopper(
    interface,
    channels=[1, 6, 11],
    dwell_time=1.0
)
hopper.start()

extractor = FeatureExtractor(use_packet_time=False)

last_extract = time.time()
extraction_round = 0

print(f"[*] Starting packet capture on {interface}")
print(f"[*] Overlapping windows: {WINDOWS}")
print(f"[*] Extraction interval: {EXTRACT_INTERVAL}s")

def packet_handler(packet):
    global last_extract, extraction_round

    features = extract_ap_features(packet)
    if not features:
        return

    extractor.observe_packet(features)

    now = time.time()
    if now - last_extract < EXTRACT_INTERVAL:
        return

    extraction_round += 1

    feature_vectors = []
    bssids = []

    for bssid in extractor.ap_observations.keys():
        buffer_len = len(extractor.ap_observations[bssid])
        if buffer_len < MIN_PACKETS:
            continue

        for w in WINDOWS:
            vec = extractor.extract_features(bssid, window_seconds=w)
            if vec:
                vec["window_seconds"] = w
                vec["extraction_round"] = extraction_round
                feature_vectors.append(vec)
                bssids.append(bssid)

    if not feature_vectors:
        last_extract = now
        return


    output_data = create_output_data(
        bssids=bssids,
        feature_vectors=feature_vectors,
        bssid_info=extractor.bssid_info
    )

	# Writes into the same file, gotta implement jsonl later on
    with open(DATASET_FILE, "r+") as f:
        dataset = json.load(f)

        dataset["extraction_rounds"].append({
            "round": extraction_round,
            "timestamp": datetime.now().isoformat(),
            "access_points": output_data["access_points"]
        })

        f.seek(0)
        json.dump(dataset, f, indent=2)
        f.truncate()

    print_save_summary(output_data, DATASET_FILE)

    last_extract = now


try:
    sniff(iface=interface, prn=packet_handler)
except KeyboardInterrupt:
    print("\n[!] Stopping capture...")
finally:
    hopper.stop()
    print("[+] Channel hopper stopped")
