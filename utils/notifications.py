import requests
import os
import logging
from dotenv import load_dotenv
import json

class TelegramNotifier:
    """
    Sends alerts to a Telegram bot
    """
    def __init__(self, token=None, chat_id=None):
        load_dotenv()
        self.token = os.getenv('TG_BOT_TOKEN')
        with open("data/config.json", "r") as r:
            config = json.load(r)
        self.chat_id = config['TG_CHAT_ID']
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            print("[Warning] Telegram notifications disabled (Token or Chat ID missing)")
        else:
            print("[+]Bot Active")

    def send_alert(self, alert_data):
        """
        Format and send alert data to Telegram
        """
        if not self.enabled:
            return False
        emoji = {'HIGH': '🚨', 'MEDIUM': '⚠️', 'LOW': '🟡'}
        level = alert_data.get('level', 'MEDIUM')
        
        message = (
            f"{emoji.get(level, '⚠️')} *THREAT DETECTED*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"*Priority:* {level}\n"
            f"*SSID:* `{alert_data.get('ssid', 'Unknown')}`\n"
            f"*BSSID:* `{alert_data.get('bssid', 'Unknown')}`\n"
            f"*Vendor:* {alert_data.get('features', {}).get('vendor', 'Unknown')}\n"
            f"*Score:* `{alert_data.get('score', 0):.3f}`\n\n"
            f"*Reasons:*\n"
        )
        
        for i, reason in enumerate(alert_data.get('reasons', []), 1):
            message += f"• {reason}\n"
            
        message += f"\n*Time:* {alert_data.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')}"

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            import threading
            thread = threading.Thread(target=self._send_request_with_retry, args=(url, payload))
            thread.daemon = True
            thread.start()
            return True
        except Exception as e:
            print(f"   Notification thread error: {e}")
            return False

    def _send_request_with_retry(self, url, payload, max_retries=5, initial_delay=5):
        """
        Background worker that attempts to send the notification with exponential backoff.
        Useful when the system temporarily loses internet (e.g., during monitor mode switching).
        """
        import time
        import requests

        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=15)
                print("[INFO]Telegram Notif Sent")
                response.raise_for_status()
                # Success!
                return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # DNS failure or timeout - likely no internet
                if attempt == max_retries:
                    print(f"  ❌ Failed to send Telegram notification after {max_retries} attempts: {e}")
                else:
                    # Exponential backoff: 5s, 10s, 20s, 40s...
                    time.sleep(delay)
                    delay *= 2
            except Exception as e:
                print(f"  ❌ Permanent Telegram error: {e}")
                break
        return False
