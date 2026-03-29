"""
	Hop between multiple channels rather than monitoring from single channel.
	Channel awareness is maintained (if this works :D )

	If it works we could add:
		- Hopping for 5 GHz channels
		- Random order for Hopping ( Stealth 100 )

"""



import time
import subprocess
import threading

class ChannelHopper:
	def __init__(self, interface, channels, dwell_time=1.0):
		"""
		Args:
			interface: monitor-mode interface (e.g. wlan0mon)
			channels: list of channels or comma-separated string (e.g. [1, 6, 11] or "1,6,11")
			dwell_time: seconds to stay on each channel
		"""
		
		self.interface = interface
		
		# Handle string input (e.g. from command line "1,6,11")
		if isinstance(channels, str):
			try:
				self.channels = [int(c.strip()) for c in channels.split(',') if c.strip()]
			except ValueError:
				print(f"[!] Invalid channel list format: {channels}")
				self.channels = [1, 6, 11] # Default fallback
		else:
			self.channels = [int(c) for c in channels]
			
		self.dwell_time = dwell_time
		self.current_channel = None
		
		self._stop_event = threading.Event()
		self._thread = None
		
	def _set_channel(self, channel):
		# Try 'iw' first (modern)
		res = subprocess.run(
			["iw", "dev", self.interface, "set", "channel", str(channel)],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.PIPE,
			text=True
		)
		
		# If iw fails, try 'iwconfig' (legacy/different drivers)
		if res.returncode != 0:
			# print(f"[!] 'iw' failed for ch {channel}: {res.stderr.strip()}")
			res2 = subprocess.run(
				["iwconfig", self.interface, "channel", str(channel)],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.PIPE,
				text=True
			)
			if res2.returncode != 0:
				# print(f"[!] 'iwconfig' also failed: {res2.stderr.strip()}")
				pass
		
		self.current_channel = channel
		print(f"[*] Now listening on channel {channel}...")
		
	def _hop_loop(self):
		if not self.channels:
			print("[!] No channels to hop. Thread exiting.")
			return
			
		while not self._stop_event.is_set():
			for ch in self.channels:
				if self._stop_event.is_set():
					break
				self._set_channel(ch)
				time.sleep(self.dwell_time)
	
	def start(self):
		"""Start channel hopping in a separate thread"""
		if self._thread and self._thread.is_alive():
			return
		
		# Ensure interface is UP (often required for 'iw')
		subprocess.run(["ip", "link", "set", self.interface, "up"], 
					 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._hop_loop, daemon=True)
		self._thread.start()
		print(f"[*] Channel hopping started on {self.interface} for channels: {self.channels}")
		
	def stop(self):
		"""Stop hopping and join thread"""
		self._stop_event.set()
		if self._thread:
			self._thread.join()
			
	def get_current_channel(self):
		"""Return the current channel for packet tagging"""
		return self.current_channel	
