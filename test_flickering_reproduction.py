#!/usr/bin/env python3

"""
Test RTI Flickering Reproduction
Send threat messages and monitor for 1ms flickering when HUD disabled
"""

import time
import threading
from cereal import messaging

class RTIFlickerTest:
    def __init__(self):
        self.monitoring = False
        self.threat_detected_frames = []

    def send_threats(self):
        """Send continuous threat messages"""
        print("Sending continuous hazard threats...")
        pm = messaging.PubMaster(['rtiStateSP'])

        message_count = 0
        while self.monitoring:
            message_count += 1

            # Create hazard threat message
            msg = messaging.new_message('rtiStateSP', valid=True)
            msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
            msg.rtiStateSP.threatAhead = True
            msg.rtiStateSP.threatDistanceM = 500.0
            msg.rtiStateSP.recommendedSpeed = 0.0
            msg.rtiStateSP.source = f'flicker_test_{message_count}'
            msg.rtiStateSP.apiStatus = 'connected'

            # Create hazard threat
            msg.rtiStateSP.init('threats', 1)
            threat_msg = msg.rtiStateSP.threats[0]
            threat_msg.id = f'hazard_{message_count}'
            threat_msg.type = 'hazard'
            threat_msg.latitude = 37.4231
            threat_msg.longitude = -122.0841
            threat_msg.distance = 500.0
            threat_msg.direction = 'ahead'
            threat_msg.confidence = 0.95
            threat_msg.speedLimitMs = 15.0

            pm.send('rtiStateSP', msg)
            time.sleep(0.05)  # 20Hz

    def monitor_threats(self):
        """Monitor for threat visibility"""
        print("Monitoring threat visibility...")
        sm = messaging.SubMaster(['rtiStateSP'])

        frame_count = 0
        while self.monitoring:
            sm.update(50)  # 50ms timeout
            frame_count += 1

            if sm.updated['rtiStateSP']:
                rti = sm['rtiStateSP']
                if len(rti.threats) > 0:
                    self.threat_detected_frames.append(frame_count)
                    timestamp = time.strftime('%H:%M:%S.%f')[:-3]
                    print(f"[{timestamp}] Frame {frame_count}: THREAT VISIBLE - {rti.threats[0].type}")

            time.sleep(0.05)  # 20Hz monitoring

    def run_test(self, duration=30):
        """Run flicker reproduction test"""
        print("=== RTI Flicker Reproduction Test ===")
        print("Parameters: RTI=ON, HUD=OFF, Audio=OFF")
        print(f"Running for {duration} seconds...")

        self.monitoring = True

        # Start threads
        sender_thread = threading.Thread(target=self.send_threats)
        monitor_thread = threading.Thread(target=self.monitor_threats)

        sender_thread.start()
        monitor_thread.start()

        # Run test
        time.sleep(duration)

        # Stop threads
        self.monitoring = False
        sender_thread.join()
        monitor_thread.join()

        # Analyze results
        print("\n=== Results ===")
        print(f"Total frames monitored: ~{duration * 20}")
        print(f"Frames with threats detected: {len(self.threat_detected_frames)}")

        if self.threat_detected_frames:
            print("Threat detection pattern:")
            for i, frame in enumerate(self.threat_detected_frames[:10]):  # Show first 10
                print(f"  Detection {i+1}: Frame {frame}")

            # Look for 1-second intervals (20 frames at 20Hz)
            intervals = []
            for i in range(1, len(self.threat_detected_frames)):
                interval = self.threat_detected_frames[i] - self.threat_detected_frames[i-1]
                intervals.append(interval)

            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                print(f"Average interval between detections: {avg_interval:.1f} frames")
                if 18 <= avg_interval <= 22:  # Around 20 frames = 1 second
                    print("PATTERN DETECTED: ~1 second intervals = flickering bug reproduced!")
        else:
            print("No threats detected - either bug is fixed or not reproduced")

if __name__ == "__main__":
    test = RTIFlickerTest()
    test.run_test(30)
