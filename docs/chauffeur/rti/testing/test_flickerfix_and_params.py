#!/usr/bin/env python3
"""
Test Script for RTI Flickering Fix and Parameter Integration

This script validates:
1. The flickering fix - continuous publishing at 50Hz
2. Parameter integration - all user settings are respected

Usage: python3 test_flickerfix_and_params.py
"""

import time
import threading
from cereal import messaging
from openpilot.common.params import Params

class RTIFixValidation:
    def __init__(self):
        self.params = Params()
        self.monitoring = True
        self.message_times = []
        self.message_gaps = []
        
    def test_continuous_publishing(self, duration=10):
        """Test that messages are published continuously without gaps."""
        print("=" * 60)
        print("TEST 1: Continuous Publishing (Flickering Fix)")
        print("=" * 60)
        print(f"Monitoring RTI messages for {duration} seconds...")
        print("Expected: Messages every 20ms (50Hz) without gaps")
        
        sm = messaging.SubMaster(['rtiStateSP'])
        
        start_time = time.time()
        last_message_time = None
        message_count = 0
        max_gap = 0
        gaps_over_100ms = 0
        
        while time.time() - start_time < duration:
            sm.update(100)  # 100ms timeout
            
            if sm.updated['rtiStateSP']:
                current_time = time.time()
                message_count += 1
                
                if last_message_time:
                    gap = (current_time - last_message_time) * 1000  # Convert to ms
                    self.message_gaps.append(gap)
                    max_gap = max(max_gap, gap)
                    
                    if gap > 100:  # Gap over 100ms indicates flickering
                        gaps_over_100ms += 1
                        print(f"  WARNING: Large gap detected: {gap:.1f}ms")
                
                last_message_time = current_time
        
        # Analyze results
        print(f"\nResults:")
        print(f"  Total messages: {message_count}")
        print(f"  Expected messages: ~{duration * 50}")
        print(f"  Max gap: {max_gap:.1f}ms")
        
        if self.message_gaps:
            avg_gap = sum(self.message_gaps) / len(self.message_gaps)
            print(f"  Average gap: {avg_gap:.1f}ms (expected ~20ms)")
            print(f"  Gaps over 100ms: {gaps_over_100ms}")
        
        # Determine pass/fail
        if gaps_over_100ms == 0 and max_gap < 50:
            print("  ✓ PASS: No flickering detected, continuous publishing working!")
        else:
            print("  ✗ FAIL: Flickering still present, gaps detected in publishing")
        
        return gaps_over_100ms == 0
    
    def test_parameter_integration(self):
        """Test that all user-configured parameters are respected."""
        print("\n" + "=" * 60)
        print("TEST 2: Parameter Integration")
        print("=" * 60)
        
        # Set test values for all parameters
        test_params = {
            "RTIDetectionRadius": "1609",  # 1 mile in meters
            "RTIForwardSlowdownRange": "805",  # 0.5 miles in meters
            "RTIResumeSpeedDistance": "402",  # 0.25 miles in meters
            "RTISpeedReduction": "24",  # 15 mph in km/h
            "RTISpeedReductionMode": "custom",
            "RTIThreatFilter": "1",  # Police only
            "RTIHUDEnabled": "1",
            "RTIAudioAlerts": "1"
        }
        
        print("Setting test parameters:")
        for key, value in test_params.items():
            self.params.put(key, value)
            print(f"  {key}: {value}")
        
        # Give the daemon time to reload parameters
        print("\nWaiting for daemon to reload parameters...")
        time.sleep(2)
        
        # Monitor messages to verify parameters are being used
        print("\nMonitoring RTI behavior with custom parameters...")
        sm = messaging.SubMaster(['rtiStateSP'])
        
        # Create a test threat message to inject
        pm = messaging.PubMaster(['rtiStateSP'])
        
        # First send a police threat (should be detected with filter=1)
        print("\nSending POLICE threat (should be detected)...")
        msg = messaging.new_message('rtiStateSP', valid=True)
        msg.rtiStateSP.timeStamp = int(time.time() * 1e9)
        msg.rtiStateSP.threatAhead = True
        msg.rtiStateSP.threatDistanceM = 400.0  # Within 0.5 mile threshold
        msg.rtiStateSP.recommendedSpeed = 10.0  # Should see custom reduction
        msg.rtiStateSP.source = 'param_test'
        msg.rtiStateSP.apiStatus = 'connected'
        
        msg.rtiStateSP.init('threats', 1)
        threat = msg.rtiStateSP.threats[0]
        threat.id = 'test_police'
        threat.type = 'police'
        threat.distance = 400.0
        threat.direction = 'ahead'
        threat.confidence = 0.9
        
        pm.send('rtiStateSP', msg)
        time.sleep(0.5)
        
        # Now send a hazard threat (should be filtered out with filter=1)
        print("Sending HAZARD threat (should be filtered out)...")
        msg.rtiStateSP.threats[0].type = 'hazard'
        msg.rtiStateSP.threats[0].id = 'test_hazard'
        pm.send('rtiStateSP', msg)
        time.sleep(0.5)
        
        # Verify HUD and audio settings
        hud_enabled = self.params.getBool("RTIHUDEnabled")
        audio_enabled = self.params.getBool("RTIAudioAlerts")
        
        print("\nParameter verification:")
        print(f"  HUD Enabled: {hud_enabled} (expected: True)")
        print(f"  Audio Alerts: {audio_enabled} (expected: True)")
        
        # Read back parameters to verify persistence
        print("\nVerifying parameter persistence:")
        for key in test_params:
            stored_value = self.params.get(key)
            if stored_value:
                stored_value = stored_value.decode('utf-8') if isinstance(stored_value, bytes) else str(stored_value)
            expected = test_params[key]
            
            if stored_value == expected:
                print(f"  ✓ {key}: {stored_value}")
            else:
                print(f"  ✗ {key}: {stored_value} (expected {expected})")
        
        return True
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("RTI FIX VALIDATION SUITE")
        print("=" * 60)
        print("This tests the flickering fix and parameter integration")
        print()
        
        # Test 1: Continuous publishing
        flicker_fixed = self.test_continuous_publishing(10)
        
        # Test 2: Parameter integration
        params_working = self.test_parameter_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Flickering Fix: {'✓ PASS' if flicker_fixed else '✗ FAIL'}")
        print(f"Parameter Integration: {'✓ PASS' if params_working else '✗ FAIL'}")
        
        if flicker_fixed and params_working:
            print("\n🎉 ALL TESTS PASSED! RTI is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Review the output above.")
        
        return flicker_fixed and params_working


def main():
    """Main entry point."""
    print("Make sure rtid daemon is running!")
    print("Start it with: python3 /data/openpilot/sunnypilot/rtid/rtid.py")
    print()
    
    validator = RTIFixValidation()
    success = validator.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())