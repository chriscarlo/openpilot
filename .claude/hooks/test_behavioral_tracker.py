#!/usr/bin/env python3
"""
Test suite for the behavioral tracking system
Run this to verify the anti-fabrication hooks are working
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from behavioral_tracker import BehaviorTracker
except ImportError:
    print("ERROR: Could not import behavioral_tracker.py")
    print("Make sure behavioral_tracker.py is in the same directory")
    sys.exit(1)

def test_initialization():
    """Test tracker initialization"""
    print("Testing initialization...")
    # Use unique PPID for test isolation
    test_ppid = os.getpid() + 1000
    tracker = BehaviorTracker(ppid=test_ppid)
    assert tracker.session is not None
    assert tracker.session["scores"]["level"] == 1
    assert tracker.session["scores"]["level_name"] == "Fabricator"
    assert tracker.session["trust_metrics"]["trust_level"] == 100
    print("  ✓ Initialization successful")
    # Clean up test file
    os.remove(tracker.session_file)

def test_verification_tracking():
    """Test verification scoring"""
    print("\nTesting verification tracking...")
    test_ppid = os.getpid() + 2000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    initial_xp = tracker.session["scores"]["verification_xp"]
    tracker.track_verification("file_read")
    assert tracker.session["scores"]["verification_xp"] > initial_xp
    print("  ✓ File read verification tracked")
    
    tracker.track_verification("context7_lookup")
    assert tracker.session["verifications"]["categories"]["context7_lookups"] == 1
    print("  ✓ Context7 lookup tracked")
    
    # Test streak multiplier
    for _ in range(3):
        tracker.track_verification("file_read")
    assert tracker.session["gamification"]["streaks"]["verification_streak"] >= 3
    print("  ✓ Streak tracking works")
    os.remove(tracker.session_file)

def test_violation_tracking():
    """Test violation penalties"""
    print("\nTesting violation tracking...")
    test_ppid = os.getpid() + 3000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    initial_trust = tracker.session["trust_metrics"]["trust_level"]
    tracker.track_fabrication("edit_without_read", "test.py")
    
    assert tracker.session["trust_metrics"]["trust_level"] < initial_trust
    assert tracker.session["violations"]["total_count"] == 1
    assert tracker.session["violations"]["categories"]["edit_without_read"] == 1
    print("  ✓ Edit without read violation tracked")
    
    # Test streak reset
    tracker.session["gamification"]["streaks"]["verification_streak"] = 5
    tracker.track_fabrication("library_without_context7", "numpy")
    assert tracker.session["gamification"]["streaks"]["verification_streak"] == 0
    print("  ✓ Streaks reset on violation")
    os.remove(tracker.session_file)
    os.remove(tracker.violations_file)

def test_trust_calculation():
    """Test trust score calculation"""
    print("\nTesting trust calculation...")
    test_ppid = os.getpid() + 4000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    # Perfect trust initially
    trust, state = tracker.calculate_trust_score()
    assert trust == 100
    assert state == "Trusted"
    print("  ✓ Initial trust is 100%")
    
    # Degrade trust with violations
    for _ in range(3):
        tracker.track_fabrication("claim_without_verification", "test")
    
    trust, state = tracker.calculate_trust_score()
    assert trust < 100
    assert state != "Trusted"
    print(f"  ✓ Trust degraded to {trust:.1f}% ({state})")
    os.remove(tracker.session_file)
    os.remove(tracker.violations_file)

def test_frustration_model():
    """Test user frustration calculation"""
    print("\nTesting frustration model...")
    test_ppid = os.getpid() + 5000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    # No frustration initially
    frustration, state = tracker.calculate_frustration()
    assert frustration == 0
    assert state == "Happy"
    print("  ✓ Initial frustration is 0%")
    
    # Increase frustration with violations
    for _ in range(2):
        tracker.track_fabrication("edit_without_read", "file.py")
    
    frustration, state = tracker.calculate_frustration()
    assert frustration > 0
    print(f"  ✓ Frustration increased to {frustration:.1f}% ({state})")
    os.remove(tracker.session_file)
    os.remove(tracker.violations_file)

def test_level_progression():
    """Test XP and level system"""
    print("\nTesting level progression...")
    test_ppid = os.getpid() + 6000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    # Earn XP through verifications
    for _ in range(15):
        tracker.track_verification("file_read")
    
    assert tracker.session["scores"]["verification_xp"] >= 100
    assert tracker.session["scores"]["level"] >= 2
    print(f"  ✓ Leveled up to {tracker.session['scores']['level_name']}")
    os.remove(tracker.session_file)

def test_session_persistence():
    """Test session save/load"""
    print("\nTesting session persistence...")
    test_ppid = os.getpid() + 7000
    
    # Create and modify a tracker
    tracker1 = BehaviorTracker(ppid=test_ppid)
    tracker1.track_verification("file_read")
    tracker1.track_fabrication("task_rush", "rushing")
    tracker1.save_session()
    
    # Load in new tracker
    tracker2 = BehaviorTracker(tracker1.ppid)
    assert tracker2.session["verifications"]["total_count"] == tracker1.session["verifications"]["total_count"]
    assert tracker2.session["violations"]["total_count"] == tracker1.session["violations"]["total_count"]
    print("  ✓ Session persisted and reloaded")
    os.remove(tracker1.session_file)
    os.remove(tracker1.violations_file)

def test_status_summary():
    """Test status summary generation"""
    print("\nTesting status summary...")
    test_ppid = os.getpid() + 8000
    tracker = BehaviorTracker(ppid=test_ppid)
    
    # Add some activity
    tracker.track_verification("context7_lookup")
    tracker.track_fabrication("overconfident_assertion", "test")
    
    status = tracker.get_status_summary()
    assert "level" in status
    assert "trust_level" in status
    assert "violations" in status
    assert "verifications" in status
    print("  ✓ Status summary generated")
    print(f"    Level: {status['level']} ({status['level_name']})")
    print(f"    Trust: {status['trust_level']:.1f}% ({status['trust_state']})")
    print(f"    Violations: {status['violations']}, Verifications: {status['verifications']}")
    os.remove(tracker.session_file)
    os.remove(tracker.violations_file)

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("BEHAVIORAL TRACKER TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_initialization,
        test_verification_tracking,
        test_violation_tracking,
        test_trust_calculation,
        test_frustration_model,
        test_level_progression,
        test_session_persistence,
        test_status_summary
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("ALL TESTS PASSED!")
        print("\nThe anti-fabrication hooks system is working correctly.")
        print("Behavioral tracking will help prevent fabrication and")
        print("encourage verification-first development.")
    else:
        print(f"\n{failed} test(s) failed. Please check the implementation.")
        sys.exit(1)
    
    print("=" * 50)

if __name__ == "__main__":
    run_all_tests()