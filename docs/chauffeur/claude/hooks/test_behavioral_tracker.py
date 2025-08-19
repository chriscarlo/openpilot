#!/usr/bin/env python3
"""Test script for behavioral_tracker.py"""

import os
import sys
import json
from behavioral_tracker import BehaviorTracker

def test_tracker():
    """Test the behavioral tracker functionality"""
    print("Testing Behavioral Tracker")
    print("=" * 50)
    
    # Initialize tracker
    tracker = BehaviorTracker(ppid=99999)  # Use test PID
    print(f"✓ Tracker initialized: {tracker.session_id}")
    
    # Test verification tracking
    print("\n1. Testing verification tracking...")
    tracker.track_verification("file_read")
    print(f"   Verifications: {tracker.session['verifications']['total_count']}")
    print(f"   XP: {tracker.session['scores']['verification_xp']}")
    
    # Test streak bonus
    print("\n2. Testing streak bonuses...")
    for i in range(4):
        tracker.track_verification("context7_lookup")
    print(f"   Verification streak: {tracker.session['gamification']['streaks']['verification_streak']}")
    print(f"   Context7 streak: {tracker.session['gamification']['streaks']['context7_streak']}")
    print(f"   XP: {tracker.session['scores']['verification_xp']}")
    
    # Test violation tracking
    print("\n3. Testing violation tracking...")
    tracker.track_fabrication("edit_without_read", "Edited main.py without reading")
    print(f"   Violations: {tracker.session['violations']['total_count']}")
    print(f"   Trust level: {tracker.session['trust_metrics']['trust_level']:.1f}%")
    print(f"   User frustration: {tracker.session['frustration_model']['user_frustration']:.1f}%")
    print(f"   Streaks reset: {tracker.session['gamification']['streaks']['verification_streak']}")
    
    # Test trust calculation
    print("\n4. Testing trust score calculation...")
    trust, state = tracker.calculate_trust_score()
    print(f"   Trust: {trust:.1f}% ({state})")
    
    # Test frustration calculation
    print("\n5. Testing frustration calculation...")
    frustration, state = tracker.calculate_frustration()
    print(f"   Frustration: {frustration:.1f}% ({state})")
    
    # Test behavioral score
    print("\n6. Testing behavioral score...")
    score = tracker.get_behavioral_score()
    print(f"   Total score: {score['total_score']:.1f}")
    print(f"   Level: {score['level']} ({score['level_name']})")
    print(f"   Progress to next: {score['progress_percent']:.1f}%")
    
    # Test feedback generation
    print("\n7. Testing feedback generation...")
    feedback = tracker.generate_feedback("verification")
    print(f"   Feedback: {feedback}")
    
    # Test hook integration
    print("\n8. Testing hook integration...")
    result = tracker.hook_pretool_use("Write", {"file_path": "test.py", "content": "test"})
    print(f"   Allow write: {result['allow']}")
    print(f"   Message: {result['message']}")
    
    # Test multiple violations to see escalation
    print("\n9. Testing violation escalation...")
    for i in range(5):
        tracker.track_fabrication("library_without_context7", f"Used library {i}")
    print(f"   Total violations: {tracker.session['violations']['total_count']}")
    print(f"   Trust level: {tracker.session['trust_metrics']['trust_level']:.1f}%")
    print(f"   Severity: {tracker.session['messages']['severity_level']}")
    
    # Test injection text
    print("\n10. Testing injection text generation...")
    injection = tracker.get_injection_text()
    print("   Injection preview:")
    for line in injection.split('\n')[:5]:
        print(f"   {line}")
    
    # Clean up test files
    test_files = [
        f"/tmp/claude_behavior_{99999}.json",
        f"/tmp/claude_violations_{99999}.txt"
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"\n✓ Cleaned up: {f}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_tracker()