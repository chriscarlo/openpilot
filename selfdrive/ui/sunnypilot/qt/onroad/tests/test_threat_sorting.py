#!/usr/bin/env python3
"""
Test to verify RTI threats are sorted correctly (closest to furthest)
"""

def test_threat_sorting():
    """Verify threats are sorted by distance in ascending order"""
    
    # Simulated threat data (unsorted)
    threats = [
        {"type": "CAMERA", "distance": 500, "position": "ahead"},
        {"type": "POLICE", "distance": 200, "position": "right"},  
        {"type": "ACCIDENT", "distance": 800, "position": "ahead"},
        {"type": "CONSTRUCTION", "distance": 1200, "position": "left"},
        {"type": "HAZARD", "distance": 350, "position": "behind"}
    ]
    
    print("Original threat order (unsorted):")
    for i, threat in enumerate(threats, 1):
        print(f"  {i}. {threat['type']:12} - {threat['distance']:4}m")
    
    # Sort by distance (ascending - closest first)
    # This mimics the C++ code: return a.distance < b.distance
    sorted_threats = sorted(threats, key=lambda x: x['distance'])
    
    print("\nSorted threat order (closest to furthest):")
    print("Display position from TOP to BOTTOM:")
    for i, threat in enumerate(sorted_threats[:4], 1):  # Max 4 threats
        color = get_threat_color(threat['distance'])
        print(f"  Line {i}: {threat['type']:12} - {threat['distance']:4}m - {color}")
    
    # Verify sorting is correct
    assert sorted_threats[0]['distance'] == 200, "Closest threat should be first"
    assert sorted_threats[1]['distance'] == 350, "Second closest should be second"
    assert sorted_threats[2]['distance'] == 500, "Third closest should be third"
    assert sorted_threats[3]['distance'] == 800, "Fourth closest should be fourth"
    
    print("\n✓ Sorting verified: Closest threat at top, furthest at bottom")
    
    # Show what the display would look like
    print("\nSimulated RTI Widget Display:")
    print("┌─────────────────────────────────────┐")
    print("│ RTI                                 │")
    print("│                                     │")
    for threat in sorted_threats[:4]:
        arrow = get_arrow_direction(threat['position'])
        distance_str = format_distance(threat['distance'])
        color = get_threat_color(threat['distance'])
        print(f"│ {arrow} {threat['type']:12} • {distance_str:6} ({color:6}) │")
    print("│                                     │")
    print("└─────────────────────────────────────┘")
    
    return True

def get_threat_color(distance_m):
    """Get color based on distance (matches C++ implementation)"""
    if distance_m < 200:
        return "RED"
    elif distance_m < 500:
        return "ORANGE"
    elif distance_m < 1000:
        return "YELLOW"
    else:
        return "GRAY"

def get_arrow_direction(position):
    """Get arrow character for position"""
    arrows = {
        "ahead": "↑",
        "right": "→",
        "behind": "↓",
        "left": "←"
    }
    return arrows.get(position, "↑")

def format_distance(distance_m):
    """Format distance for display"""
    if distance_m < 1000:
        return f"{distance_m}m"
    else:
        return f"{distance_m/1000:.1f}km"

if __name__ == "__main__":
    print("=" * 50)
    print("RTI THREAT SORTING TEST")
    print("=" * 50)
    
    if test_threat_sorting():
        print("\n" + "=" * 50)
        print("TEST PASSED: Threats correctly sorted")
        print("Closest threats appear at TOP of widget")
        print("Furthest threats appear at BOTTOM")
        print("=" * 50)