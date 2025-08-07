#!/usr/bin/env python3
"""
Test script for sigmoid-based continuous lateral acceleration function.

Requirements:
1. Replace piecewise function with continuous sigmoid
2. Keep 50+ mph range similar (curvature ≤ 0.0053) 
3. Make <50 mph range ~20% more aggressive (1.8-2.04 m/s² instead of 1.5-1.7)
4. Continuous function (no piecewise logic)

Original piecewise behavior:
- Zone 1: curvature > 0.0053 → 1.5-1.7 m/s² (tight curves, <50mph)  
- Zone 2: 0.0029 < curvature ≤ 0.0053 → 1.7-3.12 m/s² (50-70mph, exponential rise)
- Zone 3: curvature ≤ 0.0029 → 3.12 m/s² (highway, >70mph)
"""

import math
from typing import List, Tuple

def original_piecewise_lateral_acceleration(curvature: float) -> float:
    """Original piecewise function for comparison."""
    CURV_50MPH = 0.0053  # Curvature corresponding to ~50mph curves
    CURV_70MPH = 0.0029  # Curvature corresponding to ~70mph curves

    if curvature > CURV_50MPH:
        # Zone 1: Tight curves (<50mph) - CONSERVATIVE
        if curvature > 0.3:
            return 1.5
        else:
            t = (curvature - CURV_50MPH) / (0.3 - CURV_50MPH)
            return 1.7 + t * (1.5 - 1.7)
    elif curvature > CURV_70MPH:
        # Zone 2: Transition (50-70mph) - RAPID INCREASE
        t = (curvature - CURV_70MPH) / (CURV_50MPH - CURV_70MPH)
        return 1.7 + (3.12 - 1.7) * (1 - math.exp(-5 * (1 - t)))
    else:
        # Zone 3: Highway speeds (>70mph) - MAXIMUM PERFORMANCE
        return 3.12

def sigmoid_lateral_acceleration(curvature: float) -> float:
    """
    Continuous sigmoid-based lateral acceleration function.
    
    Mathematical approach:
    - Use logistic sigmoid: f(x) = L / (1 + exp(-k*(x-x0))) + offset
    - Multiple sigmoid composition for smooth transitions
    - Inverse curvature mapping (higher curvature = lower speed = lower lat accel)
    
    Key parameters:
    - L: Maximum range of sigmoid
    - k: Steepness (controls transition sharpness)  
    - x0: Midpoint (controls where transition occurs)
    - offset: Vertical shift
    """
    
    # Clamp extreme curvature values
    curvature = max(1e-8, min(curvature, 1.0))
    
    # Transform curvature to log space for better sigmoid behavior
    # Higher curvature -> lower log_curv -> lower lateral acceleration
    log_curv = -math.log(curvature)
    
    # Primary sigmoid: Maps from high performance (3.12) to moderate (1.8)
    # Transition centered around log(0.004) ≈ 5.52 (between 50-70mph range)
    k1 = 1.2  # Moderate steepness for smooth transition
    x1 = 5.2  # Transition point (corresponds to ~0.0055 curvature)
    primary_range = 3.12 - 1.8  # 1.32 m/s² range
    
    primary_sigmoid = primary_range / (1.0 + math.exp(-k1 * (log_curv - x1))) + 1.8
    
    # Secondary sigmoid: Creates the tight curve behavior
    # Additional reduction for very tight curves (>0.01 curvature)
    k2 = 2.0  # Steeper for tighter transition
    x2 = 4.6  # Earlier transition point (corresponds to ~0.01 curvature)
    secondary_range = 0.24  # 0.24 m/s² additional reduction at very tight curves
    
    secondary_reduction = secondary_range / (1.0 + math.exp(k2 * (log_curv - x2)))
    
    # Combine sigmoids
    result = primary_sigmoid - secondary_reduction
    
    # Safety clamps
    return max(1.8, min(result, 3.12))

def hyperbolic_tangent_lateral_acceleration(curvature: float) -> float:
    """
    Alternative implementation using hyperbolic tangent.
    Often smoother than sigmoid for some applications.
    """
    curvature = max(1e-8, min(curvature, 1.0))
    
    # Transform curvature
    log_curv = -math.log(curvature) 
    
    # Center around the transition zone
    shifted = log_curv - 5.2
    
    # Hyperbolic tangent mapping
    # tanh maps [-∞, +∞] to [-1, +1]
    tanh_val = math.tanh(shifted * 0.6)  # 0.6 controls steepness
    
    # Map from [-1, +1] to [1.8, 3.12]
    result = 1.8 + (3.12 - 1.8) * (tanh_val + 1) / 2
    
    # Additional tight curve adjustment
    if curvature > 0.01:
        tight_factor = min(1.0, (curvature - 0.01) / 0.05)  # Gradual reduction
        result -= tight_factor * 0.15  # Reduce by up to 0.15 m/s²
    
    return max(1.8, min(result, 3.12))

def test_functions():
    """Test both sigmoid implementations against requirements."""
    
    # Test points covering all zones
    test_curvatures = [
        # Zone 3: Highway (>70mph, curvature ≤ 0.0029)
        0.0010, 0.0020, 0.0029,
        # Zone 2: Transition (50-70mph, 0.0029 < curvature ≤ 0.0053)
        0.0035, 0.0040, 0.0050, 0.0053,
        # Zone 1: Tight curves (<50mph, curvature > 0.0053)
        0.0060, 0.0080, 0.0100, 0.0150, 0.0200, 0.0500, 0.1000, 0.2000
    ]
    
    print("Curvature | Original | Sigmoid | Tanh | Requirements Check")
    print("-" * 65)
    
    for curv in test_curvatures:
        orig = original_piecewise_lateral_acceleration(curv)
        sig = sigmoid_lateral_acceleration(curv)
        tanh_val = hyperbolic_tangent_lateral_acceleration(curv)
        
        # Check requirements
        if curv <= 0.0029:  # Highway zone - should be similar
            req_check = "✓ Highway" if abs(sig - 3.12) < 0.1 else "✗ Highway"
        elif curv <= 0.0053:  # Transition zone - should be similar  
            expected_range = (1.7, 3.12)
            req_check = "✓ Transit" if expected_range[0] <= sig <= expected_range[1] else "✗ Transit"
        else:  # Tight curves - should be 20% more aggressive (1.8-2.04 vs 1.5-1.7)
            req_check = "✓ Tight" if 1.8 <= sig <= 2.04 else "✗ Tight"
        
        print(f"{curv:8.4f} | {orig:7.2f} | {sig:6.2f} | {tanh_val:6.2f} | {req_check}")

def plot_comparison():
    """Generate comparison data for plotting (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        # Generate smooth curve data
        curvatures = np.logspace(-3.5, -0.5, 200)  # 0.0003 to 0.3
        
        original_values = [original_piecewise_lateral_acceleration(c) for c in curvatures]
        sigmoid_values = [sigmoid_lateral_acceleration(c) for c in curvatures]
        tanh_values = [hyperbolic_tangent_lateral_acceleration(c) for c in curvatures]
        
        plt.figure(figsize=(12, 8))
        plt.semilogx(curvatures, original_values, 'r-', linewidth=2, label='Original Piecewise')
        plt.semilogx(curvatures, sigmoid_values, 'b-', linewidth=2, label='Sigmoid Continuous')
        plt.semilogx(curvatures, tanh_values, 'g--', linewidth=2, label='Tanh Alternative')
        
        # Mark transition points
        plt.axvline(x=0.0029, color='gray', linestyle=':', alpha=0.7, label='70mph boundary')
        plt.axvline(x=0.0053, color='gray', linestyle=':', alpha=0.7, label='50mph boundary')
        
        plt.xlabel('Curvature (1/m)')
        plt.ylabel('Lateral Acceleration (m/s²)')
        plt.title('Continuous Sigmoid vs Original Piecewise Lateral Acceleration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(1.0, 3.5)
        
        plt.savefig('/tmp/sigmoid_comparison.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to /tmp/sigmoid_comparison.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")

if __name__ == "__main__":
    print("Testing Sigmoid-Based Continuous Lateral Acceleration Function")
    print("=" * 65)
    
    test_functions()
    plot_comparison()
    
    print("\n" + "=" * 65)
    print("MATHEMATICAL REASONING:")
    print("1. Primary sigmoid: Controls main highway->city transition") 
    print("2. Log-space transformation: Better captures curvature dynamics")
    print("3. Secondary sigmoid: Handles very tight curve behavior")
    print("4. Smooth composition: No discontinuities or sharp transitions")
    print("5. Safety bounds: Hard clamps prevent extreme values")