#!/usr/bin/env python3
"""
VTSC Comparative Performance Test Script

Tests stock vs physics-based VTSC implementations across 15 real-world scenarios.
Directly imports and calls both implementations to compare performance.

Usage:
    python comparative_performance_test.py

Results saved to: vtsc_performance_results.json
"""

import sys
import os
import json
import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Any
from collections import defaultdict

# Add stub modules to sys.modules before importing VTSC implementations
stub_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, stub_dir)

# Import and install stub modules
from stub_cereal import custom
from stub_openpilot import openpilot

sys.modules['cereal'] = type('cereal', (), {'custom': custom})
sys.modules['openpilot'] = openpilot
sys.modules['openpilot.common'] = openpilot.common
sys.modules['openpilot.common.params'] = openpilot.common.params
sys.modules['openpilot.common.conversions'] = openpilot.common.conversions
sys.modules['openpilot.common.numpy_fast'] = openpilot.common.numpy_fast
sys.modules['openpilot.selfdrive'] = openpilot.selfdrive
sys.modules['openpilot.selfdrive.car'] = openpilot.selfdrive.car
sys.modules['openpilot.selfdrive.car.cruise'] = openpilot.selfdrive.car.cruise
sys.modules['openpilot.selfdrive.modeld'] = openpilot.selfdrive.modeld
sys.modules['openpilot.selfdrive.modeld.constants'] = openpilot.selfdrive.modeld.constants
sys.modules['openpilot.selfdrive.controls'] = openpilot.selfdrive.controls
sys.modules['openpilot.selfdrive.controls.lib'] = openpilot.selfdrive.controls.lib
sys.modules['openpilot.selfdrive.controls.lib.drive_helpers'] = openpilot.selfdrive.controls.lib.drive_helpers

# Add paths for both VTSC implementations
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')
sys.path.append('/data/openpilot/docs/chauffeur/vtsc/reference')

# Import VTSC implementations
# Note: "Integrated" is now the production version, "Stock" was the old state-machine version
try:
    from vision_turn_controller_physics_based_original import VisionTurnController as PhysicsOriginalVTSC
    print("Successfully imported Physics-Based Original VTSC")
except ImportError as e:
    print(f"Failed to import Physics-Based Original VTSC: {e}")
    PhysicsOriginalVTSC = None

try:
    from vision_turn_controller import VisionTurnController as ProductionVTSC
    print("Successfully imported Current Production (Integrated) VTSC")
except ImportError as e:
    print(f"Failed to import Current Production VTSC: {e}")
    ProductionVTSC = None

# For backwards compatibility, map the old names
StockVTSC = None  # Old state-machine version no longer exists
PhysicsVTSC = PhysicsOriginalVTSC
IntegratedVTSC = ProductionVTSC

if not all([PhysicsOriginalVTSC, ProductionVTSC]):
    print("Failed to import required VTSC implementations")
    sys.exit(1)

# Mock data structures for testing
@dataclass
class MockModelData:
    """Mock modelV2 data structure"""
    orientationRate: Any = None
    velocity: Any = None
    laneLines: list = None
    laneLineProbs: list = None
    laneLineStds: list = None

    def __post_init__(self):
        if self.orientationRate is None:
            self.orientationRate = MockVector3()
        if self.velocity is None:
            self.velocity = MockVector3()
        if self.laneLines is None:
            self.laneLines = [MockLaneLine() for _ in range(4)]
        if self.laneLineProbs is None:
            self.laneLineProbs = [0.8, 0.9, 0.9, 0.8]  # Good lane confidence
        if self.laneLineStds is None:
            self.laneLineStds = [0.1, 0.1, 0.1, 0.1]  # Low standard deviation

@dataclass
class MockVector3:
    """Mock 3D vector data"""
    x: list[float] = None
    y: list[float] = None
    z: list[float] = None

    def __post_init__(self):
        if self.x is None:
            self.x = [0.0] * 33
        if self.y is None:
            self.y = [0.0] * 33
        if self.z is None:
            self.z = [0.0] * 33

@dataclass
class MockLaneLine:
    """Mock lane line data"""
    x: list[float] = None
    y: list[float] = None
    t: list[float] = None

    def __post_init__(self):
        if self.x is None:
            self.x = list(range(33))  # Distance points
        if self.y is None:
            self.y = [0.0] * 33      # Lateral offset
        if self.t is None:
            self.t = list(np.linspace(0, 5, 33))  # Time points

@dataclass
class MockCarState:
    """Mock carState data"""
    gasPressed: bool = False
    steeringAngleDeg: float = 0.0

@dataclass
class MockLateralPlan:
    """Mock lateralPlan data"""
    psis: list[float] = None
    dPathPoints: list[float] = None

    def __post_init__(self):
        if self.psis is None:
            self.psis = [0.0] * 50
        if self.dPathPoints is None:
            self.dPathPoints = [0.0] * 50

class MockSubMaster:
    """Mock SubMaster for testing"""
    def __init__(self):
        self.data = {
            'modelV2': MockModelData(),
            'carState': MockCarState(),
            'lateralPlan': MockLateralPlan()
        }
        self.valid = {
            'modelV2': True,
            'carState': True,
            'lateralPlan': True
        }

    def __getitem__(self, key):
        return self.data[key]

@dataclass
class TestScenario:
    """Defines a test scenario for VTSC comparison"""
    name: str
    description: str
    v_ego_initial: float  # m/s
    v_cruise_setpoint: float  # m/s
    curvature_profile: list[float]  # Curvature over distance
    vision_confidence: list[float]  # Vision confidence over time
    expected_behavior: str  # Description of expected behavior

class VTSCComparativeTest:
    """Comparative performance test suite for VTSC implementations"""

    def __init__(self):
        self.results = {}
        self.scenarios = self._define_test_scenarios()
        self.MAX_REASONABLE_SPEED = 200.0  # m/s (720 km/h) - cap for infinite speeds

    def _define_test_scenarios(self) -> list[TestScenario]:
        """Define 15 real-world test scenarios"""
        scenarios = []

        # 1. Gentle Highway Curve
        scenarios.append(TestScenario(
            name="gentle_highway_curve",
            description="70 mph entry into gentle sweeping highway curve",
            v_ego_initial=31.3,  # 70 mph
            v_cruise_setpoint=31.3,
            curvature_profile=self._generate_curvature_profile("gentle_curve", max_curvature=0.008),
            vision_confidence=[1.0] * 50,
            expected_behavior="Minimal speed reduction, smooth deceleration"
        ))

        # 2. Tight Mountain Hairpin
        scenarios.append(TestScenario(
            name="tight_mountain_hairpin",
            description="25 mph entry into very sharp hairpin turn",
            v_ego_initial=11.2,  # 25 mph
            v_cruise_setpoint=15.6,  # 35 mph
            curvature_profile=self._generate_curvature_profile("hairpin", max_curvature=0.15),
            vision_confidence=[1.0] * 50,
            expected_behavior="Significant deceleration, emergency levels triggered"
        ))

        # 3. Sudden Sharp Curve
        scenarios.append(TestScenario(
            name="sudden_sharp_curve",
            description="60 mph straight road with sudden sharp curve",
            v_ego_initial=26.8,  # 60 mph
            v_cruise_setpoint=26.8,
            curvature_profile=self._generate_curvature_profile("sudden_sharp", max_curvature=0.05),
            vision_confidence=[1.0] * 50,
            expected_behavior="Rapid deceleration, high emergency level"
        ))

        # 4. Highway Onramp
        scenarios.append(TestScenario(
            name="highway_onramp",
            description="45 mph onramp with consistent moderate curvature",
            v_ego_initial=20.1,  # 45 mph
            v_cruise_setpoint=26.8,  # 60 mph
            curvature_profile=self._generate_curvature_profile("onramp", max_curvature=0.02),
            vision_confidence=[1.0] * 50,
            expected_behavior="Moderate speed control, gradual adjustment"
        ))

        # 5. S-Curve Sequence
        scenarios.append(TestScenario(
            name="s_curve_sequence",
            description="Back-to-back opposite curves (S-curve)",
            v_ego_initial=22.4,  # 50 mph
            v_cruise_setpoint=22.4,
            curvature_profile=self._generate_curvature_profile("s_curve", max_curvature=0.03),
            vision_confidence=[1.0] * 50,
            expected_behavior="Complex speed modulation, apex detection"
        ))

        # 6. Vision Occlusion - Gradual
        scenarios.append(TestScenario(
            name="vision_occlusion_gradual",
            description="Moderate curve with gradually degrading vision",
            v_ego_initial=20.1,  # 45 mph
            v_cruise_setpoint=22.4,  # 50 mph
            curvature_profile=self._generate_curvature_profile("moderate_curve", max_curvature=0.025),
            vision_confidence=list(np.linspace(1.0, 0.3, 50)),  # Gradual degradation
            expected_behavior="Increasing conservatism as confidence drops"
        ))

        # 7. Vision Occlusion - Sudden
        scenarios.append(TestScenario(
            name="vision_occlusion_sudden",
            description="Highway curve with sudden vision loss",
            v_ego_initial=26.8,  # 60 mph
            v_cruise_setpoint=26.8,
            curvature_profile=self._generate_curvature_profile("highway_curve", max_curvature=0.015),
            vision_confidence=[1.0] * 20 + [0.2] * 30,  # Sudden drop
            expected_behavior="Fallback to extrapolated curvature, safety margins"
        ))

        # 8. Late Curvature Detection
        scenarios.append(TestScenario(
            name="late_curvature_detection",
            description="Sharp curve appearing late in model predictions",
            v_ego_initial=24.6,  # 55 mph
            v_cruise_setpoint=24.6,
            curvature_profile=self._generate_curvature_profile("late_detection", max_curvature=0.04),
            vision_confidence=[1.0] * 50,
            expected_behavior="Aggressive deceleration, high emergency levels"
        ))

        # 9. Curve Exceeds FOV
        scenarios.append(TestScenario(
            name="curve_exceeds_fov",
            description="Very sharp curve exceeding camera 60° FOV",
            v_ego_initial=13.4,  # 30 mph
            v_cruise_setpoint=17.9,  # 40 mph
            curvature_profile=self._generate_curvature_profile("fov_limit", max_curvature=0.12),
            vision_confidence=[1.0] * 15 + [0.5] * 20 + [0.2] * 15,  # FOV limitation
            expected_behavior="Conservative speed, occlusion handling"
        ))

        # 10. High-Speed Sweeper
        scenarios.append(TestScenario(
            name="high_speed_sweeper",
            description="80+ mph gentle sweeping curve",
            v_ego_initial=35.8,  # 80 mph
            v_cruise_setpoint=35.8,
            curvature_profile=self._generate_curvature_profile("high_speed_sweep", max_curvature=0.005),
            vision_confidence=[1.0] * 50,
            expected_behavior="Precise speed control at high speeds"
        ))

        # 11. Urban Tight Turn
        scenarios.append(TestScenario(
            name="urban_tight_turn",
            description="Low speed very tight radius urban turn",
            v_ego_initial=6.7,  # 15 mph
            v_cruise_setpoint=8.9,  # 20 mph
            curvature_profile=self._generate_curvature_profile("urban_tight", max_curvature=0.2),
            vision_confidence=[1.0] * 50,
            expected_behavior="Low speed precise control"
        ))

        # 12. Decreasing Radius Turn
        scenarios.append(TestScenario(
            name="decreasing_radius_turn",
            description="Turn that gets progressively tighter",
            v_ego_initial=17.9,  # 40 mph
            v_cruise_setpoint=17.9,
            curvature_profile=self._generate_curvature_profile("decreasing_radius", max_curvature=0.06),
            vision_confidence=[1.0] * 50,
            expected_behavior="Progressive deceleration as radius decreases"
        ))

        # 13. Increasing Radius Turn
        scenarios.append(TestScenario(
            name="increasing_radius_turn",
            description="Turn that opens up progressively",
            v_ego_initial=13.4,  # 30 mph
            v_cruise_setpoint=20.1,  # 45 mph
            curvature_profile=self._generate_curvature_profile("increasing_radius", max_curvature=0.04),
            vision_confidence=[1.0] * 50,
            expected_behavior="Initial deceleration then acceleration out"
        ))

        # 14. Emergency Braking Scenario
        scenarios.append(TestScenario(
            name="emergency_braking_scenario",
            description="Very sharp curve requiring maximum deceleration",
            v_ego_initial=26.8,  # 60 mph
            v_cruise_setpoint=26.8,
            curvature_profile=self._generate_curvature_profile("emergency_sharp", max_curvature=0.08),
            vision_confidence=[1.0] * 50,
            expected_behavior="Maximum deceleration, intervention detection"
        ))

        # 15. Vision Noise/Jitter
        scenarios.append(TestScenario(
            name="vision_noise_jitter",
            description="Moderate curve with noisy/jittery model data",
            v_ego_initial=20.1,  # 45 mph
            v_cruise_setpoint=22.4,  # 50 mph
            curvature_profile=self._generate_curvature_profile("noisy_curve", max_curvature=0.03),
            vision_confidence=[0.9, 0.7, 0.95, 0.6, 0.85] * 10,  # Jittery confidence
            expected_behavior="Stable control despite noisy inputs"
        ))

        # 16. Straight Road - Critical for Integrated VTSC Testing
        scenarios.append(TestScenario(
            name="straight_road",
            description="Perfectly straight highway - integrated VTSC should return very high speeds",
            v_ego_initial=26.8,  # 60 mph
            v_cruise_setpoint=26.8,
            curvature_profile=self._generate_curvature_profile("straight", max_curvature=0.0),
            vision_confidence=[1.0] * 50,
            expected_behavior="Integrated VTSC should return cruise setpoint or higher, allowing longitudinal planner to ignore"
        ))

        return scenarios

    def _generate_curvature_profile(self, curve_type: str, max_curvature: float) -> list[float]:
        """Generate curvature profile based on curve type"""
        points = 33  # Standard trajectory points

        if curve_type == "gentle_curve":
            # Gentle S-curve profile
            x = np.linspace(-2, 2, points)
            profile = max_curvature * np.exp(-x**2 / 2) * np.sin(x)

        elif curve_type == "hairpin":
            # Sharp hairpin - high curvature in middle
            x = np.linspace(-3, 3, points)
            profile = max_curvature * np.exp(-x**2 / 0.5)

        elif curve_type == "sudden_sharp":
            # Sudden sharp curve - straight then sharp
            profile = np.zeros(points)
            profile[15:25] = max_curvature  # Sharp section

        elif curve_type == "onramp":
            # Consistent onramp curvature
            profile = np.full(points, max_curvature * 0.8)

        elif curve_type == "s_curve":
            # S-curve - opposite curvatures
            x = np.linspace(-4, 4, points)
            profile = max_curvature * np.sin(x)

        elif curve_type == "moderate_curve":
            # Standard moderate curve
            x = np.linspace(-2, 2, points)
            profile = max_curvature * np.exp(-x**2 / 1.5)

        elif curve_type == "highway_curve":
            # Long gentle highway curve
            x = np.linspace(-3, 3, points)
            profile = max_curvature * np.exp(-x**2 / 4)

        elif curve_type == "late_detection":
            # Curve appears only in latter half
            profile = np.zeros(points)
            x = np.linspace(-2, 2, 15)
            profile[18:] = max_curvature * np.exp(-x**2 / 1)

        elif curve_type == "fov_limit":
            # Sharp curve with limited visibility
            x = np.linspace(-1, 1, points//2)
            profile = np.zeros(points)
            profile[:points//2] = max_curvature * np.exp(-x**2 / 0.3)

        elif curve_type == "high_speed_sweep":
            # Very gentle high-speed curve
            x = np.linspace(-4, 4, points)
            profile = max_curvature * np.exp(-x**2 / 8)

        elif curve_type == "urban_tight":
            # Tight urban turn - very sharp
            x = np.linspace(-1.5, 1.5, points)
            profile = max_curvature * np.exp(-x**2 / 0.2)

        elif curve_type == "decreasing_radius":
            # Progressively tighter curve
            profile = max_curvature * (0.3 + 0.7 * np.linspace(0, 1, points)**2)

        elif curve_type == "increasing_radius":
            # Progressively opening curve
            profile = max_curvature * (1.0 - 0.7 * np.linspace(0, 1, points)**2)

        elif curve_type == "emergency_sharp":
            # Emergency sharp curve
            x = np.linspace(-2, 2, points)
            profile = max_curvature * np.exp(-x**2 / 0.8)

        elif curve_type == "noisy_curve":
            # Moderate curve with noise
            x = np.linspace(-2, 2, points)
            base_profile = max_curvature * np.exp(-x**2 / 1.5)
            noise = 0.1 * max_curvature * np.random.normal(0, 1, points)
            profile = base_profile + noise

        elif curve_type == "straight":
            # Perfectly straight road - zero curvature
            profile = np.zeros(points)

        else:
            # Default: gentle curve
            x = np.linspace(-2, 2, points)
            profile = max_curvature * np.exp(-x**2 / 2)

        return np.abs(profile).tolist()  # Ensure positive curvature

    def _create_mock_sm(self, scenario: TestScenario, step: int = 0) -> MockSubMaster:
        """Create mock SubMaster data for a scenario"""
        sm = MockSubMaster()

        # Set up model data with curvature profile
        curvature_data = scenario.curvature_profile

        # Convert curvature to orientation rate and velocity for physics VTSC
        velocities = [max(scenario.v_ego_initial, 1.0)] * 33  # Avoid division by zero
        orientation_rates = [curv * vel for curv, vel in zip(curvature_data, velocities, strict=False)]

        sm.data['modelV2'].orientationRate.z = orientation_rates
        sm.data['modelV2'].velocity.x = velocities

        # CRITICAL FIX: Generate proper curved lane geometry for stock VTSC
        sm.data['modelV2'].laneLines = self._generate_curved_lane_lines(curvature_data)

        # Set vision confidence
        confidence_index = min(step, len(scenario.vision_confidence) - 1)
        confidence = scenario.vision_confidence[confidence_index]

        # Adjust lane line probabilities based on confidence
        sm.data['modelV2'].laneLineProbs = [confidence] * 4
        sm.data['modelV2'].laneLineStds = [0.1] * 4  # Low std deviation for good lanes

        # Generate curved lateral planner path as fallback
        sm.data['lateralPlan'] = self._generate_curved_lateral_plan(curvature_data)

        # Set realistic steering angle for current curvature
        if len(curvature_data) > 0:
            current_curvature = curvature_data[0]
            # Approximate steering angle from curvature: steer_angle ≈ curvature * wheelbase * steer_ratio
            wheelbase = 2.7
            steer_ratio = 15.0
            sm.data['carState'].steeringAngleDeg = current_curvature * wheelbase * steer_ratio * 57.3  # rad to deg

        # If confidence is very low, mark model as invalid
        if confidence < 0.3:
            sm.valid['modelV2'] = False

        return sm

    def _generate_curved_lane_lines(self, curvature_profile: list[float]) -> list[MockLaneLine]:
        """Generate realistic curved lane line geometry from curvature profile"""
        lane_lines = []

        # Standard distances (matching _EVAL_RANGE logic)
        distances = list(range(33))  # 0 to 32 meters ahead

        for lane_idx in range(4):  # 4 lane lines
            lane_line = MockLaneLine()
            lane_line.x = distances
            lane_line.t = list(np.linspace(0, 5, 33))

            # Generate curved path by integrating curvature
            y_positions = []
            heading_angle = 0.0
            lateral_offset = 0.0

            # Lane-specific lateral offsets (left outer, left inner, right inner, right outer)
            base_offsets = [-5.5, -1.8, 1.8, 5.5]  # meters from center
            lane_offset = base_offsets[lane_idx]

            for i, distance in enumerate(distances):
                if i < len(curvature_profile):
                    curvature = curvature_profile[i]
                else:
                    curvature = 0.0

                # Integrate curvature to get heading change
                if i > 0:
                    ds = distances[i] - distances[i-1]  # distance step
                    heading_angle += curvature * ds

                    # Integrate heading to get lateral position
                    lateral_offset += math.sin(heading_angle) * ds

                # Lane line position = base lane offset + curve-induced lateral offset
                y_positions.append(lane_offset + lateral_offset)

            lane_line.y = y_positions
            lane_lines.append(lane_line)

        return lane_lines

    def _generate_curved_lateral_plan(self, curvature_profile: list[float]) -> MockLateralPlan:
        """Generate curved lateral planner path from curvature profile"""
        lateral_plan = MockLateralPlan()

        # Generate path points by integrating curvature
        path_points = []
        heading_angle = 0.0
        lateral_offset = 0.0

        distances = list(range(50))  # Lateral planner typically has more points

        for i, distance in enumerate(distances):
            if i < len(curvature_profile):
                curvature = curvature_profile[min(i, len(curvature_profile) - 1)]
            else:
                curvature = 0.0

            if i > 0:
                ds = distances[i] - distances[i-1]
                heading_angle += curvature * ds
                lateral_offset += math.sin(heading_angle) * ds

            path_points.append(lateral_offset)

        lateral_plan.dPathPoints = path_points[:50]  # Ensure correct length
        lateral_plan.psis = list(range(50))  # Distance points

        return lateral_plan

    def _safe_speed_value(self, speed: float, source: str = "") -> float:
        """Handle infinite, NaN, or unreasonably high speed values safely."""
        if math.isnan(speed):
            print(f"WARNING: NaN speed detected in {source}, using cruise setpoint")
            return self.MAX_REASONABLE_SPEED
        elif math.isinf(speed):
            print(f"WARNING: Infinite speed detected in {source}, capping at {self.MAX_REASONABLE_SPEED} m/s")
            return self.MAX_REASONABLE_SPEED
        elif speed > self.MAX_REASONABLE_SPEED:
            print(f"WARNING: Very high speed ({speed:.1f} m/s) detected in {source}, capping at {self.MAX_REASONABLE_SPEED} m/s")
            return self.MAX_REASONABLE_SPEED
        else:
            return speed

    def _run_scenario(self, scenario: TestScenario) -> dict[str, Any]:
        """Run a single test scenario on all 3 VTSC implementations"""
        print(f"\nRunning scenario: {scenario.name}")
        print(f"   {scenario.description}")

        # Mock car parameters (simplified)
        class MockCP:
            steerRatio = 15.0
            wheelbase = 2.7

        # Initialize VTSC implementations
        physics_vtsc = PhysicsVTSC(MockCP())
        production_vtsc = IntegratedVTSC(MockCP())

        # Test parameters
        enabled = True
        a_ego = 0.0  # Assume constant speed initially

        # Results storage
        results = {
            'scenario': scenario.name,
            'description': scenario.description,
            'initial_conditions': {
                'v_ego': scenario.v_ego_initial,
                'v_cruise': scenario.v_cruise_setpoint
            },
            'physics_original_results': [],
            'production_results': [],
            'comparison': {}
        }

        # Run simulation steps
        steps = 30  # 1.5 seconds at 20Hz
        v_ego = scenario.v_ego_initial

        for step in range(steps):
            # Create mock data for this step
            sm = self._create_mock_sm(scenario, step)

            # Update both VTSC implementations
            physics_vtsc.update(sm, enabled, v_ego, a_ego, scenario.v_cruise_setpoint)
            production_vtsc.update(sm, enabled, v_ego, a_ego, scenario.v_cruise_setpoint)

            # Collect results with safe speed handling
            physics_result = {
                'step': step,
                'v_turn': self._safe_speed_value(physics_vtsc.v_turn, "Physics Original"),
                'a_target': physics_vtsc.a_target,
                'is_active': physics_vtsc.is_active,
                'state': physics_vtsc.state.name if hasattr(physics_vtsc.state, 'name') else str(physics_vtsc.state)
            }

            production_result = {
                'step': step,
                'v_turn': self._safe_speed_value(production_vtsc.v_turn, "Production"),
                'a_target': production_vtsc.a_target,
                'is_active': production_vtsc.is_active,
                'state': production_vtsc.state.name if hasattr(production_vtsc.state, 'name') else str(production_vtsc.state),
                'emergency_level': production_vtsc.emergency_level.name if hasattr(production_vtsc, 'emergency_level') else 'N/A',
                'intervention_required': production_vtsc.intervention_required if hasattr(production_vtsc, 'intervention_required') else False
            }

            results['physics_original_results'].append(physics_result)
            results['production_results'].append(production_result)

            # Update ego velocity based on acceleration (simplified physics)
            if step < steps - 1:
                dt = 0.05  # 20Hz = 50ms
                # Use average of both accelerations for next step
                avg_accel = (physics_vtsc.a_target + production_vtsc.a_target) / 2
                v_ego = max(0.1, v_ego + avg_accel * dt)  # Prevent negative speeds

        # Calculate comparison metrics
        results['comparison'] = self._calculate_comparison_metrics(
            results['physics_original_results'],
            results['production_results']
        )

        # Print summary
        physics_v_final = self._safe_speed_value(physics_vtsc.v_turn, "Physics Original Final")
        production_v_final = self._safe_speed_value(production_vtsc.v_turn, "Production Final")

        print(f"   Physics Original: v_turn={physics_v_final:.1f} m/s ({physics_v_final*2.237:.1f} mph), a_target={physics_vtsc.a_target:.2f} m/s²")
        print(f"   Production: v_turn={production_v_final:.1f} m/s ({production_v_final*2.237:.1f} mph), a_target={production_vtsc.a_target:.2f} m/s²")
        print("   Scenario completed")

        return results

    def _calculate_comparison_metrics(self, physics_results: list[dict], production_results: list[dict]) -> dict[str, float]:
        """Calculate comparison metrics between physics original and production VTSC implementations"""

        # Extract time series data
        physics_v_turn = [r['v_turn'] for r in physics_results]
        production_v_turn = [r['v_turn'] for r in production_results]

        physics_a_target = [r['a_target'] for r in physics_results]
        production_a_target = [r['a_target'] for r in production_results]

        # Calculate metrics
        metrics = {}

        # Speed differences
        v_diff = np.array(production_v_turn) - np.array(physics_v_turn)
        metrics['production_vs_physics_v_diff_mean'] = float(np.mean(v_diff))
        metrics['production_vs_physics_v_diff_max'] = float(np.max(np.abs(v_diff)))

        # Acceleration differences
        a_diff = np.array(production_a_target) - np.array(physics_a_target)
        metrics['production_vs_physics_a_diff_mean'] = float(np.mean(a_diff))
        metrics['production_vs_physics_a_diff_max'] = float(np.max(np.abs(a_diff)))

        # Smoothness metrics (standard deviation of accelerations)
        metrics['physics_smoothness'] = float(np.std(physics_a_target))
        metrics['production_smoothness'] = float(np.std(production_a_target))

        # Final values
        metrics['final_v_turn_physics'] = float(physics_v_turn[-1])
        metrics['final_v_turn_production'] = float(production_v_turn[-1])

        metrics['final_a_target_physics'] = float(physics_a_target[-1])
        metrics['final_a_target_production'] = float(production_a_target[-1])

        # Min speeds (important for curve handling)
        metrics['min_v_turn_physics'] = float(np.min(physics_v_turn))
        metrics['min_v_turn_production'] = float(np.min(production_v_turn))

        return metrics

    def run_all_tests(self) -> dict[str, Any]:
        """Run all test scenarios and return comprehensive results"""
        print("Starting VTSC Comparative Performance Test Suite")
        print(f"Running {len(self.scenarios)} test scenarios...")

        start_time = time.time()
        all_results = {
            'test_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_scenarios': len(self.scenarios),
                'stock_vtsc_path': '/data/openpilot/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py',
                'physics_vtsc_path': '/data/openpilot/docs/chauffeur/vtsc/reference/vision_turn_controller_physics_based_original.py'
            },
            'scenarios': [],
            'summary': {}
        }

        # Run each scenario
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n[{i}/{len(self.scenarios)}] ", end="")
            try:
                result = self._run_scenario(scenario)
                all_results['scenarios'].append(result)
            except Exception as e:
                print(f"FAILED: {e}")
                # Add failed result
                all_results['scenarios'].append({
                    'scenario': scenario.name,
                    'error': str(e),
                    'status': 'failed'
                })

        # Calculate summary statistics
        all_results['summary'] = self._calculate_summary_stats(all_results['scenarios'])

        end_time = time.time()
        all_results['test_info']['duration_seconds'] = end_time - start_time

        print(f"Test suite completed in {end_time - start_time:.1f} seconds")
        return all_results

    def _calculate_summary_stats(self, scenario_results: list[dict]) -> dict[str, Any]:
        """Calculate summary statistics across all scenarios"""
        successful_results = [r for r in scenario_results if 'comparison' in r]

        if not successful_results:
            return {'error': 'No successful test scenarios'}

        # Aggregate comparison metrics
        summary = {
            'successful_scenarios': len(successful_results),
            'failed_scenarios': len(scenario_results) - len(successful_results),
            'aggregate_metrics': {}
        }

        # Collect all comparison metrics
        all_metrics = defaultdict(list)
        for result in successful_results:
            for metric, value in result['comparison'].items():
                all_metrics[metric].append(value)

        # Calculate aggregate statistics
        for metric, values in all_metrics.items():
            summary['aggregate_metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        return summary

    def save_results(self, results: dict[str, Any], filename: str = 'vtsc_performance_results.json'):
        """Save test results to JSON file"""
        filepath = os.path.join('/data/openpilot/docs/chauffeur/vtsc/results', filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath

def main():
    """Main test execution"""
    print("=" * 60)
    print("VTSC COMPARATIVE PERFORMANCE TEST SUITE")
    print("=" * 60)

    # Initialize test suite
    test_suite = VTSCComparativeTest()

    # Run all tests
    results = test_suite.run_all_tests()

    # Save results
    test_suite.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    summary = results['summary']
    if 'error' not in summary:
        print(f"Successful scenarios: {summary['successful_scenarios']}")
        print(f"Failed scenarios: {summary['failed_scenarios']}")

        # Key metrics
        if 'avg_v_turn_diff' in summary['aggregate_metrics']:
            avg_speed_diff = summary['aggregate_metrics']['avg_v_turn_diff']['mean']
            print(f"Average speed difference (Physics - Stock): {avg_speed_diff:.2f} m/s")

        if 'avg_a_target_diff' in summary['aggregate_metrics']:
            avg_accel_diff = summary['aggregate_metrics']['avg_a_target_diff']['mean']
            print(f"Average acceleration difference: {avg_accel_diff:.3f} m/s²")
    else:
        print(f"Test suite failed: {summary['error']}")

    print("Use these results to validate fusion script performance against baselines")

if __name__ == "__main__":
    main()
