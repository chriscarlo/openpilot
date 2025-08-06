"""
Shared test framework for VTSC unit tests

Provides common utilities, base classes, and helpers for testing VTSC features.
Focuses on logic validation and bug detection capability.
"""

import sys
import os
import math
import time
from dataclasses import dataclass
from abc import ABC

# Add stub modules to sys.modules before importing VTSC
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

# Add path for production VTSC
sys.path.append('/data/openpilot/sunnypilot/selfdrive/controls/lib')

# Import production VTSC
try:
    from vision_turn_controller import VisionTurnController, VisionStatus, EmergencyLevel, VisionOcclusionState
    print("✓ VTSC Test Framework: Successfully imported production VTSC")
except ImportError as e:
    print(f"✗ VTSC Test Framework: Failed to import production VTSC: {e}")
    sys.exit(1)

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
            self.x = list(range(33))
        if self.y is None:
            self.y = [0.0] * 33
        if self.t is None:
            self.t = [i * 0.15 for i in range(33)]  # Time points

@dataclass
class MockModelData:
    """Mock modelV2 data structure"""
    orientationRate: MockVector3 = None
    velocity: MockVector3 = None
    laneLines: list[MockLaneLine] = None
    laneLineProbs: list[float] = None
    laneLineStds: list[float] = None

    def __post_init__(self):
        if self.orientationRate is None:
            self.orientationRate = MockVector3()
        if self.velocity is None:
            self.velocity = MockVector3()
        if self.laneLines is None:
            self.laneLines = [MockLaneLine() for _ in range(4)]
        if self.laneLineProbs is None:
            self.laneLineProbs = [0.8] * 4
        if self.laneLineStds is None:
            self.laneLineStds = [0.1] * 4

class MockSubMaster:
    """Mock SubMaster for testing"""
    def __init__(self):
        self.data = {
            'modelV2': MockModelData(),
            'carState': type('CarState', (), {'gasPressed': False})()
        }
        self.valid = {'modelV2': True}

    def __getitem__(self, key):
        return self.data[key]

class VTSCTestBase(ABC):
    """Base class for VTSC unit tests with common utilities"""

    def setUp(self):
        """Common test setup"""
        self.mock_cp = type('MockCP', (), {})()
        self.vtsc = VisionTurnController(self.mock_cp)
        self.mock_sm = MockSubMaster()
        self.test_start_time = time.time()

    def tearDown(self):
        """Common test cleanup"""

    # Assertion methods to match unittest.TestCase interface
    def assertEqual(self, first, second, msg=None):
        """Assert that two values are equal"""
        if first != second:
            error_msg = f"{first} != {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertNotEqual(self, first, second, msg=None):
        """Assert that two values are not equal"""
        if first == second:
            error_msg = f"{first} == {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertTrue(self, expr, msg=None):
        """Assert that expression is true"""
        if not expr:
            error_msg = f"Expected True, got {expr}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertFalse(self, expr, msg=None):
        """Assert that expression is false"""
        if expr:
            error_msg = f"Expected False, got {expr}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertGreater(self, first, second, msg=None):
        """Assert that first > second"""
        if not (first > second):
            error_msg = f"{first} not greater than {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertLess(self, first, second, msg=None):
        """Assert that first < second"""
        if not (first < second):
            error_msg = f"{first} not less than {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertGreaterEqual(self, first, second, msg=None):
        """Assert that first >= second"""
        if not (first >= second):
            error_msg = f"{first} not greater than or equal to {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertLessEqual(self, first, second, msg=None):
        """Assert that first <= second"""
        if not (first <= second):
            error_msg = f"{first} not less than or equal to {second}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        """Assert that two values are approximately equal"""
        if delta is not None:
            if abs(first - second) > delta:
                error_msg = f"{first} and {second} differ by more than {delta}"
                if msg:
                    error_msg = f"{msg}: {error_msg}"
                raise AssertionError(error_msg)
        else:
            if round(abs(second - first), places) != 0:
                error_msg = f"{first} != {second} within {places} places"
                if msg:
                    error_msg = f"{msg}: {error_msg}"
                raise AssertionError(error_msg)

    def assertIsInstance(self, obj, cls, msg=None):
        """Assert that obj is an instance of cls"""
        if not isinstance(obj, cls):
            error_msg = f"{obj} is not an instance of {cls}"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def subTest(self, **params):
        """Context manager for sub-tests (simplified version)"""
        from contextlib import contextmanager

        @contextmanager
        def subtest_context():
            try:
                yield
            except Exception as e:
                # Add subtest parameters to error message
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                raise AssertionError(f"SubTest failed [{param_str}]: {e}")

        return subtest_context()

    def create_mock_model_data(self, curvature: float = 0.05, vision_confidence: float = 0.9) -> MockModelData:
        """Create mock model data with specified parameters"""
        model_data = MockModelData()

        # Set curvature-based orientation rate
        velocity = 25.0  # Define velocity first
        orientation_rate = curvature * velocity  # orientation_rate = curvature * velocity
        model_data.orientationRate.z = [orientation_rate] * 33

        # Set velocity using the same value
        model_data.velocity.x = [velocity] * 33

        # Set lane line probabilities based on vision confidence
        confidence_per_line = max(0.0, min(1.0, vision_confidence))
        model_data.laneLineProbs = [confidence_per_line] * 4

        # Standard deviation inversely related to confidence
        std_dev = max(0.1, 1.0 - vision_confidence)
        model_data.laneLineStds = [std_dev] * 4

        return model_data

    def update_vtsc(self, curvature: float = 0.05, vision_confidence: float = 0.9,
                   v_ego: float = 25.0, v_cruise: float = 30.0, enabled: bool = True) -> None:
        """Update VTSC with specified parameters"""
        mock_model_data = self.create_mock_model_data(curvature, vision_confidence)
        self.mock_sm.data['modelV2'] = mock_model_data

        self.vtsc.update(
            sm=self.mock_sm,
            enabled=enabled,
            v_ego=v_ego,
            a_ego=0.0,
            v_cruise_setpoint=v_cruise
        )

    def assert_vision_status(self, expected_status: VisionStatus, message: str = ""):
        """Assert vision status matches expected value"""
        actual = self.vtsc._occlusion_state.vision_status
        if actual != expected_status:
            raise AssertionError(f"Vision status mismatch{': ' + message if message else ''}. "
                               f"Expected: {expected_status.name}, Actual: {actual.name}")

    def assert_emergency_level(self, expected_level: EmergencyLevel, message: str = ""):
        """Assert emergency level matches expected value"""
        actual = self.vtsc.emergency_level
        if actual != expected_level:
            raise AssertionError(f"Emergency level mismatch{': ' + message if message else ''}. "
                               f"Expected: {expected_level.name}, Actual: {actual.name}")

    def assert_approximately_equal(self, actual: float, expected: float, tolerance: float = 0.001, message: str = ""):
        """Assert two floats are approximately equal"""
        if abs(actual - expected) > tolerance:
            raise AssertionError(f"Values not approximately equal{': ' + message if message else ''}. "
                               f"Expected: {expected:.6f}, Actual: {actual:.6f}, Tolerance: {tolerance}")

    def assert_occlusion_start_time_updated(self, previous_time: float, message: str = ""):
        """Assert occlusion start time was updated from previous value"""
        current_time = self.vtsc._occlusion_state.occlusion_start_time
        if current_time == previous_time:
            raise AssertionError(f"Occlusion start time not updated{': ' + message if message else ''}. "
                               f"Previous: {previous_time}, Current: {current_time}")

    def assert_confidence_decay_in_range(self, min_decay: float, max_decay: float, message: str = ""):
        """Assert confidence decay factor is within expected range"""
        actual = self.vtsc._occlusion_state.confidence_decay_factor
        if not (min_decay <= actual <= max_decay):
            raise AssertionError(f"Confidence decay out of range{': ' + message if message else ''}. "
                               f"Expected range: [{min_decay:.3f}, {max_decay:.3f}], Actual: {actual:.3f}")

class TestResult:
    """Test result tracking"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message = ""
        self.execution_time = 0.0

    def mark_passed(self, execution_time: float):
        self.passed = True
        self.execution_time = execution_time

    def mark_failed(self, error_message: str, execution_time: float):
        self.passed = False
        self.error_message = error_message
        self.execution_time = execution_time

class VTSCTestRunner:
    """Test runner for VTSC unit tests"""

    def __init__(self):
        self.results: list[TestResult] = []

    def run_test(self, test_class: type, test_method_name: str) -> TestResult:
        """Run a single test method"""
        result = TestResult(f"{test_class.__name__}.{test_method_name}")
        start_time = time.time()

        try:
            # Create test instance and run
            test_instance = test_class()
            test_instance.setUp()

            # Get and run the test method
            test_method = getattr(test_instance, test_method_name)
            test_method()

            test_instance.tearDown()

            execution_time = time.time() - start_time
            result.mark_passed(execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            result.mark_failed(str(e), execution_time)

        self.results.append(result)
        return result

    def print_results(self):
        """Print test results summary"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\n{'='*80}")
        print("VTSC UNIT TEST RESULTS")
        print(f"{'='*80}")
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")

        if total > 0:
            print("\nDetailed Results:")
            for result in self.results:
                status = "PASS" if result.passed else "FAIL"
                print(f"  {status:4} | {result.test_name:50} | {result.execution_time:.3f}s")
                if not result.passed:
                    print(f"       | Error: {result.error_message}")

        print(f"{'='*80}")

        return passed == total

def calculate_expected_confidence_decay(occlusion_duration: float) -> float:
    """Calculate expected confidence decay factor for given occlusion duration"""
    return max(0.3, math.exp(-occlusion_duration / 3.0))

def calculate_vision_confidence_from_lane_probs(lane_line_probs: list[float]) -> float:
    """Calculate effective vision confidence from lane line probabilities"""
    if not lane_line_probs:
        return 0.0
    return sum(lane_line_probs) / len(lane_line_probs)
