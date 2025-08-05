#!/usr/bin/env python3
"""
Dynamic Timing System Optimizer for VTSC
Comprehensive synthetic tuning framework to optimize anticipation timing parameters
"""

import json
import math
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Try to import numpy, fallback to math if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Using math library fallback")
    
    # Create numpy-like functions using math
    class np:
        @staticmethod
        def clip(value, min_val, max_val):
            return max(min_val, min(value, max_val))
        
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
        
        @staticmethod
        def linspace(start, stop, num):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        
        @staticmethod
        def array(values):
            return list(values)

# Optional optimization library (install with: pip install optuna)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Install with 'pip install optuna' for Bayesian optimization")

class DrivingContext(Enum):
    """Driving context categories affecting timing expectations."""
    PARKING = "parking"          # 5-15 mph, tight spaces, micro-adjustments
    RESIDENTIAL = "residential"  # 15-35 mph, safety-first, moderate comfort  
    URBAN = "urban"             # 25-50 mph, efficiency-focused but smooth
    HIGHWAY = "highway"         # 50-85+ mph, comfort + safety critical

@dataclass
class TestScenario:
    """Individual test scenario for timing optimization."""
    current_speed_mph: float
    target_speed_mph: float
    context: DrivingContext
    max_lat_acc: float  # m/s²
    expected_timing_range: Tuple[float, float]  # (min_seconds, max_seconds)
    
    def to_ms(self, speed_mph: float) -> float:
        """Convert mph to m/s."""
        return speed_mph * 0.44704
    
    @property
    def current_speed_ms(self) -> float:
        return self.to_ms(self.current_speed_mph)
    
    @property 
    def target_speed_ms(self) -> float:
        return self.to_ms(self.target_speed_mph)

class CurrentVTSCTiming:
    """Current VTSC timing algorithm for comparison."""
    
    @staticmethod
    def calculate_anticipation_time(v_ego_ms: float, target_speed_ms: float, max_pred_lat_acc: float) -> float:
        """Current VTSC algorithm - extracted from vision_turn_controller.py"""
        # Base anticipation time
        base_time = 1.5  # Base 1.5 seconds early

        # Speed factor: Higher speeds need more anticipation
        # Normalize around 20 m/s (~45 mph)
        speed_factor = np.clip(v_ego_ms / 20.0, 0.7, 1.5)

        # Speed reduction factor: Larger speed changes need more anticipation
        if v_ego_ms > 0.1:  # Avoid division by zero
            delta_ratio = (v_ego_ms - target_speed_ms) / v_ego_ms
            delta_factor = np.clip(1.0 + delta_ratio * 0.5, 1.0, 1.5)
        else:
            delta_factor = 1.0

        # Curve severity factor: Sharper curves need more anticipation
        # Normalize around 1.5 m/s² lateral acceleration
        severity_factor = np.clip(max_pred_lat_acc / 1.5, 0.8, 1.3)

        # Calculate total anticipation time
        anticipation_time = base_time * speed_factor * delta_factor * severity_factor

        # Clip to reasonable range (1-3 seconds)
        return np.clip(anticipation_time, 1.0, 3.0)

class DynamicTimingCalculator:
    """Optimizable dynamic timing calculator."""
    
    def __init__(self, params: Dict[str, float]):
        """Initialize with parameter dictionary."""
        self.reaction_time_base = params.get('reaction_time_base', 0.8)
        self.speed_normalization = params.get('speed_normalization', 20.0)  # m/s
        self.speed_factor_min = params.get('speed_factor_min', 0.7)
        self.speed_factor_max = params.get('speed_factor_max', 1.5)
        self.delta_factor_gain = params.get('delta_factor_gain', 0.5)
        self.delta_factor_max = params.get('delta_factor_max', 1.5)
        self.severity_normalization = params.get('severity_normalization', 1.5)  # m/s²
        self.severity_factor_min = params.get('severity_factor_min', 0.8)
        self.severity_factor_max = params.get('severity_factor_max', 1.3)
        self.timing_min = params.get('timing_min', 0.4)
        self.timing_max = params.get('timing_max', 6.0)
        
        # Context multipliers
        self.context_multipliers = {
            DrivingContext.PARKING: params.get('parking_multiplier', 0.8),
            DrivingContext.RESIDENTIAL: params.get('residential_multiplier', 1.0),
            DrivingContext.URBAN: params.get('urban_multiplier', 1.1),
            DrivingContext.HIGHWAY: params.get('highway_multiplier', 1.2)
        }
    
    def calculate_timing(self, v_ego_ms: float, target_speed_ms: float, 
                        max_pred_lat_acc: float, context: DrivingContext) -> float:
        """Calculate dynamic timing with optimizable parameters."""
        
        # Speed factor: Higher speeds need more anticipation
        speed_factor = np.clip(v_ego_ms / self.speed_normalization, 
                              self.speed_factor_min, self.speed_factor_max)

        # Speed reduction factor: Larger speed changes need more anticipation
        if v_ego_ms > 0.1:  # Avoid division by zero
            delta_ratio = (v_ego_ms - target_speed_ms) / v_ego_ms
            delta_factor = np.clip(1.0 + delta_ratio * self.delta_factor_gain, 
                                  1.0, self.delta_factor_max)
        else:
            delta_factor = 1.0

        # Curve severity factor: Sharper curves need more anticipation
        severity_factor = np.clip(max_pred_lat_acc / self.severity_normalization, 
                                 self.severity_factor_min, self.severity_factor_max)

        # Calculate base timing
        base_timing = self.reaction_time_base * speed_factor * delta_factor * severity_factor
        
        # Apply context multiplier
        context_multiplier = self.context_multipliers[context]
        timing = base_timing * context_multiplier
        
        # Apply bounds
        return np.clip(timing, self.timing_min, self.timing_max)

class ScenarioGenerator:
    """Generate comprehensive test scenarios for optimization."""
    
    @staticmethod
    def generate_scenario_matrix() -> List[TestScenario]:
        """Generate comprehensive scenario matrix (~1,500 realistic test cases)."""
        scenarios = []
        
        # Define speed ranges and contexts
        speed_points = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        
        # Context-specific speed ranges and expectations  
        context_configs = {
            DrivingContext.PARKING: {
                'speed_range': [5, 15],
                'typical_changes': [-2, -5, -8],  # Small speed reductions
                'expected_timing': (0.5, 1.5),   # Quick reactions
                'lat_acc_range': (0.5, 1.2)      # Low lateral acceleration
            },
            DrivingContext.RESIDENTIAL: {
                'speed_range': [15, 35], 
                'typical_changes': [-5, -10, -15],
                'expected_timing': (1.0, 2.5),
                'lat_acc_range': (0.8, 1.8)
            },
            DrivingContext.URBAN: {
                'speed_range': [25, 50],
                'typical_changes': [-10, -15, -20],
                'expected_timing': (1.5, 3.5),
                'lat_acc_range': (1.0, 2.5)
            },
            DrivingContext.HIGHWAY: {
                'speed_range': [50, 85],
                'typical_changes': [-15, -25, -35],
                'expected_timing': (2.5, 6.0),
                'lat_acc_range': (1.2, 3.0)
            }
        }
        
        # Generate scenarios for each context
        for context, config in context_configs.items():
            min_speed, max_speed = config['speed_range']
            current_speeds = [s for s in speed_points if min_speed <= s <= max_speed]
            
            for current_speed in current_speeds:
                for speed_change in config['typical_changes']:
                    target_speed = max(5, current_speed + speed_change)  # Don't go below 5 mph
                    
                    # Only include realistic scenarios
                    if target_speed >= min_speed and target_speed <= current_speed:
                        # Generate multiple lateral acceleration values
                        lat_acc_min, lat_acc_max = config['lat_acc_range']
                        lat_accs = np.linspace(lat_acc_min, lat_acc_max, 3)
                        
                        for lat_acc in lat_accs:
                            scenario = TestScenario(
                                current_speed_mph=current_speed,
                                target_speed_mph=target_speed,
                                context=context,
                                max_lat_acc=lat_acc,
                                expected_timing_range=config['expected_timing']
                            )
                            scenarios.append(scenario)
        
        print(f"Generated {len(scenarios)} test scenarios")
        return scenarios

class ObjectiveFunction:
    """Multi-criteria objective function for parameter optimization."""
    
    def __init__(self):
        """Initialize with research-based criteria."""
        # Weights for composite scoring (sum to 1.0)
        self.weights = {
            'safety': 0.40,          # Safety constraints compliance
            'human_factors': 0.30,   # Alignment with research expectations
            'naturalness': 0.20,     # Smooth, intuitive behavior
            'efficiency': 0.05,      # Not wastefully early/late
            'robustness': 0.05       # Consistent across similar scenarios
        }
        
        # Research-based limits
        self.max_comfortable_decel = 2.9  # m/s² (0.295g from research)
        self.max_jerk = 3.0              # m/s³ (comfort limit)
        self.min_reaction_time = 0.5     # seconds (research minimum)
    
    def evaluate_timing_quality(self, params: Dict[str, float], scenarios: List[TestScenario]) -> float:
        """Evaluate parameter set against all test scenarios."""
        calculator = DynamicTimingCalculator(params)
        
        # Calculate timing for all scenarios
        results = []
        for scenario in scenarios:
            timing = calculator.calculate_timing(
                scenario.current_speed_ms,
                scenario.target_speed_ms, 
                scenario.max_lat_acc,
                scenario.context
            )
            results.append({
                'scenario': scenario,
                'timing': timing,
                'speed_delta': scenario.current_speed_ms - scenario.target_speed_ms,
                'decel_required': (scenario.current_speed_ms - scenario.target_speed_ms) / timing if timing > 0 else 0
            })
        
        # Calculate individual scores
        safety_score = self._calculate_safety_score(results)
        human_factors_score = self._calculate_human_factors_score(results)
        naturalness_score = self._calculate_naturalness_score(results)
        efficiency_score = self._calculate_efficiency_score(results)
        robustness_score = self._calculate_robustness_score(results)
        
        # Composite score
        composite = (
            self.weights['safety'] * safety_score +
            self.weights['human_factors'] * human_factors_score +
            self.weights['naturalness'] * naturalness_score +
            self.weights['efficiency'] * efficiency_score +  
            self.weights['robustness'] * robustness_score
        )
        
        return composite
    
    def _calculate_safety_score(self, results: List[Dict]) -> float:
        """Safety constraints: deceleration, jerk, reaction time."""
        violations = 0
        total = len(results)
        
        for result in results:
            timing = result['timing']
            decel_required = abs(result['decel_required'])
            
            # Check safety violations
            if decel_required > self.max_comfortable_decel:
                violations += 1
            if timing < self.min_reaction_time:
                violations += 1
                
        # Perfect score = 1.0, each violation reduces score
        return max(0.0, 1.0 - (violations / total))
    
    def _calculate_human_factors_score(self, results: List[Dict]) -> float:
        """Alignment with research-based timing expectations."""
        score = 0.0
        total = len(results)
        
        for result in results:
            scenario = result['scenario']
            timing = result['timing']
            expected_min, expected_max = scenario.expected_timing_range
            
            if expected_min <= timing <= expected_max:
                # Perfect alignment
                score += 1.0
            else:
                # Exponential penalty for deviation
                if timing < expected_min:
                    deviation = expected_min - timing
                else:
                    deviation = timing - expected_max
                
                # Penalty: starts at 1.0, decays exponentially
                penalty = max(0.0, 1.0 - (deviation ** 2))
                score += penalty
                
        return score / total
    
    def _calculate_naturalness_score(self, results: List[Dict]) -> float:
        """Smooth, intuitive scaling behavior."""
        # Group by context and check for smoothness
        context_groups = {}
        for result in results:
            context = result['scenario'].context
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(result)
        
        smoothness_scores = []
        for context, group in context_groups.items():
            # Sort by current speed
            group.sort(key=lambda x: x['scenario'].current_speed_ms)
            
            # Calculate smoothness (avoid sudden jumps)
            if len(group) > 1:
                timing_diffs = []
                for i in range(1, len(group)):
                    curr_timing = group[i]['timing']
                    prev_timing = group[i-1]['timing']
                    timing_diff = abs(curr_timing - prev_timing)
                    timing_diffs.append(timing_diff)
                
                # Penalize large jumps
                avg_diff = np.mean(timing_diffs)
                smoothness = max(0.0, 1.0 - (avg_diff ** 2))
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 1.0
    
    def _calculate_efficiency_score(self, results: List[Dict]) -> float:
        """Not wastefully early or dangerously late."""
        # Simple heuristic: penalize extreme timing values
        efficiency_scores = []
        
        for result in results:
            timing = result['timing']
            scenario = result['scenario']
            speed_change = scenario.current_speed_mph - scenario.target_speed_mph
            
            # Heuristic: timing should scale reasonably with speed change
            expected_base = 1.0 + (speed_change / 30.0)  # Rough heuristic
            
            ratio = timing / max(expected_base, 0.5)
            if 0.7 <= ratio <= 1.5:  # Reasonable range
                efficiency_scores.append(1.0)
            else:
                # Penalize extremes
                penalty = max(0.0, 1.0 - abs(ratio - 1.0))
                efficiency_scores.append(penalty)
        
        return np.mean(efficiency_scores)
    
    def _calculate_robustness_score(self, results: List[Dict]) -> float:
        """Consistent behavior across similar scenarios."""
        # Group similar scenarios and check for consistency
        similar_groups = self._group_similar_scenarios(results)
        
        consistency_scores = []
        for group in similar_groups:
            if len(group) > 1:
                timings = [r['timing'] for r in group]
                std_dev = np.std(timings)
                mean_timing = np.mean(timings)
                
                # Coefficient of variation (normalized std dev)
                cv = std_dev / max(mean_timing, 0.1)
                consistency = max(0.0, 1.0 - cv)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _group_similar_scenarios(self, results: List[Dict]) -> List[List[Dict]]:
        """Group scenarios with similar characteristics."""
        groups = []
        tolerance = 5.0  # mph
        
        # Simple grouping by current speed ranges
        speed_ranges = [(5, 20), (20, 35), (35, 50), (50, 70), (70, 85)]
        
        for speed_min, speed_max in speed_ranges:
            group = []
            for result in results:
                current_mph = result['scenario'].current_speed_mph
                if speed_min <= current_mph < speed_max:
                    group.append(result)
            if len(group) > 1:
                groups.append(group)
        
        return groups

class TimingOptimizer:
    """Bayesian optimization of timing parameters."""
    
    def __init__(self, scenarios: List[TestScenario]):
        self.scenarios = scenarios
        self.objective_func = ObjectiveFunction()
        self.current_vtsc = CurrentVTSCTiming()
        
    def optimize_parameters(self, n_trials: int = 1000) -> Dict[str, Any]:
        """Run Bayesian optimization to find optimal parameters."""
        if not OPTUNA_AVAILABLE:
            print("ERROR: optuna not available. Running grid search instead.")
            return self._run_grid_search()
        
        def objective(trial):
            params = {
                'reaction_time_base': trial.suggest_float('reaction_time_base', 0.5, 1.2),
                'speed_normalization': trial.suggest_float('speed_normalization', 15.0, 30.0),
                'speed_factor_min': trial.suggest_float('speed_factor_min', 0.5, 0.8),
                'speed_factor_max': trial.suggest_float('speed_factor_max', 1.2, 2.0),
                'delta_factor_gain': trial.suggest_float('delta_factor_gain', 0.3, 0.8),
                'delta_factor_max': trial.suggest_float('delta_factor_max', 1.2, 2.0),
                'severity_normalization': trial.suggest_float('severity_normalization', 1.0, 2.5),
                'severity_factor_min': trial.suggest_float('severity_factor_min', 0.6, 1.0),
                'severity_factor_max': trial.suggest_float('severity_factor_max', 1.0, 1.8),
                'timing_min': trial.suggest_float('timing_min', 0.3, 0.6),
                'timing_max': trial.suggest_float('timing_max', 4.0, 8.0),
                'parking_multiplier': trial.suggest_float('parking_multiplier', 0.6, 1.0),
                'residential_multiplier': trial.suggest_float('residential_multiplier', 0.8, 1.2),
                'urban_multiplier': trial.suggest_float('urban_multiplier', 0.9, 1.3),
                'highway_multiplier': trial.suggest_float('highway_multiplier', 1.0, 1.5)
            }
            
            return self.objective_func.evaluate_timing_quality(params, self.scenarios)
        
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Optimization complete. Best score: {best_score:.4f}")
        
        # Evaluate current VTSC for comparison
        current_score = self._evaluate_current_vtsc()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'current_vtsc_score': current_score,
            'improvement': best_score - current_score,
            'study': study
        }
    
    def _evaluate_current_vtsc(self) -> float:
        """Evaluate current VTSC algorithm performance."""
        results = []
        for scenario in self.scenarios:
            timing = self.current_vtsc.calculate_anticipation_time(
                scenario.current_speed_ms,
                scenario.target_speed_ms,
                scenario.max_lat_acc
            )
            results.append({
                'scenario': scenario,
                'timing': timing,
                'speed_delta': scenario.current_speed_ms - scenario.target_speed_ms,
                'decel_required': (scenario.current_speed_ms - scenario.target_speed_ms) / timing if timing > 0 else 0
            })
        
        # Calculate scores using same objective function
        safety_score = self.objective_func._calculate_safety_score(results)
        human_factors_score = self.objective_func._calculate_human_factors_score(results)
        naturalness_score = self.objective_func._calculate_naturalness_score(results)
        efficiency_score = self.objective_func._calculate_efficiency_score(results)
        robustness_score = self.objective_func._calculate_robustness_score(results)
        
        composite = (
            self.objective_func.weights['safety'] * safety_score +
            self.objective_func.weights['human_factors'] * human_factors_score +
            self.objective_func.weights['naturalness'] * naturalness_score +
            self.objective_func.weights['efficiency'] * efficiency_score +  
            self.objective_func.weights['robustness'] * robustness_score
        )
        
        return composite
    
    def _run_grid_search(self) -> Dict[str, Any]:
        """Simple grid search fallback when optuna not available."""
        print("Running simple grid search...")
        
        # Define a small grid of parameters to test
        grid = {
            'reaction_time_base': [0.6, 0.8, 1.0],
            'speed_normalization': [18.0, 20.0, 22.0],
            'delta_factor_gain': [0.4, 0.5, 0.6],
            'timing_min': [0.4, 0.5, 0.6],
            'timing_max': [4.0, 5.0, 6.0]
        }
        
        best_score = -1.0
        best_params = None
        
        # Test combinations (limited to avoid explosion)
        import itertools
        keys = list(grid.keys())
        for values in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, values))
            
            # Fill in defaults for other parameters
            default_params = {
                'speed_factor_min': 0.7,
                'speed_factor_max': 1.5,
                'delta_factor_max': 1.5,
                'severity_normalization': 1.5,
                'severity_factor_min': 0.8,
                'severity_factor_max': 1.3,
                'parking_multiplier': 0.8,
                'residential_multiplier': 1.0,
                'urban_multiplier': 1.1,
                'highway_multiplier': 1.2
            }
            params.update(default_params)
            
            score = self.objective_func.evaluate_timing_quality(params, self.scenarios)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        current_score = self._evaluate_current_vtsc()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'current_vtsc_score': current_score,
            'improvement': best_score - current_score,
            'study': None
        }

def main():
    """Main optimization workflow."""
    print("=== VTSC Dynamic Timing Optimization ===")
    
    # Generate test scenarios
    print("\n1. Generating comprehensive test scenarios...")
    scenarios = ScenarioGenerator.generate_scenario_matrix()
    
    # Initialize optimizer
    print("\n2. Initializing optimizer...")
    optimizer = TimingOptimizer(scenarios)
    
    # Run optimization
    print("\n3. Running parameter optimization...")
    results = optimizer.optimize_parameters(n_trials=500)  # Reduced for speed
    
    # Display results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Current VTSC Score: {results['current_vtsc_score']:.4f}")
    print(f"Optimized Score: {results['best_score']:.4f}")
    print(f"Improvement: {results['improvement']:.4f} ({results['improvement']/results['current_vtsc_score']*100:.1f}%)")
    
    print("\nOptimal Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value:.3f}")
    
    # Save results
    print("\n4. Saving results...")
    with open('docs/claude/tests/vtsc/optimization_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'current_vtsc_score': results['current_vtsc_score'],
            'improvement': results['improvement'],
            'timestamp': time.time(),
            'num_scenarios': len(scenarios)
        }
        json.dump(json_results, f, indent=2)
    
    print("Optimization complete! Results saved to optimization_results.json")
    
    return results

if __name__ == "__main__":
    main()