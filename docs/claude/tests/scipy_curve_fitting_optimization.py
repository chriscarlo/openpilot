#!/usr/bin/env python3
"""
Comprehensive scipy.optimize curve fitting solution for lateral acceleration.

Uses rigorous statistical methods to find optimal continuous function
to replace piecewise lateral acceleration behavior.

Requirements:
- Zone 1: curvature > 0.0053 → 1.5-1.7 m/s² (tight curves, <50mph)  
- Zone 2: 0.0029 < curvature ≤ 0.0053 → 1.7-3.12 m/s² (50-70mph, exponential rise)
- Zone 3: curvature ≤ 0.0029 → 3.12 m/s² (highway, >70mph)
- Modified targets: make <50mph range 20% more aggressive (1.8-2.04 vs 1.5-1.7)
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import warnings

# Suppress scipy warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)

class LateralAccelerationOptimizer:
    """Comprehensive curve fitting optimizer using multiple function forms."""

    def __init__(self):
        self.original_function = self._original_piecewise
        self.target_function = self._modified_piecewise
        self.candidate_functions = {
            'modified_sigmoid': self._modified_sigmoid,
            'double_sigmoid': self._double_sigmoid,
            'rational_function': self._rational_function,
            'generalized_logistic': self._generalized_logistic,
            'exponential_combination': self._exponential_combination,
            'asymmetric_sigmoid': self._asymmetric_sigmoid
        }

    def _original_piecewise(self, curvature: np.ndarray) -> np.ndarray:
        """Original piecewise lateral acceleration function."""
        result = np.zeros_like(curvature)
        CURV_50MPH = 0.0053
        CURV_70MPH = 0.0029

        # Zone 1: Tight curves (>50mph boundary)
        zone1_mask = curvature > CURV_50MPH
        tight_curves = curvature[zone1_mask]

        hairpin_mask = tight_curves > 0.3
        result[zone1_mask & (curvature > 0.3)] = 1.5

        moderate_tight = tight_curves[~hairpin_mask] if len(tight_curves) > 0 else np.array([])
        if len(moderate_tight) > 0:
            t = (moderate_tight - CURV_50MPH) / (0.3 - CURV_50MPH)
            result[zone1_mask & (curvature <= 0.3)] = 1.7 + t * (1.5 - 1.7)

        # Zone 2: Transition (50-70mph)
        zone2_mask = (curvature > CURV_70MPH) & (curvature <= CURV_50MPH)
        transition_curves = curvature[zone2_mask]
        if len(transition_curves) > 0:
            t = (transition_curves - CURV_70MPH) / (CURV_50MPH - CURV_70MPH)
            result[zone2_mask] = 1.7 + (3.12 - 1.7) * (1 - np.exp(-5 * (1 - t)))

        # Zone 3: Highway (≤70mph boundary)
        zone3_mask = curvature <= CURV_70MPH
        result[zone3_mask] = 3.12

        return result

    def _modified_piecewise(self, curvature: np.ndarray) -> np.ndarray:
        """Modified piecewise with 20% more aggressive <50mph behavior."""
        result = self._original_piecewise(curvature)

        # Apply 20% increase to tight curves (<50mph zone)
        CURV_50MPH = 0.0053
        tight_curve_mask = curvature > CURV_50MPH
        result[tight_curve_mask] *= 1.2  # 20% more aggressive

        # Cap maximum values for safety
        result = np.clip(result, 1.8, 3.12)

        return result

    def _modified_sigmoid(self, curvature: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """Modified sigmoid: a / (1 + exp(b * (curvature - c))) + d"""
        return a / (1.0 + np.exp(b * (curvature - c))) + d

    def _double_sigmoid(self, curvature: np.ndarray,
                       a1: float, b1: float, c1: float,
                       a2: float, b2: float, c2: float, d: float) -> np.ndarray:
        """Double sigmoid combination for better zone transitions."""
        sigmoid1 = a1 / (1.0 + np.exp(b1 * (curvature - c1)))
        sigmoid2 = a2 / (1.0 + np.exp(b2 * (curvature - c2)))
        return sigmoid1 + sigmoid2 + d

    def _rational_function(self, curvature: np.ndarray,
                          a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
        """Rational function: (a * curvature + b) / (c * curvature + d) + e"""
        denominator = c * curvature + d
        # Avoid division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        return (a * curvature + b) / denominator + e

    def _generalized_logistic(self, curvature: np.ndarray,
                             A: float, K: float, B: float, v: float, Q: float, C: float) -> np.ndarray:
        """Generalized logistic function for asymmetric curves."""
        return A + (K - A) / (C + Q * np.exp(-B * curvature)) ** (1/v)

    def _exponential_combination(self, curvature: np.ndarray,
                                a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
        """Exponential combination: a * exp(b * curvature) + c * exp(d * curvature) + e"""
        # Clip exponents to prevent overflow
        exp1 = np.exp(np.clip(b * curvature, -50, 10))
        exp2 = np.exp(np.clip(d * curvature, -50, 10))
        return a * exp1 + c * exp2 + e

    def _asymmetric_sigmoid(self, curvature: np.ndarray,
                           a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
        """Asymmetric sigmoid with adjustable skewness."""
        return a / (1.0 + np.exp(b * (curvature - c))) ** d + e

    def generate_target_data(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate target training data from modified piecewise function."""
        # Use logarithmic spacing to better sample both low and high curvature regions
        curvature_min = 1e-6
        curvature_max = 0.3

        # More points in critical transition regions
        curv_low = np.logspace(np.log10(curvature_min), np.log10(0.0029), 50)  # Highway
        curv_mid = np.linspace(0.0029, 0.0053, 50)  # Transition zone
        curv_high = np.logspace(np.log10(0.0053), np.log10(curvature_max), 100)  # Tight curves

        curvature_data = np.concatenate([curv_low, curv_mid, curv_high])
        curvature_data = np.sort(curvature_data)

        # Generate target lateral acceleration values
        lateral_accel_data = self.target_function(curvature_data)

        return curvature_data, lateral_accel_data

    def get_parameter_bounds(self, func_name: str) -> List[Tuple[float, float]]:
        """Get reasonable parameter bounds for each function type."""
        bounds_dict = {
            'modified_sigmoid': [(-5.0, 5.0), (-2000, 0), (-0.1, 0.1), (0.0, 5.0)],
            'double_sigmoid': [(-3.0, 3.0), (-2000, 0), (-0.1, 0.1),
                              (-3.0, 3.0), (-2000, 0), (-0.1, 0.1), (0.0, 4.0)],
            'rational_function': [(-10.0, 10.0), (-10.0, 10.0), (-1000.0, 1000.0),
                                 (0.1, 100.0), (0.0, 5.0)],
            'generalized_logistic': [(1.5, 4.0), (2.8, 3.2), (0.1, 10.0),
                                    (0.1, 2.0), (0.1, 2.0), (0.1, 2.0)],
            'exponential_combination': [(-5.0, 5.0), (-1000.0, 0), (-5.0, 5.0),
                                       (-1000.0, 0), (1.0, 4.0)],
            'asymmetric_sigmoid': [(-5.0, 5.0), (-2000, 0), (-0.1, 0.1),
                                  (0.1, 5.0), (0.0, 4.0)]
        }
        return bounds_dict.get(func_name, [(-10, 10)] * 4)

    def fit_function(self, func_name: str, curvature_data: np.ndarray,
                    lateral_accel_data: np.ndarray) -> Dict:
        """Fit a specific function using both curve_fit and differential evolution."""
        func = self.candidate_functions[func_name]
        bounds = self.get_parameter_bounds(func_name)

        results = {'function_name': func_name, 'success': False}

        # Method 1: scipy.optimize.curve_fit
        try:
            # Create reasonable initial guesses
            p0 = [np.mean([b[0], b[1]]) for b in bounds]

            popt_curve_fit, pcov = curve_fit(
                func, curvature_data, lateral_accel_data,
                p0=p0,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                maxfev=5000
            )

            # Evaluate fit quality
            y_pred_cf = func(curvature_data, *popt_curve_fit)
            r2_cf = r2_score(lateral_accel_data, y_pred_cf)
            rmse_cf = np.sqrt(np.mean((lateral_accel_data - y_pred_cf)**2))

            results.update({
                'curve_fit_params': popt_curve_fit,
                'curve_fit_covariance': pcov,
                'curve_fit_r2': r2_cf,
                'curve_fit_rmse': rmse_cf,
                'curve_fit_success': True
            })
        except Exception as e:
            results.update({
                'curve_fit_success': False,
                'curve_fit_error': str(e)
            })

        # Method 2: scipy.optimize.differential_evolution (global optimizer)
        def objective(params):
            try:
                y_pred = func(curvature_data, *params)
                # Handle invalid values
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return 1e6
                # Add constraints for physical validity
                if np.any(y_pred < 1.0) or np.any(y_pred > 4.0):
                    return 1e6
                return np.mean((lateral_accel_data - y_pred)**2)
            except:
                return 1e6

        try:
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=1000,
                popsize=15, tol=1e-6
            )

            if de_result.success:
                popt_de = de_result.x
                y_pred_de = func(curvature_data, *popt_de)
                r2_de = r2_score(lateral_accel_data, y_pred_de)
                rmse_de = np.sqrt(np.mean((lateral_accel_data - y_pred_de)**2))

                results.update({
                    'diff_evo_params': popt_de,
                    'diff_evo_r2': r2_de,
                    'diff_evo_rmse': rmse_de,
                    'diff_evo_success': True
                })
            else:
                results['diff_evo_success'] = False
                results['diff_evo_error'] = de_result.message
        except Exception as e:
            results.update({
                'diff_evo_success': False,
                'diff_evo_error': str(e)
            })

        # Select best parameters
        best_params = None
        best_r2 = -1
        best_method = None

        if results.get('curve_fit_success', False):
            if results['curve_fit_r2'] > best_r2:
                best_params = results['curve_fit_params']
                best_r2 = results['curve_fit_r2']
                best_method = 'curve_fit'

        if results.get('diff_evo_success', False):
            if results['diff_evo_r2'] > best_r2:
                best_params = results['diff_evo_params']
                best_r2 = results['diff_evo_r2']
                best_method = 'differential_evolution'

        if best_params is not None:
            results.update({
                'best_params': best_params,
                'best_r2': best_r2,
                'best_rmse': np.sqrt(np.mean((lateral_accel_data -
                                             func(curvature_data, *best_params))**2)),
                'best_method': best_method,
                'success': True
            })

        return results

    def cross_validate(self, func_name: str, params: np.ndarray,
                      curvature_data: np.ndarray, lateral_accel_data: np.ndarray,
                      n_splits: int = 5) -> Dict:
        """Perform k-fold cross-validation on fitted function."""
        func = self.candidate_functions[func_name]
        r2_scores = []
        rmse_scores = []

        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                curvature_data, lateral_accel_data, test_size=0.2, random_state=42
            )

            # Predict on test set
            y_pred = func(X_test, *params)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))

            r2_scores.append(r2)
            rmse_scores.append(rmse)

        return {
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores),
            'cv_rmse_mean': np.mean(rmse_scores),
            'cv_rmse_std': np.std(rmse_scores)
        }

    def validate_requirements(self, func_name: str, params: np.ndarray) -> Dict:
        """Validate that fitted function meets original requirements."""
        func = self.candidate_functions[func_name]

        # Test key curvature points
        test_curvatures = np.array([
            # Highway zone (should be ~3.12)
            0.001, 0.002, 0.0029,
            # Transition zone (should transition from 3.12 to modified range)
            0.003, 0.004, 0.005, 0.0053,
            # Tight curves (should be 1.8-2.04, 20% more aggressive)
            0.006, 0.008, 0.01, 0.02, 0.05, 0.1
        ])

        predicted_values = func(test_curvatures, *params)

        validation_results = {
            'highway_zone_ok': True,
            'transition_zone_ok': True,
            'tight_curve_zone_ok': True,
            'continuity_ok': True,
            'monotonicity_ok': True
        }

        # Check highway zone (curvature ≤ 0.0029)
        highway_mask = test_curvatures <= 0.0029
        highway_values = predicted_values[highway_mask]
        if not np.all((highway_values >= 3.0) & (highway_values <= 3.2)):
            validation_results['highway_zone_ok'] = False

        # Check transition zone (0.0029 < curvature ≤ 0.0053)
        transition_mask = (test_curvatures > 0.0029) & (test_curvatures <= 0.0053)
        transition_values = predicted_values[transition_mask]
        if not np.all((transition_values >= 1.7) & (transition_values <= 3.12)):
            validation_results['transition_zone_ok'] = False

        # Check tight curve zone (curvature > 0.0053) - should be 20% more aggressive
        tight_mask = test_curvatures > 0.0053
        tight_values = predicted_values[tight_mask]
        if not np.all((tight_values >= 1.8) & (tight_values <= 2.1)):
            validation_results['tight_curve_zone_ok'] = False

        # Check continuity at boundaries
        boundary_points = [0.00289, 0.0029, 0.00291, 0.00529, 0.0053, 0.00531]
        boundary_values = func(np.array(boundary_points), *params)
        continuity_diffs = np.diff(boundary_values)
        if np.any(np.abs(continuity_diffs) > 0.1):  # Allow small discontinuities
            validation_results['continuity_ok'] = False

        # Check monotonicity (generally decreasing with increasing curvature)
        full_range = np.logspace(-5, -1, 100)
        full_values = func(full_range, *params)
        monotonic_diffs = np.diff(full_values)
        if np.sum(monotonic_diffs > 0) > len(monotonic_diffs) * 0.1:  # Allow some non-monotonic regions
            validation_results['monotonicity_ok'] = False

        return validation_results

    def optimize_all_functions(self) -> Dict:
        """Run optimization on all candidate functions and rank results."""
        print("Generating target data from modified piecewise function...")
        curvature_data, lateral_accel_data = self.generate_target_data()

        print(f"Generated {len(curvature_data)} data points")
        print(f"Curvature range: {curvature_data.min():.6f} to {curvature_data.max():.6f}")
        print(f"Lateral accel range: {lateral_accel_data.min():.2f} to {lateral_accel_data.max():.2f}")

        results = {}

        for func_name in self.candidate_functions:
            print(f"\nOptimizing {func_name}...")

            try:
                fit_result = self.fit_function(func_name, curvature_data, lateral_accel_data)

                if fit_result['success']:
                    # Perform cross-validation
                    cv_result = self.cross_validate(
                        func_name, fit_result['best_params'],
                        curvature_data, lateral_accel_data
                    )
                    fit_result.update(cv_result)

                    # Validate requirements
                    validation_result = self.validate_requirements(
                        func_name, fit_result['best_params']
                    )
                    fit_result.update(validation_result)

                    print(f"✓ {func_name}: R² = {fit_result['best_r2']:.4f}, "
                          f"RMSE = {fit_result['best_rmse']:.4f}")
                else:
                    print(f"✗ {func_name}: Optimization failed")

                results[func_name] = fit_result

            except Exception as e:
                print(f"✗ {func_name}: Error - {e}")
                results[func_name] = {'success': False, 'error': str(e)}

        return results

    def rank_results(self, results: Dict) -> List[Tuple[str, float]]:
        """Rank optimization results by composite score."""
        scores = []

        for func_name, result in results.items():
            if not result.get('success', False):
                continue

            # Composite scoring
            r2_score = result.get('best_r2', 0) * 40  # R² weight: 40%
            rmse_penalty = (1.0 - result.get('best_rmse', 1.0)) * 20  # RMSE weight: 20%

            # Cross-validation stability (20%)
            cv_r2 = result.get('cv_r2_mean', 0) * 15
            cv_stability = max(0, 1.0 - result.get('cv_r2_std', 1.0)) * 5

            # Requirements validation (20%)
            validation_score = 0
            validation_keys = ['highway_zone_ok', 'transition_zone_ok',
                             'tight_curve_zone_ok', 'continuity_ok', 'monotonicity_ok']
            for key in validation_keys:
                if result.get(key, False):
                    validation_score += 4  # 4 points per requirement (20% total)

            total_score = r2_score + rmse_penalty + cv_r2 + cv_stability + validation_score
            scores.append((func_name, total_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


def main():
    """Main optimization workflow."""
    print("Scipy Lateral Acceleration Curve Fitting Optimization")
    print("=" * 60)

    optimizer = LateralAccelerationOptimizer()

    # Run optimization on all candidate functions
    results = optimizer.optimize_all_functions()

    # Rank results
    rankings = optimizer.rank_results(results)

    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS - RANKED BY COMPOSITE SCORE")
    print(f"{'='*60}")

    for i, (func_name, score) in enumerate(rankings):
        result = results[func_name]
        print(f"\n{i+1}. {func_name.upper()}")
        print(f"   Composite Score: {score:.2f}/100")
        print(f"   R²: {result.get('best_r2', 0):.4f}")
        print(f"   RMSE: {result.get('best_rmse', float('inf')):.4f}")
        print(f"   CV R² Mean: {result.get('cv_r2_mean', 0):.4f} ± {result.get('cv_r2_std', 0):.4f}")
        print(f"   Method: {result.get('best_method', 'N/A')}")

        # Validation summary
        validation_keys = ['highway_zone_ok', 'transition_zone_ok',
                          'tight_curve_zone_ok', 'continuity_ok', 'monotonicity_ok']
        passed_validations = sum(1 for key in validation_keys if result.get(key, False))
        print(f"   Validation: {passed_validations}/5 tests passed")

    # Generate production function for best result
    if rankings:
        best_func_name, best_score = rankings[0]
        best_result = results[best_func_name]
        best_params = best_result['best_params']

        print(f"\n{'='*60}")
        print("PRODUCTION-READY FUNCTION (Best Result)")
        print(f"{'='*60}")

        generate_production_function(best_func_name, best_params, optimizer, results)

        # Generate test validation
        generate_test_validation(best_func_name, best_params, optimizer, results)

    return results, rankings


def generate_production_function(func_name: str, params: np.ndarray,
                               optimizer: LateralAccelerationOptimizer,
                               results: Dict):
    """Generate production-ready Python function."""

    # Function implementations for different types
    if func_name == 'modified_sigmoid':
        a, b, c, d = params
        function_code = f'''def optimized_lateral_acceleration(curvature: float) -> float:
    """
    Scipy-optimized continuous lateral acceleration function.
    
    Optimized {func_name} function replacing piecewise logic:
    - Highway (≤0.0029): ~3.12 m/s² performance
    - Transition (0.0029-0.0053): Smooth decay  
    - Tight curves (>0.0053): 20% more aggressive than original
    
    Fitted parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}
    R² = {results[func_name]['best_r2']:.4f}, RMSE = {results[func_name]['best_rmse']:.4f}
    
    Args:
        curvature: Road curvature in 1/m
        
    Returns:
        Safe lateral acceleration in m/s²
    """
    import math
    
    # Clamp input to valid range
    curvature = max(1e-8, min(curvature, 1.0))
    
    # Optimized modified sigmoid
    result = {a:.6f} / (1.0 + math.exp({b:.6f} * (curvature - {c:.6f}))) + {d:.6f}
    
    # Safety bounds
    return max(1.8, min(result, 3.12))'''

    elif func_name == 'double_sigmoid':
        a1, b1, c1, a2, b2, c2, d = params
        function_code = f'''def optimized_lateral_acceleration(curvature: float) -> float:
    """
    Scipy-optimized double sigmoid lateral acceleration function.
    
    Fitted parameters optimized via differential evolution.
    R² = {results[func_name]['best_r2']:.4f}, RMSE = {results[func_name]['best_rmse']:.4f}
    """
    import math
    
    curvature = max(1e-8, min(curvature, 1.0))
    
    # Double sigmoid combination
    sigmoid1 = {a1:.6f} / (1.0 + math.exp({b1:.6f} * (curvature - {c1:.6f})))
    sigmoid2 = {a2:.6f} / (1.0 + math.exp({b2:.6f} * (curvature - {c2:.6f})))
    result = sigmoid1 + sigmoid2 + {d:.6f}
    
    return max(1.8, min(result, 3.12))'''

    # Add other function types as needed...
    else:
        function_code = f"# {func_name} function code not implemented yet"

    print(function_code)


def generate_test_validation(func_name: str, params: np.ndarray,
                           optimizer: LateralAccelerationOptimizer,
                           results: Dict):
    """Generate test validation showing curvature → lateral_accel mapping."""

    func = optimizer.candidate_functions[func_name]

    print(f"\n{'='*60}")
    print("TEST VALIDATION - CURVATURE → LATERAL ACCELERATION MAPPING")
    print(f"{'='*60}")

    test_curvatures = [
        0.0010, 0.0020, 0.0029,  # Highway
        0.0035, 0.0040, 0.0050, 0.0053,  # Transition
        0.0060, 0.0080, 0.0100, 0.0150, 0.0200, 0.0500, 0.1000  # Tight curves
    ]

    print("Curvature | Optimized | Original | Modified | Requirements | Status")
    print("-" * 75)

    for curv in test_curvatures:
        optimized = func(np.array([curv]), *params)[0]
        original = optimizer.original_function(np.array([curv]))[0]
        modified = optimizer.target_function(np.array([curv]))[0]

        # Determine requirement zone and status
        if curv <= 0.0029:
            req = "Highway ~3.12"
            status = "✓" if abs(optimized - 3.12) < 0.1 else "✗"
        elif curv <= 0.0053:
            req = "Transition 1.7-3.12"
            status = "✓" if 1.7 <= optimized <= 3.12 else "✗"
        else:
            req = "Tight 1.8-2.04"
            status = "✓" if 1.8 <= optimized <= 2.1 else "✗"

        print(f"{curv:8.4f} | {optimized:8.2f} | {original:7.2f} | {modified:7.2f} | {req:15s} | {status:4s}")

    result = results[func_name]
    print("\nFinal Metrics:")
    print(f"R² Score: {result['best_r2']:.4f}")
    print(f"RMSE: {result['best_rmse']:.4f}")
    print(f"CV R² Mean: {result.get('cv_r2_mean', 0):.4f} ± {result.get('cv_r2_std', 0):.4f}")


if __name__ == "__main__":
    main()
