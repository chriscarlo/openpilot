#!/usr/bin/env python3
"""A/B comparison: EMA prediction-corrector vs Kalman filter for lead dRel.

Runs both filters side-by-side on identical noisy synthetic lead scenarios
and prints a metrics table.  Use ``pytest -s`` to see the comparison output.

Noise model derived from real EV6 AI-model lead captures:
  - Base σ ≈ 1.5 m (Gaussian)
  - Occasional ±3–5 m outlier jumps (~5 % of frames)
  - Noise roughly proportional to sqrt(distance)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
import pytest

# Filters under test
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LeadDistanceFilter
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.lead_kalman_filter import LeadKalmanFilter

# ---------------------------------------------------------------------------
# Noise model (fitted to real EV6 captures)
# ---------------------------------------------------------------------------
# Noise model calibrated against real EV6 AI-model captures (2026-04-08).
# Real frame-to-frame |Δd| distribution:
#   ≤1m: 53%  1-5m: 39%  5-12m: 7%  >12m: 2%
# Real noise by distance band:
#   30-50m: σ≈0.7m   50-80m: σ≈2.5m   >80m: σ≈7m
OUTLIER_EXTRA_SIGMA_M = 8.0   # produces 10-20 m outliers matching real tail
DT_S = 0.05                   # 20 Hz model output rate


def _noise_sigma(true_drel: float) -> float:
  """Distance-dependent base noise σ matched to real EV6 captures.

  Calibrated against real frame-to-frame |Δd| at each distance band:
    30-50m: real mean|Δd|=0.73m → σ≈0.5m
    50-80m: real mean|Δd|=2.36m → σ≈2.0m
    >80m:   real mean|Δd|=7.73m → σ≈5.0m
  """
  d = max(true_drel, 5.0)
  if d < 40.0:
    return 0.5
  elif d < 80.0:
    return 0.5 + (d - 40.0) * 0.04  # 0.5 at 40 m → 2.1 at 80 m
  else:
    return 2.1 + (d - 80.0) * 0.15  # 2.1 at 80 m → 5.1 at 100 m


def _outlier_prob(true_drel: float) -> float:
  """Distance-dependent outlier probability.  Close = rare, far = frequent."""
  d = max(true_drel, 5.0)
  if d < 40.0:
    return 0.03
  elif d < 80.0:
    return 0.03 + (d - 40.0) * 0.001  # 3% at 40m → 7% at 80m
  else:
    return 0.07 + (d - 80.0) * 0.004  # 7% at 80m → 15% at 100m


def _noisy_drel(true_drel: float, rng: np.random.Generator) -> tuple[float, float]:
  """Add realistic AI-model noise and generate a simulated xStd.

  Returns (noisy_drel, x_std).
  """
  sigma = _noise_sigma(true_drel)
  is_outlier = rng.random() < _outlier_prob(true_drel)
  if is_outlier:
    sigma += OUTLIER_EXTRA_SIGMA_M
  noise = float(rng.normal(0.0, sigma))
  raw = true_drel + noise
  x_std = float(max(0.3, sigma * (1.0 + rng.normal(0.0, 0.3))))
  return float(raw), x_std


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
@dataclass
class ScenarioStep:
  true_drel: float
  true_vrel: float   # positive = opening, negative = closing
  raw_drel: float     # noisy measurement
  raw_vrel: float     # noisy vRel (mild noise)
  x_std: float = 1.5  # model's own uncertainty estimate for this detection


@dataclass
class Scenario:
  name: str
  steps: list[ScenarioStep]


def _build_scenario(name: str, duration_s: float,
                    d0: float, v0: float,
                    accel_profile: list[tuple[float, float]],
                    rng: np.random.Generator) -> Scenario:
  """Generate a scenario with piecewise-constant lead acceleration.

  accel_profile: list of (until_t, a_rel) pairs.
    a_rel positive = lead pulling away, negative = closing.
  """
  n = int(duration_s / DT_S)
  steps: list[ScenarioStep] = []
  d, v = float(d0), float(v0)
  profile_idx = 0
  for i in range(n):
    t = i * DT_S
    # pick acceleration for this time
    while profile_idx < len(accel_profile) - 1 and t >= accel_profile[profile_idx + 1][0]:
      profile_idx += 1
    a_rel = accel_profile[profile_idx][1]

    # integrate ground truth
    v += a_rel * DT_S
    d += v * DT_S
    d = max(d, 1.0)  # can't go through the lead

    raw_d, x_std = _noisy_drel(d, rng)
    raw_v = float(v + rng.normal(0.0, 0.3))  # mild vRel noise
    steps.append(ScenarioStep(true_drel=d, true_vrel=v, raw_drel=raw_d, raw_vrel=raw_v, x_std=x_std))
  return Scenario(name=name, steps=steps)


def _make_scenarios(seed: int = 42) -> list[Scenario]:
  rng = np.random.default_rng(seed)
  scenarios = []

  # 1. Steady follow — constant gap, pure noise rejection test
  scenarios.append(_build_scenario(
    "steady_follow_30m", 10.0, d0=30.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng))

  # 2. Steady follow far — 60 m highway, noisier measurements
  scenarios.append(_build_scenario(
    "steady_follow_60m", 10.0, d0=60.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng))

  # 3. Gentle accel away — lead pulls away at 0.5 m/s²
  scenarios.append(_build_scenario(
    "gentle_accel_away", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.5)], rng=rng))

  # 4. Gentle decel — lead brakes gently at -0.5 m/s²
  scenarios.append(_build_scenario(
    "gentle_decel", 8.0, d0=40.0, v0=0.0,
    accel_profile=[(0.0, -0.5)], rng=rng))

  # 5. Hard decel — lead brakes hard at -3.0 m/s² for 2s then holds
  scenarios.append(_build_scenario(
    "hard_decel", 6.0, d0=45.0, v0=0.0,
    accel_profile=[(0.0, -3.0), (2.0, 0.0)], rng=rng))

  # 6. Decel-to-accel transition — brakes then accelerates (the sign transition case)
  scenarios.append(_build_scenario(
    "decel_then_accel", 10.0, d0=40.0, v0=0.0,
    accel_profile=[(0.0, -1.0), (3.0, 0.0), (4.0, 1.0)], rng=rng))

  # 7. Stop and go — decel to near-stop, wait, then pull away
  scenarios.append(_build_scenario(
    "stop_and_go", 15.0, d0=25.0, v0=0.0,
    accel_profile=[(0.0, -1.5), (3.0, 0.0), (6.0, 0.0), (8.0, 1.0), (12.0, 0.0)], rng=rng))

  # 8. Cut-in — lead suddenly appears close (simulated by jump in dRel)
  base = _build_scenario(
    "cut_in", 10.0, d0=80.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  # At t=3s, lead "appears" at 20 m
  cut_frame = int(3.0 / DT_S)
  for i in range(cut_frame, len(base.steps)):
    _rd, _xs = _noisy_drel(20.0, rng)
    base.steps[i] = ScenarioStep(
      true_drel=20.0, true_vrel=-0.5,
      raw_drel=_rd, raw_vrel=float(-0.5 + rng.normal(0.0, 0.3)), x_std=_xs)
  scenarios.append(base)

  # 9. Jumpy lead — extra-noisy model output (bad lighting, occlusion)
  jumpy_rng = np.random.default_rng(seed + 99)
  jumpy = _build_scenario(
    "jumpy_lead", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=jumpy_rng)
  # Double the noise on every measurement
  for s in jumpy.steps:
    extra = float(jumpy_rng.normal(0.0, 3.0))
    s.raw_drel = s.true_drel + (s.raw_drel - s.true_drel) * 2.0 + extra
  scenarios.append(jumpy)

  # 10. Lead dropout and reappear — 1 s gap in measurements
  dropout = _build_scenario(
    "dropout_1s", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  drop_start = int(4.0 / DT_S)
  drop_end = int(5.0 / DT_S)
  # During dropout, raw_drel jumps wildly (simulates no-lead then reacquire)
  for i in range(drop_start, min(drop_end, len(dropout.steps))):
    dropout.steps[i].raw_drel = float(dropout.steps[i].true_drel + rng.normal(0.0, 15.0))
  scenarios.append(dropout)

  # --- ADVERSARIAL / EDGE-CASE SCENARIOS ---

  # 11. Single garbage spike to 0 m (the exact failure mode from real trace)
  spike_zero = _build_scenario(
    "spike_to_zero", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  spike_zero.steps[int(3.0 / DT_S)].raw_drel = 0.0  # single garbage frame
  scenarios.append(spike_zero)

  # 12. Two consecutive garbage spikes (tests confirm=2 boundary)
  spike_double = _build_scenario(
    "double_spike_to_0", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  idx = int(3.0 / DT_S)
  spike_double.steps[idx].raw_drel = 0.0
  spike_double.steps[idx + 1].raw_drel = 0.0
  scenarios.append(spike_double)

  # 13. Three consecutive garbage spikes (exceeds confirm=2 — snap fires)
  spike_triple = _build_scenario(
    "triple_spike_to_0", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  idx = int(3.0 / DT_S)
  for j in range(3):
    spike_triple.steps[idx + j].raw_drel = 0.0
  scenarios.append(spike_triple)

  # 14. Oscillating extreme noise (alternating +20m / -20m from truth)
  osc = _build_scenario(
    "oscillating_extreme", 10.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  for i, s in enumerate(osc.steps):
    s.raw_drel = s.true_drel + (20.0 if i % 2 == 0 else -20.0)
  scenarios.append(osc)

  # 15. Real hard braking that MUST be tracked (not rejected)
  scenarios.append(_build_scenario(
    "emergency_brake", 6.0, d0=50.0, v0=0.0,
    accel_profile=[(0.0, -5.0), (2.0, 0.0)], rng=rng))

  # 16. Gradual noise increase (model degrades over time)
  grad_noise = _build_scenario(
    "gradual_noise_increase", 15.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  for i, s in enumerate(grad_noise.steps):
    noise_scale = 1.0 + 4.0 * (i / len(grad_noise.steps))  # 1x → 5x noise
    extra = float(rng.normal(0.0, 1.5 * noise_scale))
    s.raw_drel = s.true_drel + extra
  scenarios.append(grad_noise)

  # 17. Lead dropout 3s (longer than 1s dropout)
  dropout_3s = _build_scenario(
    "dropout_3s", 15.0, d0=35.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  for i in range(int(4.0 / DT_S), int(7.0 / DT_S)):
    if i < len(dropout_3s.steps):
      dropout_3s.steps[i].raw_drel = float(dropout_3s.steps[i].true_drel + rng.normal(0.0, 20.0))
  scenarios.append(dropout_3s)

  # 18. Real cut-in during decel (lead appears AND is braking)
  cutin_decel = _build_scenario(
    "cutin_while_braking", 10.0, d0=80.0, v0=0.0,
    accel_profile=[(0.0, 0.0)], rng=rng)
  cut_frame = int(3.0 / DT_S)
  d_cutin = 18.0
  for i in range(cut_frame, len(cutin_decel.steps)):
    t_since = (i - cut_frame) * DT_S
    d_cutin_now = max(5.0, 18.0 - 2.0 * t_since)  # lead braking, closing
    _rd, _xs = _noisy_drel(d_cutin_now, rng)
    cutin_decel.steps[i] = ScenarioStep(
      true_drel=d_cutin_now, true_vrel=-2.0,
      raw_drel=_rd, raw_vrel=float(-2.0 + rng.normal(0.0, 0.3)), x_std=_xs)
  scenarios.append(cutin_decel)

  return scenarios


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class FilterMetrics:
  rmse_m: float           # RMS tracking error vs ground truth
  max_error_m: float      # worst-case tracking error
  mean_lag_ms: float      # average signed lag (positive = filter behind truth)
  noise_rejection_db: float  # 20*log10(input_noise_rms / output_noise_rms)
  settling_90_ms: float   # time to reach 90% of a step change (for cut-in scenario)


def _compute_metrics(true_vals: list[float], filtered_vals: list[float],
                     raw_vals: list[float]) -> FilterMetrics:
  n = len(true_vals)
  errors = [filtered_vals[i] - true_vals[i] for i in range(n)]
  raw_errors = [raw_vals[i] - true_vals[i] for i in range(n)]

  rmse = math.sqrt(sum(e * e for e in errors) / max(n, 1))
  max_err = max(abs(e) for e in errors) if errors else 0.0

  # Noise rejection: ratio of raw noise RMS to filtered noise RMS
  raw_rms = math.sqrt(sum(e * e for e in raw_errors) / max(n, 1))
  filt_rms = rmse
  if filt_rms > 1e-6:
    noise_rej_db = 20.0 * math.log10(max(raw_rms, 1e-6) / filt_rms)
  else:
    noise_rej_db = 60.0  # effectively perfect

  # Mean lag: cross-correlation peak offset (simplified: mean signed error)
  mean_lag_ms = sum(errors) / max(n, 1) / DT_S * 1000.0 if n > 0 else 0.0

  # Settling time: frames to reach within 10% of final value after largest step
  # (rough: find largest step in truth and measure filter response)
  settling_90_ms = 0.0
  max_step = 0.0
  step_idx = 0
  for i in range(1, n):
    step = abs(true_vals[i] - true_vals[i - 1])
    if step > max_step:
      max_step = step
      step_idx = i
  if max_step > 1.0:
    target = true_vals[step_idx]
    initial = filtered_vals[max(step_idx - 1, 0)]
    threshold = abs(target - initial) * 0.10
    for j in range(step_idx, n):
      if abs(filtered_vals[j] - target) <= threshold:
        settling_90_ms = (j - step_idx) * DT_S * 1000.0
        break
    else:
      settling_90_ms = (n - step_idx) * DT_S * 1000.0  # didn't settle

  return FilterMetrics(rmse_m=rmse, max_error_m=max_err, mean_lag_ms=mean_lag_ms,
                       noise_rejection_db=noise_rej_db, settling_90_ms=settling_90_ms)


# ---------------------------------------------------------------------------
# Run filters
# ---------------------------------------------------------------------------
def _run_filter(filt, scenario: Scenario) -> list[float]:
  filt.reset()
  outputs: list[float] = []
  for step in scenario.steps:
    val = filt.update(step.raw_drel, step.raw_vrel, DT_S)
    outputs.append(float(val))
  return outputs


# ---------------------------------------------------------------------------
# Parameterised sweep helper
# ---------------------------------------------------------------------------
def run_ab_comparison(kalman_q_d: float = 1.0, kalman_q_v: float = 0.0,
                      kalman_r: float = 2.5,
                      kalman_gain_max: float = 0.25,
                      kalman_deadband_m: float = 0.35,
                      ema_tau_close: float = 0.30, ema_tau_open: float = 0.60,
                      seed: int = 42, print_table: bool = True) -> dict[str, dict[str, FilterMetrics]]:
  """Run both filters on all scenarios and return metrics.

  Returns dict[scenario_name][filter_name] -> FilterMetrics
  """
  scenarios = _make_scenarios(seed=seed)

  ema = LeadDistanceFilter()
  kalman = LeadKalmanFilter(q_drel=kalman_q_d, r_drel=kalman_r,
                            kalman_gain_max=kalman_gain_max,
                            deadband_m=kalman_deadband_m)

  results: dict[str, dict[str, FilterMetrics]] = {}

  for sc in scenarios:
    true_vals = [s.true_drel for s in sc.steps]
    raw_vals = [s.raw_drel for s in sc.steps]

    # --- EMA filter ---
    ema.reset()
    ema_out: list[float] = []
    for step in sc.steps:
      val = ema.update(step.raw_drel, step.raw_vrel, DT_S,
                       tau_close=ema_tau_close, tau_open=ema_tau_open)
      ema_out.append(float(val))
    ema_metrics = _compute_metrics(true_vals, ema_out, raw_vals)

    # --- Kalman filter (fixed R) ---
    kalman.reset()
    kal_out: list[float] = []
    for step in sc.steps:
      val = kalman.update(step.raw_drel, step.raw_vrel, DT_S)
      kal_out.append(float(val))
    kal_metrics = _compute_metrics(true_vals, kal_out, raw_vals)

    # --- Kalman filter (xStd-adaptive R) ---
    kalman.reset()
    kal_xstd_out: list[float] = []
    for step in sc.steps:
      val = kalman.update(step.raw_drel, step.raw_vrel, DT_S,
                          r_meas=step.x_std * step.x_std)
      kal_xstd_out.append(float(val))
    kal_xstd_metrics = _compute_metrics(true_vals, kal_xstd_out, raw_vals)

    results[sc.name] = {"ema": ema_metrics, "kalman": kal_metrics, "kalman_xstd": kal_xstd_metrics}

  if print_table:
    _print_comparison(results)

  return results


def _print_comparison(results: dict[str, dict[str, FilterMetrics]]) -> None:
  hdr = f"{'scenario':<25} {'':5} {'RMSE(m)':>8} {'MaxErr':>8} {'Lag(ms)':>8} {'NR(dB)':>8} {'Settl90':>8}"
  sep = "-" * len(hdr)
  print("\n" + sep)
  print("  LEAD dRel FILTER A/B COMPARISON")
  print(sep)
  print(hdr)
  print(sep)

  ema_wins = 0
  kal_wins = 0

  for name, filters in results.items():
    em = filters["ema"]
    km = filters["kalman"]

    # Score: lower RMSE + higher noise rejection is better
    ema_score = em.rmse_m - em.noise_rejection_db * 0.1
    kal_score = km.rmse_m - km.noise_rejection_db * 0.1
    winner = "K" if kal_score < ema_score else "E"
    if winner == "K":
      kal_wins += 1
    else:
      ema_wins += 1

    for label, m in [("EMA", em), ("KAL", km)]:
      tag = " <" if (label == "KAL" and winner == "K") or (label == "EMA" and winner == "E") else ""
      print(f"  {name if label == 'EMA' else '':<23} {label:>5} "
            f"{m.rmse_m:8.3f} {m.max_error_m:8.3f} {m.mean_lag_ms:8.1f} "
            f"{m.noise_rejection_db:8.2f} {m.settling_90_ms:8.0f}{tag}")
    print()

  print(sep)
  print(f"  Scenario wins — EMA: {ema_wins}  Kalman: {kal_wins}")
  print(sep + "\n")


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------
class TestLeadFilterAB:
  """Run A/B comparison as a test.  Use ``pytest -s`` to see the table."""

  def test_ab_comparison_default_tuning(self):
    """Side-by-side with default tuning parameters."""
    results = run_ab_comparison(print_table=True)
    # Sanity: both filters should track within 5 m RMSE on steady follow
    assert results["steady_follow_30m"]["ema"].rmse_m < 5.0
    assert results["steady_follow_30m"]["kalman"].rmse_m < 5.0

  @pytest.mark.parametrize("q_d,q_v,r", [
    (0.1, 1.0, 5.0),    # conservative: trust prediction more
    (0.5, 4.0, 3.0),    # default
    (1.0, 8.0, 2.0),    # aggressive: trust measurements more
    (0.3, 2.0, 4.0),    # balanced
    (2.0, 10.0, 1.5),   # very aggressive
  ])
  def test_kalman_sweep(self, q_d, q_v, r):
    """Sweep Kalman Q/R and print results for each."""
    print(f"\n=== Kalman sweep: Q_d={q_d}, Q_v={q_v}, R={r} ===")
    results = run_ab_comparison(kalman_q_d=q_d, kalman_q_v=q_v, kalman_r=r,
                                print_table=True)
    # All configs should at least not diverge
    for name, filters in results.items():
      assert filters["kalman"].rmse_m < 20.0, f"Kalman diverged on {name}"


# ---------------------------------------------------------------------------
# CLI entry point for quick iteration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "sweep":
    print("Running Kalman Q/R sweep...")
    for q_d, q_v, r in [(0.1, 1.0, 5.0), (0.3, 2.0, 4.0), (0.5, 4.0, 3.0),
                         (1.0, 6.0, 2.5), (1.0, 8.0, 2.0), (2.0, 10.0, 1.5)]:
      print(f"\n{'='*60}")
      print(f"  Q_d={q_d}  Q_v={q_v}  R={r}")
      print(f"{'='*60}")
      run_ab_comparison(kalman_q_d=q_d, kalman_q_v=q_v, kalman_r=r)
  else:
    run_ab_comparison()
