# Math reference

## Runtime sigmoid

From `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` line 904:

```
a(κ) = A / (1 + exp(B · (κ − C))) + D          (clamped to [MIN, MAX])
v    = sqrt(a / κ)                              (line 963)
```

Defaults (lines 818–823):
- A = −3.260 000
- B = −6270.000 000
- C = 0.005 010
- D = 5.607 000
- MIN = 1.8
- MAX = 4.478

Both A and B are always negative by the runtime convention enforced in `vision_turn_params.py:339,342`:
```python
setattr(vtc_mod, "PHYSICS_A", -abs(clip(abs(phys_amp), 0.2, 5.0)))
setattr(vtc_mod, "PHYSICS_B", -abs(clip(abs(phys_steep), 100.0, 1e5)))
```

Asymptotes in κ space:
- κ → 0 (straight road): `exp(B · (κ − C)) = exp(−B·C)`. With B negative and C positive, this is a large positive number; `1/(1+large) → 0`; `a → D`. Then clamped to MAX.
- κ → ∞ (tight curve): exponent → −∞; `sigmoid → 1`; `a → A + D`. Then clamped to MIN (if `A + D < MIN`).

## Plain-knob ↔ raw-param mapping (`params.rs`)

Four plain-English knobs expose the 6 raw parameters:

| Plain knob | Units | Range | Raw mapping |
|---|---|---|---|
| Tight-Curve Ceiling | m/s² | [1.0, 3.0] | sets MIN and the low-κ asymptote (A+D) via `A = low − high` |
| Straight-Road Ceiling | m/s² | [2.0, 5.5] | sets MAX and high-κ asymptote (D) via `D = high` |
| Transition Speed | mph | [8, 120] | solves for C such that midpoint κ yields this speed |
| Sharpness | 0..10 | log-mapped | \|B\| = 10^(2 + 0.3·sharpness) ∈ [1e2, 1e5] |

Forward map (`to_sigmoid`):
```
low  = clamp(tight_curve_accel, [1.0, 3.0])
high = clamp(straight_road_accel, [2.0, 5.5]).max(low + 0.2)   // 0.2 = clip::ABS_A.0
D    = clamp(high, [2.0, 6.5])
A    = -clamp(high - low, [0.2, 5.0])
a_mid = 0.5·(low + high)
v_mps = transition_speed_mph · MPH_TO_MS
C    = clamp(a_mid / v²_mps, [1e-5, 0.1])
|B|  = 10^(2 + 0.3·clamp(sharpness, [0, 10]))
B    = -|B|
MIN  = low
MAX  = high
```

Inverse (`from_sigmoid`) recovers plain knobs best-effort — loses the extra degree of freedom between MIN and (A+D) (they're snapped equal in the forward map).

**What you see is what you get** — the forward map deliberately ties MIN/MAX to the asymptotes so the plotted curve's visible floor/ceiling match what the user sees as "low/high speed ceiling". A more flexible decoupled mode would need a separate raw/advanced view.

## EQ band composition (`sigmoid::apply_prepared_bands`)

Each band is `{ center_speed_mph, gain_db, q, enabled }`. For rendering / export we freeze per-band `log_kc` against the current sigmoid (so bands don't chase their own tail when composed samples are reinterpreted):

```
Band::kappa_center(params):
  a ← 0.5·(MIN + MAX)
  for _ in 0..8:
      κ ← a / v²_mps     // v from band.center_speed_mph
      a ← params.eval(κ)
  return a / v²_mps      // last estimate
```

Composition at an arbitrary κ:
```
log_k = log10(κ)
dB_total = Σ_bands (enabled) gain_db · exp(-0.5·((log_k − log_kc) / σ)²)
a_out = a_base · 10^(dB_total / 20)
```
where `σ = 0.5 / max(q, 0.1)`.

Per-band "width" visualisation (translucent Q zone) converts ±2σ in log-κ back to mph space by computing `v = sqrt(a(κ_lo) / κ_lo)` for `κ_lo/hi = 10^(log_kc ± 2σ)`.

## Monotonic-v post-process (`plot::hero_plot`)

The parametric trace `(v(κ), a(κ))` is monotonic in v unless a band's local slope satisfies:

```
da/dκ > a / κ
```

at some κ — at which point `dv/dκ = (1/(2v)) · (a − κ·da/dκ) / κ² < 0` and the curve folds back on itself.

The runtime doesn't see this (it only ever computes κ → a → v, never the other way). But the plot does, and multi-valued `a` at the same `v` is a UX bug.

Fix: walk samples in κ-descending order (naturally v-ascending) and clamp each v to the running max. Bands that bite hard then render as vertical notches — the parametric-EQ idiom — instead of folded curves.

```rust
let mut max_v = f64::NEG_INFINITY;
for s in samples.iter_mut() {
  if s.speed_mph < max_v { s.speed_mph = max_v; }
  else                   { max_v = s.speed_mph; }
}
```

Do NOT do this by taking min-a-at-each-v (envelope); that loses the sharpness visual.

## Q-curve export (`sigmoid::bands_as_q_curve_points`)

The openpilot runtime (`vtsc_curve_tuning.py:Q_CURVE_POINTS`) multiplies the *speed* `v`, not the lateral accel `a`. Since `v = sqrt(a/κ)`:

```
(a · gain) / κ = v² · gain
sqrt((a · gain) / κ) = v · sqrt(gain)
```

so `q_speed = sqrt(a_gain)`. Exported points are `(κ, q_speed)` at 64 log-uniform κ samples ∈ [1e-5, 1.0], clamped to `[0.5, 1.5]` (the runtime's accepted q range).

## Steepness wings (`plot`)

Visualised as two horizontal anchors flanking the inflection. Empirical mapping (not tied to the exact B value — intended as a "feel" knob):

```
half_width_mph = 30 · exp(-0.4 · sharpness) + 2
      at sharpness=0  → 32 mph
      at sharpness=10 →  2.5 mph
```

Inverse when dragged:
```
sharpness = -ln((half_width_mph − 2) / 30) / 0.4
```
clamped to `[0, 10]`.

## Tests (`src/sigmoid.rs`, `src/params.rs`)

9 unit tests cover:
- `matches_python_reference_at_midpoint` — `eval(C) == A/2 + D` clamped.
- `clamps_at_asymptotes` — κ→0 hits max, κ→1 stays ≥ min.
- `kappa_at_round_trip` — `v = sqrt(a/κ)` round-trips through `SigmoidParams::kappa_at`.
- `band_centered_adds_expected_gain` — +6 dB band yields `10^(6/20)` ratio at its κ_centre.
- `sample_curve_monotonic_in_speed` — default sigmoid has no fold-back even over 200 samples.
- `round_trip_default_is_stable` — `to_sigmoid(from_sigmoid(default))` stable to 1 mph / 1e-6.
- `sharpness_log_mapping` — extremes hit 100 and 1e5; monotonic.
- `transition_speed_places_midpoint_at_requested_v` — within ±0.5 mph.
- `clamps_are_respected` — absurd plain knob inputs produce in-range raw params.
