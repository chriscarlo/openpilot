# Math reference

## Runtime sigmoid

From `sunnypilot/selfdrive/controls/lib/vision_turn_controller.py`:

```
a(κ) = A / (1 + exp(B · (κ − C))) + D          (clamped to [MIN, MAX])
v    = sqrt(a / κ)                              (line 963)
```

Checked-in source authority:
- A = −2.125 961
- B = −1601.225 452
- C = 0.007 637
- D = 4.478 000
- MIN = 2.352
- MAX = 4.478

The runtime refreshes these values from persistent Params at 5 Hz. The tuner
therefore keeps the module constants, declared Params defaults, offroad-panel
reset/ensure values, generated Q base, and device migration values identical at
source precision; a mismatch fails source preflight rather than being fitted.

Both A and B are always negative by the runtime convention enforced in `vision_turn_params.py:339,342`:
```python
setattr(vtc_mod, "PHYSICS_A", -abs(clip(abs(phys_amp), 0.2, 5.0)))
setattr(vtc_mod, "PHYSICS_B", -abs(clip(abs(phys_steep), 100.0, 1e5)))
```

Asymptotes in κ space:
- κ → 0 (straight road): `exp(B · (κ − C)) = exp(−B·C)`. With B negative and C positive, this is a large positive number; `1/(1+large) → 0`; `a → D`. Then clamped to MAX.
- κ → ∞ (tight curve): exponent → −∞; `sigmoid → 1`; `a → A + D`. Then clamped to MIN (if `A + D < MIN`).

## Raw tile curvature versus runtime curvature

For three adjacent route nodes, mapd's primitive curvature is the inverse
circumradius. If `a`, `b`, and `c` are the triangle side lengths and `A` is its
area, the unsigned raw value is:

```
κ_raw = 4A / (a·b·c)
```

Schema-v1 tile `Way.safeSpeeds` apply the physics curve directly to this raw
per-vertex value. Live mapd does not publish that value unchanged. At a node
with complete route context it calculates three consecutive raw triplets over
five source nodes and arc-length-weights them:

```
κ_runtime[i] = Σ(κ_raw[j] · arc_length[j]) / Σ arc_length[j]
               for j = i-1 ... i+1
```

Before that average, `GetStateCurvatures` identifies qualifying way boundaries
(`previous.lanes < next.lanes`, plus the two-way-to-one-way split case) and
sets affected raw entries to `0.0015 1/m`. It clamps the two raw entries before
the boundary and additional backward/forward entries selected by its 15 m
proximity loops. The Swift parity estimator carries boundary indices through
stitched geometry and performs this mutation before selecting the target's
three weighted measurements. Reindexing an isolated five-node window is not
equivalent because mapd's two pre-boundary assignments depend on the boundary's
position in the complete route sequence.

This distinction matters because an isolated OSM digitization kink can have a
small raw circumradius without representing the sustained road bend. The tuner
must fit `κ_runtime`, not `κ_raw`; raw curvature is diagnostic provenance only.
The direction-feasible physical route is stitched before evaluation. At a
branch, estimator v5 accepts a unique gentle (`|κ_junction| ≤ 0.1`) exact-name
match, then a unique gentle exact-reference match, then a uniquely
least-curvature gentle partial-reference match. Ties fail closed; when those
identity priorities do not resolve the branch, only one physical continuation
is accepted rather than inventing a local tangent choice. Missing or ambiguous
five-node context therefore makes the sample ineligible instead of triggering
a raw fallback. Direction-dependent transitions on bidirectional geometry are
also ineligible because the tuner lacks the car's live route direction. Target
acceleration is then `κ_runtime·v_target²`.

For route context that estimator v5 accepts, its curvature math matches
production mapd; rejecting an order-dependent identity tie is an intentional
offline safety restriction. This is not a claim that production's unsigned
five-node average is the ideal road model. A future signed, fixed-distance,
multi-scale estimator should first run in shadow mode and gain shared Go/Swift
golden coverage for isolated jitter, irregular node spacing, true compact
bends, S-curves, way splits, and direction reversal. It must land in tile
generation and runtime together before calibration adopts it.

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

so `q_speed = sqrt(a_gain)`. Exported points are `(κ, q_speed)` at 256 log-uniform κ samples ∈ [1e-5, 1.0], clamped to `[0.5, 1.5]` (the runtime's accepted q range). The count was raised from 64 when Q expanded to 16 so narrow notches do not alias.

The native Mac app rounds every point to the source representation (`κ` as
`%.6e`, multiplier as `%.4f`) before fitting, plotting, or interpolating it.
It also evaluates candidate and final sigmoid parameters at the exact apply
precision (A/B/C/D at six fractional digits and both rails at four). This keeps
proposal diagnostics and Curve Lab numerically identical to the constants and
`Q_CURVE_POINTS` that `SourcePatcher` will write.

## Complete-curve calibration fit (`SigmoidFitter.swift`)

Real-curve targets are not restricted to the raw four-knob sigmoid envelope.
The Mac fitter projects requests to a weighted non-increasing speed sequence,
then searches the bounded sigmoid backbone with an explicit penalty whenever a
projected target would fall outside that candidate base's real `0.5...1.5`
speed-multiplier authority. It then solves a deterministic residual curve in
the same log-κ acceleration-dB space used by the EQ/Q export:

```
target_gain_db = 40 · log10(target_speed / base_effective_speed)
residual_db(κ) = Σ gain_i · exp(-0.5·((log10(κ) − log10(κ_i)) / σ_i)²)
```

One Q=4 candidate band is centered on each fixed-anchor, resolution-aware
curvature cluster. The broader generated bands make the residual curve smooth
enough to avoid narrow speed pockets. Bounded ridge coordinate descent seeds a
coordinate refinement that scores the actual source-rounded 256-point
interpolation; gains remain in `[-12, +12] dB` and the exported speed
multiplier remains clamped to `[0.5, 1.5]`. Per-sample attainable ranges are
recomputed from the final selected base, so diagnostics never claim that a
fitted target can use authority belonging to some other sigmoid candidate.
Re-fitting uses the bank's persisted original checkout as a stable backbone
anchor and creates a canonical replacement set with deterministic IDs, so
accepted fits remain idempotent across source apply and relaunch and bands
never accumulate.

Before either stage, weighted pool-adjacent-violators projection maps the
pointwise targets onto the nearest non-increasing speed sequence in increasing
curvature. This is a physical/runtime shape constraint, not an attainability
claim: diagnostics retain each original request and local conflicts remain
visible. The complete interpolated runtime curve is checked on a 4096-point log
grid, all rounded Q knots, and explicit sigmoid-transition probes. Its speed
may rise by at most 0.10 mph above the running minimum as curvature tightens;
measuring cumulative rise is required because many individually
sub-threshold knot increases can otherwise form a large hidden speed pocket.

The generated Q=4 Gaussian has `σ = 0.5 / Q = 0.125` in `log10(κ)`. Its explicit
influence collar is four standard deviations on each side of the bank:

```
curvature_ratio = 10^(4σ) = 10^0.5 = sqrt(10)
```

The sigmoid and residual may reshape the curve inside that collar so endpoint
targets retain full authority. Outside it, a 4096-point log grid, every rounded
Q knot, and explicit probes around both sigmoid transitions enforce a 2 mph
maximum change from the persisted checkout for both the base sigmoid and the
complete source-rounded runtime curve. Probing the transitions is required
because a high-|B| shelf can otherwise sit between fixed grid points. Failure
of the dense off-bank or raw monotonicity guard aborts proposal generation
rather than returning an unsafe fallback.

Attainability diagnostics use the pointwise complete base-plus-Q envelope.
They deliberately do not promise that every pointwise-reachable request is
jointly compatible with every other request or with the monotonic shape
constraint. Targets at effectively identical curvatures still share one
prediction and remain visible as conflicts; the fitter must not synthesize
unstable per-road spikes to hide them. Per-curve diagnostics show the requested
and proposed implied lateral acceleration `κ·v²` because a Q-shaped curve may
exceed the raw sigmoid's MIN/MAX acceleration rails. A proposed curve above
the 5.5 m/s² safety-review threshold requires explicit acknowledgement before
Curve Lab acceptance.

Effective predictions use the checked-in Params defaults (0 mph low-speed
bias through a 50 mph endpoint and speed factor 1.0). They are an offline
source-default model, not a snapshot of live or learned device Params.

## Runtime whole-curve profile (`whole-curve-v1`)

The production estimator does not bake a road-specific speed. Swift and Go
consume one checked-in golden corpus and run the same ordered directional-route
algorithm: deduplicate source nodes, linearly resample at approximately 5 m,
measure signed curvature over 60/100/160 m distance windows, group sustained
same-sign points into events, preserve straight shoulders, split S-curves, and
fail closed on unresolved physical forks or incomplete context. The route spans
approximately 1,200 m and carries predecessor context across current-way
rollover so event identity does not jump while the car is inside a bend.

`MapWholeCurveProfile` contains signed curvature points, event boundaries and
controlling curvature, stable IDs, confidence/flags, route fingerprint and
generation, and publish time. It deliberately contains no final speed. After
strict version/freshness/finiteness/spacing/fingerprint/proximity validation,
the Python controller evaluates each controlling magnitude with the current
source-rounded sigmoid:

```text
a_base = clip(A / (1 + exp(B * (abs(kappa) - C))) + D, MIN, MAX)
v_base = sqrt(a_base / abs(kappa))
v_q = v_base * interpolate_source_rounded_Q(abs(kappa))
```

Only the explicitly supported live VTSC modifiers are applied after that
evaluation; learned calibration scaling is fixed to 1.0 on this path. Missing,
stale, malformed, mismatched, distant, truncated, or fatally ambiguous profiles
fall back to the legacy `MapCurvatures` stream. Turning Map Lookahead off clears
the whole map-derived constraint state on that first controller tick.

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
