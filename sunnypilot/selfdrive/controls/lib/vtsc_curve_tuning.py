# Auto-generated values live here for VTSC curvature→speed tuning.
#
# Workflow:
# - Run `python3 tools/vtsc/curve_tuner.py`
# - Tweak the curve visually (baseline vs tuned)
# - Click "Apply" to write updated control points here
#
# This file is intentionally lightweight so it can be imported without pulling in any
# other VTSC dependencies.

from __future__ import annotations

# Master enable for the multiplicative tuning curve.
# When False, VTSC uses the baseline physics mapping.
Q_CURVE_ENABLED = False

# Control points for a multiplicative speed scale q(κ).
# Each point is (curvature_1_per_m, q_multiplier).
# Interpolation is linear in log10(κ).
Q_CURVE_POINTS: list[tuple[float, float]] = []

# Optional metadata (authoring note, timestamp, etc).
Q_CURVE_META = {}

