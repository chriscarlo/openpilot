package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// SigmoidCfg mirrors the runtime PHYSICS_* constants in
// /projects/chauffeur/data/openpilot/sunnypilot/selfdrive/controls/lib/vision_turn_controller.py
// (lines 818-823) plus MAX_SPEED_DEFAULT used by curvature_to_speed.
//
// The bake at tile-generation time uses the file-committed values, not any
// live runtime overrides from VisionTurnSpeedControlPhysicsBaseline/Amplitude/
// Steepness/Center params. The on-device consumer ignores baked speeds when
// any of those overrides are set, so this is consistent.
type SigmoidCfg struct {
	A               float64
	B               float64
	C               float64
	D               float64
	MinLat          float64
	MaxLat          float64
	MaxSpeedDefault float64 // m/s; speed cap when curvature is effectively zero
}

// DefaultSigmoidCfg matches the PHYSICS_* values committed in
// vision_turn_controller.py and MAX_SPEED_DEFAULT (70 m/s ≈ 156 mph).
// The flagless `mapd --generate` produces identical baseline tiles.
func DefaultSigmoidCfg() SigmoidCfg {
	return SigmoidCfg{
		A:               -2.548675,
		B:               -1024.629261,
		C:               0.006053,
		D:               4.107103,
		MinLat:          1.5584,
		MaxLat:          4.1071,
		MaxSpeedDefault: 70.0,
	}
}

// ActiveSigmoidCfg mirrors vision_turn_params.py's persistent-Param loading.
// mapd reads this once per route publication and bakes the physics-only speed
// into the v2 whole-route profile. The controller still verifies the hash and
// applies its inexpensive live bias/Q multipliers; a mismatch falls back to
// live conversion rather than consuming stale speeds.
func ActiveSigmoidCfg() SigmoidCfg {
	cfg := DefaultSigmoidCfg()
	cfg.A = -clampFloat(math.Abs(readFinitePhysicsParam("VisionTurnSpeedControlPhysicsAmplitude", cfg.A)), 0.2, 5.0)
	cfg.B = -clampFloat(math.Abs(readFinitePhysicsParam("VisionTurnSpeedControlPhysicsSteepness", cfg.B)), 100.0, 100000.0)
	cfg.C = readBoundedPhysicsParam("VisionTurnSpeedControlPhysicsCenter", cfg.C, 1.0e-5, 0.1)
	cfg.D = readBoundedPhysicsParam("VisionTurnSpeedControlPhysicsBaseline", cfg.D, 2.0, 6.5)
	cfg.MinLat = readBoundedPhysicsParam("VisionTurnSpeedControlPhysicsMinLatAccel", cfg.MinLat, 1.0, 3.0)
	cfg.MaxLat = readBoundedPhysicsParam("VisionTurnSpeedControlPhysicsMaxLatAccel", cfg.MaxLat, 2.0, 5.5)
	if cfg.MinLat > cfg.MaxLat {
		cfg.MinLat, cfg.MaxLat = cfg.MaxLat, cfg.MinLat
	}
	return cfg
}

func readFinitePhysicsParam(name string, fallback float64) float64 {
	data, err := os.ReadFile(filepath.Join(ParamsPath, name))
	if err != nil {
		return fallback
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
	if err != nil || !wholeCurveIsFinite(value) {
		return fallback
	}
	return value
}

func readBoundedPhysicsParam(name string, fallback, lower, upper float64) float64 {
	value := readFinitePhysicsParam(name, fallback)
	return clampFloat(value, lower, upper)
}

// PhysicsLatAccel computes a_lat(κ) = A/(1+exp(B·(κ-C))) + D, clamped to
// [MinLat, MaxLat]. Mirrors _physics_based_lateral_acceleration in
// vision_turn_controller.py:890-912 with low_speed_sigmoid_scale = 1.0.
func PhysicsLatAccel(curvature float64, c SigmoidCfg) float64 {
	if curvature < 1e-8 {
		curvature = 1e-8
	}
	if curvature > 1.0 {
		curvature = 1.0
	}
	exponent := c.B * (curvature - c.C)
	var raw float64
	if exponent > 700.0 {
		raw = c.D
	} else if exponent < -700.0 {
		raw = c.A + c.D
	} else {
		raw = c.A/(1.0+math.Exp(exponent)) + c.D
	}
	if raw < c.MinLat {
		return c.MinLat
	}
	if raw > c.MaxLat {
		return c.MaxLat
	}
	return raw
}

// CurvatureToSpeed maps |κ| to a safe speed in m/s using v = sqrt(a/κ).
// Q-curve and low-speed-bias multipliers from the Python runtime are
// intentionally NOT applied here — they stay live so the Q EQ remains
// tunable without rebuilding or republishing route geometry. The device-side
// consumer applies them on top of the route-baked physics speed.
func CurvatureToSpeed(curvature float64, c SigmoidCfg) float64 {
	if curvature < 1e-7 {
		return c.MaxSpeedDefault
	}
	v := math.Sqrt(PhysicsLatAccel(curvature, c) / curvature)
	if v > c.MaxSpeedDefault {
		return c.MaxSpeedDefault
	}
	return v
}

// Hash returns a stable 12-char hex prefix of SHA-256 over the canonical
// formatted sigmoid tuple. The runtime publishes this hash alongside the
// baked speeds; the openpilot consumer compares it against the in-memory
// hash of its own PHYSICS_* values to detect a stale route profile after an
// edit and falls back to live conversion until mapd republishes.
func (c SigmoidCfg) Hash() string {
	canon := fmt.Sprintf("%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.2f",
		c.A, c.B, c.C, c.D, c.MinLat, c.MaxLat, c.MaxSpeedDefault)
	sum := sha256.Sum256([]byte(canon))
	return hex.EncodeToString(sum[:6])
}
