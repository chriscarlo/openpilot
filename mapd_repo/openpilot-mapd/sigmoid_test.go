package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"capnproto.org/go/capnp/v3"
)

// Reference values precomputed in Python from the file-committed sigmoid in
// vision_turn_controller.py:818-823 + MAX_SPEED_DEFAULT=70.
// Regenerate via: python3 -c '<see plan H. Verification>'
//
// Each tuple is (κ, expected lat-accel m/s², expected speed m/s).
var sigmoidPythonReference = []struct {
	kappa    float64
	latAccel float64
	speed    float64
}{
	{1e-08, 4.1019517724780883, 70.0},
	{1e-06, 4.1019465550866467, 70.0},
	{1e-05, 4.1018988817054725, 70.0},
	{1e-04, 4.1013972770740423, 70.0},
	{1e-03, 4.0928031259267845, 63.975019546122724},
	{0.005, 3.4604845674023905, 26.307734860312053},
	{0.01, 1.60232058746272, 12.658280244419934},
	{0.05, 1.5584280000000006, 5.5828809767001131},
	{0.1, 1.5584280000000006, 3.9476929971820258},
	{0.5, 1.5584280000000006, 1.765461979199779},
	{1.0, 1.5584280000000006, 1.2483701374191873},
}

const sigmoidPythonReferenceHash = "85a608e68945"

func TestSigmoidMatchesPython(t *testing.T) {
	cfg := DefaultSigmoidCfg()

	if got := cfg.Hash(); got != sigmoidPythonReferenceHash {
		t.Fatalf("hash mismatch: got %q want %q (sigmoid drift between Go and Python)", got, sigmoidPythonReferenceHash)
	}

	for _, tc := range sigmoidPythonReference {
		gotLat := PhysicsLatAccel(tc.kappa, cfg)
		if math.Abs(gotLat-tc.latAccel) > 1e-9 {
			t.Errorf("PhysicsLatAccel(%g) = %.15g, want %.15g (Δ=%.3e)",
				tc.kappa, gotLat, tc.latAccel, gotLat-tc.latAccel)
		}
		gotSpeed := CurvatureToSpeed(tc.kappa, cfg)
		if math.Abs(gotSpeed-tc.speed) > 1e-9 {
			t.Errorf("CurvatureToSpeed(%g) = %.15g, want %.15g (Δ=%.3e)",
				tc.kappa, gotSpeed, tc.speed, gotSpeed-tc.speed)
		}
	}
}

func TestSigmoidHashStableAcrossTrivialChanges(t *testing.T) {
	// Same canonical formatting → same hash. Different value → different hash.
	cfg := DefaultSigmoidCfg()
	h1 := cfg.Hash()
	cfg2 := cfg
	cfg2.A = cfg.A + 1e-7 // sub-rounding; canonical formatting truncates to .6f
	if cfg2.Hash() != h1 {
		t.Errorf("sub-rounding change should not perturb hash (formatting truncates to %%.6f)")
	}
	cfg3 := cfg
	cfg3.A = cfg.A + 1e-3 // visible at .6f
	if cfg3.Hash() == h1 {
		t.Errorf("visible change should perturb hash")
	}
}

func TestActiveSigmoidCfgReadsAndBoundsPersistentParams(t *testing.T) {
	oldParamsPath := ParamsPath
	ParamsPath = t.TempDir()
	t.Cleanup(func() { ParamsPath = oldParamsPath })
	values := map[string]string{
		"VisionTurnSpeedControlPhysicsAmplitude":   "-3.25\n",
		"VisionTurnSpeedControlPhysicsSteepness":   "50000\n",
		"VisionTurnSpeedControlPhysicsCenter":      "0.008\n",
		"VisionTurnSpeedControlPhysicsBaseline":    "99\n",
		"VisionTurnSpeedControlPhysicsMinLatAccel": "4\n",
		"VisionTurnSpeedControlPhysicsMaxLatAccel": "2.5\n",
	}
	for name, value := range values {
		if err := os.WriteFile(filepath.Join(ParamsPath, name), []byte(value), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg := ActiveSigmoidCfg()
	if cfg.A != -3.25 || cfg.B != -50000 || cfg.C != 0.008 || cfg.D != 6.5 {
		t.Fatalf("unexpected active sigmoid: %+v", cfg)
	}
	if cfg.MinLat != 2.5 || cfg.MaxLat != 3.0 {
		t.Fatalf("bounded/sorted lateral limits incorrect: %+v", cfg)
	}
}

func TestOldReaderNewTile(t *testing.T) {
	// Round-trip a v1 tile (with safeSpeeds + schemaVersion + sigmoidHash)
	// through the SAME generated bindings the runtime uses. This verifies:
	//   1. the new fields serialize / deserialize correctly
	//   2. the standard Way fields (name, maxSpeed, nodes) are unaffected
	//   3. mixed reads (some ways with safeSpeeds, some without) don't panic
	//
	// True "old-reader new-tile" wire compat (an old binary built before the
	// schema bump) cannot be tested in-process — Cap'n Proto guarantees per
	// spec that readers ignore unknown fields, so an older binding would simply
	// not have accessor methods for the new ordinals.
	cfg := DefaultSigmoidCfg()

	arena := capnp.MultiSegment([][]byte{})
	msg, seg, err := capnp.NewMessage(arena)
	if err != nil {
		t.Fatalf("new message: %v", err)
	}
	root, err := NewRootOffline(seg)
	if err != nil {
		t.Fatalf("new root: %v", err)
	}
	root.SetMinLat(40.0)
	root.SetMinLon(-75.0)
	root.SetMaxLat(40.25)
	root.SetMaxLon(-74.75)
	root.SetOverlap(0.01)
	root.SetSchemaVersion(1)
	if err := root.SetSigmoidHash(cfg.Hash()); err != nil {
		t.Fatalf("set sigmoid hash: %v", err)
	}

	ways, err := root.NewWays(2)
	if err != nil {
		t.Fatalf("new ways: %v", err)
	}

	// Way 0: full v1 (has safeSpeeds populated).
	w0 := ways.At(0)
	if err := w0.SetName("Test Way Zero"); err != nil {
		t.Fatalf("set w0 name: %v", err)
	}
	w0.SetMaxSpeed(27.0) // 27 m/s ~= 60 mph
	nodes0, err := w0.NewNodes(4)
	if err != nil {
		t.Fatalf("new w0 nodes: %v", err)
	}
	for i := 0; i < 4; i++ {
		n := nodes0.At(i)
		n.SetLatitude(40.0 + float64(i)*0.001)
		n.SetLongitude(-75.0)
	}
	speeds0, err := w0.NewSafeSpeeds(4)
	if err != nil {
		t.Fatalf("new w0 safe speeds: %v", err)
	}
	speeds0.Set(0, cfg.MaxSpeedDefault) // endpoint
	speeds0.Set(1, 25.0)
	speeds0.Set(2, 20.0)
	speeds0.Set(3, cfg.MaxSpeedDefault) // endpoint

	// Way 1: simulate a "legacy" way with no safeSpeeds set (mixed schema).
	w1 := ways.At(1)
	if err := w1.SetName("Test Way One Legacy"); err != nil {
		t.Fatalf("set w1 name: %v", err)
	}
	w1.SetMaxSpeed(15.6) // 35 mph
	nodes1, err := w1.NewNodes(2)
	if err != nil {
		t.Fatalf("new w1 nodes: %v", err)
	}
	nodes1.At(0).SetLatitude(40.1)
	nodes1.At(0).SetLongitude(-75.0)
	nodes1.At(1).SetLatitude(40.2)
	nodes1.At(1).SetLongitude(-75.0)
	// Intentionally do NOT call w1.NewSafeSpeeds() — leave the field unset.

	packed, err := msg.MarshalPacked()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Round-trip: unmarshal and verify everything reads back correctly.
	rtMsg, err := capnp.UnmarshalPacked(packed)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	rtOffline, err := ReadRootOffline(rtMsg)
	if err != nil {
		t.Fatalf("read root: %v", err)
	}
	if v := rtOffline.SchemaVersion(); v != 1 {
		t.Errorf("schemaVersion: got %d want 1", v)
	}
	if h, _ := rtOffline.SigmoidHash(); h != cfg.Hash() {
		t.Errorf("sigmoidHash: got %q want %q", h, cfg.Hash())
	}
	rtWays, err := rtOffline.Ways()
	if err != nil {
		t.Fatalf("read ways: %v", err)
	}
	if rtWays.Len() != 2 {
		t.Fatalf("ways len: got %d want 2", rtWays.Len())
	}

	// Way 0: should have 4 baked speeds.
	rtW0 := rtWays.At(0)
	rtSpeeds0, err := rtW0.SafeSpeeds()
	if err != nil {
		t.Fatalf("read w0 safeSpeeds: %v", err)
	}
	if rtSpeeds0.Len() != 4 {
		t.Errorf("w0 safeSpeeds len: got %d want 4", rtSpeeds0.Len())
	}
	if got := rtSpeeds0.At(2); got != 20.0 {
		t.Errorf("w0 safeSpeeds[2]: got %g want 20.0", got)
	}
	rtNodes0, _ := rtW0.Nodes()
	if rtNodes0.Len() != rtSpeeds0.Len() {
		t.Errorf("w0 nodes/safeSpeeds parallel-length mismatch: %d vs %d", rtNodes0.Len(), rtSpeeds0.Len())
	}

	// Way 1: legacy-ish — safeSpeeds field unset; SafeSpeeds() returns an empty
	// list (the Cap'n Proto pointer-default for unset List fields), and standard
	// fields are still readable.
	rtW1 := rtWays.At(1)
	if name, _ := rtW1.Name(); name != "Test Way One Legacy" {
		t.Errorf("w1 name round-trip: got %q want %q", name, "Test Way One Legacy")
	}
	if got := rtW1.MaxSpeed(); got != 15.6 {
		t.Errorf("w1 maxSpeed: got %g want 15.6", got)
	}
	rtSpeeds1, err := rtW1.SafeSpeeds()
	if err != nil {
		t.Fatalf("read w1 safeSpeeds (should not error on unset): %v", err)
	}
	if rtSpeeds1.Len() != 0 {
		t.Errorf("w1 safeSpeeds (unset) should be len 0, got %d", rtSpeeds1.Len())
	}

	// Sanity: marshal-unmarshal does not change packed length materially.
	if packed2, err := rtMsg.MarshalPacked(); err == nil {
		if !bytes.Equal(packed, packed2) {
			// Allow for capnp re-arrangement; lengths should at least match.
			if len(packed) != len(packed2) {
				t.Logf("re-marshal length differs: %d vs %d (allowed; this is informational)", len(packed), len(packed2))
			}
		}
	}
}

func TestLegacyTileReadsSchemaVersionZero(t *testing.T) {
	// A v0 tile sets nothing schema-related. After round-trip, SchemaVersion()
	// must read 0 (Cap'n Proto unset-primitive default), so the on-device
	// guard in mapd.go's loop knows to skip publishing baked speeds.
	arena := capnp.MultiSegment([][]byte{})
	msg, seg, err := capnp.NewMessage(arena)
	if err != nil {
		t.Fatalf("new message: %v", err)
	}
	root, err := NewRootOffline(seg)
	if err != nil {
		t.Fatalf("new root: %v", err)
	}
	root.SetMinLat(0)
	root.SetMaxLat(1)
	// Intentionally do NOT call SetSchemaVersion or SetSigmoidHash.
	if _, err := root.NewWays(0); err != nil {
		t.Fatalf("new ways: %v", err)
	}
	packed, err := msg.MarshalPacked()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	rtMsg, err := capnp.UnmarshalPacked(packed)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	rt, err := ReadRootOffline(rtMsg)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if v := rt.SchemaVersion(); v != 0 {
		t.Errorf("legacy tile schemaVersion: got %d want 0", v)
	}
	if h, _ := rt.SigmoidHash(); h != "" {
		t.Errorf("legacy tile sigmoidHash: got %q want empty", h)
	}
}
