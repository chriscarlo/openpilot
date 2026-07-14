package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type wholeCurveGoldenCorpus struct {
	SchemaVersion    int                    `json:"schemaVersion"`
	EstimatorVersion string                 `json:"estimatorVersion"`
	Tolerances       map[string]float64     `json:"tolerances"`
	Cases            []wholeCurveGoldenCase `json:"cases"`
	SavedBank        wholeCurveSavedBank    `json:"savedBank"`
}

type wholeCurveSavedBank struct {
	SampleCount           int                       `json:"sampleCount"`
	DirectionalEventCount int                       `json:"directionalEventCount"`
	Samples               []wholeCurveSavedSample   `json:"samples"`
	Cases                 []wholeCurveSavedBankCase `json:"cases"`
}

type wholeCurveSavedSample struct {
	Number          int     `json:"number"`
	SourceKey       string  `json:"sourceKey"`
	Latitude        float64 `json:"latitude"`
	Longitude       float64 `json:"longitude"`
	DesiredSpeedMPH float64 `json:"desiredSpeedMPH"`
}

type wholeCurveSavedBankCase struct {
	Name               string                 `json:"name"`
	SampleNumbers      []int                  `json:"sampleNumbers"`
	Route              []WholeCurveInputPoint `json:"route"`
	ExpectedPointCount int                    `json:"expectedPointCount"`
	ExpectedEvent      wholeCurveGoldenEvent  `json:"expectedEvent"`
	ExpectedStudy      struct {
		TravelDirection        string    `json:"travelDirection"`
		BendDirection          string    `json:"bendDirection"`
		WholeCurveSpeedMPH     float64   `json:"wholeCurveSpeedMPH"`
		CurrentMinimumSpeedMPH float64   `json:"currentMinimumSpeedMPH"`
		CurrentMaximumSpeedMPH float64   `json:"currentMaximumSpeedMPH"`
		BankTargetsMPH         []float64 `json:"bankTargetsMPH"`
	} `json:"expectedStudy"`
}

type wholeCurveGoldenCase struct {
	Name     string                 `json:"name"`
	Features []string               `json:"features"`
	Route    []WholeCurveInputPoint `json:"route"`
	Expected struct {
		PointCount int                     `json:"pointCount"`
		Events     []wholeCurveGoldenEvent `json:"events"`
	} `json:"expected"`
}

type wholeCurveGoldenEvent struct {
	PhysicalID             string               `json:"physicalID"`
	ID                     string               `json:"id"`
	StartIndex             int                  `json:"startIndex"`
	EndIndex               int                  `json:"endIndex"`
	ApexIndex              int                  `json:"apexIndex"`
	LengthMeters           float64              `json:"lengthMeters"`
	SignedTurnRadians      float64              `json:"signedTurnRadians"`
	SignCoherence          float64              `json:"signCoherence"`
	Curvature60            *float64             `json:"curvature60"`
	Curvature100           *float64             `json:"curvature100"`
	Curvature160           *float64             `json:"curvature160"`
	ControllingCurvature   float64              `json:"controllingCurvature"`
	ScaleSpread            *float64             `json:"scaleSpread"`
	MaximumSourceGapMeters float64              `json:"maximumSourceGapMeters"`
	Confidence             WholeCurveConfidence `json:"confidence"`
	Flags                  []WholeCurveFlag     `json:"flags"`
}

func loadWholeCurveGoldenCorpus(t *testing.T) wholeCurveGoldenCorpus {
	t.Helper()
	paths := []string{
		filepath.Join("..", "..", "tools", "vtsc", "fixtures", "whole_curve_v1.json"),
		filepath.Join("testdata", "whole_curve_v1.json"),
	}
	var data []byte
	var err error
	for _, path := range paths {
		data, err = os.ReadFile(path)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Fatalf("read shared whole-curve corpus from %v: %v", paths, err)
	}
	var corpus wholeCurveGoldenCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode shared whole-curve corpus: %v", err)
	}
	return corpus
}

func TestWholeCurveSharedGoldenCorpus(t *testing.T) {
	corpus := loadWholeCurveGoldenCorpus(t)
	if corpus.SchemaVersion != 1 || corpus.EstimatorVersion != WholeCurveEstimatorVersion {
		t.Fatalf("unexpected corpus identity: schema=%d estimator=%q", corpus.SchemaVersion, corpus.EstimatorVersion)
	}
	requiredFeatures := map[string]bool{
		"straight": false, "nonuniform_node_spacing": false, "duplicate_nodes": false,
		"route_reversal": false, "s_curve": false, "compact_genuine_apex": false,
		"sparse_geometry": false, "compound_bend": false, "interpolation_overshoot_resistance": false,
		"one_node_zigzag": false, "cross_way_curve": false, "forward_direction": false,
		"reverse_direction": false, "ambiguous_fork": false, "tile_context_truncation": false,
		"current_way_rollover_mid_event": false,
	}
	for _, golden := range corpus.Cases {
		golden := golden
		t.Run(golden.Name, func(t *testing.T) {
			for _, feature := range golden.Features {
				if _, tracked := requiredFeatures[feature]; tracked {
					requiredFeatures[feature] = true
				}
			}
			estimate := EstimateWholeCurve(golden.Route, DefaultWholeCurveConfiguration())
			if len(estimate.Points) != golden.Expected.PointCount {
				t.Fatalf("point count=%d expected=%d", len(estimate.Points), golden.Expected.PointCount)
			}
			if len(estimate.Events) != len(golden.Expected.Events) {
				t.Fatalf("event count=%d expected=%d", len(estimate.Events), len(golden.Expected.Events))
			}
			for index := range estimate.Events {
				assertWholeCurveGoldenEvent(t, estimate.Events[index], golden.Expected.Events[index], corpus.Tolerances)
			}
		})
	}
	for feature, covered := range requiredFeatures {
		if !covered {
			t.Errorf("shared corpus does not cover %s", feature)
		}
	}
}

func assertWholeCurveGoldenEvent(t *testing.T, actual WholeCurveEvent, expected wholeCurveGoldenEvent, tolerance map[string]float64) {
	t.Helper()
	if actual.PhysicalID != expected.PhysicalID || actual.DirectionalID != expected.ID {
		t.Errorf("event identity=(%s,%s) expected=(%s,%s)", actual.PhysicalID, actual.DirectionalID, expected.PhysicalID, expected.ID)
	}
	if actual.StartIndex != expected.StartIndex || actual.EndIndex != expected.EndIndex || actual.ApexIndex != expected.ApexIndex {
		t.Errorf("event indices=(%d,%d,%d) expected=(%d,%d,%d)", actual.StartIndex, actual.EndIndex, actual.ApexIndex, expected.StartIndex, expected.EndIndex, expected.ApexIndex)
	}
	assertGoldenFloat(t, "length", actual.LengthMeters, expected.LengthMeters, tolerance["distanceMeters"])
	assertGoldenFloat(t, "turn", actual.SignedTurnRadians, expected.SignedTurnRadians, tolerance["turnRadians"])
	assertGoldenFloat(t, "coherence", actual.SignCoherence, expected.SignCoherence, tolerance["coherence"])
	assertGoldenOptionalFloat(t, "curvature60", actual.Curvature60, expected.Curvature60, tolerance["curvature"])
	assertGoldenOptionalFloat(t, "curvature100", actual.Curvature100, expected.Curvature100, tolerance["curvature"])
	assertGoldenOptionalFloat(t, "curvature160", actual.Curvature160, expected.Curvature160, tolerance["curvature"])
	assertGoldenFloat(t, "controlling", actual.ControllingCurvature, expected.ControllingCurvature, tolerance["curvature"])
	assertGoldenOptionalFloat(t, "scaleSpread", actual.ScaleSpread, expected.ScaleSpread, tolerance["curvature"])
	assertGoldenFloat(t, "sourceGap", actual.MaximumSourceGapMeters, expected.MaximumSourceGapMeters, tolerance["distanceMeters"])
	if actual.Confidence != expected.Confidence || !reflect.DeepEqual(actual.Flags, expected.Flags) {
		t.Errorf("confidence/flags=(%s,%v) expected=(%s,%v)", actual.Confidence, actual.Flags, expected.Confidence, expected.Flags)
	}
}

func assertGoldenFloat(t *testing.T, name string, actual, expected, tolerance float64) {
	t.Helper()
	if !wholeCurveIsFinite(actual) || math.Abs(actual-expected) > tolerance {
		t.Errorf("%s=%0.15g expected=%0.15g tolerance=%g", name, actual, expected, tolerance)
	}
}

func assertGoldenOptionalFloat(t *testing.T, name string, actual, expected *float64, tolerance float64) {
	t.Helper()
	if actual == nil || expected == nil {
		if actual != nil || expected != nil {
			t.Errorf("%s optional mismatch actual=%v expected=%v", name, actual, expected)
		}
		return
	}
	assertGoldenFloat(t, name, *actual, *expected, tolerance)
}

func TestWholeCurveGoldenReversalAndDuplicateInvariants(t *testing.T) {
	corpus := loadWholeCurveGoldenCorpus(t)
	byName := make(map[string]wholeCurveGoldenCase, len(corpus.Cases))
	for _, golden := range corpus.Cases {
		byName[golden.Name] = golden
	}
	forward := byName["broad_left"].Expected.Events[0]
	reverse := byName["broad_left_reversed"].Expected.Events[0]
	duplicate := byName["duplicate_nodes"].Expected.Events[0]
	if forward.PhysicalID != reverse.PhysicalID || forward.ID == reverse.ID || math.Signbit(forward.ControllingCurvature) == math.Signbit(reverse.ControllingCurvature) {
		t.Fatalf("reversal did not preserve physical identity and invert direction/sign: forward=%+v reverse=%+v", forward, reverse)
	}
	if forward.PhysicalID != duplicate.PhysicalID || forward.ID != duplicate.ID {
		t.Fatalf("duplicate-node filtering changed event identity: forward=%+v duplicate=%+v", forward, duplicate)
	}
}

func TestWholeCurveSavedBankGoldenCorpus(t *testing.T) {
	corpus := loadWholeCurveGoldenCorpus(t)
	bank := corpus.SavedBank
	if bank.SampleCount != 18 || bank.DirectionalEventCount != 12 || len(bank.Samples) != 18 || len(bank.Cases) != 12 {
		t.Fatalf("unexpected saved-bank dimensions: samples=%d/%d events=%d/%d", bank.SampleCount, len(bank.Samples), bank.DirectionalEventCount, len(bank.Cases))
	}
	samplesByNumber := make(map[int]wholeCurveSavedSample, len(bank.Samples))
	for _, sample := range bank.Samples {
		samplesByNumber[sample.Number] = sample
	}
	covered := make(map[int]bool, 18)
	foundSample16 := false
	for _, golden := range bank.Cases {
		estimate := EstimateWholeCurve(golden.Route, DefaultWholeCurveConfiguration())
		if len(estimate.Points) != golden.ExpectedPointCount {
			t.Errorf("%s point count=%d expected=%d", golden.Name, len(estimate.Points), golden.ExpectedPointCount)
		}
		var actual *WholeCurveEvent
		for index := range estimate.Events {
			if estimate.Events[index].DirectionalID == golden.ExpectedEvent.ID {
				actual = &estimate.Events[index]
				break
			}
		}
		if actual == nil {
			t.Errorf("%s did not reproduce event %s (got %d events)", golden.Name, golden.ExpectedEvent.ID, len(estimate.Events))
			continue
		}
		assertWholeCurveGoldenEvent(t, *actual, golden.ExpectedEvent, corpus.Tolerances)
		sourceKeys := make(map[string]bool, len(actual.SourceKeys))
		for _, sourceKey := range actual.SourceKeys {
			sourceKeys[sourceKey] = true
		}
		for _, number := range golden.SampleNumbers {
			covered[number] = true
			sample, ok := samplesByNumber[number]
			if !ok {
				t.Errorf("%s references missing sample %d", golden.Name, number)
			} else if !sourceKeys[sample.SourceKey] {
				t.Errorf("%s event %s lost exact provenance for sample %d (%s)", golden.Name, actual.DirectionalID, number, sample.SourceKey)
			}
			if number == 16 {
				foundSample16 = true
				if math.Abs(golden.ExpectedStudy.WholeCurveSpeedMPH-97.9109081492345) > 1e-9 {
					t.Errorf("sample 16 result was hidden or changed: %.15g mph", golden.ExpectedStudy.WholeCurveSpeedMPH)
				}
			}
		}
		if !wholeCurveIsFinite(golden.ExpectedStudy.WholeCurveSpeedMPH) || golden.ExpectedStudy.WholeCurveSpeedMPH <= 0 {
			t.Errorf("%s has invalid checked-in study speed %.6f", golden.Name, golden.ExpectedStudy.WholeCurveSpeedMPH)
		}
	}
	for number := 1; number <= 18; number++ {
		if !covered[number] {
			t.Errorf("saved sample %d is not covered by the 12 checked-in directional events", number)
		}
	}
	if !foundSample16 {
		t.Error("sample 16 is absent from the checked-in bank corpus")
	}
}
