package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"
)

func TestWholeCurveFingerprintCrossLanguageVector(t *testing.T) {
	points := []WholeCurveProfilePoint{
		{Latitude: 37.0, Longitude: -122.0, DistanceMeters: 0, Curvature: 0},
		{Latitude: 37.000045, Longitude: -122.0, DistanceMeters: 5, Curvature: -0.0123456785, EventID: "0123456789abcdefabcd-a"},
		{Latitude: 37.000090, Longitude: -122.0, DistanceMeters: 10, Curvature: 0.0123456785, EventID: "0123456789abcdefabcd-b"},
	}
	fingerprint, err := WholeCurveRouteFingerprint(7, points)
	if err != nil {
		t.Fatal(err)
	}
	const expected = "7a47531720b4846e1ee3094443068a42ac936554efc42b27b07796a3103b2624"
	if fingerprint != expected {
		t.Fatalf("fingerprint=%s expected=%s", fingerprint, expected)
	}
}

func TestWholeCurveProfileMaterializesSignedGeometry(t *testing.T) {
	corpus := loadWholeCurveGoldenCorpus(t)
	var golden wholeCurveGoldenCase
	for _, candidate := range corpus.Cases {
		if candidate.Name == "broad_left" {
			golden = candidate
			break
		}
	}
	route := DirectionalRoute{Generation: 9, TruncatedAfter: true}
	for index, point := range golden.Route {
		route.Nodes = append(route.Nodes, DirectionalRouteNode{
			Latitude: point.Latitude, Longitude: point.Longitude,
			Sources: []RouteNodeSource{{WayID: "golden", NodeIndex: index, IsForward: true}},
		})
	}
	recomputeRouteDistances(route.Nodes)
	pos := Position{Latitude: golden.Route[5].Latitude, Longitude: golden.Route[5].Longitude}
	now := time.Unix(1_800_000_000, 123_000_000)
	profile, estimate, err := BuildWholeCurveProfile(route, pos, WholeCurveEstimate{}, now)
	if err != nil {
		t.Fatal(err)
	}
	if profile.EstimatorVersion != WholeCurveEstimatorVersion || profile.Generation != 9 || profile.GeneratedAtUnixMillis != now.UnixMilli() {
		t.Fatalf("unexpected profile identity: %+v", profile)
	}
	if len(profile.Events) != 1 || len(estimate.Events) != 1 {
		t.Fatalf("expected one curve event, profile=%d estimate=%d", len(profile.Events), len(estimate.Events))
	}
	event := profile.Events[0]
	if event.EventID == "" || event.ControllingCurvature <= 0 {
		t.Fatalf("expected signed left event: %+v", event)
	}
	materialized := 0
	for index, point := range profile.Points {
		if point.EventID == event.EventID {
			materialized++
			if point.Curvature != event.ControllingCurvature || index < event.StartIndex || index > event.EndIndex {
				t.Fatalf("inconsistent materialized point %d: %+v event=%+v", index, point, event)
			}
		} else if point.Curvature != 0 {
			t.Fatalf("point outside event carries curvature: %+v", point)
		}
	}
	if materialized != event.EndIndex-event.StartIndex+1 {
		t.Fatalf("materialized=%d expected=%d", materialized, event.EndIndex-event.StartIndex+1)
	}
	fingerprint, err := WholeCurveRouteFingerprint(profile.Generation, profile.Points)
	if err != nil || fingerprint != profile.RouteFingerprint {
		t.Fatalf("fingerprint mismatch got=%s recomputed=%s err=%v", profile.RouteFingerprint, fingerprint, err)
	}
	data, err := json.Marshal(profile)
	if err != nil || bytes.Contains(data, []byte("NaN")) || bytes.Contains(data, []byte("Infinity")) {
		t.Fatalf("profile is not finite JSON: %s err=%v", data, err)
	}
}

func TestBuildInfoJSONAndRawMarkers(t *testing.T) {
	var output bytes.Buffer
	if err := WriteMapdBuildInfo(&output); err != nil {
		t.Fatal(err)
	}
	var info struct {
		ReleaseID        string   `json:"releaseID"`
		BuildID          string   `json:"buildID"`
		EstimatorVersion string   `json:"estimatorVersion"`
		Capabilities     []string `json:"capabilities"`
		IdentityMarkers  []string `json:"identityMarkers"`
	}
	if err := json.Unmarshal(output.Bytes(), &info); err != nil {
		t.Fatal(err)
	}
	if info.ReleaseID != MapdReleaseID || info.BuildID != MapdBuildID || info.EstimatorVersion != WholeCurveEstimatorVersion {
		t.Fatalf("unexpected build info: %+v", info)
	}
	if len(info.Capabilities) != 1 || info.Capabilities[0] != MapWholeCurveCapability {
		t.Fatalf("unexpected capabilities: %v", info.Capabilities)
	}
	wantMarkers := []string{"MapdReleaseID:" + MapdReleaseID, "MapdBuildID:" + MapdBuildID}
	if len(info.IdentityMarkers) != 2 || info.IdentityMarkers[0] != wantMarkers[0] || info.IdentityMarkers[1] != wantMarkers[1] {
		t.Fatalf("identity markers=%v expected=%v", info.IdentityMarkers, wantMarkers)
	}
}

func TestDirectionalRouteRolloverKeepsGenerationAndDetectsDivergence(t *testing.T) {
	first := makeDirectionalRouteTestWay(t, "first", [][2]float64{{0, 0}, {100, 0}, {200, 0}})
	second := makeDirectionalRouteTestWay(t, "second", [][2]float64{{200, 0}, {300, 20}, {400, 50}})
	continuation := makeDirectionalRouteTestWay(t, "continuation", [][2]float64{{400, 50}, {600, 80}, {900, 80}, {1300, 80}, {1700, 80}})
	branch := makeDirectionalRouteTestWay(t, "branch", [][2]float64{{400, 50}, {550, -20}, {900, -80}, {1300, -80}, {1700, -80}})

	firstCurrent := routeTestCurrent(first, true)
	secondNext := routeTestNext(second, true, false)
	continuationNext := routeTestNext(continuation, true, false)
	initial, err := BuildDirectionalRoute(firstCurrent, []NextWayResult{secondNext, continuationNext}, DirectionalRoute{}, routeTestPosition(50, 0))
	if err != nil {
		t.Fatal(err)
	}
	secondCurrent := routeTestCurrent(second, true)
	rolled, err := BuildDirectionalRoute(secondCurrent, []NextWayResult{continuationNext}, initial, routeTestPosition(320, 26))
	if err != nil {
		t.Fatal(err)
	}
	if rolled.Generation != initial.Generation {
		t.Fatalf("same corridor rollover changed generation %d -> %d", initial.Generation, rolled.Generation)
	}
	if len(rolled.Nodes) <= len(continuationNextNodes(t, second, continuation)) || rolled.Nodes[0].Sources[0].WayID == rolled.CurrentWayID {
		t.Fatal("rollover did not retain predecessor provenance")
	}
	diverged, err := BuildDirectionalRoute(secondCurrent, []NextWayResult{routeTestNext(branch, true, false)}, initial, routeTestPosition(320, 26))
	if err != nil {
		t.Fatal(err)
	}
	if diverged.Generation == initial.Generation {
		t.Fatalf("changed future branch retained generation %d", initial.Generation)
	}
	ambiguous, err := BuildDirectionalRoute(secondCurrent, []NextWayResult{routeTestNext(continuation, true, true)}, DirectionalRoute{}, routeTestPosition(320, 26))
	if err != nil {
		t.Fatal(err)
	}
	if !ambiguous.FatalAmbiguity {
		t.Fatal("ambiguous continuation did not fail closed")
	}
}

func TestContinuationAmbiguityRejectsOnlyUnresolvedPriorityClass(t *testing.T) {
	current := makeDirectionalRouteTestWay(t, "US 50", [][2]float64{{0, 0}, {50, 0}, {100, 0}})
	mainline := makeDirectionalRouteTestWay(t, "US 50", [][2]float64{{100, 0}, {150, 0}, {220, 0}})
	sideRoad := makeDirectionalRouteTestWay(t, "Side Road", [][2]float64{{100, 0}, {100, 50}, {100, 120}})
	_ = sideRoad.SetRef("County 12")
	currentNodes, _ := current.Nodes()
	matchBearingNode := currentNodes.At(currentNodes.Len() - 2)
	matchNode := currentNodes.At(currentNodes.Len() - 1)
	if continuationSelectionAmbiguous(current, mainline, []Way{mainline, sideRoad}, matchNode, matchBearingNode) {
		t.Fatal("unique same-name/ref mainline was incorrectly fatalized by an ordinary side-road intersection")
	}

	forkLeft := makeDirectionalRouteTestWay(t, "US 50", [][2]float64{{100, 0}, {150, 20}, {210, 50}})
	forkRight := makeDirectionalRouteTestWay(t, "US 50", [][2]float64{{100, 0}, {150, -20}, {210, -50}})
	if !continuationSelectionAmbiguous(current, forkLeft, []Way{forkLeft, forkRight}, matchNode, matchBearingNode) {
		t.Fatal("same-priority symmetric fork was not marked as a fatal unresolved branch")
	}
}

func continuationNextNodes(t *testing.T, ways ...Way) []Coordinates {
	t.Helper()
	var result []Coordinates
	for _, way := range ways {
		nodes, err := way.Nodes()
		if err != nil {
			t.Fatal(err)
		}
		for index := 0; index < nodes.Len(); index++ {
			result = append(result, nodes.At(index))
		}
	}
	return result
}

func makeDirectionalRouteTestWay(t *testing.T, name string, meters [][2]float64) Way {
	t.Helper()
	_, segment, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		t.Fatal(err)
	}
	way, err := NewRootWay(segment)
	if err != nil {
		t.Fatal(err)
	}
	if err := way.SetName(name); err != nil {
		t.Fatal(err)
	}
	if err := way.SetRef("US 50"); err != nil {
		t.Fatal(err)
	}
	way.SetLanes(2)
	nodes, err := way.NewNodes(int32(len(meters)))
	if err != nil {
		t.Fatal(err)
	}
	for index, point := range meters {
		node := nodes.At(index)
		node.SetLatitude(37.3 + point[1]/R*TO_DEGREES)
		node.SetLongitude(-120.4 + point[0]/(R*math.Cos(37.3*TO_RADIANS))*TO_DEGREES)
	}
	return way
}

func routeTestCurrent(way Way, forward bool) CurrentWay {
	start, end := GetWayStartEnd(way, forward)
	return CurrentWay{Way: way, OnWay: OnWayResult{IsForward: forward}, StartPosition: start, EndPosition: end}
}

func routeTestNext(way Way, forward, ambiguous bool) NextWayResult {
	start, end := GetWayStartEnd(way, forward)
	return NextWayResult{Way: way, IsForward: forward, StartPosition: start, EndPosition: end, CandidateCount: 1, Ambiguous: ambiguous}
}

func routeTestPosition(east, north float64) Position {
	return Position{
		Latitude:  37.3 + north/R*TO_DEGREES,
		Longitude: -120.4 + east/(R*math.Cos(37.3*TO_RADIANS))*TO_DEGREES,
	}
}

func TestBuildInfoContainsRequiredLiteralMarkers(t *testing.T) {
	joined := strings.Join(CurrentMapdBuildInfo().IdentityMarkers, "\n") + "\n" + MapWholeCurveCapability
	for _, marker := range []string{
		"MapdReleaseID:" + MapdReleaseID,
		"MapdBuildID:" + MapdBuildID,
		"MapWholeCurveProfile:whole-curve-v1",
	} {
		if !strings.Contains(joined, marker) {
			t.Fatalf("missing marker %q", marker)
		}
	}
}

func TestSeedLastGPSPositionFromPersistentValidatesBeforeWriting(t *testing.T) {
	root := t.TempDir()
	persistentDir := filepath.Join(root, "params", "d")
	memoryDir := filepath.Join(root, "mem", "d")
	if err := os.MkdirAll(persistentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(memoryDir, 0o755); err != nil {
		t.Fatal(err)
	}
	persistentPath := filepath.Join(persistentDir, "LastGPSPosition")
	memoryPath := filepath.Join(memoryDir, "LastGPSPosition")
	now := time.Unix(1_800_000_000, 0)
	valid := []byte(`{"latitude":38.73152,"longitude":-120.78821,"bearing":91.5,"altitude":1234}`)
	if err := os.WriteFile(persistentPath, valid, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(persistentPath, now, now); err != nil {
		t.Fatal(err)
	}
	position, err := seedLastGPSPositionFromPersistent(persistentPath, memoryPath, now)
	if err != nil {
		t.Fatal(err)
	}
	if position.Latitude != 38.73152 || position.Longitude != -120.78821 || position.Bearing != 91.5 {
		t.Fatalf("unexpected seeded position: %+v", position)
	}
	seeded, err := os.ReadFile(memoryPath)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodePosition(seeded)
	if err != nil || decoded != position {
		t.Fatalf("memory seed=%s decoded=%+v err=%v", seeded, decoded, err)
	}

	for name, invalid := range map[string][]byte{
		"missing": []byte(`{}`),
		"nan":     []byte(`{"latitude":"NaN","longitude":-120.7,"bearing":90}`),
		"range":   []byte(`{"latitude":91,"longitude":-120.7,"bearing":90}`),
		"bearing": []byte(`{"latitude":38.7,"longitude":-120.7,"bearing":360}`),
	} {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(persistentPath, invalid, 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Chtimes(persistentPath, now, now); err != nil {
				t.Fatal(err)
			}
			before, _ := os.ReadFile(memoryPath)
			if _, err := seedLastGPSPositionFromPersistent(persistentPath, memoryPath, now); err == nil {
				t.Fatal("invalid persistent position was accepted")
			}
			after, _ := os.ReadFile(memoryPath)
			if !bytes.Equal(before, after) {
				t.Fatalf("invalid seed mutated memory position: before=%s after=%s", before, after)
			}
		})
	}
}

func TestSeedLastGPSPositionRejectsStaleOrFutureFile(t *testing.T) {
	root := t.TempDir()
	persistentDir := filepath.Join(root, "params", "d")
	memoryDir := filepath.Join(root, "mem", "d")
	_ = os.MkdirAll(persistentDir, 0o755)
	_ = os.MkdirAll(memoryDir, 0o755)
	persistentPath := filepath.Join(persistentDir, "LastGPSPosition")
	memoryPath := filepath.Join(memoryDir, "LastGPSPosition")
	now := time.Unix(1_800_000_000, 0)
	if err := os.WriteFile(persistentPath, []byte(`{"latitude":38.7,"longitude":-120.7,"bearing":90}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, modified := range map[string]time.Time{
		"stale":  now.Add(-persistentPositionMaximumAge - time.Second),
		"future": now.Add(persistentPositionFutureSlop + time.Second),
	} {
		t.Run(name, func(t *testing.T) {
			if err := os.Chtimes(persistentPath, modified, modified); err != nil {
				t.Fatal(err)
			}
			if _, err := seedLastGPSPositionFromPersistent(persistentPath, memoryPath, now); err == nil {
				t.Fatalf("%s persistent position was accepted", name)
			}
			if _, err := os.Stat(memoryPath); !os.IsNotExist(err) {
				t.Fatalf("rejected %s seed created memory param: %v", name, err)
			}
		})
	}
}
