package main

import (
	"math"
	"testing"
)

func testTmpNodeFromMeters(originLat, originLon, eastM, northM float64) TmpNode {
	return TmpNode{
		Latitude:  originLat + (northM/R)*TO_DEGREES,
		Longitude: originLon + (eastM/(R*math.Cos(originLat*TO_RADIANS)))*TO_DEGREES,
	}
}

func testTmpNodesFromPath(points [][2]float64) []TmpNode {
	const originLat = 38.73152
	const originLon = -120.78821
	nodes := make([]TmpNode, 0, len(points))
	for _, pt := range points {
		nodes = append(nodes, testTmpNodeFromMeters(originLat, originLon, pt[0], pt[1]))
	}
	return nodes
}

func testTmpNodesFromHeadings(stepLenM float64, headingsDeg []float64) []TmpNode {
	points := make([][2]float64, 0, len(headingsDeg)+1)
	east := 0.0
	north := 0.0
	points = append(points, [2]float64{east, north})
	for _, headingDeg := range headingsDeg {
		headingRad := headingDeg * TO_RADIANS
		east += stepLenM * math.Cos(headingRad)
		north += stepLenM * math.Sin(headingRad)
		points = append(points, [2]float64{east, north})
	}
	return testTmpNodesFromPath(points)
}

func TestComputeWayWindingMetadataStraightRoadStaysLow(t *testing.T) {
	points := make([][2]float64, 0, 24)
	for x := 0.0; x <= 460.0; x += 20.0 {
		points = append(points, [2]float64{x, 0.0})
	}

	forward, backward := ComputeWayWindingMetadata(testTmpNodesFromPath(points))

	if forward.Level != 0 || backward.Level != 0 {
		t.Fatalf("expected straight road to stay level 0, got forward=%d backward=%d", forward.Level, backward.Level)
	}
	if forward.Score > 32 || backward.Score > 32 {
		t.Fatalf("expected low winding score on straight road, got forward=%d backward=%d", forward.Score, backward.Score)
	}
}

func TestComputeWayWindingMetadataSeparatedBendsStayModerate(t *testing.T) {
	points := [][2]float64{
		{0, 0}, {20, 0}, {40, 0}, {60, 8}, {80, 28}, {100, 52}, {120, 72},
		{180, 72}, {240, 72}, {300, 72},
		{320, 66}, {340, 48}, {360, 24}, {380, 0}, {400, -16}, {420, -28},
	}

	forward, backward := ComputeWayWindingMetadata(testTmpNodesFromPath(points))

	if forward.Level > 2 || backward.Level > 2 {
		t.Fatalf("expected separated bends to stay at or below level 2, got forward=%d backward=%d", forward.Level, backward.Level)
	}
}

func TestComputeWayWindingMetadataDenseSwitchbacksScoreHigh(t *testing.T) {
	separatedPoints := [][2]float64{
		{0, 0}, {20, 0}, {40, 0}, {60, 8}, {80, 28}, {100, 52}, {120, 72},
		{180, 72}, {240, 72}, {300, 72},
		{320, 66}, {340, 48}, {360, 24}, {380, 0}, {400, -16}, {420, -28},
	}
	moderateForward, moderateBackward := ComputeWayWindingMetadata(testTmpNodesFromPath(separatedPoints))

	points := [][2]float64{
		{0, 0}, {18, 0}, {36, 0}, {48, 10}, {48, 28}, {36, 38}, {18, 38}, {0, 38},
		{-12, 48}, {-12, 66}, {0, 76}, {18, 76}, {36, 76}, {48, 86}, {48, 104}, {36, 114}, {18, 114}, {0, 114},
		{-12, 124}, {-12, 142}, {0, 152}, {18, 152}, {36, 152}, {48, 162}, {48, 180}, {36, 190}, {18, 190}, {0, 190},
		{-12, 200}, {-12, 218}, {0, 228}, {18, 228}, {36, 228}, {48, 238}, {48, 256}, {36, 266}, {18, 266}, {0, 266},
	}
	forward, backward := ComputeWayWindingMetadata(testTmpNodesFromPath(points))

	if forward.Level < 2 || backward.Level < 2 {
		t.Fatalf("expected dense switchbacks to score above low winding levels, got forward=%d backward=%d", forward.Level, backward.Level)
	}
	if forward.Score <= moderateForward.Score || backward.Score <= moderateBackward.Score {
		t.Fatalf(
			"expected dense switchbacks to outrank separated bends, got dense=(%d,%d) moderate=(%d,%d)",
			forward.Score,
			backward.Score,
			moderateForward.Score,
			moderateBackward.Score,
		)
	}
	if forward.Confidence < 180 || backward.Confidence < 180 {
		t.Fatalf("expected dense switchbacks to have high confidence, got forward=%d backward=%d", forward.Confidence, backward.Confidence)
	}
}

func TestWindingLevelForMetricsSupportsHighSeverityBuckets(t *testing.T) {
	if got := windingLevelForMetrics(0.69, 4, 3, 2, 240.0, 8.5, 0.32); got != 4 {
		t.Fatalf("expected representative L4 metrics to classify as level 4, got %d", got)
	}
	if got := windingLevelForMetrics(0.85, 6, 5, 4, 360.0, 5.5, 0.46); got != 5 {
		t.Fatalf("expected representative L5 metrics to classify as level 5, got %d", got)
	}
}
