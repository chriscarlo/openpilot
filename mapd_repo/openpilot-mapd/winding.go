package main

import (
	"math"
	"sort"
)

const (
	windingMaxSpeedDefaultMps   = 45.0
	windingWindowDistanceM      = 325.0
	windingMinMeaningfulDropMps = 2.0
	windingMeaningfulDropRatio  = 0.12
	windingLocalMinEpsMps       = 0.05
	windingRearmRiseMps         = 0.75
	windingShortGapMaxM         = 120.0
	windingMinCurveCurvature    = 0.0010
)

type CompactWindingSummary struct {
	Level      uint8
	Score      uint8
	Confidence uint8
}

type windingNode struct {
	Lat float64
	Lon float64
}

type windingCurvatureSample struct {
	S        float64
	ArcM     float64
	VsafeMps float64
	TurnRad  float64
	TurnSign int
}

type windingAnchor struct {
	S        float64
	VsafeMps float64
	Sign     int
}

type windingSummary struct {
	Level      int
	Score      float64
	Confidence float64
}

func ComputeWayWindingMetadata(nodes []TmpNode) (CompactWindingSummary, CompactWindingSummary) {
	forwardNodes := directedTmpNodes(nodes, true)
	backwardNodes := directedTmpNodes(nodes, false)
	return compactWindingSummary(summarizeWinding(forwardNodes)),
		compactWindingSummary(summarizeWinding(backwardNodes))
}

func directedTmpNodes(nodes []TmpNode, forward bool) []windingNode {
	if len(nodes) == 0 {
		return nil
	}
	out := make([]windingNode, len(nodes))
	for i := 0; i < len(nodes); i++ {
		idx := i
		if !forward {
			idx = len(nodes) - 1 - i
		}
		out[i] = windingNode{Lat: nodes[idx].Latitude, Lon: nodes[idx].Longitude}
	}
	return out
}

func compactWindingSummary(summary windingSummary) CompactWindingSummary {
	return CompactWindingSummary{
		Level:      uint8(windingClampFloat(float64(summary.Level), 0.0, 5.0)),
		Score:      windingScale01ToUint8(summary.Score),
		Confidence: windingScale01ToUint8(summary.Confidence),
	}
}

func windingScale01ToUint8(v float64) uint8 {
	return uint8(math.Round(windingClip01(v) * 255.0))
}

func summarizeWinding(nodes []windingNode) windingSummary {
	samples, totalLen := buildWindingSamples(nodes)
	if len(samples) == 0 || totalLen < 25.0 {
		return windingSummary{
			Level:      0,
			Score:      0.0,
			Confidence: 0.10,
		}
	}

	best := windingSummary{
		Level:      0,
		Score:      0.0,
		Confidence: windingConfidenceForWindow(totalLen, len(samples), totalLen),
	}
	for startIdx := range samples {
		windowEndS := windingMin(totalLen, samples[startIdx].S+windingWindowDistanceM)
		endIdx := startIdx
		for endIdx+1 < len(samples) && samples[endIdx+1].S <= windowEndS+1e-6 {
			endIdx++
		}
		window := analyzeWindingWindow(samples[startIdx:endIdx+1], totalLen, samples[startIdx].S, windowEndS)
		if betterWindingSummary(window, best) {
			best = window
		}
	}
	return best
}

func betterWindingSummary(a, b windingSummary) bool {
	if a.Level != b.Level {
		return a.Level > b.Level
	}
	if math.Abs(a.Score-b.Score) > 1e-9 {
		return a.Score > b.Score
	}
	return a.Confidence > b.Confidence
}

func analyzeWindingWindow(samples []windingCurvatureSample, totalLen, startS, endS float64) windingSummary {
	if len(samples) == 0 {
		return windingSummary{}
	}
	windowLen := windingMax(1.0, endS-startS)
	vsafeList := make([]float64, len(samples))
	for i, sample := range samples {
		vsafeList[i] = sample.VsafeMps
	}
	sort.Float64s(vsafeList)
	refVsafe := windingPercentile(vsafeList, 0.85)
	drop := windingMax(windingMinMeaningfulDropMps, refVsafe*windingMeaningfulDropRatio)
	anchorThreshold := refVsafe - drop

	anchors := extractWindingAnchors(samples, anchorThreshold)
	shortGapCount := 0
	signChanges := 0
	for i := 1; i < len(anchors); i++ {
		if anchors[i].S-anchors[i-1].S <= windingShortGapMaxM+1e-6 {
			shortGapCount++
		}
		if anchors[i].Sign != 0 && anchors[i-1].Sign != 0 && anchors[i].Sign != anchors[i-1].Sign {
			signChanges++
		}
	}

	curveDist := 0.0
	headingAbsDeg := 0.0
	minAnchorVsafe := windingMaxSpeedDefaultMps
	for _, sample := range samples {
		if sample.VsafeMps <= anchorThreshold+1e-6 {
			curveDist += sample.ArcM
		}
		headingAbsDeg += math.Abs(sample.TurnRad) * TO_DEGREES
	}
	for _, anchor := range anchors {
		minAnchorVsafe = windingMin(minAnchorVsafe, anchor.VsafeMps)
	}
	if len(anchors) == 0 {
		minAnchorVsafe = refVsafe
	}

	curveFraction := curveDist / windowLen
	anchorScore := windingClip01((float64(len(anchors)) - 1.0) / 3.0)
	gapScore := windingClip01(float64(shortGapCount) / windingMax(1.0, float64(len(anchors)-1)))
	densityScore := windingClip01((curveFraction - 0.10) / 0.50)
	tightnessScore := windingClip01((refVsafe - minAnchorVsafe - 2.0) / 10.0)
	headingScore := windingClip01((headingAbsDeg - 20.0) / 220.0)
	reversalScore := windingClip01(float64(signChanges) / windingMax(1.0, float64(len(anchors)-1)))

	score := 0.25*anchorScore +
		0.20*gapScore +
		0.20*densityScore +
		0.15*tightnessScore +
		0.10*headingScore +
		0.10*reversalScore

	return windingSummary{
		Level: windingLevelForMetrics(
			score,
			len(anchors),
			shortGapCount,
			signChanges,
			headingAbsDeg,
			minAnchorVsafe,
			curveFraction,
		),
		Score:      score,
		Confidence: windingConfidenceForWindow(windowLen, len(samples), totalLen),
	}
}

func extractWindingAnchors(samples []windingCurvatureSample, threshold float64) []windingAnchor {
	if len(samples) == 0 {
		return nil
	}
	minima := make([]windingAnchor, 0, len(samples))
	for i := range samples {
		v := samples[i].VsafeMps
		prev := math.Inf(1)
		next := math.Inf(1)
		if i > 0 {
			prev = samples[i-1].VsafeMps
		}
		if i+1 < len(samples) {
			next = samples[i+1].VsafeMps
		}
		localMin := v <= prev+windingLocalMinEpsMps && v <= next+windingLocalMinEpsMps &&
			(i == 0 || i+1 == len(samples) || v < prev-windingLocalMinEpsMps || v < next-windingLocalMinEpsMps)
		if !localMin || v > threshold+1e-6 {
			continue
		}
		minima = append(minima, windingAnchor{
			S:        samples[i].S,
			VsafeMps: v,
			Sign:     samples[i].TurnSign,
		})
	}
	if len(minima) == 0 {
		return nil
	}

	accepted := []windingAnchor{minima[0]}
	for _, candidate := range minima[1:] {
		last := accepted[len(accepted)-1]
		if candidate.VsafeMps < last.VsafeMps-windingLocalMinEpsMps ||
			candidate.VsafeMps >= last.VsafeMps+windingRearmRiseMps ||
			candidate.S-last.S > windingShortGapMaxM {
			accepted = append(accepted, candidate)
		}
	}
	return accepted
}

func windingLevelForMetrics(score float64, anchors, shortGaps, signChanges int, headingAbsDeg, minAnchorVsafe, curveFraction float64) int {
	switch {
	case score >= 0.78 &&
		anchors >= 5 &&
		shortGaps >= 4 &&
		signChanges >= 3 &&
		headingAbsDeg >= 320.0 &&
		minAnchorVsafe <= 7.0 &&
		curveFraction >= 0.40:
		return 5
	case score >= 0.62 &&
		anchors >= 4 &&
		shortGaps >= 3 &&
		signChanges >= 2 &&
		headingAbsDeg >= 220.0 &&
		minAnchorVsafe <= 9.5 &&
		curveFraction >= 0.30:
		return 4
	case score >= 0.46 &&
		anchors >= 3 &&
		shortGaps >= 2 &&
		headingAbsDeg >= 110.0 &&
		minAnchorVsafe <= 14.0 &&
		curveFraction >= 0.20:
		return 3
	case score >= 0.30 &&
		anchors >= 2 &&
		headingAbsDeg >= 45.0 &&
		curveFraction >= 0.10:
		return 2
	case score >= 0.16 &&
		anchors >= 1:
		return 1
	default:
		return 0
	}
}

func windingConfidenceForWindow(windowLen float64, samples int, totalLen float64) float64 {
	coverage := windingClip01(windowLen / windingWindowDistanceM)
	density := windingClip01((float64(samples) / windingMax(1.0, windowLen)) * 30.0)
	lengthConfidence := windingClip01(totalLen / 120.0)
	return windingClip01(0.45*coverage + 0.35*density + 0.20*lengthConfidence)
}

func buildWindingSamples(nodes []windingNode) ([]windingCurvatureSample, float64) {
	if len(nodes) < 3 {
		return nil, 0.0
	}
	segCum := make([]float64, len(nodes))
	totalLen := 0.0
	for i := 1; i < len(nodes); i++ {
		totalLen += windingDistanceMeters(nodes[i-1], nodes[i])
		segCum[i] = totalLen
	}

	samples := make([]windingCurvatureSample, 0, len(nodes)-2)
	for i := 0; i+2 < len(nodes); i++ {
		curvature, arcM, turnRad := windingSignedCurvature(nodes[i], nodes[i+1], nodes[i+2])
		absCurvature := math.Abs(curvature)
		sign := 0
		if curvature > 1e-9 {
			sign = 1
		} else if curvature < -1e-9 {
			sign = -1
		}
		if absCurvature < windingMinCurveCurvature {
			sign = 0
		}
		samples = append(samples, windingCurvatureSample{
			S:        segCum[i+1],
			ArcM:     arcM,
			VsafeMps: windingCurvatureToSpeed(absCurvature),
			TurnRad:  turnRad,
			TurnSign: sign,
		})
	}
	return samples, totalLen
}

func windingCurvatureToSpeed(absCurvature float64) float64 {
	if absCurvature <= 1e-9 {
		return windingMaxSpeedDefaultMps
	}
	return windingMin(windingMaxSpeedDefaultMps, math.Sqrt(TARGET_LAT_ACCEL/absCurvature))
}

func windingSignedCurvature(a, b, c windingNode) (float64, float64, float64) {
	ab := windingDistanceMeters(a, b)
	ac := windingDistanceMeters(a, c)
	bc := windingDistanceMeters(b, c)
	if ab <= 1e-6 || ac <= 1e-6 || bc <= 1e-6 {
		return 0.0, 0.0, 0.0
	}
	semiPerimeter := (ab + ac + bc) / 2.0
	areaSq := semiPerimeter * (semiPerimeter - ab) * (semiPerimeter - ac) * (semiPerimeter - bc)
	if areaSq <= 0.0 {
		return 0.0, 0.0, 0.0
	}
	area := math.Sqrt(areaSq)
	curvature := (4.0 * area) / (ab * ac * bc)
	radius := 1.0 / windingMax(curvature, 1e-12)
	arg := ((radius * radius * 2.0) - (ac * ac)) / (2.0 * radius * radius)
	arg = windingMin(1.0, windingMax(-1.0, arg))
	turnRad := math.Acos(arg)

	ax, ay := windingLocalXY(a, b)
	cx, cy := windingLocalXY(c, b)
	cross := ax*cy - ay*cx
	if cross < 0 {
		curvature = -curvature
		turnRad = -turnRad
	}
	return curvature, radius * math.Abs(turnRad), turnRad
}

func windingLocalXY(p, origin windingNode) (float64, float64) {
	dLat := (p.Lat - origin.Lat) * TO_RADIANS
	dLon := (p.Lon - origin.Lon) * TO_RADIANS
	x := R * dLon * math.Cos(origin.Lat*TO_RADIANS)
	y := R * dLat
	return x, y
}

func windingDistanceMeters(a, b windingNode) float64 {
	latA := a.Lat * TO_RADIANS
	lonA := a.Lon * TO_RADIANS
	latB := b.Lat * TO_RADIANS
	lonB := b.Lon * TO_RADIANS
	dLat := latB - latA
	dLon := lonB - lonA
	h := math.Sin(dLat/2.0)*math.Sin(dLat/2.0) + math.Cos(latA)*math.Cos(latB)*math.Sin(dLon/2.0)*math.Sin(dLon/2.0)
	return R * 2.0 * math.Atan2(math.Sqrt(h), math.Sqrt(1.0-h))
}

func windingPercentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	if len(values) == 1 {
		return values[0]
	}
	pos := p * float64(len(values)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return values[lo]
	}
	t := pos - float64(lo)
	return values[lo] + (values[hi]-values[lo])*t
}

func windingClip01(v float64) float64 {
	return windingMin(1.0, windingMax(0.0, v))
}

func windingMin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func windingMax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func windingClampFloat(v, lo, hi float64) float64 {
	return windingMin(hi, windingMax(lo, v))
}
