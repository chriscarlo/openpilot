package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
)

const WholeCurveEstimatorVersion = "whole-curve-v1"

type WholeCurveConfiguration struct {
	ResampleSpacingMeters        float64 `json:"resampleSpacingMeters"`
	ShortSpanMeters              float64 `json:"shortSpanMeters"`
	NominalSpanMeters            float64 `json:"nominalSpanMeters"`
	LongSpanMeters               float64 `json:"longSpanMeters"`
	EnterCurvature               float64 `json:"enterCurvature"`
	ExitCurvature                float64 `json:"exitCurvature"`
	SustainedDistanceMeters      float64 `json:"sustainedDistanceMeters"`
	SameSignMergeGapMeters       float64 `json:"sameSignMergeGapMeters"`
	MinimumEventLengthMeters     float64 `json:"minimumEventLengthMeters"`
	DuplicatePointDistanceMeters float64 `json:"duplicatePointDistanceMeters"`
}

func DefaultWholeCurveConfiguration() WholeCurveConfiguration {
	return WholeCurveConfiguration{
		ResampleSpacingMeters:        5.0,
		ShortSpanMeters:              60.0,
		NominalSpanMeters:            100.0,
		LongSpanMeters:               160.0,
		EnterCurvature:               0.0015,
		ExitCurvature:                0.0010,
		SustainedDistanceMeters:      30.0,
		SameSignMergeGapMeters:       30.0,
		MinimumEventLengthMeters:     20.0,
		DuplicatePointDistanceMeters: 0.5,
	}
}

type WholeCurveConfidence string

const (
	WholeCurveConfidenceHigh   WholeCurveConfidence = "high"
	WholeCurveConfidenceReview WholeCurveConfidence = "review"
	WholeCurveConfidenceLow    WholeCurveConfidence = "low"
)

type WholeCurveFlag string

const (
	WholeCurveFlagMissingScaleContext         WholeCurveFlag = "missing_scale_context"
	WholeCurveFlagIncompleteStraightShoulder  WholeCurveFlag = "incomplete_straight_shoulder"
	WholeCurveFlagSparseSourceGeometry        WholeCurveFlag = "sparse_source_geometry"
	WholeCurveFlagUnstableTurnSign            WholeCurveFlag = "unstable_turn_sign"
	WholeCurveFlagHighScaleSpread             WholeCurveFlag = "high_scale_spread"
	WholeCurveFlagShortEvent                  WholeCurveFlag = "short_event"
	WholeCurveFlagInvalidGeometry             WholeCurveFlag = "invalid_geometry"
	WholeCurveFlagRouteContextTruncatedBefore WholeCurveFlag = "route_context_truncated_before"
	WholeCurveFlagRouteContextTruncatedAfter  WholeCurveFlag = "route_context_truncated_after"
	WholeCurveFlagAmbiguousBranch             WholeCurveFlag = "ambiguous_branch"
)

type WholeCurveInputPoint struct {
	Latitude   float64  `json:"latitude"`
	Longitude  float64  `json:"longitude"`
	SourceKey  string   `json:"sourceKey,omitempty"`
	SourceKeys []string `json:"sourceKeys,omitempty"`
}

func (p WholeCurveInputPoint) allSourceKeys() []string {
	keys := append([]string(nil), p.SourceKeys...)
	if p.SourceKey != "" {
		keys = append(keys, p.SourceKey)
	}
	return uniqueSortedStrings(keys)
}

type WholeCurveResampledPoint struct {
	Latitude         float64  `json:"latitude"`
	Longitude        float64  `json:"longitude"`
	DistanceMeters   float64  `json:"distanceMeters"`
	NearestSourceKey string   `json:"nearestSourceKey,omitempty"`
	SourceKeys       []string `json:"sourceKeys,omitempty"`
	SourceGapMeters  float64  `json:"sourceGapMeters"`
	Curvature60      *float64 `json:"curvature60"`
	Curvature100     *float64 `json:"curvature100"`
	Curvature160     *float64 `json:"curvature160"`
}

type WholeCurveEvent struct {
	PhysicalID             string               `json:"physicalID"`
	DirectionalID          string               `json:"id"`
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
	SourceKeys             []string             `json:"sourceKeys,omitempty"`
	Confidence             WholeCurveConfidence `json:"confidence"`
	Flags                  []WholeCurveFlag     `json:"flags"`
}

type WholeCurveEstimate struct {
	Points []WholeCurveResampledPoint `json:"points"`
	Events []WholeCurveEvent          `json:"events"`
}

type wholeCurvePlanarPoint struct {
	latitude, longitude float64
	x, y, distance      float64
	sourceKeys          []string
}

type wholeCurvePlanarSample struct {
	latitude, longitude float64
	x, y, distance      float64
	nearestSourceKey    string
	sourceKeys          []string
	sourceGapMeters     float64
}

type wholeCurveSignedRun struct {
	start, end int
	sign       float64
}

type wholeCurveScalePeak struct {
	value float64
	index int
}

type wholeCurveSourceTurnVertex struct {
	distance, signedCurvature, signedTurnRadians float64
}

type wholeCurveSourceTurnEvidence struct {
	supportingVertexCount int
	supportingSpanMeters  float64
	directionalCoherence  float64
}

func EstimateWholeCurve(route []WholeCurveInputPoint, cfg WholeCurveConfiguration) WholeCurveEstimate {
	if len(route) < 3 || cfg.ResampleSpacingMeters <= 0 || cfg.ShortSpanMeters <= 0 || cfg.NominalSpanMeters <= 0 || cfg.LongSpanMeters <= 0 {
		return WholeCurveEstimate{}
	}
	dedup := wholeCurveDeduplicate(route, cfg.DuplicatePointDistanceMeters)
	if len(dedup) < 3 || dedup[len(dedup)-1].distance < cfg.ShortSpanMeters {
		return WholeCurveEstimate{}
	}
	samples := wholeCurveResample(dedup, cfg.ResampleSpacingMeters)
	sourceTurns := wholeCurveSourceTurnVertices(dedup)
	points := make([]WholeCurveResampledPoint, len(samples))
	for i, sample := range samples {
		points[i] = WholeCurveResampledPoint{
			Latitude: sample.latitude, Longitude: sample.longitude,
			DistanceMeters: sample.distance, NearestSourceKey: sample.nearestSourceKey,
			SourceKeys: append([]string(nil), sample.sourceKeys...), SourceGapMeters: sample.sourceGapMeters,
		}
		points[i].Curvature60 = wholeCurveSignedCurvature(samples, sample.distance, cfg.ShortSpanMeters)
		points[i].Curvature100 = wholeCurveSignedCurvature(samples, sample.distance, cfg.NominalSpanMeters)
		points[i].Curvature160 = wholeCurveSignedCurvature(samples, sample.distance, cfg.LongSpanMeters)
	}
	runs := wholeCurveEventRuns(points, cfg)
	events := make([]WholeCurveEvent, 0, len(runs))
	for _, run := range runs {
		if event, ok := wholeCurveMakeEvent(run, points, sourceTurns, cfg); ok {
			events = append(events, event)
		}
	}
	return WholeCurveEstimate{Points: points, Events: events}
}

func wholeCurveDeduplicate(route []WholeCurveInputPoint, threshold float64) []wholeCurvePlanarPoint {
	finite := make([]WholeCurveInputPoint, 0, len(route))
	minLat, maxLat := math.Inf(1), math.Inf(-1)
	minLon, maxLon := math.Inf(1), math.Inf(-1)
	for _, point := range route {
		if !wholeCurveIsFinite(point.Latitude) || !wholeCurveIsFinite(point.Longitude) {
			continue
		}
		finite = append(finite, point)
		minLat, maxLat = math.Min(minLat, point.Latitude), math.Max(maxLat, point.Latitude)
		minLon, maxLon = math.Min(minLon, point.Longitude), math.Max(maxLon, point.Longitude)
	}
	if len(finite) == 0 {
		return nil
	}
	lat0 := 0.5 * (minLat + maxLat) * math.Pi / 180
	lon0 := 0.5 * (minLon + maxLon) * math.Pi / 180
	cosLat := math.Cos(lat0)
	result := make([]wholeCurvePlanarPoint, 0, len(finite))
	for _, point := range finite {
		lat := point.Latitude * math.Pi / 180
		lon := point.Longitude * math.Pi / 180
		x := R * (lon - lon0) * cosLat
		y := R * (lat - lat0)
		keys := point.allSourceKeys()
		if len(result) > 0 {
			previous := &result[len(result)-1]
			d := math.Hypot(x-previous.x, y-previous.y)
			if d < threshold {
				previous.sourceKeys = uniqueSortedStrings(append(previous.sourceKeys, keys...))
				continue
			}
			result = append(result, wholeCurvePlanarPoint{point.Latitude, point.Longitude, x, y, previous.distance + d, keys})
		} else {
			result = append(result, wholeCurvePlanarPoint{point.Latitude, point.Longitude, x, y, 0, keys})
		}
	}
	return result
}

func wholeCurveResample(points []wholeCurvePlanarPoint, spacing float64) []wholeCurvePlanarSample {
	total := points[len(points)-1].distance
	if total <= 0 {
		return nil
	}
	segmentCount := int(math.Round(total / spacing))
	if segmentCount < 1 {
		segmentCount = 1
	}
	uniform := total / float64(segmentCount)
	result := make([]wholeCurvePlanarSample, segmentCount+1)
	segment := 0
	for i := 0; i <= segmentCount; i++ {
		target := float64(i) * uniform
		for segment+1 < len(points)-1 && points[segment+1].distance < target {
			segment++
		}
		first, second := points[segment], points[minInt(segment+1, len(points)-1)]
		gap := second.distance - first.distance
		fraction := 0.0
		if gap > 0 {
			fraction = clampFloat((target-first.distance)/gap, 0, 1)
		}
		nearest := first
		if fraction >= 0.5 {
			nearest = second
		}
		nearestKey := ""
		if len(nearest.sourceKeys) > 0 {
			nearestKey = nearest.sourceKeys[0]
		}
		result[i] = wholeCurvePlanarSample{
			latitude:  first.latitude + (second.latitude-first.latitude)*fraction,
			longitude: first.longitude + (second.longitude-first.longitude)*fraction,
			x:         first.x + (second.x-first.x)*fraction,
			y:         first.y + (second.y-first.y)*fraction,
			distance:  target, nearestSourceKey: nearestKey,
			sourceKeys:      uniqueSortedStrings(append(append([]string{}, first.sourceKeys...), second.sourceKeys...)),
			sourceGapMeters: gap,
		}
	}
	return result
}

func wholeCurveSourceTurnVertices(points []wholeCurvePlanarPoint) []wholeCurveSourceTurnVertex {
	result := make([]wholeCurveSourceTurnVertex, 0, len(points)-2)
	for i := 1; i+1 < len(points); i++ {
		previous, current, next := points[i-1], points[i], points[i+1]
		inbound := current.distance - previous.distance
		outbound := next.distance - current.distance
		support := 0.5 * (inbound + outbound)
		if inbound <= 0 || outbound <= 0 || support <= 0 {
			continue
		}
		inHeading := math.Atan2(current.y-previous.y, current.x-previous.x)
		outHeading := math.Atan2(next.y-current.y, next.x-current.x)
		turn := normalizedAngle(outHeading - inHeading)
		if wholeCurveIsFinite(turn) {
			result = append(result, wholeCurveSourceTurnVertex{current.distance, turn / support, turn})
		}
	}
	return result
}

func wholeCurveSignedCurvature(samples []wholeCurvePlanarSample, distance, span float64) *float64 {
	half := span * 0.5
	if len(samples) == 0 || distance-half < samples[0].distance || distance+half > samples[len(samples)-1].distance {
		return nil
	}
	first, ok1 := wholeCurveInterpolate(samples, distance-half)
	middle, ok2 := wholeCurveInterpolate(samples, distance)
	last, ok3 := wholeCurveInterpolate(samples, distance+half)
	if !ok1 || !ok2 || !ok3 {
		return nil
	}
	ab := math.Hypot(middle[0]-first[0], middle[1]-first[1])
	bc := math.Hypot(last[0]-middle[0], last[1]-middle[1])
	ac := math.Hypot(last[0]-first[0], last[1]-first[1])
	if ab <= 0 || bc <= 0 || ac <= 0 {
		return nil
	}
	cross := (middle[0]-first[0])*(last[1]-first[1]) - (middle[1]-first[1])*(last[0]-first[0])
	magnitude := 2 * math.Abs(cross) / (ab * bc * ac)
	if !wholeCurveIsFinite(magnitude) {
		return nil
	}
	value := 0.0
	if magnitude >= 1e-9 {
		value = magnitude
		if cross < 0 {
			value = -value
		}
	}
	return &value
}

func wholeCurveInterpolate(samples []wholeCurvePlanarSample, distance float64) ([2]float64, bool) {
	if len(samples) == 0 || distance < samples[0].distance || distance > samples[len(samples)-1].distance {
		return [2]float64{}, false
	}
	spacing := 0.0
	if len(samples) > 1 {
		spacing = samples[1].distance - samples[0].distance
	}
	if spacing <= 0 {
		return [2]float64{samples[0].x, samples[0].y}, true
	}
	lower := int(math.Floor(distance / spacing))
	lower = clampInt(lower, 0, len(samples)-1)
	upper := minInt(lower+1, len(samples)-1)
	rangeDistance := samples[upper].distance - samples[lower].distance
	fraction := 0.0
	if rangeDistance > 0 {
		fraction = (distance - samples[lower].distance) / rangeDistance
	}
	return [2]float64{
		samples[lower].x + (samples[upper].x-samples[lower].x)*fraction,
		samples[lower].y + (samples[upper].y-samples[lower].y)*fraction,
	}, true
}

func wholeCurveSegmentationCurvature(point WholeCurveResampledPoint) *float64 {
	if point.Curvature60 != nil {
		return point.Curvature60
	}
	return point.Curvature100
}

func wholeCurveEventRuns(points []WholeCurveResampledPoint, cfg WholeCurveConfiguration) []wholeCurveSignedRun {
	raw := make([]wholeCurveSignedRun, 0)
	for index := 0; index < len(points); {
		curvature := wholeCurveSegmentationCurvature(points[index])
		if curvature == nil || math.Abs(*curvature) < cfg.ExitCurvature {
			index++
			continue
		}
		sign := math.Copysign(1, *curvature)
		start, end := index, index
		for end+1 < len(points) {
			next := wholeCurveSegmentationCurvature(points[end+1])
			if next == nil || math.Abs(*next) < cfg.ExitCurvature || math.Copysign(1, *next) != sign {
				break
			}
			end++
		}
		raw = append(raw, wholeCurveSignedRun{start, end, sign})
		index = end + 1
	}
	merged := make([]wholeCurveSignedRun, 0, len(raw))
	for _, run := range raw {
		if len(merged) == 0 {
			merged = append(merged, run)
			continue
		}
		previous := &merged[len(merged)-1]
		gap := points[run.start].DistanceMeters - points[previous.end].DistanceMeters
		opposing := wholeCurveSignedTurn(points, previous.end, run.start)
		if previous.sign == run.sign && gap <= cfg.SameSignMergeGapMeters && math.Abs(opposing) < 3*math.Pi/180 {
			previous.end = run.end
		} else {
			merged = append(merged, run)
		}
	}
	filtered := make([]wholeCurveSignedRun, 0, len(merged))
	for _, run := range merged {
		if wholeCurveHasSustainedSeed(points, run, cfg) && points[run.end].DistanceMeters-points[run.start].DistanceMeters >= cfg.MinimumEventLengthMeters {
			filtered = append(filtered, run)
		}
	}
	return filtered
}

func wholeCurveHasSustainedSeed(points []WholeCurveResampledPoint, run wholeCurveSignedRun, cfg WholeCurveConfiguration) bool {
	start := -1
	for i := run.start; i <= run.end; i++ {
		curvature := wholeCurveSegmentationCurvature(points[i])
		matches := curvature != nil && math.Abs(*curvature) >= cfg.EnterCurvature && math.Copysign(1, *curvature) == run.sign
		if matches {
			if start < 0 {
				start = i
			}
			if points[i].DistanceMeters-points[start].DistanceMeters >= cfg.SustainedDistanceMeters {
				return true
			}
		} else {
			start = -1
		}
	}
	return false
}

func wholeCurveMakeEvent(run wholeCurveSignedRun, points []WholeCurveResampledPoint, sourceTurns []wholeCurveSourceTurnVertex, cfg WholeCurveConfiguration) (WholeCurveEvent, bool) {
	if run.start < 0 || run.end >= len(points) || run.start >= run.end {
		return WholeCurveEvent{}, false
	}
	peak60 := wholeCurveRollingMedianPeak(points, func(p WholeCurveResampledPoint) *float64 { return p.Curvature60 }, run, cfg.SustainedDistanceMeters)
	peak100 := wholeCurveRollingMedianPeak(points, func(p WholeCurveResampledPoint) *float64 { return p.Curvature100 }, run, cfg.SustainedDistanceMeters)
	peak160 := wholeCurveRollingMedianPeak(points, func(p WholeCurveResampledPoint) *float64 { return p.Curvature160 }, run, cfg.SustainedDistanceMeters)
	available := make([]float64, 0, 3)
	for _, peak := range []*wholeCurveScalePeak{peak60, peak100, peak160} {
		if peak != nil {
			available = append(available, math.Abs(peak.value))
		}
	}
	if len(available) == 0 {
		return WholeCurveEvent{}, false
	}
	nominal := wholeCurveMedian(available)
	signedTurn, coherence := wholeCurveTurnStatistics(points, run.start, run.end)
	maxGap := 0.0
	for i := run.start; i <= run.end; i++ {
		maxGap = math.Max(maxGap, points[i].SourceGapMeters)
	}
	var evidence *wholeCurveSourceTurnEvidence
	if peak60 != nil {
		evidence = wholeCurveCompactApexSourceEvidence(sourceTurns, points, run, peak60.value, cfg)
	}
	compactSupported := peak60 != nil && math.Abs(peak60.value) >= nominal*1.20 && evidence != nil &&
		evidence.supportingVertexCount >= 3 && evidence.supportingSpanMeters >= cfg.SustainedDistanceMeters &&
		evidence.directionalCoherence >= 0.80 && maxGap <= 40 && coherence >= 0.80
	compactGuard := nominal
	if compactSupported && wholeCurveHasSustainedPeak(points, func(p WholeCurveResampledPoint) *float64 { return p.Curvature60 }, run, math.Abs(peak60.value)*0.85, cfg.SustainedDistanceMeters) {
		compactGuard = math.Abs(peak60.value)
	}
	controlling := run.sign * math.Max(nominal, compactGuard)
	apexIndex := run.start
	if compactGuard > nominal && peak60 != nil {
		apexIndex = peak60.index
	} else if peak100 != nil {
		apexIndex = peak100.index
	} else if peak60 != nil {
		apexIndex = peak60.index
	} else if peak160 != nil {
		apexIndex = peak160.index
	}
	var scaleSpread *float64
	if len(available) >= 2 {
		minimum, maximum := available[0], available[0]
		for _, value := range available[1:] {
			minimum, maximum = math.Min(minimum, value), math.Max(maximum, value)
		}
		spread := maximum / math.Max(minimum, 1e-12)
		scaleSpread = &spread
	}
	length := points[run.end].DistanceMeters - points[run.start].DistanceMeters
	flags := make([]WholeCurveFlag, 0)
	if len(available) < 3 {
		flags = append(flags, WholeCurveFlagMissingScaleContext)
	}
	if !wholeCurveHasStraightShoulders(points, run, cfg.SustainedDistanceMeters, cfg.ExitCurvature) {
		flags = append(flags, WholeCurveFlagIncompleteStraightShoulder)
	}
	if maxGap > 60 {
		flags = append(flags, WholeCurveFlagSparseSourceGeometry)
	}
	if coherence < 0.75 {
		flags = append(flags, WholeCurveFlagUnstableTurnSign)
	}
	if scaleSpread != nil && *scaleSpread > 1.8 {
		flags = append(flags, WholeCurveFlagHighScaleSpread)
	}
	if length < cfg.SustainedDistanceMeters {
		flags = append(flags, WholeCurveFlagShortEvent)
	}
	if !wholeCurveIsFinite(controlling) || controlling == 0 {
		flags = append(flags, WholeCurveFlagInvalidGeometry)
	}
	confidence := WholeCurveConfidenceLow
	if len(flags) == 0 && scaleSpread != nil && *scaleSpread <= 1.35 && coherence >= 0.85 && maxGap <= 40 {
		confidence = WholeCurveConfidenceHigh
	} else if !containsWholeCurveFlag(flags, WholeCurveFlagInvalidGeometry) && !containsWholeCurveFlag(flags, WholeCurveFlagUnstableTurnSign) && !containsWholeCurveFlag(flags, WholeCurveFlagMissingScaleContext) && scaleSpread != nil && *scaleSpread <= 1.8 {
		confidence = WholeCurveConfidenceReview
	}
	physicalID, directionalID := wholeCurveEventIdentifiers(points[run.start], points[apexIndex], points[run.end])
	sourceKeys := make([]string, 0)
	for i := run.start; i <= run.end; i++ {
		sourceKeys = append(sourceKeys, points[i].SourceKeys...)
	}
	event := WholeCurveEvent{
		PhysicalID: physicalID, DirectionalID: directionalID, StartIndex: run.start, EndIndex: run.end, ApexIndex: apexIndex,
		LengthMeters: length, SignedTurnRadians: signedTurn, SignCoherence: coherence,
		Curvature60: peakValuePointer(peak60), Curvature100: peakValuePointer(peak100), Curvature160: peakValuePointer(peak160),
		ControllingCurvature: controlling, ScaleSpread: scaleSpread, MaximumSourceGapMeters: maxGap,
		SourceKeys: uniqueSortedStrings(sourceKeys), Confidence: confidence, Flags: flags,
	}
	return event, true
}

func wholeCurveCompactApexSourceEvidence(sourceTurns []wholeCurveSourceTurnVertex, points []WholeCurveResampledPoint, run wholeCurveSignedRun, peak60 float64, cfg WholeCurveConfiguration) *wholeCurveSourceTurnEvidence {
	threshold := math.Abs(peak60) * 0.85
	first, last := -1, -1
	for i := run.start; i <= run.end; i++ {
		if points[i].Curvature60 != nil && math.Abs(*points[i].Curvature60) >= threshold && math.Copysign(1, *points[i].Curvature60) == run.sign {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		return nil
	}
	half := cfg.ShortSpanMeters * 0.5
	lower := math.Max(points[run.start].DistanceMeters, points[first].DistanceMeters-half)
	upper := math.Min(points[run.end].DistanceMeters, points[last].DistanceMeters+half)
	local := make([]wholeCurveSourceTurnVertex, 0)
	for _, turn := range sourceTurns {
		if turn.distance >= lower && turn.distance <= upper && math.Abs(turn.signedTurnRadians) > 1e-9 {
			local = append(local, turn)
		}
	}
	if len(local) == 0 {
		return nil
	}
	sourceThreshold := math.Max(cfg.EnterCurvature, math.Abs(peak60)*0.5)
	supporting := make([]wholeCurveSourceTurnVertex, 0)
	for _, turn := range local {
		if math.Copysign(1, turn.signedCurvature) == run.sign && math.Abs(turn.signedCurvature) >= sourceThreshold {
			supporting = append(supporting, turn)
		}
	}
	span := 0.0
	if len(supporting) > 0 {
		span = supporting[len(supporting)-1].distance - supporting[0].distance
	}
	signed, absolute := 0.0, 0.0
	for _, turn := range local {
		signed += turn.signedTurnRadians
		absolute += math.Abs(turn.signedTurnRadians)
	}
	coherence := 0.0
	if absolute > 1e-9 {
		coherence = run.sign * signed / absolute
	}
	return &wholeCurveSourceTurnEvidence{len(supporting), span, coherence}
}

func wholeCurveRollingMedianPeak(points []WholeCurveResampledPoint, get func(WholeCurveResampledPoint) *float64, run wholeCurveSignedRun, support float64) *wholeCurveScalePeak {
	if len(points) < 2 {
		return nil
	}
	spacing := math.Max(0.1, points[minInt(run.start+1, len(points)-1)].DistanceMeters-points[run.start].DistanceMeters)
	halfWindow := maxInt(1, int(math.Ceil(support/spacing/2)))
	var best *wholeCurveScalePeak
	for i := run.start; i <= run.end; i++ {
		lower, upper := maxInt(run.start, i-halfWindow), minInt(run.end, i+halfWindow)
		if points[upper].DistanceMeters-points[lower].DistanceMeters < math.Min(support, points[run.end].DistanceMeters-points[run.start].DistanceMeters) {
			continue
		}
		local := make([]float64, 0, upper-lower+1)
		supported := 0
		for j := lower; j <= upper; j++ {
			value := get(points[j])
			magnitude := 0.0
			if value != nil && math.Abs(*value) > 0 && math.Copysign(1, *value) == run.sign {
				magnitude = math.Abs(*value)
				supported++
			}
			local = append(local, magnitude)
		}
		if supported*2 <= len(local) {
			continue
		}
		candidate := &wholeCurveScalePeak{run.sign * wholeCurveMedian(local), i}
		if best == nil || math.Abs(candidate.value) > math.Abs(best.value) {
			best = candidate
		}
	}
	return best
}

func wholeCurveHasSustainedPeak(points []WholeCurveResampledPoint, get func(WholeCurveResampledPoint) *float64, run wholeCurveSignedRun, threshold, minimumDistance float64) bool {
	start := -1
	for i := run.start; i <= run.end; i++ {
		value := get(points[i])
		if value != nil && math.Abs(*value) >= threshold && math.Copysign(1, *value) == run.sign {
			if start < 0 {
				start = i
			}
			if points[i].DistanceMeters-points[start].DistanceMeters >= minimumDistance {
				return true
			}
		} else {
			start = -1
		}
	}
	return false
}

func wholeCurveHasStraightShoulders(points []WholeCurveResampledPoint, run wholeCurveSignedRun, distance, exitCurvature float64) bool {
	entry, exit := points[run.start].DistanceMeters, points[run.end].DistanceMeters
	if entry-points[0].DistanceMeters < distance || points[len(points)-1].DistanceMeters-exit < distance {
		return false
	}
	entryCount, exitCount := 0, 0
	for _, point := range points {
		if point.DistanceMeters >= entry-distance && point.DistanceMeters < entry {
			entryCount++
			curvature := wholeCurveSegmentationCurvature(point)
			if curvature == nil || math.Abs(*curvature) >= exitCurvature {
				return false
			}
		}
		if point.DistanceMeters > exit && point.DistanceMeters <= exit+distance {
			exitCount++
			curvature := wholeCurveSegmentationCurvature(point)
			if curvature == nil || math.Abs(*curvature) >= exitCurvature {
				return false
			}
		}
	}
	return entryCount > 0 && exitCount > 0
}

func wholeCurveTurnStatistics(points []WholeCurveResampledPoint, start, end int) (float64, float64) {
	if end-start < 2 {
		return 0, 0
	}
	signed, absolute := 0.0, 0.0
	for i := start + 1; i < end; i++ {
		firstHeading := wholeCurveHeading(points[i-1], points[i])
		secondHeading := wholeCurveHeading(points[i], points[i+1])
		change := normalizedAngle(secondHeading - firstHeading)
		signed += change
		absolute += math.Abs(change)
	}
	if absolute <= 1e-9 {
		return signed, 1
	}
	return signed, math.Abs(signed) / absolute
}

func wholeCurveSignedTurn(points []WholeCurveResampledPoint, start, end int) float64 {
	value, _ := wholeCurveTurnStatistics(points, start, end)
	return value
}

func wholeCurveHeading(first, second WholeCurveResampledPoint) float64 {
	latitude := 0.5 * (first.Latitude + second.Latitude) * math.Pi / 180
	x := (second.Longitude - first.Longitude) * math.Cos(latitude)
	y := second.Latitude - first.Latitude
	return math.Atan2(y, x)
}

func wholeCurveEventIdentifiers(start, apex, end WholeCurveResampledPoint) (string, string) {
	coordinate := func(point WholeCurveResampledPoint) string {
		return fmt.Sprintf("%d:%d", int64(math.Round(point.Latitude*100000)), int64(math.Round(point.Longitude*100000)))
	}
	startKey, apexKey, endKey := coordinate(start), coordinate(apex), coordinate(end)
	ends := []string{startKey, endKey}
	sort.Strings(ends)
	seed := WholeCurveEstimatorVersion + "|" + ends[0] + "|" + ends[1] + "|" + apexKey
	digest := sha256.Sum256([]byte(seed))
	physical := hex.EncodeToString(digest[:10])
	direction := "b"
	if startKey <= endKey {
		direction = "a"
	}
	return physical, physical + "-" + direction
}

func wholeCurveMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	middle := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return 0.5 * (sorted[middle-1] + sorted[middle])
	}
	return sorted[middle]
}

func normalizedAngle(value float64) float64 {
	for value > math.Pi {
		value -= 2 * math.Pi
	}
	for value < -math.Pi {
		value += 2 * math.Pi
	}
	return value
}

func peakValuePointer(peak *wholeCurveScalePeak) *float64 {
	if peak == nil {
		return nil
	}
	value := peak.value
	return &value
}
func containsWholeCurveFlag(flags []WholeCurveFlag, target WholeCurveFlag) bool {
	for _, flag := range flags {
		if flag == target {
			return true
		}
	}
	return false
}
func clampFloat(value, low, high float64) float64 { return math.Max(low, math.Min(high, value)) }
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func uniqueSortedStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value != "" {
			seen[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for value := range seen {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func wholeCurveIsFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
