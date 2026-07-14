package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	WholeCurveProfilePredecessorMeters = 110.0
	WholeCurveProfileLookaheadMeters   = 1200.0
	wholeCurveEventMatchDistanceMeters = 120.0
)

type WholeCurveProfilePoint struct {
	Latitude       float64              `json:"latitude"`
	Longitude      float64              `json:"longitude"`
	DistanceMeters float64              `json:"distanceMeters"`
	Curvature      float64              `json:"curvature"`
	EventID        string               `json:"eventID,omitempty"`
	Confidence     WholeCurveConfidence `json:"confidence,omitempty"`
	Flags          []WholeCurveFlag     `json:"flags,omitempty"`
}

type WholeCurveProfileEvent struct {
	EventID                string               `json:"eventID"`
	PhysicalID             string               `json:"physicalID"`
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

type WholeCurveProfile struct {
	EstimatorVersion      string                   `json:"estimatorVersion"`
	GeneratedAtUnixMillis int64                    `json:"generatedAtUnixMillis"`
	RouteFingerprint      string                   `json:"routeFingerprint"`
	Generation            uint64                   `json:"generation"`
	Points                []WholeCurveProfilePoint `json:"points"`
	Events                []WholeCurveProfileEvent `json:"events"`
	FatalAmbiguity        bool                     `json:"fatalAmbiguity"`
}

// BuildWholeCurveProfile runs the geometry estimator and materializes its
// signed event curvature onto a bounded route-aligned point stream. Speeds are
// deliberately not part of this contract; the controller remains the sole
// authority for sigmoid/Q conversion.
func BuildWholeCurveProfile(route DirectionalRoute, pos Position, previous WholeCurveEstimate, now time.Time) (WholeCurveProfile, WholeCurveEstimate, error) {
	if len(route.Nodes) < 3 {
		return WholeCurveProfile{}, WholeCurveEstimate{}, errors.New("not enough directional route nodes")
	}
	estimate := EstimateWholeCurve(DirectionalRouteInput(route), DefaultWholeCurveConfiguration())
	if len(estimate.Points) < 3 {
		return WholeCurveProfile{}, estimate, errors.New("whole-curve estimator produced too few points")
	}
	reconcileWholeCurveEventIDs(&estimate, previous)
	markWholeCurveRouteContext(&estimate, route)

	egoDistance := projectPositionToWholeCurvePoints(estimate.Points, pos)
	startIndex, endIndex := wholeCurveProfileWindow(estimate.Points, egoDistance)
	if endIndex-startIndex+1 < 3 {
		return WholeCurveProfile{}, estimate, errors.New("whole-curve profile has insufficient forward points")
	}

	points := make([]WholeCurveProfilePoint, endIndex-startIndex+1)
	baseDistance := estimate.Points[startIndex].DistanceMeters
	for index := range points {
		source := estimate.Points[startIndex+index]
		points[index] = WholeCurveProfilePoint{
			Latitude: source.Latitude, Longitude: source.Longitude,
			DistanceMeters: source.DistanceMeters - baseDistance,
		}
	}

	events := make([]WholeCurveProfileEvent, 0, len(estimate.Events))
	for _, event := range estimate.Events {
		if event.EndIndex < startIndex || event.StartIndex > endIndex {
			continue
		}
		clippedStart := maxInt(event.StartIndex, startIndex)
		clippedEnd := minInt(event.EndIndex, endIndex)
		clippedApex := maxInt(clippedStart, minInt(event.ApexIndex, clippedEnd))
		flags := append([]WholeCurveFlag(nil), event.Flags...)
		confidence := event.Confidence
		if event.StartIndex < startIndex {
			flags = appendUniqueWholeCurveFlag(flags, WholeCurveFlagRouteContextTruncatedBefore)
			confidence = WholeCurveConfidenceLow
		}
		if event.EndIndex > endIndex {
			flags = appendUniqueWholeCurveFlag(flags, WholeCurveFlagRouteContextTruncatedAfter)
			confidence = WholeCurveConfidenceLow
		}
		profileEvent := WholeCurveProfileEvent{
			EventID: event.DirectionalID, PhysicalID: event.PhysicalID,
			StartIndex: clippedStart - startIndex, EndIndex: clippedEnd - startIndex, ApexIndex: clippedApex - startIndex,
			LengthMeters: event.LengthMeters, SignedTurnRadians: event.SignedTurnRadians, SignCoherence: event.SignCoherence,
			Curvature60: event.Curvature60, Curvature100: event.Curvature100, Curvature160: event.Curvature160,
			ControllingCurvature: event.ControllingCurvature, ScaleSpread: event.ScaleSpread,
			MaximumSourceGapMeters: event.MaximumSourceGapMeters, SourceKeys: append([]string(nil), event.SourceKeys...),
			Confidence: confidence, Flags: flags,
		}
		events = append(events, profileEvent)
		for sourceIndex := clippedStart; sourceIndex <= clippedEnd; sourceIndex++ {
			point := &points[sourceIndex-startIndex]
			// Segmented events do not normally overlap. If malformed geometry
			// ever makes them overlap, keep the more conservative curvature.
			if point.EventID == "" || math.Abs(event.ControllingCurvature) > math.Abs(point.Curvature) {
				point.Curvature = event.ControllingCurvature
				point.EventID = event.DirectionalID
				point.Confidence = confidence
				point.Flags = append([]WholeCurveFlag(nil), flags...)
			}
		}
	}

	profile := WholeCurveProfile{
		EstimatorVersion: WholeCurveEstimatorVersion, GeneratedAtUnixMillis: now.UnixMilli(),
		Generation: route.Generation, Points: points, Events: events, FatalAmbiguity: route.FatalAmbiguity,
	}
	fingerprint, err := WholeCurveRouteFingerprint(profile.Generation, profile.Points)
	if err != nil {
		return WholeCurveProfile{}, estimate, err
	}
	profile.RouteFingerprint = fingerprint
	return profile, estimate, nil
}

func wholeCurveProfileWindow(points []WholeCurveResampledPoint, egoDistance float64) (int, int) {
	lower := math.Max(points[0].DistanceMeters, egoDistance-WholeCurveProfilePredecessorMeters)
	upper := math.Min(points[len(points)-1].DistanceMeters, egoDistance+WholeCurveProfileLookaheadMeters)
	start := 0
	for start+1 < len(points) && points[start].DistanceMeters < lower {
		start++
	}
	end := len(points) - 1
	for end > start && points[end].DistanceMeters > upper {
		end--
	}
	return start, end
}

func projectPositionToWholeCurvePoints(points []WholeCurveResampledPoint, pos Position) float64 {
	if len(points) < 2 {
		return 0
	}
	bestDistance, bestS := math.Inf(1), points[0].DistanceMeters
	for index := 0; index+1 < len(points); index++ {
		first, second := points[index], points[index+1]
		latitude, longitude := PointOnLine(first.Latitude, first.Longitude, second.Latitude, second.Longitude, pos.Latitude, pos.Longitude)
		distance := DistanceToPoint(pos.Latitude*TO_RADIANS, pos.Longitude*TO_RADIANS, latitude*TO_RADIANS, longitude*TO_RADIANS)
		if distance >= bestDistance {
			continue
		}
		segmentLength := second.DistanceMeters - first.DistanceMeters
		fraction := 0.0
		if segmentLength > 0 {
			along := DistanceToPoint(first.Latitude*TO_RADIANS, first.Longitude*TO_RADIANS, latitude*TO_RADIANS, longitude*TO_RADIANS)
			fraction = clampFloat(along/segmentLength, 0, 1)
		}
		bestDistance = distance
		bestS = first.DistanceMeters + segmentLength*fraction
	}
	return bestS
}

func reconcileWholeCurveEventIDs(current *WholeCurveEstimate, previous WholeCurveEstimate) {
	used := make(map[int]bool)
	for currentIndex := range current.Events {
		candidate := &current.Events[currentIndex]
		bestIndex, bestScore := -1, math.Inf(-1)
		for previousIndex, old := range previous.Events {
			if used[previousIndex] || math.Signbit(old.ControllingCurvature) != math.Signbit(candidate.ControllingCurvature) {
				continue
			}
			apexDistance := wholeCurveApexDistance(*candidate, current.Points, old, previous.Points)
			overlap := wholeCurveSourceKeyOverlap(candidate.SourceKeys, old.SourceKeys)
			if overlap == 0 && apexDistance > wholeCurveEventMatchDistanceMeters {
				continue
			}
			if overlap > 0 && apexDistance > 2*wholeCurveEventMatchDistanceMeters {
				continue
			}
			score := float64(overlap)*1000 - apexDistance
			if score > bestScore {
				bestIndex, bestScore = previousIndex, score
			}
		}
		if bestIndex >= 0 {
			candidate.PhysicalID = previous.Events[bestIndex].PhysicalID
			candidate.DirectionalID = previous.Events[bestIndex].DirectionalID
			used[bestIndex] = true
		}
	}
}

func wholeCurveApexDistance(first WholeCurveEvent, firstPoints []WholeCurveResampledPoint, second WholeCurveEvent, secondPoints []WholeCurveResampledPoint) float64 {
	if first.ApexIndex < 0 || first.ApexIndex >= len(firstPoints) || second.ApexIndex < 0 || second.ApexIndex >= len(secondPoints) {
		return math.Inf(1)
	}
	a, b := firstPoints[first.ApexIndex], secondPoints[second.ApexIndex]
	return DistanceToPoint(a.Latitude*TO_RADIANS, a.Longitude*TO_RADIANS, b.Latitude*TO_RADIANS, b.Longitude*TO_RADIANS)
}

func wholeCurveSourceKeyOverlap(first, second []string) int {
	keys := make(map[string]struct{}, len(first))
	for _, key := range first {
		keys[key] = struct{}{}
	}
	overlap := 0
	for _, key := range second {
		if _, ok := keys[key]; ok {
			overlap++
		}
	}
	return overlap
}

func markWholeCurveRouteContext(estimate *WholeCurveEstimate, route DirectionalRoute) {
	if len(estimate.Points) == 0 {
		return
	}
	total := estimate.Points[len(estimate.Points)-1].DistanceMeters
	for index := range estimate.Events {
		event := &estimate.Events[index]
		if route.TruncatedBefore && estimate.Points[event.StartIndex].DistanceMeters <= RouteMinimumPastContextMeters {
			event.Flags = appendUniqueWholeCurveFlag(event.Flags, WholeCurveFlagRouteContextTruncatedBefore)
			event.Confidence = WholeCurveConfidenceLow
		}
		if route.TruncatedAfter && total-estimate.Points[event.EndIndex].DistanceMeters <= RouteMinimumPastContextMeters {
			event.Flags = appendUniqueWholeCurveFlag(event.Flags, WholeCurveFlagRouteContextTruncatedAfter)
			event.Confidence = WholeCurveConfidenceLow
		}
		if route.FatalAmbiguity {
			event.Flags = appendUniqueWholeCurveFlag(event.Flags, WholeCurveFlagAmbiguousBranch)
			event.Confidence = WholeCurveConfidenceLow
		}
	}
}

func appendUniqueWholeCurveFlag(flags []WholeCurveFlag, value WholeCurveFlag) []WholeCurveFlag {
	if containsWholeCurveFlag(flags, value) {
		return flags
	}
	return append(flags, value)
}

// WholeCurveRouteFingerprint implements the cross-language, half-away-from-zero
// v1 contract consumed by vision_turn_controller.py.
func WholeCurveRouteFingerprint(generation uint64, points []WholeCurveProfilePoint) (string, error) {
	var canonical strings.Builder
	fmt.Fprintf(&canonical, "MapWholeCurveProfile|%s|%d\n", WholeCurveEstimatorVersion, generation)
	for _, point := range points {
		latitude, err := quantizeHalfAwayFromZero(point.Latitude, 1e7)
		if err != nil {
			return "", err
		}
		longitude, err := quantizeHalfAwayFromZero(point.Longitude, 1e7)
		if err != nil {
			return "", err
		}
		distance, err := quantizeHalfAwayFromZero(point.DistanceMeters, 1e3)
		if err != nil {
			return "", err
		}
		curvature, err := quantizeHalfAwayFromZero(point.Curvature, 1e9)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&canonical, "%d,%d,%d,%d,%s\n", latitude, longitude, distance, curvature, point.EventID)
	}
	digest := sha256.Sum256([]byte(canonical.String()))
	return hex.EncodeToString(digest[:]), nil
}

func quantizeHalfAwayFromZero(value, scale float64) (int64, error) {
	if !wholeCurveIsFinite(value) {
		return 0, errors.New("non-finite whole-curve fingerprint value")
	}
	magnitude := math.Floor(math.Abs(value)*scale + 0.5)
	if magnitude > math.MaxInt64 {
		return 0, errors.New("whole-curve fingerprint value overflows int64")
	}
	result := int64(magnitude)
	if value < 0 {
		result = -result
	}
	return result, nil
}
