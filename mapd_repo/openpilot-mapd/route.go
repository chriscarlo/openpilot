package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
)

const (
	RouteLookaheadMeters          = 1200.0
	RoutePredecessorRetainMeters  = 1200.0
	RouteMinimumPastContextMeters = 110.0 // 160 m scale half-span plus a 30 m straight shoulder.
)

type RouteNodeSource struct {
	WayID     string `json:"wayID"`
	NodeIndex int    `json:"nodeIndex"`
	IsForward bool   `json:"isForward"`
}

func (s RouteNodeSource) Key() string {
	return fmt.Sprintf("%s:%d", s.WayID, s.NodeIndex)
}

type DirectionalRouteNode struct {
	Latitude       float64           `json:"latitude"`
	Longitude      float64           `json:"longitude"`
	DistanceMeters float64           `json:"distanceMeters"`
	Sources        []RouteNodeSource `json:"sources"`
	BakedSpeedMPS  *float64          `json:"bakedSpeedMPS,omitempty"`
	WayBoundary    bool              `json:"wayBoundary,omitempty"`
	MergeOrSplit   bool              `json:"mergeOrSplit,omitempty"`
	Ambiguous      bool              `json:"ambiguous,omitempty"`
}

type DirectionalRoute struct {
	Generation        uint64                 `json:"generation"`
	Nodes             []DirectionalRouteNode `json:"nodes"`
	CurrentWayID      string                 `json:"currentWayID"`
	EgoDistanceMeters float64                `json:"egoDistanceMeters"`
	LookaheadMeters   float64                `json:"lookaheadMeters"`
	PredecessorMeters float64                `json:"predecessorMeters"`
	TruncatedBefore   bool                   `json:"truncatedBefore"`
	TruncatedAfter    bool                   `json:"truncatedAfter"`
	FatalAmbiguity    bool                   `json:"fatalAmbiguity"`
}

type selectedRouteWay struct {
	way       Way
	forward   bool
	ambiguous bool
}

func (state *State) directionalRoute() (DirectionalRoute, error) {
	if len(state.Route.Nodes) > 0 {
		return state.Route, nil
	}
	route, err := BuildDirectionalRoute(state.CurrentWay, state.NextWays, DirectionalRoute{}, state.Position)
	if err != nil {
		return DirectionalRoute{}, err
	}
	state.Route = route
	return route, nil
}

func BuildDirectionalRoute(current CurrentWay, next []NextWayResult, previous DirectionalRoute, pos Position) (DirectionalRoute, error) {
	if !current.Way.IsValid() {
		return DirectionalRoute{}, fmt.Errorf("current way is invalid")
	}
	currentID, err := StableWayID(current.Way)
	if err != nil {
		return DirectionalRoute{}, err
	}
	segments := []selectedRouteWay{{way: current.Way, forward: current.OnWay.IsForward}}
	for _, item := range next {
		if item.Way.IsValid() {
			segments = append(segments, selectedRouteWay{way: item.Way, forward: item.IsForward, ambiguous: item.Ambiguous})
		}
	}
	fresh, fatal, err := flattenSelectedRouteWays(segments)
	if err != nil {
		return DirectionalRoute{}, err
	}
	generation := uint64(1)
	if previous.Generation > 0 {
		generation = previous.Generation + 1
	}
	nodes := fresh
	if previous.Generation > 0 {
		if merged, matched, compatible := mergeRolloverRoute(previous.Nodes, fresh, currentID); matched {
			nodes = merged
			if compatible {
				generation = previous.Generation
			}
		}
	}
	for _, node := range nodes {
		fatal = fatal || node.Ambiguous
	}
	recomputeRouteDistances(nodes)
	egoDistance := projectPositionToRouteDistance(nodes, pos)
	if egoDistance > RoutePredecessorRetainMeters && len(nodes) > 2 {
		cutDistance := egoDistance - RoutePredecessorRetainMeters
		cut := 0
		for cut+1 < len(nodes) && nodes[cut+1].DistanceMeters < cutDistance {
			cut++
		}
		if cut > 0 {
			nodes = append([]DirectionalRouteNode(nil), nodes[cut:]...)
			recomputeRouteDistances(nodes)
			egoDistance = projectPositionToRouteDistance(nodes, pos)
		}
	}
	lookahead := 0.0
	if len(nodes) > 0 {
		lookahead = math.Max(0, nodes[len(nodes)-1].DistanceMeters-egoDistance)
	}
	predecessor := math.Max(0, egoDistance)
	return DirectionalRoute{
		Generation: generation, Nodes: nodes, CurrentWayID: currentID,
		EgoDistanceMeters: egoDistance, LookaheadMeters: lookahead, PredecessorMeters: predecessor,
		TruncatedBefore: predecessor+1e-6 < RouteMinimumPastContextMeters,
		TruncatedAfter:  lookahead+1e-6 < RouteLookaheadMeters,
		FatalAmbiguity:  fatal,
	}, nil
}

func flattenSelectedRouteWays(segments []selectedRouteWay) ([]DirectionalRouteNode, bool, error) {
	nodes := make([]DirectionalRouteNode, 0)
	fatal := false
	var previous Way
	for segmentIndex, segment := range segments {
		wayNodes, err := segment.way.Nodes()
		if err != nil {
			return nil, true, err
		}
		if wayNodes.Len() < 2 {
			continue
		}
		wayID, err := StableWayID(segment.way)
		if err != nil {
			return nil, true, err
		}
		safeSpeeds, _ := segment.way.SafeSpeeds()
		hasBaked := safeSpeeds.Len() == wayNodes.Len()
		ordered := make([]DirectionalRouteNode, wayNodes.Len())
		for offset := 0; offset < wayNodes.Len(); offset++ {
			index := offset
			if !segment.forward {
				index = wayNodes.Len() - 1 - offset
			}
			node := wayNodes.At(index)
			var baked *float64
			if hasBaked {
				value := safeSpeeds.At(index)
				if wholeCurveIsFinite(value) {
					baked = &value
				}
			}
			ordered[offset] = DirectionalRouteNode{
				Latitude: node.Latitude(), Longitude: node.Longitude(),
				Sources:       []RouteNodeSource{{WayID: wayID, NodeIndex: index, IsForward: segment.forward}},
				BakedSpeedMPS: baked, Ambiguous: segment.ambiguous,
			}
		}
		if segmentIndex == 0 || len(nodes) == 0 {
			nodes = append(nodes, ordered...)
			previous = segment.way
			fatal = fatal || segment.ambiguous
			continue
		}
		boundary := len(nodes) - 1
		if sameRouteCoordinate(nodes[boundary], ordered[0]) {
			nodes[boundary].Sources = uniqueRouteSources(append(nodes[boundary].Sources, ordered[0].Sources...))
			nodes[boundary].WayBoundary = true
			nodes[boundary].Ambiguous = nodes[boundary].Ambiguous || segment.ambiguous
			nodes[boundary].MergeOrSplit = previous.Lanes() < segment.way.Lanes() ||
				(previous.Lanes() > segment.way.Lanes() && !previous.OneWay() && segment.way.OneWay())
			nodes = append(nodes, ordered[1:]...)
		} else {
			// A selected route chain must be physically connected. Keep legacy data
			// available, but make the whole-curve profile fail closed.
			fatal = true
			ordered[0].WayBoundary = true
			ordered[0].Ambiguous = true
			nodes = append(nodes, ordered...)
		}
		fatal = fatal || segment.ambiguous
		previous = segment.way
	}
	recomputeRouteDistances(nodes)
	return nodes, fatal, nil
}

func mergeRolloverRoute(previous, fresh []DirectionalRouteNode, currentWayID string) ([]DirectionalRouteNode, bool, bool) {
	if len(previous) == 0 || len(fresh) == 0 {
		return nil, false, false
	}
	type location struct{ previousIndex, freshIndex int }
	var match *location
	previousBySource := make(map[string]int)
	for i, node := range previous {
		for _, source := range node.Sources {
			previousBySource[source.Key()] = i
		}
	}
	for freshIndex, node := range fresh {
		for _, source := range node.Sources {
			if source.WayID != currentWayID {
				continue
			}
			if previousIndex, ok := previousBySource[source.Key()]; ok {
				candidate := location{previousIndex, freshIndex}
				match = &candidate
				break
			}
		}
		if match != nil {
			break
		}
	}
	if match == nil || !sameRouteCoordinate(previous[match.previousIndex], fresh[match.freshIndex]) {
		return nil, false, false
	}
	compatible := routeSuffixCompatible(previous[match.previousIndex:], fresh[match.freshIndex:])
	merged := append([]DirectionalRouteNode(nil), previous[:match.previousIndex]...)
	overlap := fresh[match.freshIndex]
	overlap.Sources = uniqueRouteSources(append(overlap.Sources, previous[match.previousIndex].Sources...))
	overlap.WayBoundary = overlap.WayBoundary || previous[match.previousIndex].WayBoundary
	overlap.MergeOrSplit = overlap.MergeOrSplit || previous[match.previousIndex].MergeOrSplit
	overlap.Ambiguous = overlap.Ambiguous || previous[match.previousIndex].Ambiguous
	merged = append(merged, overlap)
	merged = append(merged, fresh[match.freshIndex+1:]...)
	return merged, true, compatible
}

func routeSuffixCompatible(previous, fresh []DirectionalRouteNode) bool {
	common := minInt(len(previous), len(fresh))
	for index := 0; index < common; index++ {
		if !sameRouteCoordinate(previous[index], fresh[index]) || !routeNodesShareSource(previous[index], fresh[index]) {
			return false
		}
	}
	return common > 0
}

func routeNodesShareSource(first, second DirectionalRouteNode) bool {
	keys := make(map[string]struct{}, len(first.Sources))
	for _, source := range first.Sources {
		keys[source.Key()] = struct{}{}
	}
	for _, source := range second.Sources {
		if _, ok := keys[source.Key()]; ok {
			return true
		}
	}
	return false
}

func recomputeRouteDistances(nodes []DirectionalRouteNode) {
	if len(nodes) == 0 {
		return
	}
	nodes[0].DistanceMeters = 0
	for i := 1; i < len(nodes); i++ {
		nodes[i].DistanceMeters = nodes[i-1].DistanceMeters + DistanceToPoint(
			nodes[i-1].Latitude*TO_RADIANS, nodes[i-1].Longitude*TO_RADIANS,
			nodes[i].Latitude*TO_RADIANS, nodes[i].Longitude*TO_RADIANS,
		)
	}
}

func projectPositionToRouteDistance(nodes []DirectionalRouteNode, pos Position) float64 {
	if len(nodes) < 2 {
		return 0
	}
	bestDistance, bestS := math.Inf(1), 0.0
	for i := 0; i+1 < len(nodes); i++ {
		lat, lon := PointOnLine(nodes[i].Latitude, nodes[i].Longitude, nodes[i+1].Latitude, nodes[i+1].Longitude, pos.Latitude, pos.Longitude)
		distance := DistanceToPoint(pos.Latitude*TO_RADIANS, pos.Longitude*TO_RADIANS, lat*TO_RADIANS, lon*TO_RADIANS)
		if distance < bestDistance {
			segmentLength := nodes[i+1].DistanceMeters - nodes[i].DistanceMeters
			fraction := 0.0
			if segmentLength > 0 {
				along := DistanceToPoint(nodes[i].Latitude*TO_RADIANS, nodes[i].Longitude*TO_RADIANS, lat*TO_RADIANS, lon*TO_RADIANS)
				fraction = clampFloat(along/segmentLength, 0, 1)
			}
			bestDistance = distance
			bestS = nodes[i].DistanceMeters + segmentLength*fraction
		}
	}
	return bestS
}

func DirectionalRouteInput(route DirectionalRoute) []WholeCurveInputPoint {
	result := make([]WholeCurveInputPoint, len(route.Nodes))
	for i, node := range route.Nodes {
		keys := make([]string, 0, len(node.Sources))
		for _, source := range node.Sources {
			keys = append(keys, source.Key())
		}
		result[i] = WholeCurveInputPoint{Latitude: node.Latitude, Longitude: node.Longitude, SourceKeys: uniqueSortedStrings(keys)}
	}
	return result
}

func StableWayID(way Way) (string, error) {
	name, err := way.Name()
	if err != nil {
		return "", err
	}
	reference, err := way.Ref()
	if err != nil {
		return "", err
	}
	nodes, err := way.Nodes()
	if err != nil {
		return "", err
	}
	forward := stableNodeFingerprint(nodes, false)
	reverse := stableNodeFingerprint(nodes, true)
	geometry := forward
	if reverse < forward {
		geometry = reverse
	}
	prefix := fmt.Sprintf("%s\x00%s\x00%t\x00%d\x00%.17g\x00%.17g\x00%.17g\x00%s",
		name, reference, way.OneWay(), way.Lanes(), way.MaxSpeed(), way.MaxSpeedForward(), way.MaxSpeedBackward(), geometry)
	sum := sha256.Sum256([]byte(prefix))
	return hex.EncodeToString(sum[:16]), nil
}

func stableNodeFingerprint(nodes Coordinates_List, reverse bool) string {
	var builder strings.Builder
	builder.Grow(nodes.Len() * 32)
	for offset := 0; offset < nodes.Len(); offset++ {
		index := offset
		if reverse {
			index = nodes.Len() - offset - 1
		}
		node := nodes.At(index)
		var bits [16]byte
		binary.LittleEndian.PutUint64(bits[0:8], math.Float64bits(node.Latitude()))
		binary.LittleEndian.PutUint64(bits[8:16], math.Float64bits(node.Longitude()))
		builder.WriteString(hex.EncodeToString(bits[:]))
	}
	return builder.String()
}

func sameRouteCoordinate(first, second DirectionalRouteNode) bool {
	return first.Latitude == second.Latitude && first.Longitude == second.Longitude
}

func uniqueRouteSources(values []RouteNodeSource) []RouteNodeSource {
	byKey := make(map[string]RouteNodeSource, len(values))
	for _, value := range values {
		byKey[value.Key()] = value
	}
	keys := make([]string, 0, len(byKey))
	for key := range byKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]RouteNodeSource, 0, len(keys))
	for _, key := range keys {
		result = append(result, byKey[key])
	}
	return result
}
