package main

type RouteWindingSummary struct {
	Valid             bool  `json:"valid"`
	Level             uint8 `json:"level"`
	Score             uint8 `json:"score"`
	Confidence        uint8 `json:"confidence"`
	CurrentLevel      uint8 `json:"currentLevel"`
	CurrentScore      uint8 `json:"currentScore"`
	CurrentConfidence uint8 `json:"currentConfidence"`
	WayCount          uint8 `json:"wayCount"`
}

func DirectionalWindingSummary(way Way, isForward bool) CompactWindingSummary {
	if !way.IsValid() {
		return CompactWindingSummary{}
	}
	if isForward {
		return CompactWindingSummary{
			Level:      way.WindingForwardLevel(),
			Score:      way.WindingForwardScore(),
			Confidence: way.WindingForwardConfidence(),
		}
	}
	return CompactWindingSummary{
		Level:      way.WindingBackwardLevel(),
		Score:      way.WindingBackwardScore(),
		Confidence: way.WindingBackwardConfidence(),
	}
}

func AggregateRouteWindingSummary(current CompactWindingSummary, future []CompactWindingSummary) RouteWindingSummary {
	summaries := make([]CompactWindingSummary, 0, len(future)+1)
	summaries = append(summaries, current)
	summaries = append(summaries, future...)

	summary := RouteWindingSummary{
		CurrentLevel:      current.Level,
		CurrentScore:      current.Score,
		CurrentConfidence: current.Confidence,
		WayCount:          uint8(minInt(len(summaries), 255)),
	}
	if len(summaries) == 0 {
		return summary
	}

	weights := []float64{1.00, 0.85, 0.72, 0.60, 0.50, 0.42}
	scoreMiss := 1.0
	confMiss := 1.0
	maxLevel := int(current.Level)
	valid := false

	for idx, item := range summaries {
		score01 := float64(item.Score) / 255.0
		conf01 := float64(item.Confidence) / 255.0
		if int(item.Level) > maxLevel {
			maxLevel = int(item.Level)
		}
		if item.Level > 0 || item.Score > 0 || item.Confidence > 0 {
			valid = true
		}

		weight := weights[minInt(idx, len(weights)-1)]
		scoreMiss *= (1.0 - windingClip01(score01*weight))
		confMiss *= (1.0 - windingClip01(conf01*weight))
	}

	summary.Valid = valid
	summary.Level = uint8(clampInt(maxLevel, 0, 5))
	summary.Score = windingScale01ToUint8(1.0 - scoreMiss)
	summary.Confidence = windingScale01ToUint8(1.0 - confMiss)
	return summary
}

func ComputeRouteWindingSummary(currentWay CurrentWay, nextWays []NextWayResult) RouteWindingSummary {
	current := DirectionalWindingSummary(currentWay.Way, currentWay.OnWay.IsForward)
	future := make([]CompactWindingSummary, 0, len(nextWays))
	for _, nextWay := range nextWays {
		future = append(future, DirectionalWindingSummary(nextWay.Way, nextWay.IsForward))
	}
	return AggregateRouteWindingSummary(current, future)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
