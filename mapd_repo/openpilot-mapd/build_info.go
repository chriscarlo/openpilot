package main

import (
	"encoding/json"
	"io"
)

const MapWholeCurveCapability = "MapWholeCurveProfile:whole-curve-v3"

// Release builds inject a standalone-commit identity with -X. The marker
// variables are deliberately separate because the installer verifies the raw
// binary before executing it; release tooling must inject all four together.
var (
	MapdReleaseID     = "chauffeur-whole-curve-v3"
	MapdBuildID       = "development"
	MapdReleaseMarker = "MapdReleaseID:chauffeur-whole-curve-v3"
	MapdBuildMarker   = "MapdBuildID:development"
)

type MapdBuildInfo struct {
	ReleaseID        string   `json:"releaseID"`
	BuildID          string   `json:"buildID"`
	EstimatorVersion string   `json:"estimatorVersion"`
	Capabilities     []string `json:"capabilities"`
	IdentityMarkers  []string `json:"identityMarkers"`
}

func CurrentMapdBuildInfo() MapdBuildInfo {
	return MapdBuildInfo{
		ReleaseID: MapdReleaseID, BuildID: MapdBuildID,
		EstimatorVersion: WholeCurveProfileVersion,
		Capabilities:     []string{MapWholeCurveCapability},
		IdentityMarkers:  []string{MapdReleaseMarker, MapdBuildMarker},
	}
}

func WriteMapdBuildInfo(output io.Writer) error {
	return json.NewEncoder(output).Encode(CurrentMapdBuildInfo())
}
