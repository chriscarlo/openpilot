package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"capnproto.org/go/capnp/v3"
)

type jsonBounds struct {
	MinLatitude  float64 `json:"min_latitude"`
	MinLongitude float64 `json:"min_longitude"`
	MaxLatitude  float64 `json:"max_latitude"`
	MaxLongitude float64 `json:"max_longitude"`
}

type jsonNode struct {
	Latitude      float64  `json:"latitude"`
	Longitude     float64  `json:"longitude"`
	BakedSpeedMPS *float64 `json:"baked_speed_mps"`
}

type jsonWay struct {
	StableID                  string     `json:"stable_id"`
	Name                      string     `json:"name"`
	Reference                 string     `json:"reference"`
	Bounds                    jsonBounds `json:"bounds"`
	Nodes                     []jsonNode `json:"nodes"`
	MaxSpeedMPS               float64    `json:"max_speed_mps"`
	MaxSpeedForwardMPS        float64    `json:"max_speed_forward_mps"`
	MaxSpeedBackwardMPS       float64    `json:"max_speed_backward_mps"`
	AdvisorySpeedMPS          float64    `json:"advisory_speed_mps"`
	Lanes                     uint8      `json:"lanes"`
	Hazard                    string     `json:"hazard"`
	OneWay                    bool       `json:"one_way"`
	WindingForwardLevel       uint8      `json:"winding_forward_level"`
	WindingBackwardLevel      uint8      `json:"winding_backward_level"`
	WindingForwardScore       uint8      `json:"winding_forward_score"`
	WindingBackwardScore      uint8      `json:"winding_backward_score"`
	WindingForwardConfidence  uint8      `json:"winding_forward_confidence"`
	WindingBackwardConfidence uint8      `json:"winding_backward_confidence"`
}

type jsonTile struct {
	SourcePath    string     `json:"source_path"`
	Bounds        jsonBounds `json:"bounds"`
	Overlap       float64    `json:"overlap"`
	SchemaVersion uint16     `json:"schema_version"`
	SigmoidHash   string     `json:"sigmoid_hash"`
	Ways          []jsonWay  `json:"ways"`
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "vtsc-tile-decoder: %v\n", err)
		os.Exit(1)
	}
}

func run(paths []string, output io.Writer) error {
	if len(paths) == 0 {
		return errors.New("usage: vtsc-tile-decoder <packed-tile> [packed-tile ...]")
	}
	buffered := bufio.NewWriter(output)
	defer buffered.Flush()
	encoder := json.NewEncoder(buffered)
	encoder.SetEscapeHTML(false)
	for _, path := range paths {
		tile, err := decodeTile(path)
		if err != nil {
			return fmt.Errorf("decode %q: %w", path, err)
		}
		if err := encoder.Encode(tile); err != nil {
			return fmt.Errorf("encode %q: %w", path, err)
		}
	}
	return buffered.Flush()
}

func decodeTile(path string) (jsonTile, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return jsonTile{}, err
	}
	data, err := os.ReadFile(absolutePath)
	if err != nil {
		return jsonTile{}, err
	}
	message, err := capnp.UnmarshalPacked(data)
	if err != nil {
		return jsonTile{}, fmt.Errorf("unmarshal packed Cap'n Proto: %w", err)
	}
	root, err := ReadRootOffline(message)
	if err != nil {
		return jsonTile{}, fmt.Errorf("read Offline root: %w", err)
	}
	hash, err := root.SigmoidHash()
	if err != nil {
		return jsonTile{}, fmt.Errorf("read sigmoidHash: %w", err)
	}
	tile := jsonTile{
		SourcePath: absolutePath,
		Bounds: jsonBounds{
			MinLatitude: finiteOrZero(root.MinLat()), MinLongitude: finiteOrZero(root.MinLon()),
			MaxLatitude: finiteOrZero(root.MaxLat()), MaxLongitude: finiteOrZero(root.MaxLon()),
		},
		Overlap: finiteOrZero(root.Overlap()), SchemaVersion: root.SchemaVersion(), SigmoidHash: hash,
		Ways: make([]jsonWay, 0),
	}
	ways, err := root.Ways()
	if err != nil {
		return jsonTile{}, fmt.Errorf("read ways: %w", err)
	}
	tile.Ways = make([]jsonWay, 0, ways.Len())
	for index := 0; index < ways.Len(); index++ {
		decoded, err := decodeWay(ways.At(index))
		if err != nil {
			return jsonTile{}, fmt.Errorf("way %d: %w", index, err)
		}
		tile.Ways = append(tile.Ways, decoded)
	}
	return tile, nil
}

func decodeWay(way Way) (jsonWay, error) {
	name, err := way.Name()
	if err != nil {
		return jsonWay{}, fmt.Errorf("read name: %w", err)
	}
	reference, err := way.Ref()
	if err != nil {
		return jsonWay{}, fmt.Errorf("read ref: %w", err)
	}
	hazard, err := way.Hazard()
	if err != nil {
		return jsonWay{}, fmt.Errorf("read hazard: %w", err)
	}
	nodes, err := way.Nodes()
	if err != nil {
		return jsonWay{}, fmt.Errorf("read nodes: %w", err)
	}
	safeSpeeds, err := way.SafeSpeeds()
	if err != nil {
		return jsonWay{}, fmt.Errorf("read safeSpeeds: %w", err)
	}
	if safeSpeeds.Len() != 0 && safeSpeeds.Len() != nodes.Len() {
		return jsonWay{}, fmt.Errorf("safeSpeeds length %d does not match nodes length %d", safeSpeeds.Len(), nodes.Len())
	}
	decodedNodes := make([]jsonNode, nodes.Len())
	for index := 0; index < nodes.Len(); index++ {
		node := nodes.At(index)
		latitude := node.Latitude()
		longitude := node.Longitude()
		if !isFinite(latitude) || !isFinite(longitude) {
			return jsonWay{}, fmt.Errorf("node %d has non-finite coordinates", index)
		}
		decodedNodes[index] = jsonNode{Latitude: latitude, Longitude: longitude}
		if safeSpeeds.Len() == nodes.Len() {
			value := safeSpeeds.At(index)
			if isFinite(value) {
				decodedNodes[index].BakedSpeedMPS = &value
			}
		}
	}
	decoded := jsonWay{
		Name: name, Reference: reference,
		Bounds: jsonBounds{
			MinLatitude: finiteOrZero(way.MinLat()), MinLongitude: finiteOrZero(way.MinLon()),
			MaxLatitude: finiteOrZero(way.MaxLat()), MaxLongitude: finiteOrZero(way.MaxLon()),
		},
		Nodes:       decodedNodes,
		MaxSpeedMPS: finiteOrZero(way.MaxSpeed()), MaxSpeedForwardMPS: finiteOrZero(way.MaxSpeedForward()),
		MaxSpeedBackwardMPS: finiteOrZero(way.MaxSpeedBackward()), AdvisorySpeedMPS: finiteOrZero(way.AdvisorySpeed()),
		Lanes: way.Lanes(), Hazard: hazard, OneWay: way.OneWay(),
		WindingForwardLevel: way.WindingForwardLevel(), WindingBackwardLevel: way.WindingBackwardLevel(),
		WindingForwardScore: way.WindingForwardScore(), WindingBackwardScore: way.WindingBackwardScore(),
		WindingForwardConfidence: way.WindingForwardConfidence(), WindingBackwardConfidence: way.WindingBackwardConfidence(),
	}
	decoded.StableID = stableWayID(decoded)
	return decoded, nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteOrZero(value float64) float64 {
	if !isFinite(value) {
		return 0
	}
	return value
}

func stableWayID(way jsonWay) string {
	forward := nodeFingerprint(way.Nodes, false)
	reverse := nodeFingerprint(way.Nodes, true)
	geometry := forward
	if reverse < forward {
		geometry = reverse
	}
	prefix := fmt.Sprintf("%s\x00%s\x00%t\x00%d\x00%.17g\x00%.17g\x00%.17g\x00%s",
		way.Name, way.Reference, way.OneWay, way.Lanes, way.MaxSpeedMPS,
		way.MaxSpeedForwardMPS, way.MaxSpeedBackwardMPS, geometry)
	sum := sha256.Sum256([]byte(prefix))
	return hex.EncodeToString(sum[:16])
}

func nodeFingerprint(nodes []jsonNode, reverse bool) string {
	var builder strings.Builder
	builder.Grow(len(nodes) * 33)
	for offset := range nodes {
		index := offset
		if reverse {
			index = len(nodes) - offset - 1
		}
		var bits [16]byte
		binary.LittleEndian.PutUint64(bits[0:8], math.Float64bits(nodes[index].Latitude))
		binary.LittleEndian.PutUint64(bits[8:16], math.Float64bits(nodes[index].Longitude))
		builder.WriteString(hex.EncodeToString(bits[:]))
	}
	return builder.String()
}
