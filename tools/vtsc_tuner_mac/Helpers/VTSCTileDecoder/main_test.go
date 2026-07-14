package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const goldenTileBase64 = `EQFvAStQBgLgYEFA4KBcwOCAQUDgkFzA/3sUrkfheoQ/AAEBEQV3ET1qUQQJBQAA/936n7JQXkFAA08jLZW3pFzAJZUp5iBoQUDVpMrmYJ9cwPwBAXNzx8cABDERpwIAABNSAQH/ZjlkMzhhYjMADzM1N2MRqAL/KiIe5FBeQUBT1aTK5mCfXMDd+p+yUF5BQNnG6TWHn1zAj83pDVJeQUDEbo4pjp9cwBxOTw5VXkFA607BK5OfXMAOeoA4WV5BQFprKLWXn1zAVdvyNttgQUCBOcSJ+aBcwIyN1NbjYEFAuSUoNP+gXMDaOc0C7WBBQD+Skh4GoVzAMvZvqClhQUBvNxzxP6FcwL0cdt8xYUFAdsQhG0ihXMByZgXQOWFBQBhSNMVSoVzAYzyUslZhQUApX9BCgqFcwM6zSMlhYUFAlW9sLJehXMAISsCeZWFBQEdzCSyloVzA0sA0sWVhQUBdHJWbqKFcwN4mN+VlYUFA29ZgK7KhXMDZy1saZGFBQOntdIxQolzAaBayBGNhQUCIxzSJsKJcwJgK4BxkYUFA3dRA87miXMAfRNcqZmFBQPM1FfO9olzAR7Hc0mphQUAC1xUzwqJcwJltAmeEYUFA9PEaBdKiXMCq6i8FtWFBQHoOGxvwolzA5bZ9j/phQUCEYzuEFqNcwCj4AaMmYkFAuVSlLS6jXMCl4e9ybGJBQKY09oBUo1zA5xCCn4JiQUCnLJ7VXaNcwAnREETMYkFAhmaNMHujXMCfVzz1SGNBQJA3VOeso1zAoygr4WRjQUCSBIZRtaNcwHXy2IXVY0FA1H4QbtejXMASO66hL2RBQOq9jL3yo1zAFZOSeT9kQUBHQfD49qNcwETxdPi1ZEFA+wlntxakXMDaiMwBJ2VBQOVH/Io1pFzAL3axHxhmQUCY4qqyb6RcwI4U2+4vZkFA1zj2R3OkXMCpY99fTmdBQEapc/ubpFzArXlEO11nQUCswgvHmKRcwDRaWIFrZ0FAdhw/VJqkXMAqjT0gdWdBQJCkpIehpFzAJZUp5iBoQUBPIy2Vt6RcwOCAUUD/Hkt5R5UXUUACCMxyjgKqOEAPwvqH5oI8QOCAUUDggFFA/9BT9TZXlktAAOCAUUDggFFA/8F08tCuJkdAAOCAUUDggFFA/6BSn/jTuktAAcEz26EGBUNA4IBRQOCAUUDggFFA4IBRQP+4GDMhF5k6QAJRmZXWpl4tQHEKQ6NzQUVA4IBRQOCAUUDggFFA4IBRQOCAUUDggFFA4IBRQOCAUUDggFFA4IBRQOCAUUDggFFA4IBRQOCAUUDggFFA4IBRQP+6koHX+HRQQAMCOBHONxIvQPaMVFRUHipA7u2Fp5XPR0DggFFAP1X///9VAQ==`

func writeGoldenTile(t *testing.T) string {
	t.Helper()
	data, err := base64.StdEncoding.DecodeString(goldenTileBase64)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "34.750000_-114.500000_35.000000_-114.250000")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDecodeGoldenRealTile(t *testing.T) {
	tile, err := decodeTile(writeGoldenTile(t))
	if err != nil {
		t.Fatal(err)
	}
	if tile.Bounds.MinLatitude != 34.75 || tile.Bounds.MinLongitude != -114.5 ||
		tile.Bounds.MaxLatitude != 35.0 || tile.Bounds.MaxLongitude != -114.25 {
		t.Fatalf("unexpected bounds: %+v", tile.Bounds)
	}
	if tile.SchemaVersion != 1 || tile.SigmoidHash != "f9d38ab3357c" {
		t.Fatalf("unexpected schema/hash: %d %q", tile.SchemaVersion, tile.SigmoidHash)
	}
	if len(tile.Ways) == 0 {
		t.Fatal("golden tile has no ways")
	}
	for _, way := range tile.Ways {
		if way.StableID == "" {
			t.Fatal("way stable ID is empty")
		}
		for _, node := range way.Nodes {
			if node.BakedSpeedMPS == nil {
				t.Fatal("schema-v1 golden node is missing a baked speed")
			}
		}
	}
}

func TestRunEmitsOneCompactJSONLinePerInput(t *testing.T) {
	path := writeGoldenTile(t)
	var output bytes.Buffer
	if err := run([]string{path, path}, &output); err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(output.Bytes()), []byte{'\n'})
	if len(lines) != 2 {
		t.Fatalf("got %d JSON lines, want 2", len(lines))
	}
	for _, line := range lines {
		var tile jsonTile
		if err := json.Unmarshal(line, &tile); err != nil {
			t.Fatalf("invalid compact JSON: %v", err)
		}
	}
}

func TestDecodeRejectsTruncatedTile(t *testing.T) {
	path := writeGoldenTile(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data[:len(data)/2], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeTile(path); err == nil {
		t.Fatal("truncated tile unexpectedly decoded")
	}
}

func TestNonFiniteMapMetadataIsJSONSafe(t *testing.T) {
	if isFinite(math.NaN()) {
		t.Fatal("NaN must not be treated as finite")
	}
	if got := finiteOrZero(math.Inf(1)); got != 0 {
		t.Fatalf("finiteOrZero(+Inf) = %v, want 0", got)
	}
	node := jsonNode{Latitude: 37.0, Longitude: -122.0}
	if _, err := json.Marshal(node); err != nil {
		t.Fatalf("a node with unavailable baked speed should encode as JSON: %v", err)
	}
}
