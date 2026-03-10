package main

import (
	"os"
	"path/filepath"
	"testing"
)

func withTempParamPaths(t *testing.T) {
	t.Helper()
	oldParamsPath := ParamsPath
	oldMemParamsPath := MemParamsPath
	oldBasePath := BasePath

	root := t.TempDir()
	ParamsPath = filepath.Join(root, "params", "d")
	MemParamsPath = filepath.Join(root, "mem", "d")
	BasePath = root
	EnsureParamDirectories()

	t.Cleanup(func() {
		ParamsPath = oldParamsPath
		MemParamsPath = oldMemParamsPath
		BasePath = oldBasePath
	})
}

func TestConfiguredTileBaseURLDefaults(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	if got := configuredTileBaseURL(); got != DEFAULT_TILE_BASE_URL {
		t.Fatalf("expected default tile base URL %q, got %q", DEFAULT_TILE_BASE_URL, got)
	}
}

func TestConfiguredTileBaseURLUsesPersistentParam(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	err := PutParam(ParamPath("MapdTileBaseUrl", false), []byte("https://tiles.example.com/root/"))
	if err != nil {
		t.Fatalf("failed to write persistent tile base URL: %v", err)
	}

	if got := configuredTileBaseURL(); got != "https://tiles.example.com/root" {
		t.Fatalf("expected trimmed persistent tile base URL, got %q", got)
	}
}

func TestConfiguredTileBaseURLMemParamOverridesPersistent(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	err := PutParam(ParamPath("MapdTileBaseUrl", false), []byte("https://tiles.example.com/persistent"))
	if err != nil {
		t.Fatalf("failed to write persistent tile base URL: %v", err)
	}
	err = PutParam(ParamPath("MapdTileBaseUrl", true), []byte("https://tiles.example.com/mem/"))
	if err != nil {
		t.Fatalf("failed to write memory tile base URL: %v", err)
	}

	if got := configuredTileBaseURL(); got != "https://tiles.example.com/mem" {
		t.Fatalf("expected memory tile base URL to win, got %q", got)
	}
}

func TestConfiguredTileBaseURLEnvFallback(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "https://env.example.com/maps/")

	if got := configuredTileBaseURL(); got != "https://env.example.com/maps" {
		t.Fatalf("expected env tile base URL, got %q", got)
	}
}

func TestTileArchiveURLJoinsPathCleanly(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "https://env.example.com/maps/")

	got := tileArchiveURL("/offline/38/-122.tar.gz")
	want := "https://env.example.com/maps/offline/38/-122.tar.gz"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestConfiguredTileBaseURLIgnoresMissingParamFiles(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	_ = os.Remove(ParamPath("MapdTileBaseUrl", true))
	_ = os.Remove(ParamPath("MapdTileBaseUrl", false))

	if got := configuredTileBaseURL(); got != DEFAULT_TILE_BASE_URL {
		t.Fatalf("expected default tile base URL after missing params, got %q", got)
	}
}
