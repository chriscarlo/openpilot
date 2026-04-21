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

func TestConfiguredTileBaseURLDefaultsToEmpty(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	// Chauffeur has no tile CDN; with nothing configured, tile download must
	// be disabled rather than silently falling back to a third-party host.
	if got := configuredTileBaseURL(); got != "" {
		t.Fatalf("expected empty default tile base URL, got %q", got)
	}
	if DEFAULT_TILE_BASE_URL != "" {
		t.Fatalf("DEFAULT_TILE_BASE_URL must stay empty; got %q", DEFAULT_TILE_BASE_URL)
	}
}

func TestTileArchiveURLIsEmptyWhenNoBaseConfigured(t *testing.T) {
	withTempParamPaths(t)
	t.Setenv("MAPD_TILE_BASE_URL", "")

	if got := tileArchiveURL("/offline/38/-122.tar.gz"); got != "" {
		t.Fatalf("expected empty tile URL when no base configured, got %q", got)
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

	if got := configuredTileBaseURL(); got != "" {
		t.Fatalf("expected empty tile base URL after missing params, got %q", got)
	}
}
