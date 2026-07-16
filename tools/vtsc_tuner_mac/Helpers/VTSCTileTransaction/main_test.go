package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVerifyRejectsUnexpectedFileAndAcceptsEmptyRsyncPartial(t *testing.T) {
	root := newTestRoot(t)
	stage := writeStage(t, root, "fresh")
	partial := filepath.Join(stage, activeOfflineName, ".rsync-partial")
	if err := os.Mkdir(partial, 0o755); err != nil {
		t.Fatalf("create empty rsync partial directory: %v", err)
	}

	manifest, err := verifyArtifactRoot(stage, "fresh", false)
	if err != nil {
		t.Fatalf("verify stage with empty rsync partial directory: %v", err)
	}
	if manifest.FileCount != 1 {
		t.Fatalf("verified file count = %d, want 1", manifest.FileCount)
	}
	if err := os.WriteFile(filepath.Join(partial, "unfinished"), []byte("not canonical"), 0o644); err != nil {
		t.Fatalf("write partial tile: %v", err)
	}
	if _, err := verifyArtifactRoot(stage, "fresh", false); err == nil || !strings.Contains(err.Error(), "unexpected tile file") {
		t.Fatalf("verify nonempty rsync partial error = %v, want unexpected tile file", err)
	}
	if err := os.Remove(filepath.Join(partial, "unfinished")); err != nil {
		t.Fatalf("remove partial tile: %v", err)
	}

	if err := os.WriteFile(filepath.Join(stage, activeOfflineName, "38", "-121", "unexpected"), []byte("not in the manifest"), 0o644); err != nil {
		t.Fatalf("write unexpected tile: %v", err)
	}
	if _, err := verifyArtifactRoot(stage, "fresh", false); err == nil || !strings.Contains(err.Error(), "unexpected tile file") {
		t.Fatalf("verify unexpected file error = %v, want unexpected tile file", err)
	}
}

func TestRunVerifyWritesMachineReadableJSON(t *testing.T) {
	root := newTestRoot(t)
	stage := writeStage(t, root, "fresh")
	var output bytes.Buffer
	if err := run([]string{"verify", "--root", stage, "--tile-set-id", "fresh"}, &output); err != nil {
		t.Fatalf("run verify: %v", err)
	}
	var result transactionResult
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatalf("decode verify result %q: %v", output.String(), err)
	}
	if result.Operation != "verify" || result.TileSetID != "fresh" || result.FileCount != 1 || result.TotalBytes == 0 {
		t.Fatalf("unexpected verify result: %+v", result)
	}
}

func TestTransactionLockSerializesConcurrentMutations(t *testing.T) {
	root := newTestRoot(t)
	engine := testEngine(t, root)
	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		_, err := engine.withExclusiveTransactionLock(func() (transactionResult, error) {
			close(firstEntered)
			<-releaseFirst
			return transactionResult{}, nil
		})
		firstDone <- err
	}()
	<-firstEntered

	secondEntered := make(chan struct{})
	secondDone := make(chan error, 1)
	go func() {
		_, err := engine.withExclusiveTransactionLock(func() (transactionResult, error) {
			close(secondEntered)
			return transactionResult{}, nil
		})
		secondDone <- err
	}()

	select {
	case <-secondEntered:
		t.Fatal("second transaction entered while the first held the lock")
	case <-time.After(50 * time.Millisecond):
	}
	close(releaseFirst)
	if err := <-firstDone; err != nil {
		t.Fatalf("first locked mutation: %v", err)
	}
	select {
	case <-secondEntered:
	case <-time.After(time.Second):
		t.Fatal("second transaction did not acquire the released lock")
	}
	if err := <-secondDone; err != nil {
		t.Fatalf("second locked mutation: %v", err)
	}
}

func TestLockWaitStateFlipRejectsActivationBeforeAnyExchange(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "old")
	makeMinimalGeneration(t, root, "older")
	linkGeneration(t, root, activeOfflineName, "old")
	linkGeneration(t, root, previousOfflineName, "older")
	stage := writeStage(t, root, "fresh")
	engine := testEngine(t, root)

	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	go func() {
		_, _ = engine.withExclusiveTransactionLock(func() (transactionResult, error) {
			close(firstEntered)
			<-releaseFirst
			return transactionResult{}, nil
		})
	}()
	<-firstEntered

	result := make(chan error, 1)
	go func() {
		_, err := engine.activate(stage, "fresh", "")
		result <- err
	}()
	if err := os.WriteFile(filepath.Join(engine.paramsDir, "IsOnroad"), []byte("1"), 0o600); err != nil {
		t.Fatalf("flip onroad Param while helper waits: %v", err)
	}
	close(releaseFirst)
	if err := <-result; err == nil || !strings.Contains(err.Error(), "IsOnroad") {
		t.Fatalf("activation after state flip error = %v, want exact parked-state rejection", err)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/old/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/older/offline")
}

func TestLockWaitStateFlipRejectsRollbackBeforeAnyExchange(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "fresh")
	makeMinimalGeneration(t, root, "old")
	linkGeneration(t, root, activeOfflineName, "fresh")
	linkGeneration(t, root, previousOfflineName, "old")
	engine := testEngine(t, root)

	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	go func() {
		_, _ = engine.withExclusiveTransactionLock(func() (transactionResult, error) {
			close(firstEntered)
			<-releaseFirst
			return transactionResult{}, nil
		})
	}()
	<-firstEntered

	result := make(chan error, 1)
	go func() {
		_, err := engine.rollback("fresh", "old", "")
		result <- err
	}()
	if err := os.WriteFile(filepath.Join(engine.paramsDir, "MTSCLookaheadEnabled"), []byte("1"), 0o600); err != nil {
		t.Fatalf("flip lookahead Param while helper waits: %v", err)
	}
	close(releaseFirst)
	if err := <-result; err == nil || !strings.Contains(err.Error(), "MTSCLookaheadEnabled") {
		t.Fatalf("rollback after state flip error = %v, want exact parked-state rejection", err)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/old/offline")
}

func TestActivateRollbackAndRecovery(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "old")
	makeMinimalGeneration(t, root, "older")
	linkGeneration(t, root, activeOfflineName, "old")
	linkGeneration(t, root, previousOfflineName, "older")
	stage := writeStage(t, root, "fresh")
	if err := os.Mkdir(filepath.Join(stage, activeOfflineName, ".rsync-partial"), 0o755); err != nil {
		t.Fatalf("create rsync partial directory: %v", err)
	}
	engine := testEngine(t, root)

	activated, err := engine.activate(stage, "fresh", "")
	if err != nil {
		t.Fatalf("activate: %v", err)
	}
	if activated.ActivatedTileSetID != "fresh" || activated.Recovered {
		t.Fatalf("unexpected activation result: %+v", activated)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/old/offline")
	if _, err := os.Lstat(filepath.Join(root, generationDirectory, "fresh", activeOfflineName, ".rsync-partial")); !os.IsNotExist(err) {
		t.Fatalf("staging rsync partial directory survived promotion: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(root, transactionFileName)); !os.IsNotExist(err) {
		t.Fatalf("activation journal still present: %v", err)
	}

	rolledBack, err := engine.rollback("fresh", "old", "")
	if err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if rolledBack.RolledBackTileSetID != "old" || rolledBack.Recovered {
		t.Fatalf("unexpected rollback result: %+v", rolledBack)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/old/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/fresh/offline")
}

func TestActivationRecoveryAfterPointerExchange(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "old")
	makeMinimalGeneration(t, root, "older")
	linkGeneration(t, root, activeOfflineName, "old")
	linkGeneration(t, root, previousOfflineName, "older")
	stage := writeStage(t, root, "fresh")
	engine := testEngine(t, root)

	if _, err := engine.activate(stage, "fresh", "after_switch"); err == nil {
		t.Fatal("activation with injected post-switch failure unexpectedly succeeded")
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")

	recovered, err := engine.activate(stage, "fresh", "")
	if err != nil {
		t.Fatalf("recover activation: %v", err)
	}
	if !recovered.Recovered || recovered.ActivatedTileSetID != "fresh" {
		t.Fatalf("unexpected recovery result: %+v", recovered)
	}
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/old/offline")
	if _, err := os.Lstat(filepath.Join(root, transactionFileName)); !os.IsNotExist(err) {
		t.Fatalf("recovery journal still present: %v", err)
	}
}

func TestRollbackRejectsDriftedPreviousGenerationBeforeExchange(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "fresh")
	makeMinimalGeneration(t, root, "unrelated")
	linkGeneration(t, root, activeOfflineName, "fresh")
	linkGeneration(t, root, previousOfflineName, "unrelated")
	engine := testEngine(t, root)

	if _, err := engine.rollback("fresh", "old", ""); err == nil {
		t.Fatal("rollback accepted a previous generation that differed from the recorded identity")
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/unrelated/offline")
	if _, err := os.Lstat(filepath.Join(root, transactionFileName)); !os.IsNotExist(err) {
		t.Fatalf("rollback wrote a transaction before rejecting previous identity drift: %v", err)
	}
}

func TestActivationMigratesLegacyDirectoryBeforeExchange(t *testing.T) {
	root := newTestRoot(t)
	legacyTile := filepath.Join(root, activeOfflineName, "38", "-121", "legacy-tile")
	if err := os.MkdirAll(filepath.Dir(legacyTile), 0o755); err != nil {
		t.Fatalf("create legacy tree: %v", err)
	}
	if err := os.WriteFile(legacyTile, bytes.Repeat([]byte("legacy"), 20), 0o644); err != nil {
		t.Fatalf("write legacy tile: %v", err)
	}
	stage := writeStage(t, root, "fresh")
	engine := testEngine(t, root)

	if _, err := engine.activate(stage, "fresh", ""); err != nil {
		t.Fatalf("activate from legacy directory: %v", err)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")
	previousTarget, isLink, err := symlinkTarget(filepath.Join(root, previousOfflineName))
	if err != nil || !isLink || !strings.HasPrefix(previousTarget, "tile-generations/legacy-") {
		t.Fatalf("legacy previous target = %q (isLink=%t err=%v), want immutable legacy generation", previousTarget, isLink, err)
	}
	if _, err := os.Stat(filepath.Join(root, previousTarget, "38", "-121", "legacy-tile")); err != nil {
		t.Fatalf("migrated legacy tile is unavailable: %v", err)
	}
	retained, err := filepath.Glob(filepath.Join(root, ".retained-pre-generation-fresh-*"))
	if err != nil || len(retained) != 1 {
		t.Fatalf("legacy source was not retained after exchange: paths=%v err=%v", retained, err)
	}
}

func TestLegacyActivationRecoversBeforePointerExchange(t *testing.T) {
	root := newTestRoot(t)
	legacyTile := filepath.Join(root, activeOfflineName, "38", "-121", "legacy-tile")
	if err := os.MkdirAll(filepath.Dir(legacyTile), 0o755); err != nil {
		t.Fatalf("create legacy tree: %v", err)
	}
	if err := os.WriteFile(legacyTile, bytes.Repeat([]byte("legacy"), 20), 0o644); err != nil {
		t.Fatalf("write legacy tile: %v", err)
	}
	stage := writeStage(t, root, "fresh")
	engine := testEngine(t, root)

	if _, err := engine.activate(stage, "fresh", "after_journal"); err == nil {
		t.Fatal("legacy activation with injected pre-switch failure unexpectedly succeeded")
	}
	if _, isLink, err := symlinkTarget(filepath.Join(root, activeOfflineName)); err != nil || isLink {
		t.Fatalf("legacy active tree changed before pointer exchange: isLink=%t err=%v", isLink, err)
	}

	activated, err := engine.activate(stage, "fresh", "")
	if err != nil {
		t.Fatalf("recover legacy activation: %v", err)
	}
	if activated.Recovered || activated.ActivatedTileSetID != "fresh" {
		t.Fatalf("unexpected legacy recovery result: %+v", activated)
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/fresh/offline")
	previousTarget, isLink, err := symlinkTarget(filepath.Join(root, previousOfflineName))
	if err != nil || !isLink || !strings.HasPrefix(previousTarget, "tile-generations/legacy-") {
		t.Fatalf("legacy recovery previous target = %q (isLink=%t err=%v)", previousTarget, isLink, err)
	}
}

func TestRollbackRecoveryAfterPointerExchange(t *testing.T) {
	root := newTestRoot(t)
	makeMinimalGeneration(t, root, "old")
	makeMinimalGeneration(t, root, "older")
	linkGeneration(t, root, activeOfflineName, "old")
	linkGeneration(t, root, previousOfflineName, "older")
	stage := writeStage(t, root, "fresh")
	engine := testEngine(t, root)
	if _, err := engine.activate(stage, "fresh", ""); err != nil {
		t.Fatalf("activate: %v", err)
	}

	if _, err := engine.rollback("fresh", "old", "after_switch"); err == nil {
		t.Fatal("rollback with injected post-switch failure unexpectedly succeeded")
	}
	assertLinkTarget(t, filepath.Join(root, activeOfflineName), "tile-generations/old/offline")
	assertLinkTarget(t, filepath.Join(root, previousOfflineName), "tile-generations/fresh/offline")

	recovered, err := engine.rollback("fresh", "old", "")
	if err != nil {
		t.Fatalf("recover rollback: %v", err)
	}
	if !recovered.Recovered || recovered.RolledBackTileSetID != "old" {
		t.Fatalf("unexpected rollback recovery result: %+v", recovered)
	}
	if _, err := os.Lstat(filepath.Join(root, transactionFileName)); !os.IsNotExist(err) {
		t.Fatalf("rollback recovery journal still present: %v", err)
	}
}

func writeStage(t *testing.T, root, tileSetID string) string {
	t.Helper()
	stage := filepath.Join(root, ".tileset-"+tileSetID+".partial")
	tilePath := filepath.Join(stage, activeOfflineName, "38", "-121", "tile")
	contents := bytes.Repeat([]byte("canonical-map-tile"), 12)
	if err := os.MkdirAll(filepath.Dir(tilePath), 0o755); err != nil {
		t.Fatalf("create tile directory: %v", err)
	}
	if err := os.WriteFile(tilePath, contents, 0o644); err != nil {
		t.Fatalf("write tile: %v", err)
	}
	digest := sha256.Sum256(contents)
	manifest := tileManifest{
		TileSetID:  tileSetID,
		FileCount:  1,
		TotalBytes: uint64(len(contents)),
		Files: []manifestFile{{
			Path:      "offline/38/-121/tile",
			ByteCount: uint64(len(contents)),
			SHA256:    hex.EncodeToString(digest[:]),
		}},
	}
	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("encode manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(stage, "manifest.json"), append(manifestBytes, '\n'), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return stage
}

func makeMinimalGeneration(t *testing.T, root, tileSetID string) {
	t.Helper()
	generation := filepath.Join(root, generationDirectory, tileSetID, activeOfflineName)
	if err := os.MkdirAll(generation, 0o755); err != nil {
		t.Fatalf("create generation %s: %v", tileSetID, err)
	}
	manifest := fmt.Sprintf("{\"tile_set_id\":%q}\n", tileSetID)
	if err := os.WriteFile(filepath.Join(generation, embeddedManifestName), []byte(manifest), 0o444); err != nil {
		t.Fatalf("write generation manifest %s: %v", tileSetID, err)
	}
}

func linkGeneration(t *testing.T, root, name, tileSetID string) {
	t.Helper()
	target := filepath.ToSlash(filepath.Join(generationDirectory, tileSetID, activeOfflineName))
	if err := os.Symlink(target, filepath.Join(root, name)); err != nil {
		t.Fatalf("link %s to %s: %v", name, target, err)
	}
}

func testEngine(t *testing.T, root string) *transactionEngine {
	t.Helper()
	paramsDir := filepath.Join(root, "test-params")
	if err := os.MkdirAll(paramsDir, 0o755); err != nil {
		t.Fatalf("create test Params directory: %v", err)
	}
	for key, value := range map[string]string{
		"IsOffroad": "1",
		"IsOnroad": "0",
		"MTSCLookaheadEnabled": "0",
	} {
		if err := os.WriteFile(filepath.Join(paramsDir, key), []byte(value), 0o600); err != nil {
			t.Fatalf("write test safety Param %s: %v", key, err)
		}
	}
	engine, err := newTransactionEngine(root, paramsDir)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	engine.exchange = emulatedRenameExchange
	engine.syncDir = func(string) error { return nil }
	engine.now = func() time.Time { return time.Unix(1_700_000_000, 42) }
	return engine
}

func emulatedRenameExchange(first, second string) error {
	temporary, err := os.CreateTemp(filepath.Dir(first), ".test-rename-exchange-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Remove(temporaryPath); err != nil {
		return err
	}
	if err := os.Rename(first, temporaryPath); err != nil {
		return err
	}
	if err := os.Rename(second, first); err != nil {
		_ = os.Rename(temporaryPath, first)
		return err
	}
	if err := os.Rename(temporaryPath, second); err != nil {
		return err
	}
	return nil
}

func assertLinkTarget(t *testing.T, link, expected string) {
	t.Helper()
	target, isLink, err := symlinkTarget(link)
	if err != nil || !isLink || target != expected {
		t.Fatalf("symlink %s = %q (isLink=%t err=%v), want %q", link, target, isLink, err, expected)
	}
}

func newTestRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Cleanup(func() {
		if err := filepath.WalkDir(root, func(current string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return nil
			}
			if info.IsDir() {
				return os.Chmod(current, 0o700)
			}
			return os.Chmod(current, 0o600)
		}); err != nil {
			t.Errorf("restore test fixture permissions: %v", err)
		}
	})
	return root
}
