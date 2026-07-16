package main

// This helper is intentionally small and narrow: the Mac Swift app owns
// deployment policy, release identity, and staging. The tici receives only a
// verified artifact tree and uses this binary for Linux-only durable filesystem
// operations that a POSIX shell cannot perform, notably RENAME_EXCHANGE.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	embeddedManifestName = ".tileset-manifest.json"
	transactionFileName  = ".tileset-transaction.json"
	activeOfflineName    = "offline"
	previousOfflineName  = "offline.previous"
	generationDirectory  = "tile-generations"
	transactionLockName  = ".tileset-transaction.lock"
)

type manifestFile struct {
	Path      string `json:"path"`
	ByteCount uint64 `json:"byte_count"`
	SHA256    string `json:"sha256"`
}

// TileManifest deliberately retains only the fields needed on-device. Swift
// validates the complete manifest schema and identity before transfer; the
// helper independently validates the canonical remote file tree before it can
// become active.
type tileManifest struct {
	TileSetID  string         `json:"tile_set_id"`
	FileCount  int            `json:"file_count"`
	TotalBytes uint64         `json:"total_bytes"`
	Files      []manifestFile `json:"files"`
}

type manifestIdentity struct {
	TileSetID                      string `json:"tile_set_id"`
	Legacy                         bool   `json:"legacy,omitempty"`
	LegacyMigrationTargetTileSetID string `json:"legacy_migration_target_tile_set_id,omitempty"`
	LegacyTreeSHA256               string `json:"legacy_tree_sha256,omitempty"`
}

// Every path that a new activation may create is authorized before the first
// creation. A pointer makes absence of this block distinguishable from false
// boolean values in legacy transaction JSON, which must fail closed rather
// than guessing ownership during cleanup.
type activationArtifact struct {
	Path        string `json:"path"`
	Tombstone   string `json:"tombstone,omitempty"`
	Preexisting bool   `json:"preexisting"`
	TreeSHA256  string `json:"treeSHA256,omitempty"`
}

type activationArtifactAuthority struct {
	Schema                    int                 `json:"schema"`
	TargetGeneration          activationArtifact  `json:"targetGeneration"`
	TargetBuilding            activationArtifact  `json:"targetBuilding"`
	LegacyMigrationPlanned    bool                `json:"legacyMigrationPlanned"`
	LegacyGeneration          *activationArtifact `json:"legacyGeneration,omitempty"`
	LegacyBuilding            *activationArtifact `json:"legacyBuilding,omitempty"`
	Switch                    activationArtifact  `json:"switch"`
	PreviousTemporary         activationArtifact  `json:"previousTemporary"`
	RetainedPrefix            string              `json:"retainedPrefix"`
	RetainedTombstone         string              `json:"retainedTombstone"`
	PreviousPointerTarget     string              `json:"previousPointerTarget,omitempty"`
	PreviousPointerID         string              `json:"previousPointerID,omitempty"`
	PreviousPointerTreeSHA256 string              `json:"previousPointerTreeSHA256,omitempty"`
}

// The field names deliberately match the pre-existing Python journal so a
// helper can recover a transaction written by the old implementation during a
// rolling upgrade.
type tileTransaction struct {
	Kind           string                       `json:"kind"`
	NewID          string                       `json:"newID,omitempty"`
	NewTarget      string                       `json:"newTarget,omitempty"`
	PreviousTarget string                       `json:"previousTarget,omitempty"`
	PreviousID     string                       `json:"previousID,omitempty"`
	SwitchPath     string                       `json:"switchPath,omitempty"`
	ActiveTarget   string                       `json:"activeTarget,omitempty"`
	ActiveID       string                       `json:"activeID,omitempty"`
	Artifacts      *activationArtifactAuthority `json:"artifacts,omitempty"`
}

type transactionResult struct {
	Operation                 string `json:"operation"`
	TileSetID                 string `json:"tile_set_id,omitempty"`
	ActivatedTileSetID        string `json:"activated_tile_set_id,omitempty"`
	RolledBackTileSetID       string `json:"rolled_back_tile_set_id,omitempty"`
	PreviousTileSetID         string `json:"previous_tile_set_id,omitempty"`
	PreviousTileSetProvenance string `json:"previous_tile_set_provenance,omitempty"`
	PreviousTileSetTargetID   string `json:"previous_tile_set_target_id,omitempty"`
	TargetAlreadyActive       bool   `json:"target_already_active,omitempty"`
	ActiveTileSetID           string `json:"active_tile_set_id,omitempty"`
	FileCount                 int    `json:"file_count,omitempty"`
	TotalBytes                uint64 `json:"total_bytes,omitempty"`
	Recovered                 bool   `json:"recovered,omitempty"`
	TileActivationNotSwitched bool   `json:"tile_activation_not_switched,omitempty"`
	TileActivationNotObserved bool   `json:"tile_activation_not_observed,omitempty"`
}

type legacyMigrationPlan struct {
	ContainerID  string
	PreviousID   string
	Target       string
	TreeSHA256   string
	ManifestData []byte
}

func legacyContainerID(info fs.FileInfo, targetTileSetID, treeSHA256 string) string {
	seed := fmt.Sprintf("%d:%d:%s:%s", info.ModTime().UnixNano(), info.Size(), targetTileSetID, treeSHA256)
	digest := sha256.Sum256([]byte(seed))
	return "legacy-" + hex.EncodeToString(digest[:])[:16]
}

const legacyMigrationProvenance = "legacy-migration-v1"

type exchangeFunction func(string, string) error
type directorySyncFunction func(string) error
type gitIdentityFunction func(string) (string, string, bool, error)

type transactionEngine struct {
	root        string
	paramsDir   string
	exchange    exchangeFunction
	syncDir     directorySyncFunction
	now         func() time.Time
	gitIdentity gitIdentityFunction
}

func newTransactionEngine(root, paramsDir string) (*transactionEngine, error) {
	cleanRoot, err := validateRoot(root)
	if err != nil {
		return nil, err
	}
	cleanParams, err := validateRoot(paramsDir)
	if err != nil {
		return nil, fmt.Errorf("invalid Params directory: %w", err)
	}
	return &transactionEngine{
		root:        cleanRoot,
		paramsDir:   cleanParams,
		exchange:    renameExchange,
		syncDir:     fsyncDirectory,
		now:         time.Now,
		gitIdentity: currentGitIdentity,
	}, nil
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "vtsc-tile-transaction: %v\n", err)
		os.Exit(1)
	}
}

func run(arguments []string, output io.Writer) error {
	if len(arguments) == 0 {
		return usageError()
	}
	command := arguments[0]
	arguments = arguments[1:]

	switch command {
	case "verify":
		flags := flag.NewFlagSet("verify", flag.ContinueOnError)
		flags.SetOutput(io.Discard)
		root := flags.String("root", "", "artifact root containing manifest.json and offline/")
		tileSetID := flags.String("tile-set-id", "", "expected immutable tile-set identifier")
		if err := flags.Parse(arguments); err != nil || flags.NArg() != 0 {
			return usageError()
		}
		if !isSafeID(*tileSetID) {
			return fmt.Errorf("invalid --tile-set-id")
		}
		manifest, err := verifyArtifactRoot(*root, *tileSetID, false)
		if err != nil {
			return err
		}
		return encodeResult(output, transactionResult{
			Operation: "verify", TileSetID: manifest.TileSetID,
			FileCount: manifest.FileCount, TotalBytes: manifest.TotalBytes,
		})

	case "activate":
		flags := flag.NewFlagSet("activate", flag.ContinueOnError)
		flags.SetOutput(io.Discard)
		root := flags.String("root", "", "OSM root")
		paramsDir := flags.String("params-dir", "", "Params data directory")
		stage := flags.String("stage", "", "verified remote staging root")
		tileSetID := flags.String("tile-set-id", "", "expected immutable tile-set identifier")
		expectedCurrentID := flags.String("expected-current-tile-set-id", "", "optional exact active identity captured by the host")
		injectedFailure := flags.String("inject-failure", "", "test-only failure point")
		if err := flags.Parse(arguments); err != nil || flags.NArg() != 0 {
			return usageError()
		}
		engine, err := newTransactionEngine(*root, *paramsDir)
		if err != nil {
			return err
		}
		if *expectedCurrentID != "" && !isSafeID(*expectedCurrentID) {
			return fmt.Errorf("invalid --expected-current-tile-set-id")
		}
		result, err := engine.activate(*stage, *tileSetID, *expectedCurrentID, *injectedFailure)
		if err != nil {
			return err
		}
		return encodeResult(output, result)

	case "rollback":
		flags := flag.NewFlagSet("rollback", flag.ContinueOnError)
		flags.SetOutput(io.Discard)
		root := flags.String("root", "", "OSM root")
		paramsDir := flags.String("params-dir", "", "Params data directory")
		expectedID := flags.String("expected-tile-set-id", "", "optional tile-set ID expected to be active")
		expectedPreviousID := flags.String("expected-previous-tile-set-id", "", "optional tile-set ID required in offline.previous")
		repoRoot := flags.String("repo-root", "", "openpilot Git checkout root")
		expectedGitBranch := flags.String("expected-git-branch", "", "exact branch permitted at rollback")
		expectedGitHead := flags.String("expected-git-head", "", "exact Git HEAD permitted at rollback")
		injectedFailure := flags.String("inject-failure", "", "test-only failure point")
		if err := flags.Parse(arguments); err != nil || flags.NArg() != 0 {
			return usageError()
		}
		if *expectedID != "" && !isSafeID(*expectedID) {
			return fmt.Errorf("invalid --expected-tile-set-id")
		}
		if *expectedPreviousID != "" && !isSafeID(*expectedPreviousID) {
			return fmt.Errorf("invalid --expected-previous-tile-set-id")
		}
		cleanRepo, err := validateRoot(*repoRoot)
		if err != nil {
			return fmt.Errorf("invalid --repo-root: %w", err)
		}
		if !isSafeGitBranch(*expectedGitBranch) {
			return fmt.Errorf("invalid --expected-git-branch")
		}
		if !isSafeGitHead(*expectedGitHead) {
			return fmt.Errorf("invalid --expected-git-head")
		}
		engine, err := newTransactionEngine(*root, *paramsDir)
		if err != nil {
			return err
		}
		result, err := engine.rollbackBound(
			*expectedID,
			*expectedPreviousID,
			cleanRepo,
			*expectedGitBranch,
			*expectedGitHead,
			*injectedFailure,
		)
		if err != nil {
			return err
		}
		return encodeResult(output, result)
	default:
		return usageError()
	}
}

func usageError() error {
	return errors.New("usage: vtsc-tile-transaction verify --root <artifact-root> --tile-set-id <id> | activate --root <osm-root> --params-dir <params-data-dir> --stage <artifact-root> --tile-set-id <id> [--expected-current-tile-set-id <id>] [--inject-failure <point>] | rollback --root <osm-root> --params-dir <params-data-dir> --repo-root <checkout> --expected-git-branch <branch> --expected-git-head <head> [--expected-tile-set-id <id>] [--expected-previous-tile-set-id <id>] [--inject-failure <point>]")
}

func currentGitIdentity(repoRoot string) (string, string, bool, error) {
	branchOutput, err := exec.Command("git", "-C", repoRoot, "branch", "--show-current").Output()
	if err != nil {
		return "", "", false, fmt.Errorf("read rollback Git branch: %w", err)
	}
	headOutput, err := exec.Command("git", "-C", repoRoot, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", "", false, fmt.Errorf("read rollback Git HEAD: %w", err)
	}
	statusOutput, err := exec.Command("git", "-C", repoRoot, "status", "--porcelain").Output()
	if err != nil {
		return "", "", false, fmt.Errorf("read rollback Git status: %w", err)
	}
	return strings.TrimSpace(string(branchOutput)), strings.TrimSpace(string(headOutput)), len(statusOutput) != 0, nil
}

func isSafeGitHead(value string) bool {
	if len(value) != 40 {
		return false
	}
	for _, character := range value {
		if !((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')) {
			return false
		}
	}
	return true
}

func isSafeGitBranch(value string) bool {
	if value == "" || strings.HasPrefix(value, "/") || strings.Contains(value, "..") {
		return false
	}
	for _, character := range value {
		if !((character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			strings.ContainsRune("._/-", character)) {
			return false
		}
	}
	return true
}

func encodeResult(output io.Writer, result transactionResult) error {
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(result)
}

func validateRoot(root string) (string, error) {
	if root == "" || !filepath.IsAbs(root) {
		return "", fmt.Errorf("root must be an absolute path")
	}
	clean := filepath.Clean(root)
	if clean == string(filepath.Separator) {
		return "", fmt.Errorf("root cannot be the filesystem root")
	}
	return clean, nil
}

func (e *transactionEngine) activePath() string {
	return filepath.Join(e.root, activeOfflineName)
}

func (e *transactionEngine) previousPath() string {
	return filepath.Join(e.root, previousOfflineName)
}

func (e *transactionEngine) generationsPath() string {
	return filepath.Join(e.root, generationDirectory)
}

func (e *transactionEngine) transactionPath() string {
	return filepath.Join(e.root, transactionFileName)
}

// withExclusiveTransactionLock serializes every pointer/journal mutation
// across independently launched helper processes. renameat2 makes each name
// exchange atomic, but it cannot make two whole multi-step transactions
// atomic with respect to one another.
func (e *transactionEngine) withExclusiveTransactionLock(
	operation func() (transactionResult, error),
) (transactionResult, error) {
	lock, err := os.OpenFile(filepath.Join(e.root, transactionLockName), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return transactionResult{}, fmt.Errorf("open tile transaction lock: %w", err)
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return transactionResult{}, fmt.Errorf("lock tile transaction: %w", err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
	if err := e.requireExactParkedState(); err != nil {
		return transactionResult{}, err
	}
	return operation()
}

func (e *transactionEngine) requireExactParkedState() error {
	required := map[string]string{
		"IsOffroad":            "1",
		"IsOnroad":             "0",
		"MTSCLookaheadEnabled": "0",
	}
	for key, expected := range required {
		value, err := os.ReadFile(filepath.Join(e.paramsDir, key))
		if err != nil {
			return fmt.Errorf("read parked-state Param %s: %w", key, err)
		}
		if string(value) != expected {
			return fmt.Errorf("refusing tile mutation: %s must equal %q exactly", key, expected)
		}
	}
	return nil
}

func (e *transactionEngine) generationPath(tileSetID string) string {
	return filepath.Join(e.generationsPath(), tileSetID)
}

func (e *transactionEngine) generationTarget(tileSetID string) (string, error) {
	target, err := filepath.Rel(e.root, filepath.Join(e.generationPath(tileSetID), activeOfflineName))
	if err != nil {
		return "", err
	}
	target = filepath.ToSlash(target)
	if !isGenerationTarget(target) {
		return "", fmt.Errorf("unsafe generated target %q", target)
	}
	return target, nil
}

func (e *transactionEngine) stagePath(tileSetID string) string {
	return filepath.Join(e.root, ".tileset-"+tileSetID+".partial")
}

func (e *transactionEngine) switchPath(tileSetID string) string {
	return filepath.Join(e.root, ".offline-switch-"+tileSetID)
}

func verifyArtifactRoot(root, expectedTileSetID string, allowEmbeddedManifest bool) (tileManifest, error) {
	if !isSafeID(expectedTileSetID) {
		return tileManifest{}, fmt.Errorf("invalid tile-set identifier")
	}
	cleanRoot, err := validateRoot(root)
	if err != nil {
		return tileManifest{}, err
	}
	rootInfo, err := os.Lstat(cleanRoot)
	if err != nil {
		return tileManifest{}, fmt.Errorf("stat artifact root: %w", err)
	}
	if rootInfo.Mode()&os.ModeSymlink != 0 || !rootInfo.IsDir() {
		return tileManifest{}, fmt.Errorf("artifact root is not a real directory")
	}
	manifestPath := filepath.Join(cleanRoot, "manifest.json")
	manifestInfo, err := os.Lstat(manifestPath)
	if err != nil {
		return tileManifest{}, fmt.Errorf("stat manifest: %w", err)
	}
	if manifestInfo.Mode()&os.ModeSymlink != 0 || !manifestInfo.Mode().IsRegular() {
		return tileManifest{}, fmt.Errorf("manifest is not a regular file")
	}
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return tileManifest{}, fmt.Errorf("read manifest: %w", err)
	}
	var manifest tileManifest
	if err := decodeSingleJSON(manifestData, &manifest); err != nil {
		return tileManifest{}, fmt.Errorf("parse manifest: %w", err)
	}
	if err := validateManifest(manifest, expectedTileSetID); err != nil {
		return tileManifest{}, err
	}

	offlineRoot := filepath.Join(cleanRoot, activeOfflineName)
	info, err := os.Lstat(offlineRoot)
	if err != nil {
		return tileManifest{}, fmt.Errorf("stat offline tree: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return tileManifest{}, fmt.Errorf("offline tree is not a real directory")
	}

	expectedFiles := make(map[string]manifestFile, len(manifest.Files))
	expectedDirectories := map[string]struct{}{activeOfflineName: {}}
	partialDirectory := path.Join(activeOfflineName, ".rsync-partial")
	for _, entry := range manifest.Files {
		expectedFiles[entry.Path] = entry
		components := strings.Split(entry.Path, "/")
		for index := 1; index < len(components); index++ {
			expectedDirectories[strings.Join(components[:index], "/")] = struct{}{}
		}
	}
	if allowEmbeddedManifest {
		expectedFiles[path.Join(activeOfflineName, embeddedManifestName)] = manifestFile{Path: path.Join(activeOfflineName, embeddedManifestName)}
	}

	seenFiles := make(map[string]struct{}, len(expectedFiles))
	err = filepath.WalkDir(offlineRoot, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(cleanRoot, current)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink is forbidden in tile tree: %s", relative)
		}
		if info.IsDir() {
			// rsync uses this directory while staging. A successful transfer can
			// leave the empty directory behind; it is removed before promotion.
			if !allowEmbeddedManifest && relative == partialDirectory {
				return nil
			}
			if _, ok := expectedDirectories[relative]; !ok {
				return fmt.Errorf("unexpected tile directory: %s", relative)
			}
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("non-regular tile entry: %s", relative)
		}
		expected, ok := expectedFiles[relative]
		if !ok {
			return fmt.Errorf("unexpected tile file: %s", relative)
		}
		if relative == path.Join(activeOfflineName, embeddedManifestName) && allowEmbeddedManifest {
			if err := verifyEmbeddedManifest(current, expectedTileSetID, manifestData); err != nil {
				return err
			}
			seenFiles[relative] = struct{}{}
			return nil
		}
		if uint64(info.Size()) != expected.ByteCount {
			return fmt.Errorf("tile size mismatch: %s", relative)
		}
		digest, err := sha256File(current)
		if err != nil {
			return fmt.Errorf("digest %s: %w", relative, err)
		}
		if digest != expected.SHA256 {
			return fmt.Errorf("tile digest mismatch: %s", relative)
		}
		seenFiles[relative] = struct{}{}
		return nil
	})
	if err != nil {
		return tileManifest{}, err
	}
	if len(seenFiles) != len(expectedFiles) {
		return tileManifest{}, fmt.Errorf("remote tile file set differs from manifest")
	}
	return manifest, nil
}

func verifyEmbeddedManifest(filePath, expectedID string, expectedBytes []byte) error {
	actual, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("read embedded manifest: %w", err)
	}
	var embedded manifestIdentity
	if err := decodeSingleJSON(actual, &embedded); err != nil {
		return fmt.Errorf("parse embedded manifest: %w", err)
	}
	if embedded.TileSetID != expectedID {
		return fmt.Errorf("embedded manifest tile-set identity mismatch")
	}
	if !bytes.Equal(actual, expectedBytes) {
		return fmt.Errorf("embedded manifest differs from generation manifest")
	}
	return nil
}

func validateManifest(manifest tileManifest, expectedTileSetID string) error {
	if manifest.TileSetID != expectedTileSetID || !isSafeID(manifest.TileSetID) {
		return fmt.Errorf("tile-set identity mismatch")
	}
	if len(manifest.Files) == 0 || manifest.FileCount != len(manifest.Files) {
		return fmt.Errorf("manifest file-count mismatch")
	}
	var total uint64
	previousPath := ""
	for _, entry := range manifest.Files {
		if !isSafeTilePath(entry.Path) {
			return fmt.Errorf("unsafe manifest path: %q", entry.Path)
		}
		if !isLowerSHA256(entry.SHA256) {
			return fmt.Errorf("invalid tile digest: %s", entry.Path)
		}
		if entry.ByteCount <= 64 {
			return fmt.Errorf("placeholder tile: %s", entry.Path)
		}
		if previousPath != "" && entry.Path <= previousPath {
			return fmt.Errorf("manifest paths are not canonical")
		}
		if ^uint64(0)-total < entry.ByteCount {
			return fmt.Errorf("manifest byte-count overflow")
		}
		total += entry.ByteCount
		previousPath = entry.Path
	}
	if total != manifest.TotalBytes {
		return fmt.Errorf("manifest total-bytes mismatch")
	}
	return nil
}

func manifestsEqual(first, second tileManifest) bool {
	if first.TileSetID != second.TileSetID || first.FileCount != second.FileCount ||
		first.TotalBytes != second.TotalBytes || len(first.Files) != len(second.Files) {
		return false
	}
	for index := range first.Files {
		if first.Files[index] != second.Files[index] {
			return false
		}
	}
	return true
}

func isSafeID(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') || character == '.' || character == '_' || character == '-' {
			if index == 0 && (character == '.' || character == '_' || character == '-') {
				return false
			}
			continue
		}
		return false
	}
	return true
}

func isSafeTilePath(value string) bool {
	if value == "" || strings.HasPrefix(value, "/") || strings.Contains(value, "\\") || strings.ContainsRune(value, '\x00') || path.Clean(value) != value {
		return false
	}
	components := strings.Split(value, "/")
	if len(components) < 4 || components[0] != activeOfflineName {
		return false
	}
	for _, component := range components {
		if component == "" || component == "." || component == ".." {
			return false
		}
	}
	if _, err := strconv.Atoi(components[1]); err != nil {
		return false
	}
	if _, err := strconv.Atoi(components[2]); err != nil {
		return false
	}
	return true
}

func isGenerationTarget(value string) bool {
	if strings.HasPrefix(value, "/") || path.Clean(value) != value {
		return false
	}
	components := strings.Split(value, "/")
	return len(components) == 3 && components[0] == generationDirectory && isSafeID(components[1]) && components[2] == activeOfflineName
}

func isLowerSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if !((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')) {
			return false
		}
	}
	return true
}

func sha256File(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func treeSHA256(root, excludedRelativePath string) (string, error) {
	info, err := os.Lstat(root)
	if err != nil {
		return "", err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("tree root is not a real directory")
	}
	hash := sha256.New()
	err = filepath.WalkDir(root, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(root, current)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if relative == "." || relative == excludedRelativePath {
			return nil
		}
		entryInfo, err := entry.Info()
		if err != nil {
			return err
		}
		if entryInfo.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("tree contains a symlink: %s", relative)
		}
		if entryInfo.IsDir() {
			_, _ = io.WriteString(hash, "D\x00"+relative+"\x00")
			return nil
		}
		if !entryInfo.Mode().IsRegular() {
			return fmt.Errorf("tree contains a non-regular entry: %s", relative)
		}
		digest, err := sha256File(current)
		if err != nil {
			return err
		}
		_, _ = io.WriteString(hash, "F\x00"+relative+"\x00"+strconv.FormatInt(entryInfo.Size(), 10)+"\x00"+digest+"\x00")
		return nil
	})
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func verifyDirectOfflineTree(root string, manifest tileManifest) error {
	offlineRoot := filepath.Join(root, activeOfflineName)
	info, err := os.Lstat(offlineRoot)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("active offline tree is not a direct directory")
	}
	expectedFiles := make(map[string]manifestFile, len(manifest.Files))
	expectedDirectories := map[string]struct{}{activeOfflineName: {}}
	for _, entry := range manifest.Files {
		expectedFiles[entry.Path] = entry
		components := strings.Split(entry.Path, "/")
		for index := 1; index < len(components); index++ {
			expectedDirectories[strings.Join(components[:index], "/")] = struct{}{}
		}
	}
	seenFiles := make(map[string]struct{}, len(expectedFiles))
	err = filepath.WalkDir(offlineRoot, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(root, current)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		entryInfo, err := entry.Info()
		if err != nil {
			return err
		}
		if entryInfo.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("direct active tree contains a symlink: %s", relative)
		}
		if entryInfo.IsDir() {
			if _, ok := expectedDirectories[relative]; !ok {
				return fmt.Errorf("direct active tree has an unexpected directory: %s", relative)
			}
			return nil
		}
		if !entryInfo.Mode().IsRegular() {
			return fmt.Errorf("direct active tree has a non-regular entry: %s", relative)
		}
		expected, ok := expectedFiles[relative]
		if !ok {
			return fmt.Errorf("direct active tree has an unexpected file: %s", relative)
		}
		if uint64(entryInfo.Size()) != expected.ByteCount {
			return fmt.Errorf("direct active tile size mismatch: %s", relative)
		}
		digest, err := sha256File(current)
		if err != nil {
			return err
		}
		if digest != expected.SHA256 {
			return fmt.Errorf("direct active tile digest mismatch: %s", relative)
		}
		seenFiles[relative] = struct{}{}
		return nil
	})
	if err != nil {
		return err
	}
	if len(seenFiles) != len(expectedFiles) {
		return fmt.Errorf("direct active tile file set differs from requested target")
	}
	return nil
}

func decodeSingleJSON(data []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

func fsyncFile(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()
	return file.Sync()
}

func fsyncDirectory(directory string) error {
	file, err := os.Open(directory)
	if err != nil {
		return err
	}
	defer file.Close()
	return file.Sync()
}

func durableWriteFile(destination string, contents []byte, mode fs.FileMode, syncDirectory directorySyncFunction) error {
	directory := filepath.Dir(destination)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(destination)+".tmp-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(mode); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(contents); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, destination); err != nil {
		return err
	}
	return syncDirectory(directory)
}

func (e *transactionEngine) writeTransaction(transaction tileTransaction) error {
	if err := validateTransaction(e.root, transaction); err != nil {
		return err
	}
	contents, err := json.Marshal(transaction)
	if err != nil {
		return err
	}
	contents = append(contents, '\n')
	return durableWriteFile(e.transactionPath(), contents, 0o600, e.syncDir)
}

func (e *transactionEngine) readTransaction() (tileTransaction, bool, error) {
	contents, err := os.ReadFile(e.transactionPath())
	if errors.Is(err, os.ErrNotExist) {
		return tileTransaction{}, false, nil
	}
	if err != nil {
		return tileTransaction{}, false, err
	}
	var transaction tileTransaction
	if err := decodeSingleJSON(contents, &transaction); err != nil {
		return tileTransaction{}, false, fmt.Errorf("parse tile transaction: %w", err)
	}
	if err := validateTransaction(e.root, transaction); err != nil {
		return tileTransaction{}, false, err
	}
	return transaction, true, nil
}

func validateTransaction(root string, transaction tileTransaction) error {
	switch transaction.Kind {
	case "activation":
		if !isSafeID(transaction.NewID) || !isGenerationTarget(transaction.NewTarget) || !isGenerationTarget(transaction.PreviousTarget) {
			return fmt.Errorf("unsafe activation transaction")
		}
		if transaction.PreviousID != "" && !isSafeID(transaction.PreviousID) {
			return fmt.Errorf("unsafe activation previous identity")
		}
		if filepath.Clean(transaction.SwitchPath) != filepath.Join(root, ".offline-switch-"+transaction.NewID) {
			return fmt.Errorf("unsafe activation switch path")
		}
		if err := validateActivationArtifactAuthority(root, transaction); err != nil {
			return err
		}
	case "rollback":
		if !isGenerationTarget(transaction.ActiveTarget) || !isGenerationTarget(transaction.PreviousTarget) {
			return fmt.Errorf("unsafe rollback transaction")
		}
		if transaction.ActiveID != "" && !isSafeID(transaction.ActiveID) {
			return fmt.Errorf("unsafe rollback active identity")
		}
		if transaction.PreviousID != "" && !isSafeID(transaction.PreviousID) {
			return fmt.Errorf("unsafe rollback previous identity")
		}
	default:
		return fmt.Errorf("unknown tile transaction kind")
	}
	return nil
}

func generationContainerPath(root, target string) (string, error) {
	if !isGenerationTarget(target) {
		return "", fmt.Errorf("unsafe generation target")
	}
	components := strings.Split(target, "/")
	return filepath.Join(root, generationDirectory, components[1]), nil
}

func cleanupTombstone(parent, tileSetID, label string) string {
	return filepath.Join(parent, ".vtsc-cleanup-"+tileSetID+"-"+label)
}

func validActivationArtifact(
	artifact activationArtifact,
	expectedPath, expectedTombstone string,
	allowPreexisting bool,
) bool {
	if filepath.Clean(artifact.Path) != expectedPath || artifact.Tombstone != expectedTombstone {
		return false
	}
	if artifact.Preexisting {
		return allowPreexisting && isLowerSHA256(artifact.TreeSHA256)
	}
	return artifact.TreeSHA256 == ""
}

func validateActivationArtifactAuthority(root string, transaction tileTransaction) error {
	authority := transaction.Artifacts
	if authority == nil {
		return fmt.Errorf("activation transaction lacks durable artifact ownership authority")
	}
	if authority.Schema != 1 {
		return fmt.Errorf("unsupported activation artifact authority schema")
	}
	targetGeneration := filepath.Join(root, generationDirectory, transaction.NewID)
	targetBuilding := filepath.Join(root, generationDirectory, "."+transaction.NewID+".building")
	if !validActivationArtifact(
		authority.TargetGeneration,
		targetGeneration,
		cleanupTombstone(filepath.Dir(targetGeneration), transaction.NewID, "target-generation"),
		true,
	) || !validActivationArtifact(
		authority.TargetBuilding,
		targetBuilding,
		cleanupTombstone(filepath.Dir(targetBuilding), transaction.NewID, "target-building"),
		false,
	) {
		return fmt.Errorf("activation target artifact ownership/path mismatch")
	}
	if authority.TargetBuilding.Preexisting {
		return fmt.Errorf("activation target build artifact cannot preexist its transaction")
	}
	if !validActivationArtifact(authority.Switch, transaction.SwitchPath, "", false) || authority.Switch.Preexisting {
		return fmt.Errorf("activation switch ownership/path mismatch")
	}
	previousTemporary := filepath.Join(root, ".offline-previous-link-"+transaction.NewID)
	if !validActivationArtifact(authority.PreviousTemporary, previousTemporary, "", false) || authority.PreviousTemporary.Preexisting {
		return fmt.Errorf("activation previous-link ownership/path mismatch")
	}
	retainedPrefix := filepath.Join(root, ".retained-pre-generation-"+transaction.NewID+"-")
	retainedTombstone := cleanupTombstone(root, transaction.NewID, "retained")
	if authority.RetainedPrefix != retainedPrefix || authority.RetainedTombstone != retainedTombstone {
		return fmt.Errorf("activation retained artifact ownership/path mismatch")
	}
	if authority.LegacyMigrationPlanned {
		if authority.LegacyGeneration == nil || authority.LegacyBuilding == nil {
			return fmt.Errorf("activation legacy artifact ownership is incomplete")
		}
		legacyGeneration, err := generationContainerPath(root, transaction.PreviousTarget)
		if err != nil {
			return err
		}
		legacyID := filepath.Base(legacyGeneration)
		legacyBuilding := filepath.Join(filepath.Dir(legacyGeneration), "."+legacyID+".building")
		if !validActivationArtifact(
			*authority.LegacyGeneration,
			legacyGeneration,
			cleanupTombstone(filepath.Dir(legacyGeneration), transaction.NewID, "legacy-generation"),
			true,
		) || !validActivationArtifact(
			*authority.LegacyBuilding,
			legacyBuilding,
			cleanupTombstone(filepath.Dir(legacyBuilding), transaction.NewID, "legacy-building"),
			false,
		) || authority.LegacyBuilding.Preexisting {
			return fmt.Errorf("activation legacy artifact ownership/path mismatch")
		}
	} else if authority.LegacyGeneration != nil || authority.LegacyBuilding != nil {
		return fmt.Errorf("activation has legacy artifact ownership without a migration")
	}
	if authority.PreviousPointerTarget == "" || authority.PreviousPointerID == "" || authority.PreviousPointerTreeSHA256 == "" {
		if authority.PreviousPointerTarget != "" || authority.PreviousPointerID != "" || authority.PreviousPointerTreeSHA256 != "" {
			return fmt.Errorf("activation previous pointer authority is incomplete")
		}
	} else if !isGenerationTarget(authority.PreviousPointerTarget) ||
		!isSafeID(authority.PreviousPointerID) ||
		!isLowerSHA256(authority.PreviousPointerTreeSHA256) {
		return fmt.Errorf("activation previous pointer authority is unsafe")
	}
	if authority.PreviousPointerTarget == transaction.NewTarget && !authority.TargetGeneration.Preexisting {
		return fmt.Errorf("activation ownership contradicts its preexisting previous pointer")
	}
	return nil
}

func (e *transactionEngine) unchangedDirectLegacyTree() (bool, error) {
	activeInfo, err := os.Lstat(e.activePath())
	if err != nil {
		return false, err
	}
	if !activeInfo.IsDir() || activeInfo.Mode()&os.ModeSymlink != 0 {
		return false, nil
	}
	if _, present, err := readManifestIdentity(e.activePath()); err != nil {
		return false, err
	} else if present {
		return false, nil
	}
	if _, err := os.Lstat(e.previousPath()); err == nil {
		return false, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}
	if entries, err := os.ReadDir(e.generationsPath()); err == nil {
		if len(entries) != 0 {
			return false, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}
	for _, pattern := range []string{
		".offline-*",
		".retained-*",
		filepath.Join(generationDirectory, ".*.building"),
	} {
		matches, err := filepath.Glob(filepath.Join(e.root, pattern))
		if err != nil {
			return false, err
		}
		if len(matches) != 0 {
			return false, nil
		}
	}
	return true, nil
}

func (e *transactionEngine) adjacentManifestIdentity() (manifestIdentity, bool, error) {
	contents, err := os.ReadFile(filepath.Join(e.root, "offline.manifest.json"))
	if errors.Is(err, os.ErrNotExist) {
		return manifestIdentity{}, false, nil
	}
	if err != nil {
		return manifestIdentity{}, false, err
	}
	var manifest manifestIdentity
	if err := decodeSingleJSON(contents, &manifest); err != nil {
		return manifestIdentity{}, false, fmt.Errorf("parse adjacent offline manifest: %w", err)
	}
	if !isSafeID(manifest.TileSetID) {
		return manifestIdentity{}, false, fmt.Errorf("adjacent offline manifest has an invalid tile identity")
	}
	return manifest, true, nil
}

func (e *transactionEngine) verifyExactSameTargetDirectTree(expectedTileSetID string) error {
	unchanged, err := e.unchangedDirectLegacyTree()
	if err != nil {
		return err
	}
	if !unchanged {
		return fmt.Errorf("same-ID direct tree has ambiguous transaction or generation evidence")
	}
	contents, err := os.ReadFile(filepath.Join(e.root, "offline.manifest.json"))
	if err != nil {
		return fmt.Errorf("read adjacent same-ID manifest: %w", err)
	}
	var manifest tileManifest
	if err := decodeSingleJSON(contents, &manifest); err != nil {
		return fmt.Errorf("parse adjacent same-ID manifest: %w", err)
	}
	if err := validateManifest(manifest, expectedTileSetID); err != nil {
		return fmt.Errorf("validate adjacent same-ID manifest: %w", err)
	}
	if err := verifyDirectOfflineTree(e.root, manifest); err != nil {
		return fmt.Errorf("same-ID direct tree differs from requested target: %w", err)
	}
	return nil
}

func (e *transactionEngine) verifyUnchangedDirectTreeIdentity(expectedPreviousTileSetID string) error {
	if !isSafeID(expectedPreviousTileSetID) {
		return fmt.Errorf("recorded direct-tree identity is invalid")
	}
	unchanged, err := e.unchangedDirectLegacyTree()
	if err != nil {
		return err
	}
	if !unchanged {
		return fmt.Errorf("direct tile tree has ambiguous transaction or pointer evidence")
	}
	adjacent, present, err := e.adjacentManifestIdentity()
	if err != nil {
		return err
	}
	if !present || adjacent.TileSetID != expectedPreviousTileSetID {
		return fmt.Errorf("direct tile tree differs from the recorded adjacent previous identity")
	}
	return nil
}

func requirePathAbsent(path, detail string) error {
	if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	return fmt.Errorf("%s already exists before transaction authority", detail)
}

func (e *transactionEngine) verifyGenerationTargetIdentity(target, expectedID string) (string, error) {
	container, err := generationContainerPath(e.root, target)
	if err != nil {
		return "", err
	}
	manifest, present, err := readManifestIdentity(filepath.Join(e.root, target))
	if err != nil || !present || manifest.TileSetID != expectedID {
		if err != nil {
			return "", err
		}
		return "", fmt.Errorf("generation target differs from its recorded identity")
	}
	if manifest.Legacy {
		if !isSafeID(manifest.LegacyMigrationTargetTileSetID) {
			return "", fmt.Errorf("legacy generation target binding is invalid")
		}
		if err := verifyLegacyMigrationTree(
			filepath.Join(e.root, target), manifest, manifest.LegacyMigrationTargetTileSetID,
		); err != nil {
			return "", err
		}
	} else {
		if filepath.Base(container) != expectedID {
			return "", fmt.Errorf("canonical generation container differs from its manifest identity")
		}
		if _, err := verifyArtifactRoot(container, expectedID, true); err != nil {
			return "", err
		}
	}
	return treeSHA256(filepath.Join(e.root, target), "")
}

func (e *transactionEngine) activationArtifactAuthority(
	transaction tileTransaction,
	legacyPlan *legacyMigrationPlan,
) (*activationArtifactAuthority, error) {
	targetGeneration := e.generationPath(transaction.NewID)
	targetBuilding := filepath.Join(e.generationsPath(), "."+transaction.NewID+".building")
	target := activationArtifact{
		Path:      targetGeneration,
		Tombstone: cleanupTombstone(e.generationsPath(), transaction.NewID, "target-generation"),
	}
	if _, err := os.Lstat(targetGeneration); err == nil {
		if err := e.verifyGeneration(transaction.NewID); err != nil {
			return nil, fmt.Errorf("verify preexisting target generation: %w", err)
		}
		digest, err := treeSHA256(targetGeneration, "")
		if err != nil {
			return nil, err
		}
		target.Preexisting = true
		target.TreeSHA256 = digest
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	targetBuild := activationArtifact{
		Path:      targetBuilding,
		Tombstone: cleanupTombstone(e.generationsPath(), transaction.NewID, "target-building"),
	}
	if err := requirePathAbsent(targetBuilding, "target generation build directory"); err != nil {
		return nil, err
	}
	for path, detail := range map[string]string{
		target.Tombstone:       "target generation cleanup tombstone",
		targetBuild.Tombstone:  "target build cleanup tombstone",
		transaction.SwitchPath: "activation switch",
		filepath.Join(e.root, ".offline-previous-link-"+transaction.NewID): "activation previous-link temporary",
	} {
		if err := requirePathAbsent(path, detail); err != nil {
			return nil, err
		}
	}
	authority := &activationArtifactAuthority{
		Schema:           1,
		TargetGeneration: target,
		TargetBuilding:   targetBuild,
		Switch: activationArtifact{
			Path: transaction.SwitchPath,
		},
		PreviousTemporary: activationArtifact{
			Path: filepath.Join(e.root, ".offline-previous-link-"+transaction.NewID),
		},
		RetainedPrefix:    filepath.Join(e.root, ".retained-pre-generation-"+transaction.NewID+"-"),
		RetainedTombstone: cleanupTombstone(e.root, transaction.NewID, "retained"),
	}
	retained, err := filepath.Glob(authority.RetainedPrefix + "*")
	if err != nil {
		return nil, err
	}
	if len(retained) != 0 {
		return nil, fmt.Errorf("retained activation artifact predates transaction authority")
	}
	if err := requirePathAbsent(authority.RetainedTombstone, "retained cleanup tombstone"); err != nil {
		return nil, err
	}

	if legacyPlan != nil {
		authority.LegacyMigrationPlanned = true
		legacyGeneration := e.generationPath(legacyPlan.ContainerID)
		legacyBuilding := filepath.Join(e.generationsPath(), "."+legacyPlan.ContainerID+".building")
		legacy := activationArtifact{
			Path:      legacyGeneration,
			Tombstone: cleanupTombstone(e.generationsPath(), transaction.NewID, "legacy-generation"),
		}
		if _, err := os.Lstat(legacyGeneration); err == nil {
			manifest, present, err := readManifestIdentity(filepath.Join(legacyGeneration, activeOfflineName))
			if err != nil || !present || manifest.TileSetID != legacyPlan.PreviousID {
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("preexisting legacy generation differs from its migration plan")
			}
			if err := verifyLegacyMigrationTree(
				filepath.Join(legacyGeneration, activeOfflineName), manifest, transaction.NewID,
			); err != nil {
				return nil, err
			}
			digest, err := treeSHA256(legacyGeneration, "")
			if err != nil {
				return nil, err
			}
			legacy.Preexisting = true
			legacy.TreeSHA256 = digest
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		legacyBuild := activationArtifact{
			Path:      legacyBuilding,
			Tombstone: cleanupTombstone(e.generationsPath(), transaction.NewID, "legacy-building"),
		}
		for path, detail := range map[string]string{
			legacyBuilding:        "legacy generation build directory",
			legacy.Tombstone:      "legacy generation cleanup tombstone",
			legacyBuild.Tombstone: "legacy build cleanup tombstone",
		} {
			if err := requirePathAbsent(path, detail); err != nil {
				return nil, err
			}
		}
		authority.LegacyGeneration = &legacy
		authority.LegacyBuilding = &legacyBuild
	}

	previousTarget, previousIsLink, err := symlinkTarget(e.previousPath())
	if err != nil {
		return nil, err
	}
	if previousIsLink {
		if !isGenerationTarget(previousTarget) {
			return nil, fmt.Errorf("existing previous tile pointer is unsafe")
		}
		manifest, present, err := readManifestIdentity(e.previousPath())
		if err != nil || !present || !isSafeID(manifest.TileSetID) {
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("existing previous tile pointer has no exact identity")
		}
		digest, err := e.verifyGenerationTargetIdentity(previousTarget, manifest.TileSetID)
		if err != nil {
			return nil, fmt.Errorf("verify existing previous tile pointer: %w", err)
		}
		authority.PreviousPointerTarget = previousTarget
		authority.PreviousPointerID = manifest.TileSetID
		authority.PreviousPointerTreeSHA256 = digest
	} else if _, err := os.Lstat(e.previousPath()); err == nil {
		return nil, fmt.Errorf("existing previous tile path is not a canonical generation pointer")
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if authority.PreviousPointerTarget == transaction.NewTarget && !authority.TargetGeneration.Preexisting {
		return nil, fmt.Errorf("previous pointer references a target generation not recorded as preexisting")
	}
	return authority, nil
}

func (e *transactionEngine) clearTransaction() error {
	err := os.Remove(e.transactionPath())
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return e.syncDir(e.root)
}

func validateOwnedRemovalTree(tree string) error {
	info, err := os.Lstat(tree)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("transaction-owned cleanup path is not a real directory: %s", tree)
	}
	return filepath.WalkDir(tree, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		entryInfo, err := entry.Info()
		if err != nil {
			return err
		}
		if entryInfo.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("transaction-owned cleanup tree contains a symlink: %s", current)
		}
		if !entryInfo.IsDir() && !entryInfo.Mode().IsRegular() {
			return fmt.Errorf("transaction-owned cleanup tree contains a non-regular entry: %s", current)
		}
		return nil
	})
}

func pathExists(path string) (bool, error) {
	if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}

func validateOwnedTreeArtifactState(
	artifact activationArtifact,
	validateSource func(string) error,
) error {
	sourceExists, err := pathExists(artifact.Path)
	if err != nil {
		return err
	}
	tombstoneExists, err := pathExists(artifact.Tombstone)
	if err != nil {
		return err
	}
	if sourceExists && tombstoneExists {
		return fmt.Errorf("owned cleanup source and tombstone both exist: %s", artifact.Path)
	}
	if artifact.Preexisting {
		if !sourceExists || tombstoneExists {
			return fmt.Errorf("preexisting generation topology changed before cleanup: %s", artifact.Path)
		}
		if err := validateSource(artifact.Path); err != nil {
			return err
		}
		digest, err := treeSHA256(artifact.Path, "")
		if err != nil {
			return err
		}
		if digest != artifact.TreeSHA256 {
			return fmt.Errorf("preexisting generation content changed before cleanup: %s", artifact.Path)
		}
		return nil
	}
	if sourceExists {
		return validateSource(artifact.Path)
	}
	// A tombstone is already transaction-bound by its exact durable path. Its
	// source was validated before the same-filesystem rename, so a retry must
	// not demand intact content after an interrupted recursive delete.
	return nil
}

func makeTreeWritable(root string) error {
	var directories []string
	if err := filepath.WalkDir(root, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			directories = append(directories, current)
		}
		return nil
	}); err != nil {
		return err
	}
	for _, directory := range directories {
		if err := os.Chmod(directory, 0o755); err != nil {
			return err
		}
	}
	return nil
}

func removeOneTombstoneFile(tombstone string, syncDirectory directorySyncFunction) error {
	var selected string
	err := filepath.WalkDir(tombstone, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if selected == "" && !entry.IsDir() {
			selected = current
		}
		return nil
	})
	if err != nil {
		return err
	}
	if selected != "" {
		if err := os.Remove(selected); err != nil {
			return err
		}
		return syncDirectory(filepath.Dir(selected))
	}
	return syncDirectory(tombstone)
}

func (e *transactionEngine) cleanupOwnedTreeArtifact(
	artifact activationArtifact,
	label, injectedFailure string,
) error {
	if artifact.Preexisting {
		return nil
	}
	sourceExists, err := pathExists(artifact.Path)
	if err != nil {
		return err
	}
	tombstoneExists, err := pathExists(artifact.Tombstone)
	if err != nil {
		return err
	}
	if sourceExists && tombstoneExists {
		return fmt.Errorf("owned cleanup source and tombstone both exist: %s", artifact.Path)
	}
	if sourceExists {
		if err := os.Rename(artifact.Path, artifact.Tombstone); err != nil {
			return err
		}
		if err := e.syncDir(filepath.Dir(artifact.Path)); err != nil {
			return err
		}
		tombstoneExists = true
	}
	if !tombstoneExists {
		return nil
	}
	if err := makeTreeWritable(artifact.Tombstone); err != nil {
		return err
	}
	if injectedFailure == "during_cleanup_"+label {
		if err := removeOneTombstoneFile(artifact.Tombstone, e.syncDir); err != nil {
			return err
		}
		return fmt.Errorf("injected tile cleanup failure: %s", label)
	}
	if err := os.RemoveAll(artifact.Tombstone); err != nil {
		return err
	}
	return e.syncDir(filepath.Dir(artifact.Tombstone))
}

func (e *transactionEngine) validateOwnedSymlink(link, expectedTarget string) error {
	actual, isLink, err := symlinkTarget(link)
	if err != nil {
		return err
	}
	if !isLink {
		if _, err := os.Lstat(link); errors.Is(err, os.ErrNotExist) {
			return nil
		} else if err != nil {
			return err
		}
		return fmt.Errorf("transaction-owned cleanup link is not a symlink: %s", link)
	}
	if actual != expectedTarget {
		return fmt.Errorf("transaction-owned cleanup link has an unexpected target: %s", link)
	}
	return nil
}

func (e *transactionEngine) removeOwnedSymlinkDurably(link string) error {
	if err := os.Remove(link); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return e.syncDir(filepath.Dir(link))
}

func symlinkTarget(link string) (string, bool, error) {
	info, err := os.Lstat(link)
	if errors.Is(err, os.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return "", false, nil
	}
	target, err := os.Readlink(link)
	if err != nil {
		return "", false, err
	}
	return filepath.ToSlash(target), true, nil
}

func (e *transactionEngine) removeOrRetain(target, label string) error {
	info, err := os.Lstat(target)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || info.Mode().IsRegular() {
		if err := os.Remove(target); err != nil {
			return err
		}
	} else if info.IsDir() {
		retained := filepath.Join(e.root, ".retained-"+label+"-"+strconv.FormatInt(e.now().UnixNano(), 10))
		if _, err := os.Lstat(retained); err == nil {
			return fmt.Errorf("retained tile path already exists: %s", retained)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err := os.Rename(target, retained); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("refusing to remove non-file tile path: %s", target)
	}
	return e.syncDir(e.root)
}

func (e *transactionEngine) installPreviousLink(target, tileSetID string) error {
	if !isGenerationTarget(target) || !isSafeID(tileSetID) {
		return fmt.Errorf("unsafe previous tile target")
	}
	temporary := filepath.Join(e.root, ".offline-previous-link-"+tileSetID)
	if err := e.removeOrRetain(temporary, "previous-temp"); err != nil {
		return err
	}
	if err := os.Symlink(target, temporary); err != nil {
		return err
	}
	if err := e.syncDir(e.root); err != nil {
		return err
	}
	if err := e.removeOrRetain(e.previousPath(), "older-previous"); err != nil {
		return err
	}
	if err := os.Rename(temporary, e.previousPath()); err != nil {
		return err
	}
	return e.syncDir(e.root)
}

func (e *transactionEngine) cleanupExchange(exchangePath, tileSetID string) error {
	if !isSafeID(tileSetID) {
		return fmt.Errorf("unsafe tile-set identifier")
	}
	return e.removeOrRetain(exchangePath, "pre-generation-"+tileSetID)
}

func (e *transactionEngine) ensureGeneration(stage, tileSetID string) (tileManifest, error) {
	if !isSafeID(tileSetID) {
		return tileManifest{}, fmt.Errorf("invalid tile-set identifier")
	}
	cleanStage, err := validateRoot(stage)
	if err != nil {
		return tileManifest{}, err
	}
	if cleanStage != e.stagePath(tileSetID) {
		return tileManifest{}, fmt.Errorf("unexpected tile staging root")
	}
	if err := os.MkdirAll(e.generationsPath(), 0o755); err != nil {
		return tileManifest{}, err
	}
	if err := e.syncDir(e.root); err != nil {
		return tileManifest{}, err
	}
	generation := e.generationPath(tileSetID)
	if _, err := os.Lstat(generation); err == nil {
		return verifyArtifactRoot(generation, tileSetID, true)
	} else if !errors.Is(err, os.ErrNotExist) {
		return tileManifest{}, err
	}

	if _, err := verifyArtifactRoot(cleanStage, tileSetID, false); err != nil {
		return tileManifest{}, err
	}
	if err := removeEmptyStagingPartial(filepath.Join(cleanStage, activeOfflineName), e.syncDir); err != nil {
		return tileManifest{}, err
	}
	manifestBytes, err := os.ReadFile(filepath.Join(cleanStage, "manifest.json"))
	if err != nil {
		return tileManifest{}, err
	}
	embedded := filepath.Join(cleanStage, activeOfflineName, embeddedManifestName)
	if err := durableWriteFile(embedded, manifestBytes, 0o444, e.syncDir); err != nil {
		return tileManifest{}, err
	}
	if err := makeTreeReadOnly(cleanStage, e.syncDir); err != nil {
		return tileManifest{}, err
	}
	if err := os.Rename(cleanStage, generation); err != nil {
		return tileManifest{}, err
	}
	if err := e.syncDir(e.generationsPath()); err != nil {
		return tileManifest{}, err
	}
	// The stage is a child of root while the generation is a child of
	// tile-generations, so durably record both directory-entry changes.
	if err := e.syncDir(e.root); err != nil {
		return tileManifest{}, err
	}
	return verifyArtifactRoot(generation, tileSetID, true)
}

func removeEmptyStagingPartial(offlineRoot string, syncDirectory directorySyncFunction) error {
	partial := filepath.Join(offlineRoot, ".rsync-partial")
	info, err := os.Lstat(partial)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return fmt.Errorf("rsync partial path is not a directory")
	}
	entries, err := os.ReadDir(partial)
	if err != nil {
		return err
	}
	if len(entries) != 0 {
		return fmt.Errorf("rsync partial directory is not empty")
	}
	if err := os.Remove(partial); err != nil {
		return err
	}
	return syncDirectory(offlineRoot)
}

func makeTreeReadOnly(root string, syncDirectory directorySyncFunction) error {
	var entries []string
	err := filepath.WalkDir(root, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink is forbidden in immutable generation: %s", current)
		}
		// Keep the generation container writable. The legacy Python workflow
		// freezes the manifest and offline tree, then renames the container;
		// preserving that shape also leaves durable recovery cleanup possible.
		if current != root {
			entries = append(entries, current)
		}
		return nil
	})
	if err != nil {
		return err
	}
	sort.Slice(entries, func(first, second int) bool { return len(entries[first]) > len(entries[second]) })
	for _, entry := range entries {
		info, err := os.Lstat(entry)
		if err != nil {
			return err
		}
		if info.IsDir() {
			if err := os.Chmod(entry, 0o555); err != nil {
				return err
			}
			if err := syncDirectory(entry); err != nil {
				return err
			}
		} else if info.Mode().IsRegular() {
			if err := os.Chmod(entry, 0o444); err != nil {
				return err
			}
			if err := fsyncFile(entry); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("non-regular generation entry: %s", entry)
		}
	}
	return syncDirectory(root)
}

func (e *transactionEngine) verifyGeneration(tileSetID string) error {
	_, err := verifyArtifactRoot(e.generationPath(tileSetID), tileSetID, true)
	return err
}

func (e *transactionEngine) planLegacyMigration(targetTileSetID string) (legacyMigrationPlan, error) {
	if !isSafeID(targetTileSetID) {
		return legacyMigrationPlan{}, fmt.Errorf("invalid legacy migration target tile-set identifier")
	}
	active := e.activePath()
	info, err := os.Stat(active)
	if err != nil {
		return legacyMigrationPlan{}, err
	}
	legacyTreeSHA256, err := treeSHA256(active, "")
	if err != nil {
		return legacyMigrationPlan{}, fmt.Errorf("hash direct legacy tree: %w", err)
	}
	// The generation container is target-bound even when the adjacent legacy
	// manifest preserves a canonical tile_set_id. A failed pre-switch attempt
	// for one target therefore cannot be mistaken for provenance for another.
	legacyID := legacyContainerID(info, targetTileSetID, legacyTreeSHA256)
	target, err := e.generationTarget(legacyID)
	if err != nil {
		return legacyMigrationPlan{}, err
	}
	manifestData, previousID, err := e.legacyManifestBytes(legacyID, targetTileSetID, legacyTreeSHA256)
	if err != nil {
		return legacyMigrationPlan{}, err
	}
	if previousID == targetTileSetID {
		return legacyMigrationPlan{}, fmt.Errorf("direct legacy prior identity equals requested target")
	}
	return legacyMigrationPlan{
		ContainerID: legacyID, PreviousID: previousID, Target: target,
		TreeSHA256: legacyTreeSHA256, ManifestData: manifestData,
	}, nil
}

func (e *transactionEngine) materializeLegacyMigration(plan legacyMigrationPlan, targetTileSetID string) error {
	if !isSafeID(plan.ContainerID) || !isSafeID(plan.PreviousID) || !isGenerationTarget(plan.Target) {
		return fmt.Errorf("invalid legacy migration plan")
	}
	active := e.activePath()
	legacy := e.generationPath(plan.ContainerID)
	if _, err := os.Lstat(legacy); errors.Is(err, os.ErrNotExist) {
		building := filepath.Join(e.generationsPath(), "."+plan.ContainerID+".building")
		if err := os.RemoveAll(building); err != nil {
			return err
		}
		if err := copyDirectory(active, filepath.Join(building, activeOfflineName)); err != nil {
			return err
		}
		if err := durableWriteFile(filepath.Join(building, "manifest.json"), plan.ManifestData, 0o444, e.syncDir); err != nil {
			return err
		}
		if err := durableWriteFile(filepath.Join(building, activeOfflineName, embeddedManifestName), plan.ManifestData, 0o444, e.syncDir); err != nil {
			return err
		}
		if err := makeTreeReadOnly(building, e.syncDir); err != nil {
			return err
		}
		if err := os.Rename(building, legacy); err != nil {
			return err
		}
		if err := e.syncDir(e.generationsPath()); err != nil {
			return err
		}
	} else if err != nil {
		return err
	}
	manifest, present, err := readManifestIdentity(filepath.Join(legacy, activeOfflineName))
	if err != nil || !present || manifest.TileSetID == "" {
		if err != nil {
			return err
		}
		return fmt.Errorf("legacy generation manifest is missing")
	}
	if manifest.TileSetID != plan.PreviousID {
		return fmt.Errorf("legacy generation identity differs from its durable plan")
	}
	if err := verifyLegacyMigrationTree(filepath.Join(legacy, activeOfflineName), manifest, targetTileSetID); err != nil {
		return err
	}
	return nil
}

func (e *transactionEngine) legacyManifestBytes(
	legacyID, targetTileSetID, legacyTreeSHA256 string,
) ([]byte, string, error) {
	payload := map[string]any{"tile_set_id": legacyID, "legacy": true}
	adjacent := filepath.Join(e.root, "offline.manifest.json")
	if contents, err := os.ReadFile(adjacent); err == nil {
		var candidate map[string]any
		if decodeSingleJSON(contents, &candidate) == nil {
			payload = candidate
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, "", err
	}
	if tileSetID, ok := payload["tile_set_id"].(string); !ok || tileSetID == "" {
		payload["tile_set_id"] = legacyID
	}
	previousID, ok := payload["tile_set_id"].(string)
	if !ok || !isSafeID(previousID) {
		return nil, "", fmt.Errorf("legacy adjacent manifest has an invalid tile identity")
	}
	payload["legacy"] = true
	payload["legacy_migration_target_tile_set_id"] = targetTileSetID
	payload["legacy_tree_sha256"] = legacyTreeSHA256
	contents, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	return append(contents, '\n'), previousID, nil
}

func copyDirectory(source, destination string) error {
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return err
	}
	return filepath.WalkDir(source, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, current)
		if err != nil {
			return err
		}
		if relative == "." {
			return nil
		}
		target := filepath.Join(destination, relative)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("legacy tile tree contains a symlink: %s", current)
		}
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("legacy tile tree contains non-regular entry: %s", current)
		}
		return copyRegularFile(current, target, info.Mode().Perm())
	})
}

func copyRegularFile(source, destination string, mode fs.FileMode) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		output.Close()
		return err
	}
	if err := output.Sync(); err != nil {
		output.Close()
		return err
	}
	return output.Close()
}

func manifestID(tree string) (string, bool, error) {
	manifest, present, err := readManifestIdentity(tree)
	return manifest.TileSetID, present, err
}

func readManifestIdentity(tree string) (manifestIdentity, bool, error) {
	contents, err := os.ReadFile(filepath.Join(tree, embeddedManifestName))
	if errors.Is(err, os.ErrNotExist) {
		return manifestIdentity{}, false, nil
	}
	if err != nil {
		return manifestIdentity{}, false, err
	}
	var manifest manifestIdentity
	if err := decodeSingleJSON(contents, &manifest); err != nil {
		return manifestIdentity{}, false, err
	}
	if manifest.TileSetID == "" || !isSafeID(manifest.TileSetID) {
		return manifestIdentity{}, false, fmt.Errorf("invalid embedded tile-set identifier")
	}
	return manifest, true, nil
}

func requireLegacyMigrationProof(manifest manifestIdentity, expectedTargetTileSetID string) error {
	if !manifest.Legacy ||
		manifest.LegacyMigrationTargetTileSetID == "" ||
		manifest.LegacyMigrationTargetTileSetID != expectedTargetTileSetID ||
		!isLowerSHA256(manifest.LegacyTreeSHA256) {
		return fmt.Errorf("previous generation lacks exact target-bound legacy migration provenance")
	}
	return nil
}

func verifyLegacyMigrationTree(
	tree string,
	manifest manifestIdentity,
	expectedTargetTileSetID string,
) error {
	if err := requireLegacyMigrationProof(manifest, expectedTargetTileSetID); err != nil {
		return err
	}
	digest, err := treeSHA256(tree, embeddedManifestName)
	if err != nil {
		return err
	}
	if digest != manifest.LegacyTreeSHA256 {
		return fmt.Errorf("legacy migration tree digest differs from its immutable provenance")
	}
	return nil
}

func attachLegacyMigrationProof(
	result transactionResult,
	manifest manifestIdentity,
	expectedTargetTileSetID string,
) (transactionResult, error) {
	if err := requireLegacyMigrationProof(manifest, expectedTargetTileSetID); err != nil {
		return transactionResult{}, err
	}
	result.PreviousTileSetProvenance = legacyMigrationProvenance
	result.PreviousTileSetTargetID = expectedTargetTileSetID
	return result, nil
}

func (e *transactionEngine) validateExactPostRollbackTopology(expectedTargetTileSetID string) (manifestIdentity, error) {
	activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
	if err != nil {
		return manifestIdentity{}, err
	}
	previousTarget, previousIsLink, err := symlinkTarget(e.previousPath())
	if err != nil {
		return manifestIdentity{}, err
	}
	if !activeIsLink || !previousIsLink ||
		!isGenerationTarget(activeTarget) || !isGenerationTarget(previousTarget) ||
		activeTarget == previousTarget {
		return manifestIdentity{}, fmt.Errorf("target-inactive recovery lacks exact post-rollback pointer topology")
	}
	previousManifest, present, err := readManifestIdentity(e.previousPath())
	if err != nil || !present || previousManifest.TileSetID != expectedTargetTileSetID {
		if err != nil {
			return manifestIdentity{}, err
		}
		return manifestIdentity{}, fmt.Errorf("post-rollback previous pointer does not contain the expected deployed target")
	}
	previousContainer := strings.Split(previousTarget, "/")[1]
	if previousContainer != expectedTargetTileSetID {
		return manifestIdentity{}, fmt.Errorf("post-rollback previous pointer targets an unexpected immutable generation")
	}
	if err := e.verifyGeneration(expectedTargetTileSetID); err != nil {
		return manifestIdentity{}, fmt.Errorf("verify post-rollback deployed target generation: %w", err)
	}
	activeManifest, present, err := readManifestIdentity(e.activePath())
	if err != nil || !present || activeManifest.TileSetID == "" || activeManifest.TileSetID == expectedTargetTileSetID {
		if err != nil {
			return manifestIdentity{}, err
		}
		return manifestIdentity{}, fmt.Errorf("post-rollback active generation lacks a distinct prior identity")
	}
	if err := verifyLegacyMigrationTree(
		filepath.Join(e.root, activeTarget), activeManifest, expectedTargetTileSetID,
	); err != nil {
		return manifestIdentity{}, err
	}
	for _, pattern := range []string{".offline-*", filepath.Join(generationDirectory, ".*.building")} {
		matches, err := filepath.Glob(filepath.Join(e.root, pattern))
		if err != nil {
			return manifestIdentity{}, err
		}
		if len(matches) != 0 {
			return manifestIdentity{}, fmt.Errorf("post-rollback topology contains conflicting temporary artifacts")
		}
	}
	retained, err := filepath.Glob(filepath.Join(e.root, ".retained-*"))
	if err != nil {
		return manifestIdentity{}, err
	}
	expectedPrefix := ".retained-pre-generation-" + expectedTargetTileSetID + "-"
	if len(retained) != 1 || !strings.HasPrefix(filepath.Base(retained[0]), expectedPrefix) {
		return manifestIdentity{}, fmt.Errorf("post-rollback topology lacks its exact retained direct-tree evidence")
	}
	retainedDigest, err := treeSHA256(retained[0], "")
	if err != nil {
		return manifestIdentity{}, err
	}
	if retainedDigest != activeManifest.LegacyTreeSHA256 {
		return manifestIdentity{}, fmt.Errorf("retained direct-tree evidence differs from the restored prior generation")
	}
	retainedInfo, err := os.Stat(retained[0])
	if err != nil {
		return manifestIdentity{}, err
	}
	activeContainer := strings.Split(activeTarget, "/")[1]
	if activeContainer != legacyContainerID(retainedInfo, expectedTargetTileSetID, retainedDigest) {
		return manifestIdentity{}, fmt.Errorf("restored prior generation container differs from target-bound migration evidence")
	}
	return activeManifest, nil
}

func (e *transactionEngine) validatePreviousPointerAuthority(authority *activationArtifactAuthority) error {
	target, isLink, err := symlinkTarget(e.previousPath())
	if err != nil {
		return err
	}
	if authority.PreviousPointerTarget == "" {
		if isLink {
			return fmt.Errorf("previous tile pointer appeared after transaction authority")
		}
		if _, err := os.Lstat(e.previousPath()); err == nil {
			return fmt.Errorf("previous tile path appeared after transaction authority")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if !isLink || target != authority.PreviousPointerTarget {
		return fmt.Errorf("previous tile pointer topology changed during activation")
	}
	digest, err := e.verifyGenerationTargetIdentity(target, authority.PreviousPointerID)
	if err != nil {
		return err
	}
	if digest != authority.PreviousPointerTreeSHA256 {
		return fmt.Errorf("previous tile pointer content changed during activation")
	}
	return nil
}

func (e *transactionEngine) retainedCleanupArtifact(
	authority *activationArtifactAuthority,
	legacyPlan *legacyMigrationPlan,
) (*activationArtifact, error) {
	sources, err := filepath.Glob(authority.RetainedPrefix + "*")
	if err != nil {
		return nil, err
	}
	if len(sources) > 1 {
		return nil, fmt.Errorf("activation has multiple retained cleanup sources")
	}
	tombstoneExists, err := pathExists(authority.RetainedTombstone)
	if err != nil {
		return nil, err
	}
	if len(sources) == 0 && !tombstoneExists {
		return nil, nil
	}
	if legacyPlan == nil {
		return nil, fmt.Errorf("activation has a retained artifact without legacy migration authority")
	}
	artifact := activationArtifact{Path: authority.RetainedPrefix, Tombstone: authority.RetainedTombstone}
	if len(sources) == 1 {
		artifact.Path = sources[0]
		if tombstoneExists {
			return nil, fmt.Errorf("retained cleanup source and tombstone both exist")
		}
		if err := validateOwnedRemovalTree(artifact.Path); err != nil {
			return nil, err
		}
		digest, err := treeSHA256(artifact.Path, "")
		if err != nil {
			return nil, err
		}
		if digest != legacyPlan.TreeSHA256 {
			return nil, fmt.Errorf("retained direct-tree artifact differs from transaction authority")
		}
	}
	return &artifact, nil
}

// settlePreExchangeActivation removes only artifacts whose exact paths and
// pre-transaction state are durably authorized. Each owned tree is first
// atomically renamed to a deterministic same-filesystem tombstone; recursive
// deletion may then be interrupted and replayed without revalidating partial
// content. The transaction is cleared only after all source names and
// tombstones are absent and their parent directories have been synced.
func (e *transactionEngine) settlePreExchangeActivation(
	transaction tileTransaction,
	injectedFailure string,
) error {
	if transaction.Kind != "activation" {
		return fmt.Errorf("pre-exchange settlement requires an activation transaction")
	}
	authority := transaction.Artifacts
	if authority == nil {
		return fmt.Errorf("activation cleanup ownership is unknown")
	}
	activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
	if err != nil {
		return err
	}
	var legacyPlan *legacyMigrationPlan
	if activeIsLink {
		if activeTarget != transaction.PreviousTarget {
			return fmt.Errorf("active tile pointer differs from the pre-exchange transaction baseline")
		}
		manifest, present, err := readManifestIdentity(e.activePath())
		if err != nil || !present || manifest.TileSetID != transaction.PreviousID {
			if err != nil {
				return err
			}
			return fmt.Errorf("pre-exchange active generation differs from the durable previous identity")
		}
	} else {
		info, err := os.Lstat(e.activePath())
		if err != nil {
			return err
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("pre-exchange active tile tree is not the original direct directory")
		}
		if _, present, err := readManifestIdentity(e.activePath()); err != nil || present {
			if err != nil {
				return err
			}
			return fmt.Errorf("pre-exchange direct tile tree unexpectedly has an embedded manifest")
		}
		plan, err := e.planLegacyMigration(transaction.NewID)
		if err != nil {
			return err
		}
		if plan.Target != transaction.PreviousTarget || plan.PreviousID != transaction.PreviousID {
			return fmt.Errorf("pre-exchange direct tree differs from the durable migration authority")
		}
		legacyPlan = &plan
	}
	if authority.LegacyMigrationPlanned != (legacyPlan != nil) {
		return fmt.Errorf("activation legacy ownership differs from active topology")
	}
	if err := e.validatePreviousPointerAuthority(authority); err != nil {
		return err
	}

	stage := e.stagePath(transaction.NewID)
	if _, err := os.Lstat(stage); err == nil {
		if _, err := verifyArtifactRoot(stage, transaction.NewID, false); err != nil {
			return fmt.Errorf("verify transaction-owned target stage: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := validateOwnedTreeArtifactState(authority.TargetGeneration, func(source string) error {
		_, err := verifyArtifactRoot(source, transaction.NewID, true)
		return err
	}); err != nil {
		return err
	}
	if err := validateOwnedTreeArtifactState(authority.TargetBuilding, validateOwnedRemovalTree); err != nil {
		return err
	}

	var legacyGeneration, legacyBuilding *activationArtifact
	if legacyPlan != nil {
		legacyGeneration = authority.LegacyGeneration
		legacyBuilding = authority.LegacyBuilding
		if legacyGeneration == nil || legacyBuilding == nil {
			return fmt.Errorf("activation legacy cleanup ownership is missing")
		}
		if err := validateOwnedTreeArtifactState(*legacyGeneration, func(source string) error {
			manifest, present, err := readManifestIdentity(filepath.Join(source, activeOfflineName))
			if err != nil || !present || manifest.TileSetID != transaction.PreviousID {
				if err != nil {
					return err
				}
				return fmt.Errorf("legacy generation differs from its durable previous identity")
			}
			return verifyLegacyMigrationTree(filepath.Join(source, activeOfflineName), manifest, transaction.NewID)
		}); err != nil {
			return err
		}
		if err := validateOwnedTreeArtifactState(*legacyBuilding, validateOwnedRemovalTree); err != nil {
			return err
		}
	}

	if err := e.validateOwnedSymlink(transaction.SwitchPath, transaction.NewTarget); err != nil {
		return err
	}
	previousTemporary := filepath.Join(e.root, ".offline-previous-link-"+transaction.NewID)
	if err := e.validateOwnedSymlink(previousTemporary, transaction.PreviousTarget); err != nil {
		return err
	}
	retained, err := e.retainedCleanupArtifact(authority, legacyPlan)
	if err != nil {
		return err
	}

	if err := e.removeOwnedSymlinkDurably(transaction.SwitchPath); err != nil {
		return err
	}
	if err := e.removeOwnedSymlinkDurably(previousTemporary); err != nil {
		return err
	}
	// The verified transfer stage predates the helper transaction and remains
	// available for an activation retry. Preexisting immutable generations and
	// the recorded offline.previous topology are likewise preserved.
	if err := e.cleanupOwnedTreeArtifact(authority.TargetBuilding, "target_building", injectedFailure); err != nil {
		return err
	}
	if err := e.cleanupOwnedTreeArtifact(authority.TargetGeneration, "target_generation", injectedFailure); err != nil {
		return err
	}
	if legacyBuilding != nil {
		if err := e.cleanupOwnedTreeArtifact(*legacyBuilding, "legacy_building", injectedFailure); err != nil {
			return err
		}
	}
	if legacyGeneration != nil {
		if err := e.cleanupOwnedTreeArtifact(*legacyGeneration, "legacy_generation", injectedFailure); err != nil {
			return err
		}
	}
	if retained != nil {
		if err := e.cleanupOwnedTreeArtifact(*retained, "retained", injectedFailure); err != nil {
			return err
		}
	}
	return e.clearTransaction()
}

func (e *transactionEngine) recoverActivation(
	tileSetID, expectedCurrentTileSetID string,
) (bool, transactionResult, error) {
	transaction, exists, err := e.readTransaction()
	if err != nil || !exists {
		return false, transactionResult{}, err
	}
	if transaction.Kind != "activation" || transaction.NewID != tileSetID {
		return false, transactionResult{}, fmt.Errorf("another tile transaction requires recovery")
	}
	if expectedCurrentTileSetID != "" && transaction.PreviousID != expectedCurrentTileSetID {
		return false, transactionResult{}, fmt.Errorf("activation recovery baseline differs from the host snapshot")
	}
	activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
	if err != nil {
		return false, transactionResult{}, err
	}
	if activeIsLink && activeTarget == transaction.NewTarget {
		previousManifest, present, err := readManifestIdentity(filepath.Join(e.root, transaction.PreviousTarget))
		if err != nil || !present || previousManifest.TileSetID == "" {
			if err != nil {
				return false, transactionResult{}, err
			}
			return false, transactionResult{}, fmt.Errorf("recovered activation previous generation identity is missing")
		}
		if transaction.PreviousID == "" || previousManifest.TileSetID != transaction.PreviousID {
			return false, transactionResult{}, fmt.Errorf("recovered activation previous identity differs from its durable helper transaction")
		}
		if previousManifest.Legacy {
			if err := verifyLegacyMigrationTree(
				filepath.Join(e.root, transaction.PreviousTarget), previousManifest, tileSetID,
			); err != nil {
				return false, transactionResult{}, err
			}
		}
		if err := e.verifyGeneration(tileSetID); err != nil {
			return false, transactionResult{}, err
		}
		if err := e.installPreviousLink(transaction.PreviousTarget, tileSetID); err != nil {
			return false, transactionResult{}, err
		}
		if err := e.cleanupExchange(transaction.SwitchPath, tileSetID); err != nil {
			return false, transactionResult{}, err
		}
		if err := e.clearTransaction(); err != nil {
			return false, transactionResult{}, err
		}
		return true, transactionResult{
			Operation: "activate", TileSetID: tileSetID, ActivatedTileSetID: tileSetID,
			PreviousTileSetID: previousManifest.TileSetID, Recovered: true,
		}, nil
	}
	if activeIsLink && activeTarget != transaction.PreviousTarget {
		return false, transactionResult{}, fmt.Errorf("active tile pointer diverged during recovery")
	}
	if err := e.settlePreExchangeActivation(transaction, ""); err != nil {
		return false, transactionResult{}, err
	}
	return false, transactionResult{}, nil
}

func (e *transactionEngine) activate(
	stage, tileSetID, expectedCurrentTileSetID, injectedFailure string,
) (transactionResult, error) {
	return e.withExclusiveTransactionLock(func() (transactionResult, error) {
		return e.activateLocked(stage, tileSetID, expectedCurrentTileSetID, injectedFailure)
	})
}

func (e *transactionEngine) activateLocked(
	stage, tileSetID, expectedCurrentTileSetID, injectedFailure string,
) (transactionResult, error) {
	if !isSafeID(tileSetID) {
		return transactionResult{}, fmt.Errorf("invalid tile-set identifier")
	}
	if expectedCurrentTileSetID != "" && !isSafeID(expectedCurrentTileSetID) {
		return transactionResult{}, fmt.Errorf("invalid expected current tile-set identifier")
	}
	if recovered, result, err := e.recoverActivation(tileSetID, expectedCurrentTileSetID); err != nil {
		return transactionResult{}, err
	} else if recovered {
		return result, nil
	}
	stagedManifest, err := verifyArtifactRoot(stage, tileSetID, false)
	if err != nil {
		return transactionResult{}, err
	}
	if activeID, present, err := manifestID(e.activePath()); err != nil {
		return transactionResult{}, err
	} else if present && activeID == tileSetID {
		if expectedCurrentTileSetID != tileSetID {
			return transactionResult{}, fmt.Errorf("already-active target differs from the host-recorded baseline")
		}
		activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
		if err != nil || !activeIsLink || activeTarget != filepath.ToSlash(filepath.Join(generationDirectory, tileSetID, activeOfflineName)) {
			if err != nil {
				return transactionResult{}, err
			}
			return transactionResult{}, fmt.Errorf("already-active target is not the exact canonical generation pointer")
		}
		generationManifest, err := verifyArtifactRoot(e.generationPath(tileSetID), tileSetID, true)
		if err != nil {
			return transactionResult{}, err
		}
		if !manifestsEqual(stagedManifest, generationManifest) {
			return transactionResult{}, fmt.Errorf("already-active canonical generation differs from the staged target")
		}
		return transactionResult{
			Operation: "activate", TileSetID: tileSetID, ActivatedTileSetID: tileSetID,
			ActiveTileSetID: tileSetID, PreviousTileSetID: tileSetID,
			TargetAlreadyActive: true, TileActivationNotSwitched: true,
		}, nil
	}

	newTarget, err := e.generationTarget(tileSetID)
	if err != nil {
		return transactionResult{}, err
	}
	active := e.activePath()
	previousTarget, activeIsLink, err := symlinkTarget(active)
	if err != nil {
		return transactionResult{}, err
	}
	var previousID string
	var legacyPlan *legacyMigrationPlan
	directHasAdjacentIdentity := false
	if activeIsLink {
		if !isGenerationTarget(previousTarget) {
			return transactionResult{}, fmt.Errorf("active tile pointer is unsafe")
		}
		previousManifest, present, err := readManifestIdentity(active)
		if err != nil || !present || previousManifest.TileSetID == "" {
			if err != nil {
				return transactionResult{}, err
			}
			return transactionResult{}, fmt.Errorf("previous tile generation identity is missing")
		}
		previousID = previousManifest.TileSetID
		if expectedCurrentTileSetID == "" || expectedCurrentTileSetID != previousID {
			return transactionResult{}, fmt.Errorf("canonical active identity differs from the host-recorded baseline")
		}
	} else {
		info, err := os.Lstat(active)
		if err != nil {
			return transactionResult{}, fmt.Errorf("no complete active tile generation is available: %w", err)
		}
		if !info.IsDir() {
			return transactionResult{}, fmt.Errorf("no complete active tile generation is available")
		}
		adjacent, present, err := e.adjacentManifestIdentity()
		if err != nil {
			return transactionResult{}, err
		}
		if present && adjacent.TileSetID == tileSetID {
			if expectedCurrentTileSetID != tileSetID {
				return transactionResult{}, fmt.Errorf("same-ID direct tree differs from the host-recorded baseline")
			}
			if err := e.verifyExactSameTargetDirectTree(tileSetID); err != nil {
				return transactionResult{}, err
			}
			if err := verifyDirectOfflineTree(e.root, stagedManifest); err != nil {
				return transactionResult{}, fmt.Errorf("same-ID direct tree differs from staged target: %w", err)
			}
			return transactionResult{
				Operation: "activate", TileSetID: tileSetID, ActivatedTileSetID: tileSetID,
				ActiveTileSetID: tileSetID, PreviousTileSetID: tileSetID,
				TargetAlreadyActive: true, TileActivationNotSwitched: true,
			}, nil
		}
		plan, err := e.planLegacyMigration(tileSetID)
		if err != nil {
			return transactionResult{}, err
		}
		legacyPlan = &plan
		previousTarget = plan.Target
		previousID = plan.PreviousID
		directHasAdjacentIdentity = present
		if directHasAdjacentIdentity {
			if expectedCurrentTileSetID != previousID {
				return transactionResult{}, fmt.Errorf("direct active identity differs from the host-recorded baseline")
			}
		} else if expectedCurrentTileSetID != "" {
			return transactionResult{}, fmt.Errorf("unidentified direct active tree conflicts with the host-recorded baseline")
		}
	}
	if previousID == tileSetID {
		return transactionResult{}, fmt.Errorf("previous tile identity equals requested target before activation")
	}

	switchPath := e.switchPath(tileSetID)
	if _, err := os.Lstat(switchPath); err == nil {
		return transactionResult{}, fmt.Errorf("activation switch path exists without matching durable transaction authority")
	} else if !errors.Is(err, os.ErrNotExist) {
		return transactionResult{}, err
	}
	if injectedFailure == "before_journal" {
		return transactionResult{}, errors.New("injected tile activation failure: before_journal")
	}
	transaction := tileTransaction{
		Kind: "activation", NewID: tileSetID, NewTarget: newTarget,
		PreviousTarget: previousTarget, PreviousID: previousID, SwitchPath: switchPath,
	}
	authority, err := e.activationArtifactAuthority(transaction, legacyPlan)
	if err != nil {
		return transactionResult{}, err
	}
	transaction.Artifacts = authority
	if err := e.writeTransaction(transaction); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_journal" {
		return transactionResult{}, errors.New("injected tile activation failure: after_journal")
	}
	if _, err := e.ensureGeneration(stage, tileSetID); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_generation" {
		return transactionResult{}, errors.New("injected tile activation failure: after_generation")
	}
	if legacyPlan != nil {
		if err := e.materializeLegacyMigration(*legacyPlan, tileSetID); err != nil {
			return transactionResult{}, err
		}
	}
	if injectedFailure == "after_legacy" {
		return transactionResult{}, errors.New("injected tile activation failure: after_legacy")
	}
	if err := os.Symlink(newTarget, switchPath); err != nil {
		return transactionResult{}, err
	}
	if err := e.syncDir(e.root); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_switch_publish" {
		return transactionResult{}, errors.New("injected tile activation failure: after_switch_publish")
	}
	if err := e.requireExactParkedState(); err != nil {
		return transactionResult{}, err
	}
	if err := e.exchange(switchPath, active); err != nil {
		return transactionResult{}, fmt.Errorf("renameat2 exchange activation: %w", err)
	}
	if err := e.syncDir(e.root); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_switch" {
		return transactionResult{}, errors.New("injected tile activation failure: after_switch")
	}
	if err := e.installPreviousLink(previousTarget, tileSetID); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_previous" {
		return transactionResult{}, errors.New("injected tile activation failure: after_previous")
	}
	if err := e.cleanupExchange(switchPath, tileSetID); err != nil {
		return transactionResult{}, err
	}
	if err := e.clearTransaction(); err != nil {
		return transactionResult{}, err
	}
	return transactionResult{
		Operation: "activate", TileSetID: tileSetID, ActivatedTileSetID: tileSetID,
		PreviousTileSetID: previousID,
	}, nil
}

func (e *transactionEngine) rollback(expectedTileSetID, expectedPreviousTileSetID, injectedFailure string) (transactionResult, error) {
	return e.withExclusiveTransactionLock(func() (transactionResult, error) {
		return e.rollbackLocked(expectedTileSetID, expectedPreviousTileSetID, injectedFailure)
	})
}

func (e *transactionEngine) rollbackBound(
	expectedTileSetID, expectedPreviousTileSetID, repoRoot, expectedBranch, expectedHead, injectedFailure string,
) (transactionResult, error) {
	return e.withExclusiveTransactionLock(func() (transactionResult, error) {
		branch, head, dirty, err := e.gitIdentity(repoRoot)
		if err != nil {
			return transactionResult{}, err
		}
		if branch != expectedBranch || head != expectedHead || dirty {
			return transactionResult{}, fmt.Errorf("rollback Git identity changed before tile mutation")
		}
		return e.rollbackLocked(expectedTileSetID, expectedPreviousTileSetID, injectedFailure)
	})
}

func (e *transactionEngine) rollbackLocked(expectedTileSetID, expectedPreviousTileSetID, injectedFailure string) (transactionResult, error) {
	transaction, exists, err := e.readTransaction()
	if err != nil {
		return transactionResult{}, err
	}
	if exists {
		switch transaction.Kind {
		case "activation":
			if expectedTileSetID == "" || transaction.NewID != expectedTileSetID {
				return transactionResult{}, fmt.Errorf("activation recovery target differs from the recorded deployment target")
			}
			activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
			if err != nil {
				return transactionResult{}, err
			}
			if !activeIsLink || activeTarget == transaction.PreviousTarget {
				if expectedPreviousTileSetID != "" && transaction.PreviousID != expectedPreviousTileSetID {
					return transactionResult{}, fmt.Errorf("activation recovery previous identity differs from the recorded journal")
				}
				if err := e.settlePreExchangeActivation(transaction, injectedFailure); err != nil {
					return transactionResult{}, err
				}
				result := transactionResult{
					Operation: "rollback", TileActivationNotSwitched: true, Recovered: true,
				}
				if expectedPreviousTileSetID != "" {
					result.ActiveTileSetID = expectedPreviousTileSetID
					result.PreviousTileSetID = expectedPreviousTileSetID
				}
				return result, nil
			}
			if activeTarget != transaction.NewTarget {
				return transactionResult{}, fmt.Errorf("activation pointers diverged during rollback recovery")
			}
			previousManifest, present, err := readManifestIdentity(filepath.Join(e.root, transaction.PreviousTarget))
			if err != nil || !present || previousManifest.TileSetID == "" {
				if err != nil {
					return transactionResult{}, err
				}
				return transactionResult{}, fmt.Errorf("activation recovery previous generation identity is missing")
			}
			if expectedPreviousTileSetID != "" {
				if previousManifest.TileSetID != expectedPreviousTileSetID {
					return transactionResult{}, fmt.Errorf("activation recovery previous identity differs from the recorded journal")
				}
			} else {
				if transaction.PreviousID == "" || previousManifest.TileSetID != transaction.PreviousID {
					return transactionResult{}, fmt.Errorf("activation recovery previous identity differs from its durable helper transaction")
				}
				if err := verifyLegacyMigrationTree(
					filepath.Join(e.root, transaction.PreviousTarget), previousManifest, expectedTileSetID,
				); err != nil {
					return transactionResult{}, err
				}
			}
			if err := e.installPreviousLink(transaction.PreviousTarget, transaction.NewID); err != nil {
				return transactionResult{}, err
			}
			if err := e.cleanupExchange(transaction.SwitchPath, transaction.NewID); err != nil {
				return transactionResult{}, err
			}
			if err := e.clearTransaction(); err != nil {
				return transactionResult{}, err
			}
		case "rollback":
			if expectedTileSetID != "" && transaction.ActiveID != "" && transaction.ActiveID != expectedTileSetID {
				return transactionResult{}, fmt.Errorf("rollback transaction target differs from the recorded deployment target")
			}
			activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
			if err != nil {
				return transactionResult{}, err
			}
			previousTarget, previousIsLink, err := symlinkTarget(e.previousPath())
			if err != nil {
				return transactionResult{}, err
			}
			if activeIsLink && previousIsLink && activeTarget == transaction.PreviousTarget && previousTarget == transaction.ActiveTarget {
				manifest, present, err := readManifestIdentity(e.activePath())
				if err != nil || !present || manifest.TileSetID == "" {
					if err == nil {
						err = fmt.Errorf("recovered rollback active identity is missing")
					}
					return transactionResult{}, err
				}
				id := manifest.TileSetID
				if transaction.PreviousID != "" && id != transaction.PreviousID {
					return transactionResult{}, fmt.Errorf("recovered rollback identity differs from the durable helper transaction")
				}
				if expectedPreviousTileSetID != "" && id != expectedPreviousTileSetID {
					return transactionResult{}, fmt.Errorf("recovered rollback tile-set ID differs from recorded previous identity")
				}
				result := transactionResult{
					Operation: "rollback", RolledBackTileSetID: id,
					PreviousTileSetID: id, Recovered: true,
				}
				if expectedPreviousTileSetID == "" {
					if transaction.ActiveID != expectedTileSetID || transaction.PreviousID != id {
						return transactionResult{}, fmt.Errorf("recovered rollback lacks exact target-bound helper transaction identity")
					}
					if err := verifyLegacyMigrationTree(
						filepath.Join(e.root, activeTarget), manifest, expectedTileSetID,
					); err != nil {
						return transactionResult{}, err
					}
					result, err = attachLegacyMigrationProof(result, manifest, expectedTileSetID)
					if err != nil {
						return transactionResult{}, err
					}
				}
				if err := e.clearTransaction(); err != nil {
					return transactionResult{}, err
				}
				return result, nil
			}
			if !activeIsLink || !previousIsLink || activeTarget != transaction.ActiveTarget || previousTarget != transaction.PreviousTarget {
				return transactionResult{}, fmt.Errorf("tile rollback pointers diverged during recovery")
			}
		default:
			return transactionResult{}, fmt.Errorf("unknown tile transaction requires manual recovery")
		}
	}

	if expectedTileSetID != "" {
		activeManifest, present, err := readManifestIdentity(e.activePath())
		if err != nil {
			return transactionResult{}, err
		}
		if !present || activeManifest.TileSetID != expectedTileSetID {
			if !present && expectedPreviousTileSetID != "" {
				if err := e.verifyUnchangedDirectTreeIdentity(expectedPreviousTileSetID); err != nil {
					return transactionResult{}, err
				}
				return transactionResult{
					Operation: "rollback", TileActivationNotSwitched: true,
					ActiveTileSetID: expectedPreviousTileSetID, PreviousTileSetID: expectedPreviousTileSetID,
				}, nil
			}
			if !present && expectedPreviousTileSetID == "" {
				unchanged, err := e.unchangedDirectLegacyTree()
				if err != nil {
					return transactionResult{}, err
				}
				if unchanged {
					return transactionResult{
						Operation: "rollback", TileActivationNotSwitched: true,
					}, nil
				}
			}
			if present && expectedPreviousTileSetID != "" {
				if activeManifest.TileSetID != expectedPreviousTileSetID {
					return transactionResult{}, fmt.Errorf("target activation is absent without the recorded previous identity")
				}
				return transactionResult{
					Operation: "rollback", TileActivationNotObserved: true,
					ActiveTileSetID: activeManifest.TileSetID, PreviousTileSetID: activeManifest.TileSetID,
				}, nil
			}
			if !present {
				return transactionResult{}, fmt.Errorf("target activation is absent without a manifest identity")
			}
			activeManifest, err = e.validateExactPostRollbackTopology(expectedTileSetID)
			if err != nil {
				return transactionResult{}, err
			}
			return attachLegacyMigrationProof(transactionResult{
				Operation: "rollback", TileActivationNotObserved: true,
				ActiveTileSetID: activeManifest.TileSetID, PreviousTileSetID: activeManifest.TileSetID,
			}, activeManifest, expectedTileSetID)
		}
	}
	activeTarget, activeIsLink, err := symlinkTarget(e.activePath())
	if err != nil {
		return transactionResult{}, err
	}
	previousTarget, previousIsLink, err := symlinkTarget(e.previousPath())
	if err != nil {
		return transactionResult{}, err
	}
	if !activeIsLink || !previousIsLink || !isGenerationTarget(activeTarget) || !isGenerationTarget(previousTarget) {
		return transactionResult{}, fmt.Errorf("rollback requires two complete immutable tile generations")
	}
	if activeID, present, err := manifestID(e.activePath()); err != nil || !present || activeID == "" {
		if err != nil {
			return transactionResult{}, err
		}
		return transactionResult{}, fmt.Errorf("active generation manifest is missing")
	}
	previousManifest, present, err := readManifestIdentity(e.previousPath())
	if err != nil || !present || previousManifest.TileSetID == "" {
		if err != nil {
			return transactionResult{}, err
		}
		return transactionResult{}, fmt.Errorf("previous generation manifest is missing")
	}
	previousID := previousManifest.TileSetID
	if expectedPreviousTileSetID != "" && previousID != expectedPreviousTileSetID {
		return transactionResult{}, fmt.Errorf("previous generation tile-set ID differs from recorded rollback identity")
	}
	var previousProof transactionResult
	if expectedPreviousTileSetID == "" {
		if err := verifyLegacyMigrationTree(
			filepath.Join(e.root, previousTarget), previousManifest, expectedTileSetID,
		); err != nil {
			return transactionResult{}, err
		}
		previousProof, err = attachLegacyMigrationProof(
			transactionResult{
				PreviousTileSetID: previousID,
			},
			previousManifest,
			expectedTileSetID,
		)
		if err != nil {
			return transactionResult{}, err
		}
	}

	transaction = tileTransaction{
		Kind: "rollback", ActiveTarget: activeTarget, ActiveID: expectedTileSetID,
		PreviousTarget: previousTarget, PreviousID: previousID,
	}
	if err := e.writeTransaction(transaction); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_journal" {
		return transactionResult{}, errors.New("injected tile rollback failure: after_journal")
	}
	if err := e.requireExactParkedState(); err != nil {
		return transactionResult{}, err
	}
	if err := e.exchange(e.activePath(), e.previousPath()); err != nil {
		return transactionResult{}, fmt.Errorf("renameat2 exchange rollback: %w", err)
	}
	if err := e.syncDir(e.root); err != nil {
		return transactionResult{}, err
	}
	if injectedFailure == "after_switch" {
		return transactionResult{}, errors.New("injected tile rollback failure: after_switch")
	}
	if err := e.clearTransaction(); err != nil {
		return transactionResult{}, err
	}
	id, _, err := manifestID(e.activePath())
	if err != nil {
		return transactionResult{}, err
	}
	result := transactionResult{
		Operation: "rollback", RolledBackTileSetID: id, PreviousTileSetID: id,
	}
	if expectedPreviousTileSetID == "" {
		result.PreviousTileSetProvenance = previousProof.PreviousTileSetProvenance
		result.PreviousTileSetTargetID = previousProof.PreviousTileSetTargetID
	}
	return result, nil
}
