import Foundation
import Testing
@testable import VTSCTunerCore

@Test func releaseArtifactRequiresExactVersionedARM64Identity() throws {
  let directory = temporaryDirectory("vtsc-release")
  defer { try? FileManager.default.removeItem(at: directory) }
  let binary = directory.appendingPathComponent("mapd")
  var bytes = Data(repeating: 0, count: 128)
  bytes.replaceSubrange(0 ..< 6, with: [0x7f, 0x45, 0x4c, 0x46, 2, 1])
  bytes[18] = 183
  bytes[19] = 0
  for marker in [
    "MapdReleaseID:whole-curve-release-v1", "MapdBuildID:build-abc123",
    "MapWholeCurveProfile:whole-curve-v1",
  ] {
    bytes.append(Data(marker.utf8))
    bytes.append(0)
  }
  try bytes.write(to: binary)
  let digest = MapdReleaseArtifact.sha256Hex(bytes)
  let artifact = MapdReleaseArtifact(
    releaseID: "whole-curve-release-v1",
    buildID: "build-abc123",
    binaryURL: binary,
    sha256: digest
  )
  let validated = try artifact.validated()
  #expect(validated.artifact.capability == "MapWholeCurveProfile:whole-curve-v1")
  #expect(validated.byteCount == UInt64(bytes.count))
  #expect(validated.persistentCacheFileName.hasPrefix("mapd-"))
  #expect(validated.persistentCacheFileName.hasSuffix(String(digest.prefix(16))))
  let install = TiciDeploymentCommandBuilder.installReleaseCommand(
    release: validated,
    stagedPath: "/tmp/mapd.partial",
    rollbackPath: "/tmp/mapd.rollback"
  )
  #expect(install.contains("MapdReleaseID:whole-curve-release-v1"))
  #expect(install.contains("MapdBuildID:build-abc123"))
  #expect(install.contains("[str(staged), \"--build-info\"]"))

  var wrongDigest = artifact
  wrongDigest.sha256 = String(repeating: "0", count: 64)
  #expect(throws: MapdReleaseArtifactError.digestMismatch(expected: wrongDigest.sha256, actual: digest)) {
    try wrongDigest.validated()
  }
}

@Test func tuneDeploymentIdentityUsesExactSourceRoundedPhysicsAndQ() {
  let tune = Tune(params: .checkoutFallback)
  let identity = TuneDeploymentIdentity(tune: tune)
  #expect(identity.physics.map(\.value) == [
    "-1.658965", "-1395.055546", "0.005397", "4.107103", "2.4481", "4.1071",
  ])
  #expect(!identity.qCurveEnabled)
  #expect(identity.qCurvePointCount == 0)
  #expect(identity.tileSigmoidHash == "22bd4f8d9bae")
  #expect(identity.identitySHA256.count == 64)
}

@Test func canonicalManifestFullDecodesEveryExactHashedTile() async throws {
  let root = temporaryDirectory("vtsc-tiles")
  defer { try? FileManager.default.removeItem(at: root) }
  let region = root.appendingPathComponent("offline/32/-118", isDirectory: true)
  try FileManager.default.createDirectory(at: region, withIntermediateDirectories: true)
  let tileURL = region.appendingPathComponent("32.000000_-118.000000_32.250000_-117.750000")
  try Data(repeating: 0x5a, count: 256).write(to: tileURL)
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  let entry = TileSetManifest.FileEntry(
    path: "offline/32/-118/\(tileURL.lastPathComponent)",
    byteCount: 256,
    sha256: try FileSHA256.hex(tileURL)
  )
  let manifest = TileSetManifest(
    tileSchemaVersion: 1,
    estimatorVersion: "whole-curve-v1",
    tuneIdentitySHA256: identity.identitySHA256,
    tileSigmoidHash: identity.tileSigmoidHash,
    mapdReleaseID: "whole-curve-release-v1",
    mapdBuildID: "build-abc123",
    mapdBinarySHA256: String(repeating: "a", count: 64),
    pbfSHA256: String(repeating: "b", count: 64),
    pbfDate: "2026-07-13",
    regions: [RegionBox(minLatitude: 32, minLongitude: -118)],
    files: [entry]
  )
  try manifest.write(to: root)
  let decoder = ManifestTileDecoder(sigmoidHash: identity.tileSigmoidHash)
  let validated = try await TileSetValidator(decoder: decoder, batchSize: 1).validate(
    CanonicalTileSetArtifact(rootURL: root, manifest: manifest)
  )
  #expect(validated.manifest.fileCount == 1)
  #expect(await decoder.decodedURLs == [tileURL.standardizedFileURL])

  try Data(repeating: 0x33, count: 256).write(to: tileURL)
  do {
    _ = try await TileSetValidator(decoder: decoder).validate(validated)
    Issue.record("corrupt tile unexpectedly passed its manifest")
  } catch let error as TileSetError {
    guard case .fileDigestMismatch = error else {
      Issue.record("unexpected validation error: \(error)")
      return
    }
  }
}

@Test func tileDeploymentOnlyRsyncsToStagingAndUsesRenameExchange() async throws {
  let root = temporaryDirectory("vtsc-stage")
  defer { try? FileManager.default.removeItem(at: root) }
  let offline = root.appendingPathComponent("offline/32/-118", isDirectory: true)
  try FileManager.default.createDirectory(at: offline, withIntermediateDirectories: true)
  let tile = offline.appendingPathComponent("32.000000_-118.000000_32.250000_-117.750000")
  try Data(repeating: 0x7a, count: 128).write(to: tile)
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  let manifest = TileSetManifest(
    tileSchemaVersion: 1,
    estimatorVersion: "whole-curve-v1",
    tuneIdentitySHA256: identity.identitySHA256,
    tileSigmoidHash: identity.tileSigmoidHash,
    mapdReleaseID: "release-v1",
    mapdBuildID: "build-v1",
    mapdBinarySHA256: String(repeating: "c", count: 64),
    pbfSHA256: String(repeating: "d", count: 64),
    pbfDate: "2026-07-13",
    regions: [RegionBox(minLatitude: 32, minLongitude: -118)],
    files: [TileSetManifest.FileEntry(
      path: "offline/32/-118/\(tile.lastPathComponent)",
      byteCount: 128,
      sha256: try FileSHA256.hex(tile)
    )]
  )
  try manifest.write(to: root)
  let runner = DeploymentRecordingRunner(identity: manifest.tileSetID)
  let service = TiciTileSetDeploymentService(processRunner: runner)
  let staged = try await service.stageAndVerify(
    artifact: CanonicalTileSetArtifact(rootURL: root, manifest: manifest),
    profile: "commaAdb"
  )
  _ = try await service.activate(staged)
  let requests = await runner.requests
  let rsyncRequests = requests.filter { $0.executableURL == TiciTileSetDeploymentService.rsyncURL }
  #expect(rsyncRequests.count == 2)
  #expect(rsyncRequests[0].arguments.contains("--delete"))
  #expect(rsyncRequests[0].arguments.last?.contains(".tileset-\(manifest.tileSetID).partial/offline/") == true)
  #expect(!rsyncRequests.flatMap(\.arguments).contains("commaAdb:/data/media/0/osm/offline/"))
  let activation = try #require(requests.last?.arguments.last)
  #expect(activation.contains("renameat2"))
  #expect(activation.contains("os.fsencode(active), 2"))
  #expect(TiciTileSetDeploymentService.atomicRollbackCommand().contains("renameat2"))
}

@Test func immutableTileGenerationRecoversEveryInjectedSwitchBoundary() {
  let previous = "tile-generations/old/offline"
  let next = "tile-generations/new/offline"
  #expect(TileTransactionRecovery.activation(
    pendingID: "new", expectedID: "new", activeTarget: previous,
    newTarget: next, previousTarget: previous
  ) == .restartBeforeSwitch)
  #expect(TileTransactionRecovery.activation(
    pendingID: "new", expectedID: "new", activeTarget: next,
    newTarget: next, previousTarget: previous
  ) == .finishAfterSwitch)
  #expect(TileTransactionRecovery.activation(
    pendingID: "new", expectedID: "new", activeTarget: "unexpected",
    newTarget: next, previousTarget: previous
  ) == .manualRecoveryRequired)
  #expect(TileTransactionRecovery.rollback(
    activeTarget: previous, previousTarget: next,
    recordedActiveTarget: next, recordedPreviousTarget: previous
  ) == .alreadyRolledBack)
  #expect(TileTransactionRecovery.rollback(
    activeTarget: next, previousTarget: previous,
    recordedActiveTarget: next, recordedPreviousTarget: previous
  ) == .performRollback)

  let activation = TiciTileSetDeploymentService.atomicActivationCommand(
    stagingRoot: "/data/media/0/osm/.tileset-new.partial",
    tileSetID: "new",
    injectedFailurePoint: "after_switch"
  )
  #expect(activation.contains("injected tile activation failure"))
  #expect(activation.contains(".tileset-manifest.json"))
  #expect(activation.contains("tile-generations"))
  #expect(activation.contains(".tileset-transaction.json"))
  #expect(activation.contains("fsync_dir(root)"))
  #expect(!activation.contains("offline.manifest.previous.json"))
  let rollback = TiciTileSetDeploymentService.atomicRollbackCommand(
    expectedActivatedTileSetID: "new",
    injectedFailurePoint: "after_switch"
  )
  #expect(rollback.contains("injected tile rollback failure"))
  #expect(rollback.contains("manifest_id(active)"))
  #expect(rollback.contains("pending.get(\"kind\") == \"activation\""))
  #expect(rollback.contains("tile_activation_not_observed"))
}

@Test func productionCommandsPinSafetyBranchAndExactCommit() {
  let fastForward = TiciDeploymentCommandBuilder.exactFastForwardCommand(
    branch: "chauffeur-exp01",
    head: String(repeating: "a", count: 40)
  )
  #expect(fastForward.contains("git fetch --no-tags origin refs/heads/chauffeur-exp01"))
  #expect(fastForward.contains("git merge --ff-only"))
  #expect(fastForward.contains("git status --porcelain"))
  let inspection = TiciDeploymentCommandBuilder.preflightInspectionCommand()
  #expect(inspection.contains("IsOffroad"))
  #expect(inspection.contains("MTSCLookaheadEnabled"))
  #expect(inspection.contains("q_curve_sha256"))
  let suite = ProductionVerificationSuite.requests(repositoryRoot: URL(fileURLWithPath: "/repo"))
  let go = suite.first { $0.executableURL.lastPathComponent == "go" }
  #expect(go?.executableURL.path == "/repo/tools/vtsc_tuner_mac/.build-tools/go-1.26.5-darwin-arm64/bin/go")
  #expect(go?.environment["CGO_ENABLED"] == "0")
}

@Test func generatedRemotePythonCommandsAreSyntacticallyValid() async throws {
  let artifact = MapdReleaseArtifact(
    releaseID: "release-v1",
    buildID: "build-v1",
    binaryURL: URL(fileURLWithPath: "/tmp/mapd"),
    sha256: String(repeating: "a", count: 64)
  )
  let release = ValidatedMapdReleaseArtifact(
    artifact: artifact,
    byteCount: 1_000_000,
    persistentCacheFileName: "mapd-version-artifact"
  )
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  let journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "b", count: 40),
    previousPhysicsParams: [:],
    previousQCurveSHA256: String(repeating: "c", count: 64),
    previousMapdReleaseVersion: "old-release",
    previousMapdVersion: "old-release",
    previousActiveMapdSHA256: String(repeating: "d", count: 64),
    previousCachedMapdPath: "/tmp/old-cache",
    mapdRollbackPath: "/tmp/rollback"
  )
  let commands = [
    TiciDeploymentCommandBuilder.preflightInspectionCommand(),
    TiciDeploymentCommandBuilder.installReleaseCommand(
      release: release, stagedPath: "/tmp/staged", rollbackPath: "/tmp/rollback"
    ),
    TiciDeploymentCommandBuilder.qCurveVerificationCommand(identity: identity),
    TiciDeploymentCommandBuilder.postflightInspectionCommand(
      head: String(repeating: "e", count: 40),
      release: release,
      identity: identity,
      expectedTileSetID: nil
    ),
    TiciDeploymentCommandBuilder.rollbackCommand(journal: journal, rollbackMapdPath: "/tmp/rollback"),
    TiciDeploymentCommandBuilder.rollbackVerificationCommand(journal: journal, tilesWereTouched: true),
    TiciTileSetDeploymentService.remoteManifestVerificationCommand(
      stagingRoot: "/tmp/stage", tileSetID: String(repeating: "f", count: 64)
    ),
    TiciTileSetDeploymentService.atomicActivationCommand(
      stagingRoot: "/tmp/stage", tileSetID: String(repeating: "f", count: 64)
    ),
    TiciTileSetDeploymentService.atomicRollbackCommand(),
  ]
  let runner = SystemProcessRunner()
  for command in commands {
    let body = try #require(pythonHeredocBody(command))
    let result = try await runner.run(ProcessRequest(
      executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
      arguments: ["-c", "import ast, sys; ast.parse(sys.argv[1])", body],
      timeout: 10
    ))
    #expect(result.succeeded, Comment(rawValue: result.combinedOutput))
  }
}

@Test func generatedPostflightScriptExecutesWithStubbedRuntimeNames() async throws {
  let directory = temporaryDirectory("vtsc-postflight-script")
  defer { try? FileManager.default.removeItem(at: directory) }
  let qURL = directory.appendingPathComponent(
    "sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py",
    isDirectory: false
  )
  try FileManager.default.createDirectory(
    at: qURL.deletingLastPathComponent(),
    withIntermediateDirectories: true
  )
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  try TuneDeploymentIdentity.canonicalQCurveSource(
    parameters: .checkoutFallback,
    bands: []
  ).write(to: qURL, atomically: true, encoding: .utf8)

  let activeBinary = directory.appendingPathComponent("mapd")
  let cachedBinary = directory.appendingPathComponent("mapd-cache")
  var binaryData = Data(repeating: 0, count: 128)
  binaryData.replaceSubrange(0 ..< 6, with: [0x7f, 0x45, 0x4c, 0x46, 2, 1])
  binaryData[18] = 183
  binaryData.append(Data("MapWholeCurveProfile:whole-curve-v1".utf8))
  try binaryData.write(to: activeBinary)
  try binaryData.write(to: cachedBinary)
  let digest = MapdReleaseArtifact.sha256Hex(binaryData)
  let artifact = MapdReleaseArtifact(
    releaseID: "release-v1",
    buildID: "build-v1",
    binaryURL: activeBinary,
    sha256: digest
  )
  let release = ValidatedMapdReleaseArtifact(
    artifact: artifact,
    byteCount: UInt64(binaryData.count),
    persistentCacheFileName: "mapd-version-artifact"
  )
  let head = String(repeating: "e", count: 40)
  let command = TiciDeploymentCommandBuilder.postflightInspectionCommand(
    head: head,
    release: release,
    identity: identity,
    expectedTileSetID: nil
  )
  var body = try #require(pythonHeredocBody(command))
  let physics = Dictionary(uniqueKeysWithValues: identity.physics.map { ($0.paramKey, $0.value) })
  let physicsJSON = String(data: try JSONEncoder().encode(physics), encoding: .utf8)!
  let profile: [String: Any] = [
    "estimatorVersion": "whole-curve-v1",
    "generatedAtUnixMillis": Date().timeIntervalSince1970 * 1_000,
    "generation": 1,
    "routeFingerprint": String(repeating: "f", count: 64),
    "fatalAmbiguity": false,
    "points": [
      ["latitude": 34.0, "longitude": -118.0, "distanceMeters": 0.0, "curvature": 0.0],
      ["latitude": 34.0001, "longitude": -118.0, "distanceMeters": 11.1, "curvature": 0.0],
      ["latitude": 34.0002, "longitude": -118.0, "distanceMeters": 22.2, "curvature": 0.0],
    ],
    "events": [],
  ]
  let profileJSON = String(data: try JSONSerialization.data(withJSONObject: profile), encoding: .utf8)!
  let paramsStub = """
  class Params:
    values = json.loads(\(pythonTestLiteral(physicsJSON)))
    values.update({"MapdReleaseVersion": "release-v1", "MapdVersion": "release-v1"})
    profile = json.loads(\(pythonTestLiteral(profileJSON)))
    gps = {"latitude": 34.0, "longitude": -118.0, "bearing": 90.0}
    def __init__(self, *_args): pass
    def get(self, key):
      if key == "MapWholeCurveProfile": return json.dumps(self.profile).encode()
      if key == "LastGPSPosition": return json.dumps(self.gps).encode()
      value = self.values.get(key)
      return value.encode() if value is not None else None
    def get_bool(self, key):
      return key == "IsOffroad"
  """
  let runtimeStub = """
  class VisionTurnController:
    @classmethod
    def _parse_map_whole_curve_profile(cls, raw, gps_pose):
      payload = json.loads(raw)
      if len(payload.get("points", [])) < 3 or len(gps_pose) != 3: raise ValueError("invalid profile")
      return payload
  class _RunResult:
    returncode = 0
  def _stub_check_output(arguments, **_kwargs):
    if arguments[0] == "git": return \(pythonTestLiteral(head + "\n"))
    return json.dumps({
      "releaseID": "release-v1",
      "buildID": "build-v1",
      "estimatorVersion": "whole-curve-v1",
      "capabilities": ["MapWholeCurveProfile:whole-curve-v1"],
    })
  subprocess.check_output = _stub_check_output
  subprocess.run = lambda *_args, **_kwargs: _RunResult()
  """
  body = body.replacingOccurrences(
    of: "from openpilot.common.params import Params",
    with: paramsStub
  )
  body = body.replacingOccurrences(
    of: "from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController",
    with: runtimeStub
  )
  body = body.replacingOccurrences(
    of: "/data/openpilot/third_party/mapd/mapd",
    with: activeBinary.path
  )
  body = body.replacingOccurrences(
    of: "/data/media/0/osm/binaries/mapd-version-artifact",
    with: cachedBinary.path
  )
  body = body.replacingOccurrences(
    of: "/data/media/0/osm/offline.manifest.json",
    with: directory.appendingPathComponent("missing-manifest.json").path
  )
  let result = try await SystemProcessRunner().run(ProcessRequest(
    executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
    arguments: ["-c", body],
    currentDirectoryURL: directory,
    timeout: 20
  ))
  #expect(result.succeeded, Comment(rawValue: result.combinedOutput))
  let decoded = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
    TiciDeploymentPostflight.self,
    output: result.standardOutput
  )
  #expect(decoded.profileFresh)
  #expect(decoded.gpsStatus == "valid")
  #expect(decoded.profileValidationStatus == "valid")
  #expect(decoded.profilePointCount == 3)
  #expect(decoded.profileEventCount == 0)
  #expect(decoded.activeELFARM64)
  #expect(decoded.buildInfoMatches)

  let pendingGPSBody = body.replacingOccurrences(
    of: "if key == \"LastGPSPosition\": return json.dumps(self.gps).encode()",
    with: "if key == \"LastGPSPosition\": return None"
  )
  let pendingResult = try await SystemProcessRunner().run(ProcessRequest(
    executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
    arguments: ["-c", pendingGPSBody],
    currentDirectoryURL: directory,
    timeout: 20
  ))
  #expect(pendingResult.succeeded, Comment(rawValue: pendingResult.combinedOutput))
  let pending = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
    TiciDeploymentPostflight.self,
    output: pendingResult.standardOutput
  )
  #expect(pending.gpsStatus == "pending")
  #expect(pending.profileValidationStatus == "pending_real_gps")
  #expect(!pending.profileFresh)

  let farGPSBody = body.replacingOccurrences(
    of: "gps = {\"latitude\": 34.0, \"longitude\": -118.0, \"bearing\": 90.0}",
    with: "gps = {\"latitude\": 36.0, \"longitude\": -118.0, \"bearing\": 90.0}"
  )
  let farResult = try await SystemProcessRunner().run(ProcessRequest(
    executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
    arguments: ["-c", farGPSBody],
    currentDirectoryURL: directory,
    timeout: 20
  ))
  #expect(farResult.succeeded, Comment(rawValue: farResult.combinedOutput))
  let far = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
    TiciDeploymentPostflight.self,
    output: farResult.standardOutput
  )
  #expect(far.gpsStatus == "valid")
  #expect(far.profileValidationStatus.contains("route_not_near_real_gps"))
  #expect(!far.profileFresh)
}

@Test func generatedRollbackVerificationChecksExactRestoredIdentity() async throws {
  let directory = temporaryDirectory("vtsc-rollback-verification")
  defer { try? FileManager.default.removeItem(at: directory) }
  let qURL = directory.appendingPathComponent(
    "sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py",
    isDirectory: false
  )
  try FileManager.default.createDirectory(at: qURL.deletingLastPathComponent(), withIntermediateDirectories: true)
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  try TuneDeploymentIdentity.canonicalQCurveSource(parameters: .checkoutFallback, bands: [])
    .write(to: qURL, atomically: true, encoding: .utf8)
  let active = directory.appendingPathComponent("mapd")
  let cache = directory.appendingPathComponent("mapd-cache")
  let binary = Data(repeating: 0x7a, count: 256)
  try binary.write(to: active)
  try binary.write(to: cache)
  let digest = MapdReleaseArtifact.sha256Hex(binary)
  let head = String(repeating: "b", count: 40)
  let physics = Dictionary(uniqueKeysWithValues: identity.physics.map { ($0.paramKey, Optional($0.value)) })
  let journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: head,
    previousPhysicsParams: physics,
    previousQCurveSHA256: identity.qCurveSHA256,
    previousMapdReleaseVersion: nil,
    previousMapdVersion: "old-release",
    previousActiveMapdSHA256: digest,
    previousCachedMapdPath: cache.path,
    previousCachedMapdSHA256: digest,
    mapdRollbackPath: "/tmp/rollback"
  )
  var body = try #require(pythonHeredocBody(
    TiciDeploymentCommandBuilder.rollbackVerificationCommand(
      journal: journal,
      tilesWereTouched: false
    )
  ))
  let physicsJSON = String(data: try JSONEncoder().encode(physics), encoding: .utf8)!
  let paramsStub = """
  class Params:
    values = json.loads(\(pythonTestLiteral(physicsJSON)))
    values["MapdVersion"] = "old-release"
    def get(self, key):
      value = self.values.get(key)
      return value.encode() if value is not None else None
    def get_bool(self, key): return key == "IsOffroad"
  """
  let processStub = """
  class _RunResult:
    returncode = 0
  def _stub_check_output(arguments, **_kwargs):
    if arguments[:3] == ["git", "rev-parse", "HEAD"]: return \(pythonTestLiteral(head + "\n"))
    if arguments[:3] == ["git", "status", "--porcelain"]: return ""
    raise RuntimeError(arguments)
  subprocess.check_output = _stub_check_output
  subprocess.run = lambda *_args, **_kwargs: _RunResult()
  """
  body = body.replacingOccurrences(of: "from openpilot.common.params import Params", with: paramsStub)
  body = body.replacingOccurrences(
    of: "params = Params()",
    with: "params = Params()\n" + processStub
  )
  body = body.replacingOccurrences(of: "/data/openpilot/third_party/mapd/mapd", with: active.path)
  let runner = SystemProcessRunner()
  let request = ProcessRequest(
    executableURL: URL(fileURLWithPath: "/usr/bin/python3"),
    arguments: ["-c", body],
    currentDirectoryURL: directory,
    timeout: 20
  )
  let result = try await runner.run(request)
  #expect(result.succeeded, Comment(rawValue: result.combinedOutput))
  #expect(result.standardOutput.contains("\"rollback_verified\": true"))

  try Data(repeating: 0x33, count: 256).write(to: active)
  let corrupt = try await runner.run(request)
  #expect(!corrupt.succeeded)
  #expect(corrupt.combinedOutput.contains("active_mapd"))
}

@Test func onroadPreflightStopsBeforeTuneOrSourceMutation() async throws {
  let repository = temporaryDirectory("vtsc-onroad-preflight")
  defer { try? FileManager.default.removeItem(at: repository) }
  try FileManager.default.createDirectory(at: repository.appendingPathComponent(".git"), withIntermediateDirectories: true)
  for relativePath in [
    RepositoryLocator.physicsRelativePath,
    RepositoryLocator.qCurveRelativePath,
    RepositoryLocator.paramsDefaultsRelativePath,
    RepositoryLocator.physicsPanelRelativePath,
  ] {
    let url = repository.appendingPathComponent(relativePath)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let entries = VTSCPhysicsAuthority.entries(for: .checkoutFallback)
    let contents: String
    switch relativePath {
    case RepositoryLocator.physicsRelativePath:
      contents = entries.map { "\($0.moduleName) = \($0.formattedValue)" }.joined(separator: "\n") + "\n"
    case RepositoryLocator.qCurveRelativePath:
      contents = TuneDeploymentIdentity.canonicalQCurveSource(parameters: .checkoutFallback, bands: [])
    case RepositoryLocator.paramsDefaultsRelativePath:
      contents = entries.map {
        "{\"\($0.paramKey)\", {PERSISTENT | BACKUP, FLOAT, \"\($0.formattedValue)\"}},"
      }.joined(separator: "\n") + "\n"
    default:
      contents = entries.flatMap {
        [
          "params.put(\"\($0.paramKey)\", \"\($0.formattedValue)\");",
          "ensure(\"\($0.paramKey)\", \"\($0.formattedValue)\");",
        ]
      }.joined(separator: "\n") + "\n"
    }
    try contents.write(to: url, atomically: true, encoding: .utf8)
  }
  let binary = repository.appendingPathComponent("mapd-arm64")
  var binaryData = Data(repeating: 0, count: 128)
  binaryData.replaceSubrange(0 ..< 6, with: [0x7f, 0x45, 0x4c, 0x46, 2, 1])
  binaryData[18] = 183
  for marker in [
    "MapdReleaseID:release-v1",
    "MapdBuildID:build-v1",
    "MapWholeCurveProfile:whole-curve-v1",
  ] { binaryData.append(Data(marker.utf8)) }
  try binaryData.write(to: binary)
  let releaseURL = repository.appendingPathComponent("mapd-release.json")
  let release = MapdReleaseArtifact(
    releaseID: "release-v1",
    buildID: "build-v1",
    binaryURL: binary,
    sha256: MapdReleaseArtifact.sha256Hex(binaryData)
  )
  try JSONEncoder().encode(release).write(to: releaseURL)
  let tuneURL = repository.appendingPathComponent("must-not-be-written.json")
  let runner = OnroadPreflightRunner()
  let events = DeploymentEventCollector()
  let succeeded = await ApplyPipeline(processRunner: runner).apply(ApplyRequest(
    action: .pullOnTici,
    tune: Tune(params: .checkoutFallback),
    repositoryRoot: repository,
    tuneURL: tuneURL,
    preferredTiciProfile: "commaAdb",
    mapdReleaseManifestURL: releaseURL,
    verificationRequests: []
  )) { event in
    await events.append(event)
  }
  #expect(!succeeded)
  #expect(!FileManager.default.fileExists(atPath: tuneURL.path))
  #expect(!(await events.events).contains { event in
    if case let .step(step) = event { step.id == 1 || step.id == 2 } else { false }
  })
  #expect((await runner.requests).contains { request in
    request.arguments.last?.contains("IsOffroad") == true
  })
  #expect(!(await runner.requests).contains { request in
    request.executableURL == ApplyPipeline.gitURL && request.arguments.first == "fetch"
  })
}

@Test func rollbackRefusesAnyMutationWhenFreshParkedStateIsUnsafe() async throws {
  let runner = RollbackSafetyRunner(states: [
    .init(isOffroad: false, isOnroad: true, mapLookaheadEnabled: false),
  ])
  let pipeline = ApplyPipeline(processRunner: runner)
  let preflight = makeRollbackPreflight(rebootSent: true)
  let result = await pipeline.rollbackProductionDeployment(preflight: preflight, tilesActivated: false)
  #expect(!result.success)
  #expect(result.detail.contains("journal retained"))
  let requests = await runner.requests
  #expect(requests.count == 1)
  #expect(!requests.contains { $0.arguments.last?.contains("rolled_back_head") == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func rollbackRechecksStateAndSuppressesRebootIfCarStateChanges() async throws {
  let runner = RollbackSafetyRunner(states: [
    .init(isOffroad: true, isOnroad: false, mapLookaheadEnabled: false),
    .init(isOffroad: false, isOnroad: true, mapLookaheadEnabled: false),
  ])
  let pipeline = ApplyPipeline(processRunner: runner)
  let result = await pipeline.rollbackProductionDeployment(
    preflight: makeRollbackPreflight(rebootSent: true),
    tilesActivated: false
  )
  #expect(!result.success)
  #expect(result.detail.contains("rollback reboot or post-rollback verification failed"))
  let requests = await runner.requests
  #expect(requests.contains { $0.arguments.last?.contains("rolled_back_head") == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

private func temporaryDirectory(_ prefix: String) -> URL {
  let url = FileManager.default.temporaryDirectory
    .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
  try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  return url
}

private func pythonHeredocBody(_ command: String) -> String? {
  guard let start = command.range(of: "<<'PY'\n"),
        let end = command.range(of: "\nPY", options: .backwards),
        start.upperBound <= end.lowerBound
  else { return nil }
  return String(command[start.upperBound ..< end.lowerBound])
}

private func pythonTestLiteral(_ value: String) -> String {
  "'" + value.replacingOccurrences(of: "\\", with: "\\\\")
    .replacingOccurrences(of: "'", with: "\\'")
    .replacingOccurrences(of: "\n", with: "\\n") + "'"
}

private func makeRollbackPreflight(rebootSent: Bool) -> RuntimeDeploymentPreflight {
  let artifact = MapdReleaseArtifact(
    releaseID: "release-v1",
    buildID: "build-v1",
    binaryURL: URL(fileURLWithPath: "/tmp/mapd"),
    sha256: String(repeating: "a", count: 64)
  )
  var journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "b", count: 40),
    previousPhysicsParams: [:],
    previousQCurveSHA256: String(repeating: "c", count: 64),
    previousMapdReleaseVersion: "old-release",
    previousMapdVersion: "old-release",
    previousActiveMapdSHA256: String(repeating: "d", count: 64),
    previousCachedMapdPath: "/tmp/old-cache",
    mapdRollbackPath: "/tmp/rollback"
  )
  journal.rebootSent = rebootSent
  return RuntimeDeploymentPreflight(
    git: GitDeploymentPreflight(
      branch: "chauffeur-exp01",
      localHead: String(repeating: "b", count: 40),
      originHead: String(repeating: "b", count: 40),
      upstream: "origin/chauffeur-exp01"
    ),
    profile: "commaAdb",
    release: ValidatedMapdReleaseArtifact(
      artifact: artifact,
      byteCount: 1_000_000,
      persistentCacheFileName: "mapd-version-artifact"
    ),
    tileSet: nil,
    journal: journal,
    journalURL: URL(fileURLWithPath: "/tmp/rollback-journal.json")
  )
}

private actor ManifestTileDecoder: MapTileDecoding {
  let sigmoidHash: String
  private(set) var decodedURLs: [URL] = []

  init(sigmoidHash: String) { self.sigmoidHash = sigmoidHash }

  func decode(fileURLs: [URL]) async throws -> [MapTile] {
    decodedURLs += fileURLs.map(\.standardizedFileURL)
    return fileURLs.map { url in
      MapTile(
        sourcePath: url.standardizedFileURL.path,
        bounds: MapTileIndex.parseBounds(filename: url.lastPathComponent)!,
        overlap: 0.01,
        schemaVersion: 1,
        sigmoidHash: sigmoidHash,
        ways: []
      )
    }
  }
}

private actor DeploymentRecordingRunner: ProcessRunning {
  let identity: String
  private(set) var requests: [ProcessRequest] = []

  init(identity: String) { self.identity = identity }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    return ProcessResult(terminationStatus: 0, standardOutput: identity, standardError: "")
  }
}

private actor DeploymentEventCollector {
  private(set) var events: [ApplyEvent] = []
  func append(_ event: ApplyEvent) { events.append(event) }
}

private actor OnroadPreflightRunner: ProcessRunning {
  private(set) var requests: [ProcessRequest] = []
  private let head = String(repeating: "a", count: 40)

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == ApplyPipeline.gitURL {
      let arguments = request.arguments
      if arguments == ["branch", "--show-current"] {
        return success("chauffeur-exp01\n")
      }
      if arguments == ["status", "--porcelain"] { return success("") }
      if arguments == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"] {
        return success("origin/chauffeur-exp01\n")
      }
      if arguments == ["rev-parse", "HEAD"] {
        return success(head + "\n")
      }
      if arguments == ["ls-remote", "--heads", "origin", "refs/heads/chauffeur-exp01"] {
        return success("\(head)\trefs/heads/chauffeur-exp01\n")
      }
      return success("")
    }
    if request.executableURL == ApplyPipeline.networkSetupURL {
      return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unavailable")
    }
    if request.executableURL == ApplyPipeline.sshURL {
      if request.arguments.last == "true" { return success("") }
      let snapshot: [String: Any] = [
        "is_offroad": false,
        "is_onroad": true,
        "map_lookahead_enabled": false,
        "branch": "chauffeur-exp01",
        "head": head,
        "dirty": false,
        "physics_params": [:],
        "q_curve_sha256": String(repeating: "b", count: 64),
        "mapd_release_version": "old",
        "mapd_version": "old",
        "active_mapd_sha256": String(repeating: "c", count: 64),
        "cached_mapd_path": "/tmp/cache",
        "active_tile_set_id": NSNull(),
      ]
      return success(String(data: try JSONSerialization.data(withJSONObject: snapshot), encoding: .utf8)! + "\n")
    }
    return success("")
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }
}

private struct RollbackSafetyState: Sendable {
  var isOffroad: Bool
  var isOnroad: Bool
  var mapLookaheadEnabled: Bool
}

private actor RollbackSafetyRunner: ProcessRunning {
  private var states: [RollbackSafetyState]
  private(set) var requests: [ProcessRequest] = []

  init(states: [RollbackSafetyState]) { self.states = states }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if command.contains("\"is_offroad\"") {
      let state = states.isEmpty
        ? RollbackSafetyState(isOffroad: false, isOnroad: true, mapLookaheadEnabled: false)
        : states.removeFirst()
      let snapshot: [String: Any] = [
        "is_offroad": state.isOffroad,
        "is_onroad": state.isOnroad,
        "map_lookahead_enabled": state.mapLookaheadEnabled,
        "branch": "chauffeur-exp01",
        "head": String(repeating: "b", count: 40),
        "dirty": false,
        "physics_params": [:],
        "q_curve_sha256": String(repeating: "c", count: 64),
        "mapd_release_version": "old-release",
        "mapd_version": "old-release",
        "active_mapd_sha256": String(repeating: "d", count: 64),
        "cached_mapd_path": "/tmp/old-cache",
        "active_tile_set_id": NSNull(),
      ]
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: String(data: try JSONSerialization.data(withJSONObject: snapshot), encoding: .utf8)! + "\n",
        standardError: ""
      )
    }
    return ProcessResult(terminationStatus: 0, standardOutput: "ok\n", standardError: "")
  }
}
