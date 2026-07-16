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
    "MapWholeCurveProfile:whole-curve-v2",
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
  #expect(validated.artifact.capability == "MapWholeCurveProfile:whole-curve-v2")
  #expect(validated.byteCount == UInt64(bytes.count))
  #expect(validated.persistentCacheFileName.hasPrefix("mapd-"))
  #expect(validated.persistentCacheFileName.hasSuffix(String(digest.prefix(16))))
  let install = try TiciMapdReleaseTransactionCommandBuilder.installCommand(
    release: validated,
    stagedPath: "/data/media/0/osm/binaries/.mapd-release-01234567-89ab-cdef-0123-456789abcdef.partial",
    rollbackPath: "/data/media/0/osm/binaries/mapd-rollback-01234567-89ab-cdef-0123-456789abcdef"
  )
  #expect(install.contains("MapdReleaseID:whole-curve-release-v1"))
  #expect(install.contains("MapdBuildID:build-abc123"))
  #expect(install.contains("\"$staged\" --build-info"))
  #expect(!install.lowercased().contains("python"))

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
    estimatorVersion: "whole-curve-v2",
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

@Test func tileDeploymentOnlyRsyncsToStagingAndUsesTheVerifiedNativeExchangeHelper() async throws {
  let root = temporaryDirectory("vtsc-stage")
  defer { try? FileManager.default.removeItem(at: root) }
  let offline = root.appendingPathComponent("offline/32/-118", isDirectory: true)
  try FileManager.default.createDirectory(at: offline, withIntermediateDirectories: true)
  let tile = offline.appendingPathComponent("32.000000_-118.000000_32.250000_-117.750000")
  try Data(repeating: 0x7a, count: 128).write(to: tile)
  let identity = TuneDeploymentIdentity(tune: Tune(params: .checkoutFallback))
  let manifest = TileSetManifest(
    tileSchemaVersion: 1,
    estimatorVersion: "whole-curve-v2",
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
  let runner = DeploymentRecordingRunner(
    identity: manifest.tileSetID,
    fileCount: manifest.fileCount,
    totalBytes: manifest.totalBytes
  )
  let helper = try transactionHelperFixture(in: root)
  let service = TiciTileSetDeploymentService(
    processRunner: runner,
    transactionHelperURL: helper
  )
  let staged = try await service.stageAndVerify(
    artifact: CanonicalTileSetArtifact(rootURL: root, manifest: manifest),
    profile: "commaAdb"
  )
  _ = try await service.activate(staged)
  let requests = await runner.requests
  let rsyncRequests = requests.filter { $0.executableURL == TiciTileSetDeploymentService.rsyncURL }
  #expect(rsyncRequests.count == 3)
  #expect(rsyncRequests[0].arguments.contains("--delete"))
  #expect(rsyncRequests[0].arguments.last?.contains(".tileset-\(manifest.tileSetID).partial/offline/") == true)
  #expect(!rsyncRequests.flatMap(\.arguments).contains("commaAdb:/data/media/0/osm/offline/"))
  let activation = try #require(requests.last?.arguments.last)
  #expect(activation.contains("vtsc-tile-transaction-"))
  #expect(activation.contains(" activate --root "))
  #expect(activation.contains("MTSCLookaheadEnabled"))
  #expect(!activation.lowercased().contains("python"))
}

@Test func immutableTileGenerationRecoversEveryInjectedSwitchBoundary() throws {
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

  let tileSetID = String(repeating: "f", count: 64)
  let helperPath = "/data/media/0/osm/binaries/vtsc-tile-transaction-\(String(repeating: "a", count: 16))"
  let activation = try TiciTileSetDeploymentService.atomicActivationCommand(
    helperPath: helperPath,
    stagingRoot: TiciTileSetDeploymentService.stagingRoot(tileSetID: tileSetID),
    tileSetID: tileSetID,
    injectedFailurePoint: "after_switch"
  )
  #expect(activation.contains("vtsc-tile-transaction-"))
  #expect(activation.contains("--inject-failure 'after_switch'"))
  #expect(activation.contains("refusing tile mutation unless tici is offroad"))
  let rollback = try TiciTileSetDeploymentService.atomicRollbackCommand(
    helperPath: helperPath,
    expectedActivatedTileSetID: tileSetID,
    injectedFailurePoint: "after_switch"
  )
  #expect(rollback.contains(" rollback --root "))
  #expect(rollback.contains("--expected-tile-set-id '\(tileSetID)'"))
  #expect(rollback.contains("--inject-failure 'after_switch'"))
  #expect(!rollback.lowercased().contains("python"))
}

@Test func swiftOwnedDeploymentCommandsKeepPinnedSafetyContracts() throws {
  let head = String(repeating: "a", count: 40)
  let fastForward = try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
    branch: "chauffeur-exp01",
    head: head
  )
  let snapshotCommand = TiciSnapshotWireCommandBuilder.inspectionCommand(includeRuntimePostflight: true)
  let mapdRecovery = TiciMapdReleaseTransactionCommandBuilder.recoveryCommand()

  let releaseID = "release-v1"
  let releaseSHA = String(repeating: "a", count: 64)
  let cacheName = "mapd-" + String(MapdReleaseArtifact.sha256Hex(Data(releaseID.utf8)).prefix(16))
    + "-" + String(releaseSHA.prefix(16))
  let release = ValidatedMapdReleaseArtifact(
    artifact: MapdReleaseArtifact(
      releaseID: releaseID,
      buildID: "build-v1",
      binaryURL: URL(fileURLWithPath: "/tmp/mapd"),
      sha256: releaseSHA
    ),
    byteCount: 1,
    persistentCacheFileName: cacheName
  )
  let stagedPath = "/data/media/0/osm/binaries/.mapd-release-v1-0123456789.partial"
  let rollbackPath = "/data/media/0/osm/binaries/mapd-rollback-v1-0123456789"
  let probe = try TiciMapdReleaseTransactionCommandBuilder.probeCommand(
    release: release,
    stagedPath: stagedPath
  )
  let install = try TiciMapdReleaseTransactionCommandBuilder.installCommand(
    release: release,
    stagedPath: stagedPath,
    rollbackPath: rollbackPath
  )
  let physics = Dictionary(uniqueKeysWithValues: VTSCPhysicsAuthority.entries(for: .checkoutFallback).map {
    ($0.paramKey, Optional.some($0.formattedValue))
  })
  let journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "b", count: 40),
    previousPhysicsParams: physics,
    previousQCurveSHA256: String(repeating: "c", count: 64),
    previousMapdReleaseVersion: "old-release",
    previousMapdVersion: "old-release",
    previousActiveMapdSHA256: String(repeating: "d", count: 64),
    previousCachedMapdPath: "/data/media/0/osm/binaries/mapd-old",
    mapdRollbackPath: rollbackPath
  )
  let parameterTransaction = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(
    parameters: .checkoutFallback
  )
  let rollback = try TiciProductionRollbackCommandBuilder.command(journal: journal)

  let commands = [fastForward, snapshotCommand, mapdRecovery, probe, install, parameterTransaction, rollback]
  for command in commands {
    #expect(!command.lowercased().contains("python"))
    #expect(!command.contains("openpilot.common.params"))
  }

  #expect(fastForward.contains("git fetch --no-tags origin refs/heads/chauffeur-exp01"))
  #expect(fastForward.contains("git merge --ff-only"))
  #expect(fastForward.contains("git status --porcelain"))
  #expect(snapshotCommand.contains("/bin/sh <<'VTSC_SNAPSHOT_SH'"))
  #expect(snapshotCommand.contains(TiciSnapshotWireField.isOffroad.rawValue))
  #expect(snapshotCommand.contains(TiciSnapshotWireField.qCurveFile.rawValue))
  #expect(snapshotCommand.contains(TiciSnapshotWireField.activeMapdBuildInfo.rawValue))
  #expect(mapdRecovery.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker))
  #expect(mapdRecovery.contains(TiciMapdReleaseTransactionCommandBuilder.transactionDirectory))
  #expect(mapdRecovery.contains("flock -x 8"))
  #expect(probe.contains(TiciMapdReleaseTransactionCommandBuilder.probeMarker))
  #expect(install.contains(TiciMapdReleaseTransactionCommandBuilder.resultMarker))
  #expect(install.contains("/data/params/.lock"))
  #expect(parameterTransaction.contains("VisionTurnSpeedControlPhysicsMaxLatAccel"))
  #expect(parameterTransaction.contains("flock -x 9"))
  #expect(rollback.contains(TiciProductionRollbackCommandBuilder.resultMarker))
  #expect(rollback.contains("git reset --hard"))

  let encoded = TiciSnapshotWireCodec.encode(.init(rawValues: [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data(head.utf8),
    .dirty: Data("0".utf8),
  ]))
  let decoded = try TiciSnapshotWireCodec.decode(encoded)
  #expect(decoded.text(for: .branch) == "chauffeur-exp01")
  #expect(decoded.text(for: .head) == head)

  let suite = ProductionVerificationSuite.requests(repositoryRoot: URL(fileURLWithPath: "/repo"))
  let go = suite.first { $0.executableURL.lastPathComponent == "go" }
  #expect(go?.executableURL.path == "/repo/tools/vtsc_tuner_mac/.build-tools/go-1.26.5-darwin-arm64/bin/go")
  #expect(go?.environment["CGO_ENABLED"] == "0")
}

@Test func tileTransactionHelperCommandsArePinnedToTheVerifiedHelper() throws {
  let tileSetID = String(repeating: "f", count: 64)
  let helperPath = "/data/media/0/osm/binaries/vtsc-tile-transaction-aaaaaaaaaaaaaaaa"
  let stagingRoot = TiciTileSetDeploymentService.stagingRoot(tileSetID: tileSetID)
  let verify = try TiciTileSetDeploymentService.remoteManifestVerificationCommand(
    helperPath: helperPath,
    stagingRoot: stagingRoot,
    tileSetID: tileSetID
  )
  let activate = try TiciTileSetDeploymentService.atomicActivationCommand(
    helperPath: helperPath,
    stagingRoot: stagingRoot,
    tileSetID: tileSetID
  )
  let rollback = try TiciTileSetDeploymentService.atomicRollbackCommand(
    helperPath: helperPath,
    expectedActivatedTileSetID: tileSetID
  )
  for command in [verify, activate, rollback] {
    #expect(command.contains(helperPath))
    #expect(!command.lowercased().contains("python"))
  }
  #expect(verify.contains(" verify --root "))
  #expect(activate.contains(" activate --root "))
  #expect(rollback.contains(" rollback --root "))
}

@Test func tileRollbackAcceptsVerifiedOldGenerationAndRejectsActivationMismatch() async throws {
  let root = temporaryDirectory("vtsc-tile-rollback")
  defer { try? FileManager.default.removeItem(at: root) }
  let helper = try transactionHelperFixture(in: root)
  let expectedTileSetID = String(repeating: "f", count: 64)

  let succeeded = TiciTileSetDeploymentService(
    processRunner: TileRollbackRunner(output: #"{"operation":"rollback","rolled_back_tile_set_id":"old","tile_activation_not_observed":false}"#),
    transactionHelperURL: helper
  )
  try await succeeded.rollback(profile: "commaAdb", expectedActivatedTileSetID: expectedTileSetID)

  let mismatch = TiciTileSetDeploymentService(
    processRunner: TileRollbackRunner(output: #"{"operation":"rollback","rolled_back_tile_set_id":"old","tile_activation_not_observed":true}"#),
    transactionHelperURL: helper
  )
  do {
    try await mismatch.rollback(profile: "commaAdb", expectedActivatedTileSetID: expectedTileSetID)
    Issue.record("expected a tile activation mismatch to stop rollback verification")
  } catch let error as TiciTileSetDeploymentError {
    #expect(error == .activationIdentityMissing(expectedTileSetID))
  } catch {
    Issue.record("unexpected tile rollback error: \(error)")
  }
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
    "MapWholeCurveProfile:whole-curve-v2",
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
  #expect((await runner.requests).contains { request in
    request.arguments.last?.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker) == true
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
  #expect(!requests.contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
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
  #expect(requests.contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

private func temporaryDirectory(_ prefix: String) -> URL {
  let url = FileManager.default.temporaryDirectory
    .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
  try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  return url
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
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-01234567-89ab-cdef-0123-456789abcdef"
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
    mapdRecoveryOutcome: .clean,
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
  let fileCount: Int
  let totalBytes: UInt64
  private(set) var requests: [ProcessRequest] = []

  init(identity: String, fileCount: Int, totalBytes: UInt64) {
    self.identity = identity
    self.fileCount = fileCount
    self.totalBytes = totalBytes
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if command.contains(" verify --root ") {
      return success("""
      {"operation":"verify","tile_set_id":"\(identity)","file_count":\(fileCount),"total_bytes":\(totalBytes)}
      """)
    }
    if command.contains(" activate --root ") {
      return success("""
      {"operation":"activate","activated_tile_set_id":"\(identity)"}
      """)
    }
    return success("")
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output + "\n", standardError: "")
  }
}

private actor TileRollbackRunner: ProcessRunning {
  let output: String

  init(output: String) { self.output = output }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    let command = request.arguments.last ?? ""
    let standardOutput = command.contains(" rollback --root ") ? output + "\n" : ""
    return ProcessResult(terminationStatus: 0, standardOutput: standardOutput, standardError: "")
  }
}

private func transactionHelperFixture(in directory: URL) throws -> URL {
  let helper = directory.appendingPathComponent("vtsc-tile-transaction")
  try Data("#!/bin/sh\nexit 0\n".utf8).write(to: helper)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helper.path)
  return helper
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
      if request.arguments.last?.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker) == true {
        return success("\(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker)\tclean\n")
      }
      return success(deploymentSnapshotWire(
        isOffroad: false,
        isOnroad: true,
        mapLookaheadEnabled: false,
        head: head
      ))
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
    if command.contains(TiciProductionRollbackCommandBuilder.resultMarker) {
      let head = String(repeating: "b", count: 40)
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: "\(TiciProductionRollbackCommandBuilder.resultMarker)\t\(Data(head.utf8).base64EncodedString())\n",
        standardError: ""
      )
    }
    if command.contains("is_offroad") {
      let state = states.isEmpty
        ? RollbackSafetyState(isOffroad: false, isOnroad: true, mapLookaheadEnabled: false)
        : states.removeFirst()
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: deploymentSnapshotWire(
          isOffroad: state.isOffroad,
          isOnroad: state.isOnroad,
          mapLookaheadEnabled: state.mapLookaheadEnabled,
          head: String(repeating: "b", count: 40)
        ),
        standardError: ""
      )
    }
    return ProcessResult(terminationStatus: 0, standardOutput: "ok\n", standardError: "")
  }
}

private func deploymentSnapshotWire(
  isOffroad: Bool,
  isOnroad: Bool,
  mapLookaheadEnabled: Bool,
  head: String
) -> String {
  let qCurve = TuneDeploymentIdentity.canonicalQCurveSource(parameters: .checkoutFallback, bands: [])
  let values: [TiciSnapshotWireField: Data] = [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data(head.utf8),
    .dirty: Data("0".utf8),
    .isOffroad: Data((isOffroad ? "1" : "0").utf8),
    .isOnroad: Data((isOnroad ? "1" : "0").utf8),
    .mapLookaheadEnabled: Data((mapLookaheadEnabled ? "1" : "0").utf8),
    .qCurveFile: Data(qCurve.utf8),
    .activeMapdSHA256: Data(String(repeating: "d", count: 64).utf8),
    .mapdReleaseVersion: Data("old-release".utf8),
    .mapdVersion: Data("old-release".utf8),
    .mapdCacheListing: Data("/tmp/old-cache\t\(String(repeating: "d", count: 64))\n".utf8),
  ]
  return TiciSnapshotWireCodec.encode(.init(rawValues: values)) + "\n"
}
