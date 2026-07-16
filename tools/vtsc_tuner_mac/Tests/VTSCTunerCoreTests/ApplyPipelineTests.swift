import Foundation
import Testing
@testable import VTSCTunerCore

@Test func applyActionsKeepRustSupersetOrder() {
  #expect(ApplyAction.allCases == [
    .local,
    .commit,
    .push,
    .pullOnTici,
    .rebuildTilesAndReboot,
  ])
  #expect(ApplyAction.pullOnTici.label.contains("keep current tiles"))
  #expect(ApplyAction.rebuildTilesAndReboot.label.contains("build canonical tiles"))
}

@Test func productionTilePlansSeparateRuntimePullFromCanonicalRebuild() {
  let tune = Tune(params: .checkoutFallback)
  let repository = URL(fileURLWithPath: "/repo")
  let explicit = URL(fileURLWithPath: "/tmp/exact-tile-set")
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .pullOnTici,
    tune: tune,
    repositoryRoot: repository
  )) == .unchanged)
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: repository
  )) == .generateCanonical)
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .pullOnTici,
    tune: tune,
    repositoryRoot: repository,
    tileSetArtifactURL: explicit
  )) == .validateExplicit(explicit))
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: repository,
    tileSetArtifactURL: explicit
  )) == .validateExplicit(explicit))
}

@Test func ticiPreflightAcceptsOnlyAnExactHeadOrProvenFastForwardAncestor() async throws {
  let currentHead = String(repeating: "a", count: 40)
  let targetHead = String(repeating: "b", count: 40)
  let repository = URL(fileURLWithPath: "/repo")
  let git = GitDeploymentPreflight(
    branch: "chauffeur-exp01",
    localHead: targetHead,
    originHead: targetHead,
    upstream: "origin/chauffeur-exp01"
  )
  let snapshot = TiciDeploymentSnapshot(
    isOffroad: true,
    isOnroad: false,
    mapLookaheadEnabled: false,
    branch: "chauffeur-exp01",
    head: currentHead,
    dirty: false,
    physicsParams: [:],
    qCurveSHA256: "",
    mapdReleaseVersion: nil,
    mapdVersion: nil,
    activeMapdSHA256: "",
    cachedMapdPath: "",
    cachedMapdSHA256: nil,
    activeTileSetID: nil
  )

  let acceptedRunner = FastForwardRelationshipRunner(status: 0)
  try await ApplyPipeline(processRunner: acceptedRunner).validateTiciPreflight(
    snapshot,
    git: git,
    repositoryRoot: repository
  )
  let acceptedRequests = await acceptedRunner.requests
  #expect(acceptedRequests.count == 1)
  #expect(acceptedRequests[0].executableURL.path == "/usr/bin/git")
  #expect(acceptedRequests[0].arguments == ["merge-base", "--is-ancestor", currentHead, targetHead])
  #expect(acceptedRequests[0].currentDirectoryURL == repository)

  let rejectedRunner = FastForwardRelationshipRunner(status: 1)
  await #expect(throws: ApplyPipelineError.self) {
    try await ApplyPipeline(processRunner: rejectedRunner).validateTiciPreflight(
      snapshot,
      git: git,
      repositoryRoot: repository
    )
  }
}

@Test func resumePostflightCapturesOnroadControllerEvidenceThenCompletesOffroadWithoutErasingIt() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let runner = ResumePostflightRunner(fixture: fixture)
  let collector = ApplyEventCollector()

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 1,
      pollInterval: 0.001
    )
  ) { event in
    await collector.append(event)
  }

  #expect(succeeded)
  let completed = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(completed.completed)
  #expect(completed.targetHead == fixture.deployedTargetHead)
  #expect(completed.deploymentID == fixture.journal.deploymentID)
  #expect(completed.completedAt != nil)
  #expect(completed.completedToolingHead == fixture.toolingHead)
  #expect(completed.completionHostOnlyPaths == fixture.changedPaths.sorted())

  let requests = await runner.requests
  #expect(requests.contains {
    $0.executableURL == ApplyPipeline.gitURL &&
      $0.arguments == ["merge-base", "--is-ancestor", fixture.deployedTargetHead, fixture.toolingHead]
  })
  #expect(requests.contains {
    $0.executableURL == ApplyPipeline.gitURL &&
      $0.arguments.contains("\(fixture.deployedTargetHead)..\(fixture.toolingHead)")
  })
  #expect(requests.contains {
    $0.executableURL == ApplyPipeline.sshURL &&
      $0.arguments.last?.contains("memory_params_root=/dev/shm/params/d") == true
  })
  #expect(!requests.contains { $0.executableURL == ApplyPipeline.rsyncURL })
  #expect(!requests.contains { request in
    let command = request.arguments.joined(separator: " ")
    return command.contains("sudo reboot") || command.contains("git fetch --no-tags") ||
      command.contains("flock -x 9") || command.contains(".mapd-release-")
  })
  #expect(await runner.runtimeReadCount >= 3)
  let events = await collector.events
  let captureIndex = try #require(events.firstIndex { event in
    if case let .step(step) = event {
      return step.text == "Controller-ready profile captured — turn ignition off now"
    }
    return false
  })
  let completionIndex = try #require(events.firstIndex { event in
    if case let .step(step) = event {
      return step.text == "Controller-ready evidence and the subsequent clean offroad identity both passed"
    }
    return false
  })
  #expect(captureIndex < completionIndex)
}

@Test func resumePostflightRequiresControllerReadyRoadGeometryBeforeOffroadCompletion() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.controllerRoadGeometryValid = false
  let runner = ResumePostflightRunner(fixture: fixture)
  let before = try Data(contentsOf: fixture.journalURL)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func resumePostflightRejectsStaleOrFutureControllerMessages() async throws {
  for delta in [UInt64(1_500_000_001), UInt64.max] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    if delta == UInt64.max {
      fixture.controllerLogMonoTimeNs = 2_000_000_000
      fixture.controllerSampleMonoTimeNs = 1_999_999_999
    } else {
      fixture.controllerLogMonoTimeNs = 1_000_000_000
      fixture.controllerSampleMonoTimeNs = 1_000_000_000 + delta
    }
    let before = try Data(contentsOf: fixture.journalURL)
    let succeeded = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
      .resumePendingPostflight(
        ResumePostflightRequest(
          tune: fixture.tune,
          repositoryRoot: fixture.repository,
          mapdReleaseManifestURL: fixture.releaseURL,
          journalURL: fixture.journalURL,
          timeout: 0
        )
      ) { _ in }
    #expect(!succeeded)
    #expect(try Data(contentsOf: fixture.journalURL) == before)
  }
}

@Test func resumePostflightRejectsRoadStateThatStraddlesEitherPhase() async throws {
  for phase in ["controller", "offroad"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    if phase == "controller" {
      fixture.controllerEndIsOffroad = true
      fixture.controllerEndIsOnroad = false
    } else {
      fixture.offroadEndIsOffroad = false
      fixture.offroadEndIsOnroad = true
    }
    let before = try Data(contentsOf: fixture.journalURL)
    let succeeded = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
      .resumePendingPostflight(
        ResumePostflightRequest(
          tune: fixture.tune,
          repositoryRoot: fixture.repository,
          mapdReleaseManifestURL: fixture.releaseURL,
          journalURL: fixture.journalURL,
          timeout: phase == "controller" ? 0 : 0.1,
          pollInterval: 0.001
        )
      ) { _ in }
    #expect(!succeeded)
    #expect(try Data(contentsOf: fixture.journalURL) == before)
  }
}

@Test func resumePostflightRejectsLookaheadThatTurnsOnDuringRuntimeBlock() async throws {
  for phase in ["controller", "offroad"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    if phase == "controller" { fixture.controllerEndMapLookaheadEnabled = true }
    else { fixture.offroadEndMapLookaheadEnabled = true }
    let before = try Data(contentsOf: fixture.journalURL)
    let succeeded = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
      .resumePendingPostflight(
        ResumePostflightRequest(
          tune: fixture.tune,
          repositoryRoot: fixture.repository,
          mapdReleaseManifestURL: fixture.releaseURL,
          journalURL: fixture.journalURL,
          timeout: phase == "controller" ? 0 : 0.1,
          pollInterval: 0.001
        )
      ) { _ in }
    #expect(!succeeded)
    #expect(try Data(contentsOf: fixture.journalURL) == before)
  }
}

@Test func resumePostflightRejectsMalformedOffroadProfileWithCopiedTopLevelIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  var root = try JSONSerialization.jsonObject(with: fixture.profileData) as! [String: Any]
  var points = root["points"] as! [[String: Any]]
  points[2]["curvatureCoefficient"] = "not-a-number"
  root["points"] = points
  fixture.offroadProfileData = try JSONSerialization.data(withJSONObject: root)
  let before = try Data(contentsOf: fixture.journalURL)

  let succeeded = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
    .resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 0.1,
        pollInterval: 0.001
      )
    ) { _ in }

  #expect(!succeeded)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func resumePostflightClassifiesEmptyIndoorProfileBeforeVersionOrHash() async throws {
  var fixture = try resumePostflightFixture(validGPS: false)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.profileData = Data("{}".utf8)
  let runner = ResumePostflightRunner(fixture: fixture)
  let collector = ApplyEventCollector()

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { event in
    await collector.append(event)
  }

  #expect(!succeeded)
  let events = await collector.events
  #expect(events.contains { event in
    if case let .step(step) = event {
      return step.status == .failed && step.detail.contains("pending real GPS/profile") &&
        !step.detail.contains("estimator version") && !step.detail.contains("sigmoid hash")
    }
    return false
  })
}

@Test func cancellingResumedPostflightKeepsThePendingJournalByteForByte() async throws {
  let fixture = try resumePostflightFixture(validGPS: false)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let before = try Data(contentsOf: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)
  let pipeline = ApplyPipeline(processRunner: runner)
  let task = Task {
    await pipeline.resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 120
      )
    ) { _ in }
  }
  try await Task.sleep(for: .milliseconds(50))
  task.cancel()
  #expect(await task.value == false)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func completionLockAllowsOnlyOneTunerInstancePerJournal() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-journal-lock-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let journalURL = root.appendingPathComponent("pending.json")
  try Data("pending".utf8).write(to: journalURL)

  let first = try DeploymentRollbackJournal.acquireCompletionLock(for: journalURL)
  #expect(throws: DeploymentRollbackJournalError.completionLocked(journalURL.standardizedFileURL)) {
    try DeploymentRollbackJournal.acquireCompletionLock(for: journalURL)
  }
  first.unlock()
  let second = try DeploymentRollbackJournal.acquireCompletionLock(for: journalURL)
  second.unlock()
}

@Test func resumePostflightFailsClosedForNonToolingFollowupAndRetainsJournal() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.changedPaths = ["selfdrive/controls/lib/longitudinal_planner.py"]
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(!requests.contains { $0.executableURL == ApplyPipeline.sshURL })
}

@Test func resumePostflightIndoorsRetainsJournalWithoutMutationOrReboot() async throws {
  let fixture = try resumePostflightFixture(validGPS: false)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(requests.contains {
    $0.executableURL == ApplyPipeline.sshURL &&
      $0.arguments.last?.contains("remote_epoch_milliseconds") == true
  })
  #expect(!requests.contains { request in
    let command = request.arguments.joined(separator: " ")
    return command.contains("sudo reboot") || command.contains("git fetch --no-tags") ||
      command.contains("flock -x 9") || command.contains(".mapd-release-")
  })
}

@Test func resumePostflightRejectsADeviceThatHasNotReachedTheExactToolingHead() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.runtimeHead = fixture.deployedTargetHead
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test func resumePostflightRejectsDetachedOrDirtyTiciAndRetainsJournal() async throws {
  for mutation in ["detached", "dirty"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    if mutation == "detached" { fixture.runtimeBranch = "" }
    if mutation == "dirty" { fixture.runtimeDirty = true }
    let runner = ResumePostflightRunner(fixture: fixture)

    let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 0
      )
    ) { _ in }

    #expect(!succeeded)
    #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  }
}

@Test func resumePostflightRejectsSelfConsistentProfileForTheWrongTuneHash() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let wrongHash = "0123456789ab"
  #expect(wrongHash != TuneDeploymentIdentity(tune: fixture.tune).tileSigmoidHash)
  fixture.profileData = try resumeWholeCurveProfileData(
    now: Date(timeIntervalSince1970: 1_800_000_000),
    sigmoidHash: wrongHash
  )
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test func pendingPostflightJournalMustBeSchemaOneRebootedIncompleteAndTargeted() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-pending-journal-contract-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  var journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "a", count: 40),
    targetHead: String(repeating: "b", count: 40),
    previousPhysicsParams: [:],
    previousQCurveSHA256: String(repeating: "c", count: 64),
    previousMapdReleaseVersion: nil,
    previousMapdVersion: nil,
    previousActiveMapdSHA256: String(repeating: "d", count: 64),
    previousCachedMapdPath: "",
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-contract"
  )
  let url = root.appendingPathComponent("journal.json")

  try journal.write(to: url)
  #expect(throws: DeploymentRollbackJournalError.notAwaitingPostflight) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: url)
  }

  journal.rebootSent = true
  journal.completed = true
  try journal.write(to: url)
  #expect(throws: DeploymentRollbackJournalError.notAwaitingPostflight) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: url)
  }

  journal.completed = false
  journal.targetHead = nil
  try journal.write(to: url)
  #expect(throws: DeploymentRollbackJournalError.invalidTargetHead("")) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: url)
  }

  journal.targetHead = String(repeating: "b", count: 40)
  journal.schema = 2
  try journal.write(to: url)
  #expect(throws: DeploymentRollbackJournalError.unsupportedSchema(2)) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: url)
  }
}

@Test func mapdConfigDecodesRustCompatiblePaths() throws {
  let data = #"""
  {
    "pbf_path": "/tmp/ready.osm.pbf",
    "mapd_repo_path": "/tmp/openpilot-mapd",
    "mapd_binary_path": "/tmp/mapd-darwin",
    "regions_override": [{"min_lat": 32, "min_lon": -118}]
  }
  """#.data(using: .utf8)!
  let config = try JSONDecoder().decode(MapdConfig.self, from: data)
  #expect(config.pbfURL.path == "/tmp/ready.osm.pbf")
  #expect(config.effectiveBinaryURL.path == "/tmp/mapd-darwin")
  #expect(config.regionsOverride == [RegionBox(minLatitude: 32, minLongitude: -118)])
}

@Test func mapdConfigRejectsAnExecutableELFBinary() throws {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-mapd-test-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let binaryURL = directory.appendingPathComponent("mapd")
  try Data([0x7f, 0x45, 0x4c, 0x46, 0, 0, 0, 0]).write(to: binaryURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binaryURL.path)
  let config = MapdConfig(
    pbfURL: directory.appendingPathComponent("map.osm.pbf"),
    mapdRepositoryURL: directory,
    mapdBinaryURL: binaryURL
  )
  #expect(throws: MapdConfigError.notDarwinBinary(binaryURL)) {
    try config.validateDarwinBinary()
  }
}

@Test func mapdGenerationPassesEveryTunedPhysicsValue() {
  let parameters = SigmoidParameters.checkoutFallback
  let arguments = MapdCommandBuilder.generationArguments(
    parameters: parameters,
    region: RegionBox(minLatitude: 32, minLongitude: -118)
  )
  #expect(arguments.contains("--minlat=32"))
  #expect(arguments.contains("--maxlon=-116"))
  #expect(arguments.contains("--phys-a=-1.658965"))
  #expect(arguments.contains("--phys-b=-1395.055546"))
  #expect(arguments.contains("--phys-c=0.005397"))
  #expect(arguments.contains("--phys-d=4.107103"))
  #expect(arguments.contains("--phys-min-lat=2.4481"))
  #expect(arguments.contains("--phys-max-lat=4.1071"))
}

@Test func rebuildGenerationPersistsAndFullDecodesCanonicalArtifact() async throws {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-canonical-generation-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }

  let mapdRepository = directory.appendingPathComponent("openpilot-mapd", isDirectory: true)
  try FileManager.default.createDirectory(at: mapdRepository, withIntermediateDirectories: true)
  try "schema".write(
    to: mapdRepository.appendingPathComponent("offline.capnp"),
    atomically: true,
    encoding: .utf8
  )
  let pbfURL = directory.appendingPathComponent("prepared.osm.pbf")
  try Data(repeating: 0x51, count: 512).write(to: pbfURL)
  let generatorURL = directory.appendingPathComponent("mapd-darwin")
  try Data([0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0]).write(to: generatorURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: generatorURL.path)
  let decoderURL = directory.appendingPathComponent("vtsc-tile-decoder")
  try Data("#!/bin/sh\nexit 0\n".utf8).write(to: decoderURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: decoderURL.path)

  let region = RegionBox(minLatitude: 32, minLongitude: -118)
  let config = try MapdConfig(
    pbfURL: pbfURL,
    mapdRepositoryURL: mapdRepository,
    mapdBinaryURL: generatorURL,
    regionsOverride: [region]
  ).validated()
  try config.validateDarwinBinary()
  let tune = Tune(params: .checkoutFallback)
  let identity = TuneDeploymentIdentity(tune: tune)
  let release = ValidatedMapdReleaseArtifact(
    artifact: MapdReleaseArtifact(
      releaseID: "whole-curve-release-v1",
      buildID: "build-abc123",
      binaryURL: directory.appendingPathComponent("mapd-arm64"),
      sha256: String(repeating: "a", count: 64)
    ),
    byteCount: 1_000_000,
    persistentCacheFileName: "mapd-release"
  )
  let setsRoot = directory.appendingPathComponent("map_tiles/sets", isDirectory: true)
  let request = ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: URL(fileURLWithPath: "/repo"),
    tileSetsRootURL: setsRoot,
    tileDecoderURL: decoderURL
  )
  let runner = CanonicalGenerationRunner(
    generatorURL: generatorURL,
    decoderURL: decoderURL,
    sigmoidHash: identity.tileSigmoidHash
  )
  let artifact = try await ApplyPipeline(processRunner: runner).generateCanonicalTileSet(
    request: request,
    config: config,
    regions: [region],
    release: release,
    tuneIdentity: identity,
    decoderURL: decoderURL
  )

  #expect(artifact.rootURL.deletingLastPathComponent() == setsRoot.standardizedFileURL)
  #expect(artifact.manifest.regions == [region])
  #expect(artifact.manifest.fileCount == 1)
  #expect(artifact.manifest.tuneIdentitySHA256 == identity.identitySHA256)
  #expect(artifact.manifest.tileSigmoidHash == identity.tileSigmoidHash)
  #expect(artifact.manifest.pbfSHA256 == (try FileSHA256.hex(pbfURL)))
  #expect(FileManager.default.fileExists(atPath: artifact.offlineURL.path))
  let requests = await runner.requests
  let generation = try #require(requests.first { $0.executableURL == generatorURL })
  #expect(generation.arguments == MapdCommandBuilder.generationArguments(
    parameters: .checkoutFallback,
    region: region
  ))
  #expect(requests.filter { $0.executableURL == decoderURL }.count == 2)
}

@Test func ticiPhysicsMigrationPassesEverySourceRoundedValue() throws {
  let command = try TiciPhysicsCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)
  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsAmplitude') printf '%s' '-1.658965'"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsSteepness') printf '%s' '-1395.055546'"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsCenter') printf '%s' '0.005397'"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsBaseline') printf '%s' '4.107103'"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsMinLatAccel') printf '%s' '2.4481'"))
  #expect(command.contains("'VisionTurnSpeedControlPhysicsMaxLatAccel') printf '%s' '4.1071'"))
}

@Test func tuneSaveFailureStopsBeforeSourceMutation() async {
  let collector = ApplyEventCollector()
  let pipeline = ApplyPipeline()
  let request = ApplyRequest(
    action: .local,
    tune: Tune(created: "2026-07-12T00:00:00-07:00", params: .checkoutFallback),
    repositoryRoot: URL(fileURLWithPath: "/definitely/not/a/repository"),
    tuneURL: URL(fileURLWithPath: "/dev/null/current.tune.json")
  )
  let succeeded = await pipeline.apply(request) { event in
    await collector.append(event)
  }
  let events = await collector.events
  #expect(!succeeded)
  #expect(events.contains(.step(ApplyStepEvent(
    id: 1,
    status: .running,
    text: "Saving the tune file…"
  ))))
  #expect(events.contains { event in
    if case let .step(step) = event { step.id == 1 && step.status == .failed } else { false }
  })
  #expect(!events.contains { event in
    if case let .step(step) = event { step.id == 2 } else { false }
  })
  #expect(events.last == .finished(success: false))
}

@Test func rebuildPreflightFailsBeforeWritingTuneOrTouchingGit() async {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-rebuild-preflight-\(UUID().uuidString)", isDirectory: true)
  try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let tuneURL = directory.appendingPathComponent("should-not-exist.json")
  let collector = ApplyEventCollector()
  let pipeline = ApplyPipeline()
  let succeeded = await pipeline.apply(ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: Tune(created: "2026-07-12T00:00:00-07:00", params: .checkoutFallback),
    repositoryRoot: URL(fileURLWithPath: "/definitely/not/a/repository"),
    tuneURL: tuneURL,
    mapdConfigURL: directory.appendingPathComponent("missing-mapd.json")
  )) { event in
    await collector.append(event)
  }
  let events = await collector.events
  #expect(!succeeded)
  #expect(!FileManager.default.fileExists(atPath: tuneURL.path))
  #expect(events.contains { event in
    if case let .step(step) = event { step.id == 0 && step.status == .failed } else { false }
  })
  #expect(!events.contains { event in
    if case let .step(step) = event { step.id == 1 } else { false }
  })
}

private actor ApplyEventCollector {
  private(set) var events: [ApplyEvent] = []

  func append(_ event: ApplyEvent) {
    events.append(event)
  }
}

private actor FastForwardRelationshipRunner: ProcessRunning {
  let status: Int32
  private(set) var requests: [ProcessRequest] = []

  init(status: Int32) {
    self.status = status
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    return ProcessResult(terminationStatus: status, standardOutput: "", standardError: "")
  }
}

private struct ResumePostflightFixture: Sendable {
  var root: URL
  var repository: URL
  var tune: Tune
  var releaseURL: URL
  var release: MapdReleaseArtifact
  var journalURL: URL
  var journal: DeploymentRollbackJournal
  var deployedTargetHead: String
  var toolingHead: String
  var runtimeHead: String
  var runtimeBranch: String
  var runtimeDirty: Bool
  var controllerRoadGeometryValid: Bool
  var controllerLogMonoTimeNs: UInt64
  var controllerSampleMonoTimeNs: UInt64
  var controllerEndIsOffroad: Bool
  var controllerEndIsOnroad: Bool
  var offroadEndIsOffroad: Bool
  var offroadEndIsOnroad: Bool
  var controllerEndMapLookaheadEnabled: Bool
  var offroadEndMapLookaheadEnabled: Bool
  var changedPaths: [String]
  var profileData: Data
  var offroadProfileData: Data?
  var gpsData: Data?
}

private func resumePostflightFixture(validGPS: Bool) throws -> ResumePostflightFixture {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-resume-postflight-\(UUID().uuidString)", isDirectory: true)
  let repository = root.appendingPathComponent("chauffeur", isDirectory: true)
  try FileManager.default.createDirectory(
    at: repository.appendingPathComponent(".git", isDirectory: true),
    withIntermediateDirectories: true
  )
  let tune = Tune(params: .checkoutFallback)
  let entries = VTSCPhysicsAuthority.entries(for: tune.params)
  for relativePath in [
    RepositoryLocator.physicsRelativePath,
    RepositoryLocator.qCurveRelativePath,
    RepositoryLocator.paramsDefaultsRelativePath,
    RepositoryLocator.physicsPanelRelativePath,
  ] {
    let url = repository.appendingPathComponent(relativePath)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let contents: String
    switch relativePath {
    case RepositoryLocator.physicsRelativePath:
      contents = entries.map { "\($0.moduleName) = \($0.formattedValue)" }.joined(separator: "\n") + "\n"
    case RepositoryLocator.qCurveRelativePath:
      contents = TuneDeploymentIdentity.canonicalQCurveSource(parameters: tune.params, bands: tune.bands)
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

  let releaseID = "chauffeur-whole-curve-v3"
  let buildID = "tree-host-tooling-test"
  let binaryURL = root.appendingPathComponent("mapd")
  var binary = Data(repeating: 0, count: 128)
  binary.replaceSubrange(0 ..< 6, with: [0x7f, 0x45, 0x4c, 0x46, 2, 1])
  binary[18] = 183
  for marker in [
    "MapdReleaseID:\(releaseID)",
    "MapdBuildID:\(buildID)",
    MapdReleaseArtifact.defaultCapability,
  ] { binary.append(Data(marker.utf8)) }
  try binary.write(to: binaryURL)
  let release = MapdReleaseArtifact(
    releaseID: releaseID,
    buildID: buildID,
    binaryURL: binaryURL,
    sha256: MapdReleaseArtifact.sha256Hex(binary)
  )
  let releaseURL = root.appendingPathComponent("mapd-release.json")
  try JSONEncoder().encode(release).write(to: releaseURL)

  let deployedTargetHead = String(repeating: "a", count: 40)
  let toolingHead = String(repeating: "b", count: 40)
  var journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "9", count: 40),
    targetHead: deployedTargetHead,
    previousPhysicsParams: Dictionary(uniqueKeysWithValues: entries.map { ($0.paramKey, Optional($0.formattedValue)) }),
    previousQCurveSHA256: TuneDeploymentIdentity(tune: tune).qCurveSHA256,
    previousMapdReleaseVersion: "chauffeur-whole-curve-v1",
    previousMapdVersion: "chauffeur-whole-curve-v1",
    previousActiveMapdSHA256: String(repeating: "c", count: 64),
    previousCachedMapdPath: "/data/media/0/osm/binaries/mapd-old",
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-test"
  )
  journal.rebootSent = true
  let journalURL = root.appendingPathComponent("journals", isDirectory: true)
    .appendingPathComponent("\(journal.deploymentID.uuidString).json")
  try journal.write(to: journalURL)

  let now = Date(timeIntervalSince1970: 1_800_000_000)
  return ResumePostflightFixture(
    root: root,
    repository: repository,
    tune: tune,
    releaseURL: releaseURL,
    release: release,
    journalURL: journalURL,
    journal: journal,
    deployedTargetHead: deployedTargetHead,
    toolingHead: toolingHead,
    runtimeHead: toolingHead,
    runtimeBranch: "chauffeur-exp01",
    runtimeDirty: false,
    controllerRoadGeometryValid: true,
    controllerLogMonoTimeNs: 123_456_789,
    controllerSampleMonoTimeNs: 123_456_999,
    controllerEndIsOffroad: false,
    controllerEndIsOnroad: true,
    offroadEndIsOffroad: true,
    offroadEndIsOnroad: false,
    controllerEndMapLookaheadEnabled: false,
    offroadEndMapLookaheadEnabled: false,
    changedPaths: [
      ".codex/skills/vtsc-tuner-app/references/changelog.md",
      "tools/vtsc_tuner_mac/Sources/VTSCTunerCore/ApplyPipeline.swift",
      "tools/vtsc_tuner_mac/Sources/VTSCTunerCore/TiciSnapshotWireCodec.swift",
    ],
    profileData: try resumeWholeCurveProfileData(
      now: now,
      sigmoidHash: TuneDeploymentIdentity(tune: tune).tileSigmoidHash
    ),
    offroadProfileData: nil,
    gpsData: validGPS ? try JSONSerialization.data(withJSONObject: [
      "latitude": 37.0,
      "longitude": -122.0,
      "bearing": 90.0,
    ]) : nil
  )
}

private actor ResumePostflightRunner: ProcessRunning {
  let fixture: ResumePostflightFixture
  private(set) var requests: [ProcessRequest] = []
  private(set) var runtimeReadCount = 0

  init(fixture: ResumePostflightFixture) { self.fixture = fixture }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == ApplyPipeline.gitURL {
      switch request.arguments {
      case ["branch", "--show-current"]:
        return success("chauffeur-exp01\n")
      case ["status", "--porcelain"]:
        return success("")
      case ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
        return success("origin/chauffeur-exp01\n")
      case ["rev-parse", "HEAD"]:
        return success(fixture.toolingHead + "\n")
      case ["ls-remote", "--heads", "origin", "refs/heads/chauffeur-exp01"]:
        return success("\(fixture.toolingHead)\trefs/heads/chauffeur-exp01\n")
      case ["merge-base", "--is-ancestor", fixture.deployedTargetHead, fixture.toolingHead]:
        return success("")
      default:
        if request.arguments.starts(with: ["diff", "--no-ext-diff", "--name-only", "-z"]) {
          return success(fixture.changedPaths.joined(separator: "\0") + "\0")
        }
        return failure("unexpected git request: \(request.arguments)")
      }
    }
    if request.executableURL == ApplyPipeline.sshURL {
      if request.arguments.last == "true" { return success("") }
      if request.arguments.last?.contains("remote_epoch_milliseconds") == true {
        runtimeReadCount += 1
        return success(runtimeWire(readIndex: runtimeReadCount))
      }
      return failure("unexpected ssh request")
    }
    return failure("unexpected executable: \(request.executableURL.path)")
  }

  private func runtimeWire(readIndex: Int) -> String {
    let identity = TuneDeploymentIdentity(tune: fixture.tune)
    let releaseDigest = TuneDeploymentIdentity.sha256Hex(Data(fixture.release.releaseID.utf8))
    let cachePath = "/data/media/0/osm/binaries/mapd-\(releaseDigest.prefix(16))-\(fixture.release.sha256.prefix(16))"
    let buildInfo = TiciMapdReleaseBuildInfo(
      releaseID: fixture.release.releaseID,
      buildID: fixture.release.buildID,
      estimatorVersion: fixture.release.estimatorVersion,
      capabilities: [fixture.release.capability],
      identityMarkers: [
        "MapdReleaseID:\(fixture.release.releaseID)",
        "MapdBuildID:\(fixture.release.buildID)",
      ]
    )
    let controllerPhase = readIndex == 1
    var fields: [TiciSnapshotWireField: Data] = [
      .branch: Data(fixture.runtimeBranch.utf8),
      .head: Data(fixture.runtimeHead.utf8),
      .dirty: Data((fixture.runtimeDirty ? "1" : "0").utf8),
      .isOffroad: Data((controllerPhase ? "0" : "1").utf8),
      .isOnroad: Data((controllerPhase ? "1" : "0").utf8),
      .mapLookaheadEnabled: Data("0".utf8),
      .mapdReleaseVersion: Data(fixture.release.releaseID.utf8),
      .mapdVersion: Data(fixture.release.releaseID.utf8),
      .activeMapdSHA256: Data(fixture.release.sha256.utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: fixture.tune.params,
        bands: fixture.tune.bands
      ).utf8),
      .mapdCacheListing: Data("\(cachePath)\t\(fixture.release.sha256)\n".utf8),
      .activeMapdBuildInfo: try! JSONEncoder().encode(buildInfo),
      .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
      .mapdRunning: Data("1".utf8),
      .remoteEpochMilliseconds: Data("1800000000000".utf8),
      .memoryWholeCurveProfile: controllerPhase ? fixture.profileData : (fixture.offroadProfileData ?? fixture.profileData),
      .liveMapDataControllerStatus: Data((controllerPhase
        ? "1|1|\(fixture.controllerLogMonoTimeNs)|\(fixture.controllerRoadGeometryValid ? 1 : 0)|\(fixture.controllerSampleMonoTimeNs)"
        : "1|0|123456790|0|123456999").utf8),
      .runtimeEndIsOffroad: Data(((controllerPhase
        ? fixture.controllerEndIsOffroad : fixture.offroadEndIsOffroad) ? "1" : "0").utf8),
      .runtimeEndIsOnroad: Data(((controllerPhase
        ? fixture.controllerEndIsOnroad : fixture.offroadEndIsOnroad) ? "1" : "0").utf8),
      .runtimeEndMapLookaheadEnabled: Data(((controllerPhase
        ? fixture.controllerEndMapLookaheadEnabled : fixture.offroadEndMapLookaheadEnabled) ? "1" : "0").utf8),
    ]
    if let gpsData = fixture.gpsData { fields[.memoryLastGPSPosition] = gpsData }
    for physics in identity.physics {
      let field: TiciSnapshotWireField = switch physics.paramKey {
      case "VisionTurnSpeedControlPhysicsAmplitude": .physicsAmplitude
      case "VisionTurnSpeedControlPhysicsSteepness": .physicsSteepness
      case "VisionTurnSpeedControlPhysicsCenter": .physicsCenter
      case "VisionTurnSpeedControlPhysicsBaseline": .physicsBaseline
      case "VisionTurnSpeedControlPhysicsMinLatAccel": .physicsMinLatAccel
      default: .physicsMaxLatAccel
      }
      fields[field] = Data(physics.value.utf8)
    }
    return TiciSnapshotWireCodec.encode(.init(rawValues: fields)) + "\n"
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }

  private func failure(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 1, standardOutput: "", standardError: output)
  }
}

private func resumeWholeCurveProfileData(now: Date, sigmoidHash: String) throws -> Data {
  let stepDegrees = 5.0 / 6_371_007.2 * 180 / Double.pi
  let eventID = "0123456789abcdefabcd-a"
  let points: [[String: Any]] = (0..<5).map { index in
    [
      "latitude": 37.0 + Double(index) * stepDegrees,
      "longitude": -122.0,
      "distanceMeters": Double(index) * 5,
      "curvature": [0.0, 0.003, 0.006, 0.003, 0.0][index],
      "curvatureCoefficient": [1.0, 0.5, 1.0, 0.5, 1.0][index],
      "baseSafeSpeedMPS": [70.0, 20.0, 14.0, 20.0, 70.0][index],
      "eventID": (1...3).contains(index) ? eventID : "",
      "confidence": 1.0,
      "flags": [],
    ]
  }
  let fingerprint = try TiciWholeCurvePostflightValidator.routeFingerprint(
    generation: 7,
    sigmoidHash: sigmoidHash,
    points: points.map {
      ($0["latitude"] as! Double, $0["longitude"] as! Double, $0["distanceMeters"] as! Double,
       $0["curvature"] as! Double, $0["curvatureCoefficient"] as! Double,
       $0["baseSafeSpeedMPS"] as! Double, $0["eventID"] as! String)
    }
  )
  return try JSONSerialization.data(withJSONObject: [
    "estimatorVersion": "whole-curve-v3",
    "generatedAtUnixMillis": now.timeIntervalSince1970 * 1_000,
    "routeFingerprint": fingerprint,
    "sigmoidHash": sigmoidHash,
    "generation": 7,
    "points": points,
    "events": [[
      "eventID": eventID,
      "startIndex": 1,
      "endIndex": 3,
      "apexIndex": 2,
      "profileApexIndex": 2,
      "controllingCurvature": 0.006,
      "maximumApexCoefficient": 1.0,
      "confidence": 1.0,
      "flags": [],
    ]],
    "fatalAmbiguity": false,
  ])
}

private actor CanonicalGenerationRunner: ProcessRunning {
  let generatorURL: URL
  let decoderURL: URL
  let sigmoidHash: String
  private(set) var requests: [ProcessRequest] = []

  init(generatorURL: URL, decoderURL: URL, sigmoidHash: String) {
    self.generatorURL = generatorURL
    self.decoderURL = decoderURL
    self.sigmoidHash = sigmoidHash
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == generatorURL {
      let latitude = try argumentValue("--minlat=", in: request.arguments)
      let longitude = try argumentValue("--minlon=", in: request.arguments)
      let output = try #require(request.currentDirectoryURL)
        .appendingPathComponent("offline", isDirectory: true)
        .appendingPathComponent(String(latitude), isDirectory: true)
        .appendingPathComponent(String(longitude), isDirectory: true)
      try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
      let name = String(
        format: "%d.000000_%d.000000_%d.000000_%d.000000",
        latitude, longitude, latitude + 2, longitude + 2
      )
      try Data(repeating: 0x77, count: 256).write(to: output.appendingPathComponent(name))
      return ProcessResult(terminationStatus: 0, standardOutput: "generated\n", standardError: "")
    }
    if request.executableURL == decoderURL {
      let encoder = JSONEncoder()
      let lines = try request.arguments.map { path -> String in
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let bounds = try #require(MapTileIndex.parseBounds(filename: url.lastPathComponent))
        let tile = MapTile(
          sourcePath: url.path,
          bounds: bounds,
          overlap: 0.01,
          schemaVersion: 1,
          sigmoidHash: sigmoidHash,
          ways: []
        )
        return String(decoding: try encoder.encode(tile), as: UTF8.self)
      }
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: lines.joined(separator: "\n") + "\n",
        standardError: ""
      )
    }
    return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unexpected executable")
  }

  private func argumentValue(_ prefix: String, in arguments: [String]) throws -> Int {
    let value = try #require(arguments.first { $0.hasPrefix(prefix) })
    return try #require(Int(value.dropFirst(prefix.count)))
  }
}
