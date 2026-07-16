import Darwin
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
      return step.text == "Profile acceptance proven — turn ignition off now"
    }
    return false
  })
  let completionIndex = try #require(events.firstIndex { event in
    if case let .step(step) = event {
      return step.text == "Profile acceptance and the subsequent clean offroad identity both passed"
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

@Test func resumePostflightRetriesOnlyExpectedTransportAndControllerWaitStates() async throws {
  for waitKind in ["ssh_255", "timeout", "profile_pending"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    if waitKind == "ssh_255" { fixture.transientTransportFailures = 1 }
    else if waitKind == "timeout" { fixture.throwTimedOutOnce = true }
    else { fixture.controllerPendingReads = 1 }
    let runner = ResumePostflightRunner(fixture: fixture)

    let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 0.2,
        pollInterval: 0.001
      )
    ) { _ in }

    #expect(succeeded)
    #expect(await runner.runtimeReadCount >= 4)
  }
}

@Test func resumedPostflightTreatsRemoteSnapshotExitOneAsImmediateTerminal() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.transientTransportFailures = 1
  fixture.transientTransportStatus = 1
  let runner = ResumePostflightRunner(fixture: fixture)
  let before = try Data(contentsOf: fixture.journalURL)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 1,
      pollInterval: 0.001
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(await runner.runtimeReadCount == 1)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func resumePostflightFailsImmediatelyOnHardIdentityViolationEvenIfNextReadWouldRecover() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.firstRuntimeHead = String(repeating: "e", count: 40)
  let runner = ResumePostflightRunner(fixture: fixture)
  let before = try Data(contentsOf: fixture.journalURL)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 1,
      pollInterval: 0.001
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(await runner.runtimeReadCount == 1)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func resumedPostflightGivesTheOffroadPhaseItsOwnFullDeadline() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.firstRuntimeDelay = .milliseconds(45)
  fixture.offroadWaitReads = 1
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0.05,
      pollInterval: 0.03
    )
  ) { _ in }

  #expect(succeeded)
  #expect(await runner.runtimeReadCount >= 4)
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

  let runner = ResumePostflightRunner(fixture: fixture)
  let succeeded = await ApplyPipeline(processRunner: runner)
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
  #expect(await runner.runtimeReadCount == 2)
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

@Test func cancellingAtTheFinalizingEventStillPrecedesTheJournalCommitPoint() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let before = try Data(contentsOf: fixture.journalURL)
  let task = Task {
    await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture)).resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 1,
        pollInterval: 0.001
      )
    ) { event in
      if case let .step(step) = event,
         step.id == 2, step.status == .running {
        withUnsafeCurrentTask { $0?.cancel() }
      }
    }
  }

  #expect(await task.value == false)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func cancellationAfterTheAtomicCommitReportsCompletedSuccess() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let task = Task {
    await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture)).resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 1,
        pollInterval: 0.001
      )
    ) { event in
      if case let .step(step) = event,
         step.id == 2, step.status == .succeeded,
         step.text == "The existing deployment journal is now complete" {
        withUnsafeCurrentTask { $0?.cancel() }
      }
    }
  }

  #expect(await task.value)
  let completed = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(completed.completed)
  #expect(completed.effectiveResolution == .completed)
}

@Test func completedResumeJournalPreventsAStaleOriginalRollbackWithoutMutation() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  var preparedCompletion = fixture.journal
  preparedCompletion.completed = true
  preparedCompletion.completedAt = "2026-07-16T00:00:00Z"
  preparedCompletion.completedToolingHead = fixture.toolingHead
  preparedCompletion.completionHostOnlyPaths = fixture.changedPaths.sorted()
  preparedCompletion.resolution = .completed
  let completed = preparedCompletion
  try completed.write(to: fixture.journalURL)
  let before = try Data(contentsOf: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)

  let result = await ApplyPipeline(processRunner: runner).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  )

  guard case let .alreadyCompleted(detail) = result else {
    Issue.record("expected alreadyCompleted rollback resolution")
    return
  }
  #expect(detail.contains("rollback skipped"))
  #expect(await runner.requests.isEmpty)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func originalCompletionWinningBeforeResumeCASIsIdempotentAndDoesNotRewrite() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  var preparedCompletion = fixture.journal
  preparedCompletion.completed = true
  preparedCompletion.completedAt = "2026-07-16T00:00:00Z"
  preparedCompletion.completedToolingHead = fixture.deployedTargetHead
  preparedCompletion.completionHostOnlyPaths = []
  preparedCompletion.resolution = .completed
  let completed = preparedCompletion
  let expectedURL = fixture.root.appendingPathComponent("expected-completed.json")
  try completed.write(to: expectedURL)
  let expectedBytes = try Data(contentsOf: expectedURL)

  let succeeded = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
    .resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 1,
        pollInterval: 0.001
      )
    ) { event in
      if case let .step(step) = event,
         step.id == 1, step.status == .succeeded {
        try! completed.write(to: fixture.journalURL)
      }
    }

  #expect(succeeded)
  #expect(try Data(contentsOf: fixture.journalURL) == expectedBytes)
}

@Test func twoResumeInstancesSerializeToOneAtomicCompletionWrite() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let barrier = TwoPartyBarrier()
  let request = ResumePostflightRequest(
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    mapdReleaseManifestURL: fixture.releaseURL,
    journalURL: fixture.journalURL,
    timeout: 1,
    pollInterval: 0.001
  )
  func runOne() async -> Bool {
    await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
      .resumePendingPostflight(request) { event in
        if case let .step(step) = event,
           step.id == 1, step.status == .succeeded {
          await barrier.arriveAndWait()
        }
      }
  }

  async let first = runOne()
  async let second = runOne()
  let results = await [first, second]

  #expect(results == [true, true])
  let completed = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(completed.effectiveResolution == .completed)
  #expect(completed.completed)
  #expect(completed.completedAt != nil)
}

@Test func DurableRollbackClaimPermanentlyBlocksResumeCompletion() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let before = try Data(contentsOf: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 1
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(await runner.requests.isEmpty)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
}

@Test func rollbackClaimIsNonPendingToAnOlderSchemaOneReader() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)

  let legacy = try JSONDecoder().decode(
    LegacySchemaOnePendingReader.self,
    from: Data(contentsOf: fixture.journalURL)
  )
  #expect(legacy.schema == 1)
  #expect(!legacy.isPendingPostflight)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rollbackInProgress)
}

@Test func orphanedRollbackClaimCanBeTakenOverAndSettlesPartialFailure() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let runner = RollbackClaimRecoveryRunner(failRemoteRollback: true)

  let result = await ApplyPipeline(processRunner: runner).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  )

  guard case .rollbackFailed = result else {
    Issue.record("expected rollbackFailed resolution")
    return
  }
  #expect((await runner.requests).contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.completed)
  #expect(settled.effectiveResolution == .rollbackFailed)
}

@Test func freshProcessRecoveryUsesOnlyDurableTileIdentitiesAndDoesNotInventAPreReboot() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let previousTileSetID = String(repeating: "1", count: 64)
  let targetTileSetID = String(repeating: "2", count: 64)
  let previousRelease = "chauffeur-whole-curve-v1"
  let releaseDigest = TuneDeploymentIdentity.sha256Hex(Data(previousRelease.utf8))
  let previousMapdSHA = String(repeating: "c", count: 64)
  let previousCacheSHA = String(repeating: "d", count: 64)
  fixture.journal.previousMapdReleaseVersion = previousRelease
  fixture.journal.previousMapdVersion = previousRelease
  fixture.journal.previousActiveMapdSHA256 = previousMapdSHA
  fixture.journal.previousCachedMapdPath =
    "/data/media/0/osm/binaries/mapd-\(releaseDigest.prefix(16))-\(previousCacheSHA.prefix(16))"
  fixture.journal.previousCachedMapdSHA256 = previousCacheSHA
  fixture.journal.previousTileSetID = previousTileSetID
  fixture.journal.targetTileSetID = targetTileSetID
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)

  let succeeded = await ApplyPipeline(processRunner: runner).recoverPendingRollback(
    RollbackRecoveryRequest(
      repositoryRoot: fixture.repository,
      journalURL: fixture.journalURL
    )
  ) { _ in }

  #expect(succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rolledBack)
  let requests = await runner.requests
  #expect(requests.contains { request in
    request.arguments.last?.contains("rollback --root") == true &&
      request.arguments.last?.contains("--expected-tile-set-id '\(targetTileSetID)'") == true
  })
  #expect(!requests.contains { $0.arguments.joined(separator: " ").contains("sudo reboot") })
}

@Test func liveRollbackOwnerExcludesASecondMutatorUntilTerminalSettlement() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let preflight = try resumeRuntimePreflight(fixture)
  let ownerRunner = BlockingRollbackOwnerRunner()
  let ownerTask = Task {
    await ApplyPipeline(processRunner: ownerRunner).rollbackProductionDeploymentIfJournalPending(
      preflight: preflight,
      tilesActivated: false
    )
  }
  await ownerRunner.waitUntilRemoteMutationStarts()
  let claimed = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(claimed.completed)
  #expect(claimed.effectiveResolution == .rollbackInProgress)

  let contenderRunner = RollbackClaimRecoveryRunner(failRemoteRollback: false)
  let contender = await ApplyPipeline(processRunner: contenderRunner)
    .rollbackProductionDeploymentIfJournalPending(
      preflight: preflight,
      tilesActivated: false
    )
  guard case .rollbackFailed = contender else {
    Issue.record("live rollback owner should exclude the contender")
    await ownerRunner.releaseRemoteMutation()
    _ = await ownerTask.value
    return
  }
  #expect(await contenderRunner.requests.isEmpty)

  await ownerRunner.releaseRemoteMutation()
  guard case .rollbackFailed = await ownerTask.value else {
    Issue.record("injected owner failure should settle rollbackFailed")
    return
  }
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rollbackFailed)
}

@Test func inconsistentExplicitResolutionRequiresOrRepairsLegacyCompletionSentinel() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = false
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  #expect(throws: DeploymentRollbackJournalError.inconsistentResolution(
    "rollbackInProgress requires the legacy completed=true sentinel"
  )) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: fixture.journalURL)
  }

  let runner = RollbackClaimRecoveryRunner(failRemoteRollback: true)
  guard case .rollbackFailed = await ApplyPipeline(processRunner: runner)
    .rollbackProductionDeploymentIfJournalPending(
      preflight: try resumeRuntimePreflight(fixture),
      tilesActivated: false
    )
  else {
    Issue.record("expected repaired rollback replay to settle failure")
    return
  }
  let repaired = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(repaired.completed)
  #expect(repaired.effectiveResolution == .rollbackFailed)

  var invalidCompleted = fixture.journal
  invalidCompleted.resolution = .completed
  invalidCompleted.completed = false
  try invalidCompleted.write(to: fixture.journalURL)
  #expect(throws: DeploymentRollbackJournalError.inconsistentResolution(
    "completed requires the legacy completed=true sentinel"
  )) {
    try DeploymentRollbackJournal.loadPendingPostflight(from: fixture.journalURL)
  }
  let beforeInvalidCompletion = try Data(contentsOf: fixture.journalURL)
  let noProofResult = await ApplyPipeline(processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false))
    .rollbackProductionDeploymentIfJournalPending(
      preflight: try resumeRuntimePreflight(fixture),
      tilesActivated: false
    )
  guard case .rollbackFailed = noProofResult else {
    Issue.record("corrupt completed resolution must fail closed")
    return
  }
  #expect(try Data(contentsOf: fixture.journalURL) == beforeInvalidCompletion)

  var tornRolledBack = fixture.journal
  tornRolledBack.resolution = .rolledBack
  tornRolledBack.completed = false
  try tornRolledBack.write(to: fixture.journalURL)
  #expect(try DeploymentRollbackJournal.loadRecoverableRollback(from: fixture.journalURL).journal == tornRolledBack)
  let replayRunner = RollbackClaimRecoveryRunner(failRemoteRollback: true)
  guard case .rollbackFailed = await ApplyPipeline(processRunner: replayRunner)
    .rollbackProductionDeploymentIfJournalPending(
      preflight: try resumeRuntimePreflight(fixture),
      tilesActivated: false
    )
  else {
    Issue.record("torn rolledBack resolution must replay restoration")
    return
  }
  #expect((await replayRunner.requests).contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rollbackFailed)
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

@Test func durableJournalWriterReportsVisibleButIndeterminateAfterDirectorySyncFailure() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-journal-durability-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let url = root.appendingPathComponent("transaction.json")
  let expected = Data("{\"resolution\":\"rollbackInProgress\"}\n".utf8)

  #expect(throws: DeploymentRollbackJournalError.committedButNotDurable(
    url.standardizedFileURL,
    "directory fsync failed after three attempts: Input/output error; exact_visible=1"
  )) {
    try DeploymentRollbackJournal.durablyReplace(
      expected,
      at: url,
      directorySync: { _ in
        errno = EIO
        return -1
      }
    )
  }
  #expect(try Data(contentsOf: url) == expected)
}

@Test func indeterminateRollbackClaimNeverStartsRemoteMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = false
  fixture.journal.resolution = .rollbackFailed
  try fixture.journal.write(to: fixture.journalURL)
  let runner = RollbackClaimRecoveryRunner(failRemoteRollback: false)
  let pipeline = ApplyPipeline(processRunner: runner) { journal, url in
    try journal.write(to: url)
    if journal.effectiveResolution == .rollbackInProgress {
      throw DeploymentRollbackJournalError.committedButNotDurable(url, "injected")
    }
  }
  guard case .rollbackFailed = await pipeline.rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  ) else { Issue.record("indeterminate claim must fail closed"); return }
  #expect(!(await runner.requests).contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
}

@Test func journalWriteCreatesNestedRecoveryDirectoryBeforeItsDurableTransition() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-journal-parent-durability-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let url = root.appendingPathComponent("new/deployment-journals/transaction.json")
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress

  try fixture.journal.write(to: url)

  #expect(try DeploymentRollbackJournal.load(from: url) == fixture.journal)
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

@Test func newCarFacingDeploymentIsBlockedByAnyRecoverableRollbackJournal() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackFailed
  try fixture.journal.write(to: fixture.journalURL)
  let tuneURL = fixture.root.appendingPathComponent("must-not-save.json")
  let succeeded = await ApplyPipeline().apply(ApplyRequest(
    action: .pullOnTici,
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    tuneURL: tuneURL,
    rollbackJournalDirectoryURL: fixture.journalURL.deletingLastPathComponent()
  )) { _ in }
  #expect(!succeeded)
  #expect(!FileManager.default.fileExists(atPath: tuneURL.path))
}

@Test func multipleRollbackJournalsAreEnumeratedForExactSelection() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackFailed
  try fixture.journal.write(to: fixture.journalURL)
  var second = fixture.journal
  second.deploymentID = UUID()
  second.createdAt = "2026-07-16T03:00:00Z"
  let secondURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(second.deploymentID.uuidString).json")
  try second.write(to: secondURL)

  let candidates = try DeploymentRollbackJournal.recoverableRollbacks(
    directory: fixture.journalURL.deletingLastPathComponent()
  )
  #expect(candidates.map(\.url) == [fixture.journalURL, secondURL].map(\.standardizedFileURL).sorted { $0.path < $1.path })
  #expect(try DeploymentRollbackJournal.loadRecoverableRollback(from: secondURL).journal.deploymentID == second.deploymentID)
}

private actor ApplyEventCollector {
  private(set) var events: [ApplyEvent] = []

  func append(_ event: ApplyEvent) {
    events.append(event)
  }
}

private actor TwoPartyBarrier {
  private var arrivals = 0
  private var waiters: [CheckedContinuation<Void, Never>] = []

  func arriveAndWait() async {
    arrivals += 1
    if arrivals == 2 {
      let pending = waiters
      waiters.removeAll()
      pending.forEach { $0.resume() }
      return
    }
    await withCheckedContinuation { continuation in
      waiters.append(continuation)
    }
  }
}

private struct LegacySchemaOnePendingReader: Decodable {
  var schema: Int
  var rebootSent: Bool
  var completed: Bool

  var isPendingPostflight: Bool { schema == 1 && rebootSent && !completed }
}

private actor FreshProcessRollbackRecoveryRunner: ProcessRunning {
  let journal: DeploymentRollbackJournal
  private(set) var requests: [ProcessRequest] = []

  init(journal: DeploymentRollbackJournal) {
    self.journal = journal
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if request.executableURL == ApplyPipeline.sshURL, command == "true" {
      return success("")
    }
    if request.executableURL == ApplyPipeline.rsyncURL {
      return success("")
    }
    if command.contains("rollback --root") {
      return success(String(decoding: try JSONSerialization.data(withJSONObject: [
        "operation": "rollback",
        "tile_activation_not_observed": true,
        "active_tile_set_id": journal.previousTileSetID ?? "",
      ]), as: UTF8.self) + "\n")
    }
    if command.contains(TiciProductionRollbackCommandBuilder.resultMarker) {
      return success(
        "\(TiciProductionRollbackCommandBuilder.resultMarker)\t" +
          Data(journal.previousHead.utf8).base64EncodedString() + "\n"
      )
    }
    if command.contains("params_root=/data/params/d") {
      return success(runtimeWire())
    }
    if request.executableURL == ApplyPipeline.sshURL {
      return success("")
    }
    return ProcessResult(
      terminationStatus: 1,
      standardOutput: "",
      standardError: "unexpected recovery request: \(request.executableURL.path) \(request.arguments)"
    )
  }

  private func runtimeWire() -> String {
    var fields: [TiciSnapshotWireField: Data] = [
      .branch: Data(journal.branch.utf8),
      .head: Data(journal.previousHead.utf8),
      .dirty: Data("0".utf8),
      .isOffroad: Data("1".utf8),
      .isOnroad: Data("0".utf8),
      .mapLookaheadEnabled: Data("0".utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: .checkoutFallback,
        bands: []
      ).utf8),
      .activeMapdSHA256: Data(journal.previousActiveMapdSHA256.utf8),
      .mapdCacheListing: Data((journal.previousCachedMapdSHA256.map {
        "\(journal.previousCachedMapdPath)\t\($0)\n"
      } ?? "").utf8),
      .activeMapdBuildInfo: Data("{}".utf8),
      .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1]),
      .mapdRunning: Data("1".utf8),
      .remoteEpochMilliseconds: Data("1800000000000".utf8),
      .liveMapDataControllerStatus: Data("0|0|0|0|0".utf8),
      .runtimeEndIsOffroad: Data("1".utf8),
      .runtimeEndIsOnroad: Data("0".utf8),
      .runtimeEndMapLookaheadEnabled: Data("0".utf8),
    ]
    if let release = journal.previousMapdReleaseVersion {
      fields[.mapdReleaseVersion] = Data(release.utf8)
    }
    if let version = journal.previousMapdVersion {
      fields[.mapdVersion] = Data(version.utf8)
    }
    if let tileSetID = journal.previousTileSetID {
      fields[.tileManifest] = try! JSONSerialization.data(withJSONObject: ["tile_set_id": tileSetID])
    }
    let physicsFields: [String: TiciSnapshotWireField] = [
      "VisionTurnSpeedControlPhysicsAmplitude": .physicsAmplitude,
      "VisionTurnSpeedControlPhysicsSteepness": .physicsSteepness,
      "VisionTurnSpeedControlPhysicsCenter": .physicsCenter,
      "VisionTurnSpeedControlPhysicsBaseline": .physicsBaseline,
      "VisionTurnSpeedControlPhysicsMinLatAccel": .physicsMinLatAccel,
      "VisionTurnSpeedControlPhysicsMaxLatAccel": .physicsMaxLatAccel,
    ]
    for (key, field) in physicsFields {
      if let value = journal.previousPhysicsParams[key] ?? nil {
        fields[field] = Data(value.utf8)
      }
    }
    return TiciSnapshotWireCodec.encode(.init(rawValues: fields)) + "\n"
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }
}

private actor RollbackClaimRecoveryRunner: ProcessRunning {
  let failRemoteRollback: Bool
  private(set) var requests: [ProcessRequest] = []

  init(failRemoteRollback: Bool) {
    self.failRemoteRollback = failRemoteRollback
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if command.contains(TiciProductionRollbackCommandBuilder.resultMarker) {
      return failRemoteRollback
        ? ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "partial rollback failure")
        : ProcessResult(terminationStatus: 0, standardOutput: "", standardError: "")
    }
    if command.contains("is_offroad") {
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: rollbackSafetyWire(),
        standardError: ""
      )
    }
    return ProcessResult(terminationStatus: 0, standardOutput: "ok\n", standardError: "")
  }

  private func rollbackSafetyWire() -> String {
    TiciSnapshotWireCodec.encode(.init(rawValues: [
      .branch: Data("chauffeur-exp01".utf8),
      .head: Data(String(repeating: "9", count: 40).utf8),
      .dirty: Data("0".utf8),
      .isOffroad: Data("1".utf8),
      .isOnroad: Data("0".utf8),
      .mapLookaheadEnabled: Data("0".utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: .checkoutFallback,
        bands: []
      ).utf8),
      .activeMapdSHA256: Data(String(repeating: "c", count: 64).utf8),
      .mapdCacheListing: Data("".utf8),
    ])) + "\n"
  }
}

private actor BlockingRollbackOwnerRunner: ProcessRunning {
  private var startedWaiters: [CheckedContinuation<Void, Never>] = []
  private var releaseWaiters: [CheckedContinuation<Void, Never>] = []
  private var remoteMutationStarted = false
  private var released = false

  func waitUntilRemoteMutationStarts() async {
    if remoteMutationStarted { return }
    await withCheckedContinuation { startedWaiters.append($0) }
  }

  func releaseRemoteMutation() {
    released = true
    let waiters = releaseWaiters
    releaseWaiters.removeAll()
    waiters.forEach { $0.resume() }
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    let command = request.arguments.last ?? ""
    if command.contains(TiciProductionRollbackCommandBuilder.resultMarker) {
      remoteMutationStarted = true
      let waiters = startedWaiters
      startedWaiters.removeAll()
      waiters.forEach { $0.resume() }
      if !released {
        await withCheckedContinuation { releaseWaiters.append($0) }
      }
      return ProcessResult(
        terminationStatus: 1,
        standardOutput: "",
        standardError: "injected owner rollback failure"
      )
    }
    if command.contains("is_offroad") {
      let runner = RollbackClaimRecoveryRunner(failRemoteRollback: false)
      return try await runner.run(request)
    }
    return ProcessResult(terminationStatus: 0, standardOutput: "ok\n", standardError: "")
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
  var transientTransportFailures: Int
  var controllerPendingReads: Int
  var offroadWaitReads: Int
  var firstRuntimeHead: String?
  var firstRuntimeDelay: Duration?
  var transientTransportStatus: Int32
  var throwTimedOutOnce: Bool
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
    transientTransportFailures: 0,
    controllerPendingReads: 0,
    offroadWaitReads: 0,
    firstRuntimeHead: nil,
    firstRuntimeDelay: nil,
    transientTransportStatus: 255,
    throwTimedOutOnce: false,
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

private func resumeRuntimePreflight(
  _ fixture: ResumePostflightFixture
) throws -> RuntimeDeploymentPreflight {
  RuntimeDeploymentPreflight(
    git: GitDeploymentPreflight(
      branch: fixture.journal.branch,
      localHead: fixture.toolingHead,
      originHead: fixture.toolingHead,
      upstream: "origin/\(fixture.journal.branch)"
    ),
    profile: fixture.journal.profile,
    mapdRecoveryOutcome: .clean,
    release: try fixture.release.validated(),
    tileSet: nil,
    journal: fixture.journal,
    journalURL: fixture.journalURL
  )
}

private actor ResumePostflightRunner: ProcessRunning {
  let fixture: ResumePostflightFixture
  private(set) var requests: [ProcessRequest] = []
  private(set) var runtimeReadCount = 0
  private var successfulRuntimeReadCount = 0

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
        if fixture.throwTimedOutOnce, runtimeReadCount == 1 {
          throw ProcessRunnerError.timedOut(request.executableURL, request.timeout ?? 30)
        }
        if runtimeReadCount <= fixture.transientTransportFailures {
          return ProcessResult(
            terminationStatus: fixture.transientTransportStatus,
            standardOutput: "",
            standardError: "tici runtime still starting"
          )
        }
        successfulRuntimeReadCount += 1
        if successfulRuntimeReadCount == 1, let delay = fixture.firstRuntimeDelay {
          try await Task.sleep(for: delay)
        }
        return success(runtimeWire(readIndex: successfulRuntimeReadCount))
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
    let evidenceReadIndex = fixture.controllerPendingReads + 1
    let pendingControllerPhase = readIndex <= fixture.controllerPendingReads
    let controllerPhase = readIndex <= evidenceReadIndex
    let waitingOnroadPhase = readIndex > evidenceReadIndex &&
      readIndex <= evidenceReadIndex + fixture.offroadWaitReads
    let stableOnroadPhase = controllerPhase || waitingOnroadPhase
    var fields: [TiciSnapshotWireField: Data] = [
      .branch: Data(fixture.runtimeBranch.utf8),
      .head: Data(((readIndex == 1 ? fixture.firstRuntimeHead : nil) ?? fixture.runtimeHead).utf8),
      .dirty: Data((fixture.runtimeDirty ? "1" : "0").utf8),
      .isOffroad: Data((stableOnroadPhase ? "0" : "1").utf8),
      .isOnroad: Data((stableOnroadPhase ? "1" : "0").utf8),
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
      .memoryWholeCurveProfile: pendingControllerPhase
        ? Data("{}".utf8)
        : controllerPhase ? fixture.profileData : (fixture.offroadProfileData ?? fixture.profileData),
      .liveMapDataControllerStatus: Data((controllerPhase && !pendingControllerPhase
        ? "1|1|\(fixture.controllerLogMonoTimeNs)|\(fixture.controllerRoadGeometryValid ? 1 : 0)|\(fixture.controllerSampleMonoTimeNs)"
        : "1|0|123456790|0|123456999").utf8),
      .runtimeEndIsOffroad: Data(((stableOnroadPhase
        ? fixture.controllerEndIsOffroad : fixture.offroadEndIsOffroad) ? "1" : "0").utf8),
      .runtimeEndIsOnroad: Data(((stableOnroadPhase
        ? fixture.controllerEndIsOnroad : fixture.offroadEndIsOnroad) ? "1" : "0").utf8),
      .runtimeEndMapLookaheadEnabled: Data(((stableOnroadPhase
        ? fixture.controllerEndMapLookaheadEnabled : fixture.offroadEndMapLookaheadEnabled) ? "1" : "0").utf8),
    ]
    if !pendingControllerPhase, let gpsData = fixture.gpsData {
      fields[.memoryLastGPSPosition] = gpsData
    }
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
