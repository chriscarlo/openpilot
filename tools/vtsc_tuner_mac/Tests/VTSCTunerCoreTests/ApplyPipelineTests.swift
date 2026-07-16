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
    activeTileSetID: nil,
    activeTileTopology: .directUnidentified
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
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
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
  #expect(completed.completedDeviceHead == fixture.deployedTargetHead)
  #expect(completed.completionHostOnlyPaths == [])

  let requests = await runner.requests
  #expect(requests.contains {
    $0.executableURL == ApplyPipeline.gitURL &&
      $0.arguments == ["merge-base", "--is-ancestor", fixture.deployedTargetHead, fixture.toolingHead]
  })
  #expect(!requests.contains {
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

@Test func newOutdoorResumeRequiresChangedDeploymentBootIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  fixture.bootIDsByRead = [fixture.journal.deploymentPreRebootBootID]
  try fixture.journal.write(to: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)

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
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test func legacyB174ShapeAtCompletionCompatibleSuccessorRemainsResumable() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let compatibleSuccessor = String(repeating: "c", count: 40)
  fixture.journal.resolution = nil
  fixture.journal.deploymentPreRebootBootID = nil
  fixture.runtimeHead = compatibleSuccessor
  try fixture.journal.write(to: fixture.journalURL)
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
    ) { _ in }
  #expect(succeeded)
  let completed = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(completed.effectiveResolution == .completed)
  #expect(completed.completedDeviceHead == compatibleSuccessor)
  #expect(completed.completionHostOnlyPaths == fixture.changedPaths.sorted())
}

@Test func explicitAwaitingPostflightWithoutBootIdentityIsNotLegacy() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingPostflight
  fixture.journal.deploymentPreRebootBootID = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)
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
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test func awaitingOutdoorPostflightWithoutBootIdentityIsNotTreatedAsLegacy() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  fixture.journal.deploymentPreRebootBootID = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = ResumePostflightRunner(fixture: fixture)
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
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test(arguments: [[false], [true, false]])
func resumePostflightRequiresManagerForControllerAndFinalOffroadProof(managerByRead: [Bool]) async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  let before = try Data(contentsOf: fixture.journalURL)
  fixture.managerRunningByRead = managerByRead
  let runner = ResumePostflightRunner(fixture: fixture)

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
  #expect(await runner.runtimeReadCount == managerByRead.count)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
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
        // Leave enough wall-clock headroom for the full parallel suite; this
        // test verifies retry classification, not deadline exhaustion.
        timeout: 2,
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
  fixture.changedPaths = ["selfdrive/controls/lib/longitudinal_planner.py"]
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
  // Keep the first phase close enough to its deadline that one phase-wide
  // deadline would expire during the offroad poll, while leaving enough wall
  // clock margin for this timing contract to remain stable in the full
  // parallel test suite.
  fixture.firstRuntimeDelay = .milliseconds(400)
  fixture.offroadWaitReads = 1
  let runner = ResumePostflightRunner(fixture: fixture)

  let succeeded = await ApplyPipeline(processRunner: runner).resumePendingPostflight(
    ResumePostflightRequest(
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      mapdReleaseManifestURL: fixture.releaseURL,
      journalURL: fixture.journalURL,
      timeout: 0.5,
      pollInterval: 0.15
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

@Test func released5caProductionPreflightFailsClosedForEveryNewUnresolvedLifecycle() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  for resolution in [
    DeploymentRollbackJournal.Resolution.preflightReserved,
    .mutationInProgress,
    .awaitingOutdoorPostflight,
  ] {
    fixture.journal.resolution = resolution
    fixture.journal.completed = resolution != .awaitingOutdoorPostflight
    fixture.journal.rebootSent = resolution == .awaitingOutdoorPostflight
    try fixture.journal.write(to: fixture.journalURL)
    #expect(throws: (any Error).self) {
      _ = try JSONDecoder().decode(
        Released5caDeploymentJournal.self,
        from: Data(contentsOf: fixture.journalURL)
      )
    }
  }

  for resolution in [
    DeploymentRollbackJournal.Resolution.rollbackInProgress,
    .rollbackFailed,
  ] {
    fixture.journal.resolution = resolution
    fixture.journal.completed = true
    fixture.journal.rebootSent = false
    try fixture.journal.write(to: fixture.journalURL)
    let old = try JSONDecoder().decode(
      Released5caDeploymentJournal.self,
      from: Data(contentsOf: fixture.journalURL)
    )
    #expect(old.blocksProductionPreflight)
  }
}

@Test func orphanedCurrentReservationIsRemovedByTheNextGlobalOwnerWithoutAgeDelay() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.targetHead = nil
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .preflightReserved
  try fixture.journal.write(to: fixture.journalURL)

  let owner = try DeploymentRollbackJournal.acquireProductionOwnerLock(
    directory: fixture.journalURL.deletingLastPathComponent()
  )
  defer { owner.unlock() }
  try DeploymentRollbackJournal.removeAbandonedPreflightReservations(
    directory: fixture.journalURL.deletingLastPathComponent()
  )
  #expect(!FileManager.default.fileExists(atPath: fixture.journalURL.path))
}

@Test func targetBearingPreflightReservationIsRemovedOnlyAfterOwnerCrash() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .preflightReserved
  try fixture.journal.write(to: fixture.journalURL)
  let directory = fixture.journalURL.deletingLastPathComponent()

  let liveOwner = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  #expect(throws: DeploymentRollbackJournalError.productionOwnerLocked(
    directory.appendingPathComponent(".production-owner.lock").standardizedFileURL
  )) {
    _ = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  }
  #expect(FileManager.default.fileExists(atPath: fixture.journalURL.path))
  liveOwner.unlock()

  let recoveryOwner = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  defer { recoveryOwner.unlock() }
  try DeploymentRollbackJournal.removeAbandonedPreflightReservations(directory: directory)
  #expect(!FileManager.default.fileExists(atPath: fixture.journalURL.path))
}

@Test func peerTunerProcessDetectionIsExactAndExcludesCurrentPID() {
  let lines = """
      100 /Applications/VTSC Tuner.app/Contents/MacOS/VTSCTuner
      101 /tmp/VTSCTuner-helper
      102 /Applications/Other.app/Contents/MacOS/Other --label VTSCTuner
      103 /Users/test/VTSC Tuner.app/Contents/MacOS/VTSCTuner
  """
  let peers = ApplyPipeline.otherTunerProcessLines(lines, excluding: 100)
  #expect(peers.count == 1)
  #expect(peers[0].hasPrefix("pid=103 "))
}

@Test func liveOldTunerPeerPreventsLegacyPruningAndAnyPreflightWork() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.targetHead = nil
  fixture.journal.rebootSent = false
  fixture.journal.completed = false
  fixture.journal.resolution = nil
  fixture.journal.createdAt = Date(timeIntervalSinceNow: -3_600).ISO8601Format()
  try fixture.journal.write(to: fixture.journalURL)
  let runner = ProductionPreflightRunner(
    head: fixture.toolingHead,
    processList: "999 /Applications/VTSC Tuner.app/Contents/MacOS/VTSCTuner\n"
  )
  await #expect(throws: ApplyPipelineError.self) {
    _ = try await ApplyPipeline(
      processRunner: runner,
      adbURL: nil,
      journalWriter: { journal, url in try journal.write(to: url) },
      currentProcessID: 100
    ).productionPreflight(ApplyRequest(
      action: .pullOnTici,
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      preferredTiciProfile: "commaAdb",
      mapdReleaseManifestURL: fixture.releaseURL,
      rollbackJournalDirectoryURL: fixture.journalURL.deletingLastPathComponent(),
      verificationRequests: []
    ))
  }
  #expect(FileManager.default.fileExists(atPath: fixture.journalURL.path))
  #expect(!(await runner.requests).contains { $0.executableURL == ApplyPipeline.sshURL })
}

@Test func unknownReservationPrecedesPeerScanAndFreshBaselineStopsOldFirstDrift() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let journalDirectory = fixture.root.appendingPathComponent("gap-journals", isDirectory: true)
  let driftedHead = String(repeating: "d", count: 40)
  let runner = OldFirstGapRunner(
    baselineHead: fixture.toolingHead,
    driftedHead: driftedHead,
    journalDirectory: journalDirectory
  )
  let pipeline = ApplyPipeline(processRunner: runner)
  let preflight = try await pipeline.productionPreflight(ApplyRequest(
    action: .pullOnTici,
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    preferredTiciProfile: "commaAdb",
    mapdReleaseManifestURL: fixture.releaseURL,
    rollbackJournalDirectoryURL: journalDirectory,
    verificationRequests: []
  ))
  defer {
    preflight.productionOwnerLock?.unlock()
    try? FileManager.default.removeItem(at: preflight.journalURL)
  }

  #expect(await runner.reservationWasPublishedBeforeFirstPeerScan)
  await #expect(throws: ApplyPipelineError.self) {
    try await pipeline.validateExclusiveProductionMutationClaim(preflight)
  }
  #expect(await runner.staticSnapshotReadCount == 2)
  let retained = try DeploymentRollbackJournal.load(from: preflight.journalURL)
  #expect(retained.effectiveResolution == .preflightReserved)
  #expect(retained.previousHead == fixture.toolingHead)
}

@Test func rollbackVerificationFailsImmediatelyOnHardMismatchEvenIfNextReadWouldHeal() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeBranches: ["wrong-branch", fixture.journal.branch],
    runtimeProcessReady: [false, true]
  )
  do {
    _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
      context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
      tilesWereTouched: false,
      timeout: 1,
      pollInterval: 0.001
    )
    Issue.record("Expected hard rollback branch mismatch to fail immediately")
  } catch {}
  #expect(await runner.runtimeReadCount == 1)
}

@Test func rollbackVerificationRetriesTransportThenAcceptsExactIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeTransportFailures: 1
  )
  _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
    context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
    tilesWereTouched: false,
    timeout: 5,
    pollInterval: 0.001
  )
  #expect(await runner.runtimeReadCount == 2)
}

@Test func rollbackVerificationRetriesOnlyRuntimeStartupThenAcceptsExactIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeProcessReady: [false, true]
  )
  _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
    context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
    tilesWereTouched: false,
    timeout: 1,
    pollInterval: 0.001
  )
  #expect(await runner.runtimeReadCount == 2)
}

@Test func staticPostRebootInstallationRetriesStartupAndRequiresExactIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.runtimeHead = fixture.toolingHead
  fixture.managerRunningByRead = [false, true]
  let runner = ResumePostflightRunner(fixture: fixture)
  _ = try await ApplyPipeline(processRunner: runner).waitForStaticInstalledIdentity(
    preflight: try resumeRuntimePreflight(fixture),
    targetHead: fixture.toolingHead,
    identity: TuneDeploymentIdentity(tune: fixture.tune),
    expectedActiveTileSetID: nil,
    timeout: 1,
    pollInterval: 0.001
  )
  #expect(await runner.runtimeReadCount == 2)

  fixture.runtimeBranch = "wrong-branch"
  fixture.managerRunningByRead = [true, true]
  let hardRunner = ResumePostflightRunner(fixture: fixture)
  await #expect(throws: ApplyPipelineError.self) {
    _ = try await ApplyPipeline(processRunner: hardRunner).waitForStaticInstalledIdentity(
      preflight: try resumeRuntimePreflight(fixture),
      targetHead: fixture.toolingHead,
      identity: TuneDeploymentIdentity(tune: fixture.tune),
      expectedActiveTileSetID: nil,
      timeout: 1,
      pollInterval: 0.001
    )
  }
  #expect(await hardRunner.runtimeReadCount == 1)
}

@Test func staticInstallRequiresANewBootIdentityAndBoundsNoOpRebootWait() async throws {
  let oldBootID = "11111111-1111-4111-8111-111111111111"
  let newBootID = "22222222-2222-4222-8222-222222222222"
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.runtimeHead = fixture.toolingHead
  fixture.journal.deploymentPreRebootBootID = oldBootID
  fixture.bootIDsByRead = [oldBootID, newBootID]
  let runner = ResumePostflightRunner(fixture: fixture)
  _ = try await ApplyPipeline(processRunner: runner).waitForStaticInstalledIdentity(
    preflight: try resumeRuntimePreflight(fixture),
    targetHead: fixture.toolingHead,
    identity: TuneDeploymentIdentity(tune: fixture.tune),
    expectedActiveTileSetID: nil,
    timeout: 1,
    pollInterval: 0.001
  )
  #expect(await runner.runtimeReadCount == 2)

  fixture.bootIDsByRead = [oldBootID]
  let noOpRunner = ResumePostflightRunner(fixture: fixture)
  do {
    _ = try await ApplyPipeline(processRunner: noOpRunner).waitForStaticInstalledIdentity(
      preflight: try resumeRuntimePreflight(fixture),
      targetHead: fixture.toolingHead,
      identity: TuneDeploymentIdentity(tune: fixture.tune),
      expectedActiveTileSetID: nil,
      timeout: 0.004,
      pollInterval: 0.001
    )
    Issue.record("Expected unchanged boot identity to time out")
  } catch {
    #expect(error.localizedDescription.contains("boot identity has not changed"))
  }
  #expect(await noOpRunner.runtimeReadCount >= 1)
}

@Test func staticInstallRetriesMissingPostBootIdentityButRejectsLegacyMissingPreBootProof() async throws {
  let newBootID = "22222222-2222-4222-8222-222222222222"
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.runtimeHead = fixture.toolingHead
  fixture.bootIDsByRead = [nil, newBootID]
  let missingThenReady = ResumePostflightRunner(fixture: fixture)
  _ = try await ApplyPipeline(processRunner: missingThenReady).waitForStaticInstalledIdentity(
    preflight: try resumeRuntimePreflight(fixture),
    targetHead: fixture.toolingHead,
    identity: TuneDeploymentIdentity(tune: fixture.tune),
    expectedActiveTileSetID: nil,
    timeout: 1,
    pollInterval: 0.001
  )
  #expect(await missingThenReady.runtimeReadCount == 2)

  fixture.journal.deploymentPreRebootBootID = nil
  fixture.bootIDsByRead = [newBootID]
  let legacy = ResumePostflightRunner(fixture: fixture)
  do {
    _ = try await ApplyPipeline(processRunner: legacy).waitForStaticInstalledIdentity(
      preflight: try resumeRuntimePreflight(fixture),
      targetHead: fixture.toolingHead,
      identity: TuneDeploymentIdentity(tune: fixture.tune),
      expectedActiveTileSetID: nil,
      timeout: 1,
      pollInterval: 0.001
    )
    Issue.record("Expected legacy journal without pre-reboot proof to fail closed")
  } catch {
    #expect(error.localizedDescription.contains("predates durable boot-transition proof"))
  }
  #expect(await legacy.runtimeReadCount == 1)
}

@Test func rollbackUsesStaticSnapshotAndRequiresItsOwnBootTransition() async throws {
  let oldBootID = "11111111-1111-4111-8111-111111111111"
  let newBootID = "22222222-2222-4222-8222-222222222222"
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  fixture.journal.rollbackPreRebootBootID = oldBootID
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeBootIDs: [oldBootID, newBootID],
    runtimeBuildEvidenceAvailable: false
  )
  _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
    context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
    tilesWereTouched: false,
    timeout: 1,
    pollInterval: 0.001
  )
  #expect(await runner.runtimeReadCount == 2)
  let requests = await runner.requests
  #expect(requests.contains { request in
    let command = request.arguments.last ?? ""
    return command.contains("manager_running") &&
      !command.contains("active_mapd_build_info") &&
      !command.contains("live_map_data_controller_status") &&
      !command.contains("remote_epoch_milliseconds")
  })

  let noOpRunner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeBootIDs: [oldBootID]
  )
  do {
    _ = try await ApplyPipeline(processRunner: noOpRunner).waitForRollbackVerification(
      context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
      tilesWereTouched: false,
      timeout: 0.004,
      pollInterval: 0.001
    )
    Issue.record("Expected rollback with unchanged boot identity to time out")
  } catch {
    #expect(error.localizedDescription.contains("boot identity has not changed"))
  }
  #expect(await noOpRunner.runtimeReadCount >= 1)
}

@Test func failedStaticPostRebootProofRetainsPendingJournalWithoutRollback() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .preflightReserved
  try fixture.journal.write(to: fixture.journalURL)
  let pipeline = ApplyPipeline(processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false))
  var claimed = try await pipeline.claimProductionMutation(expected: fixture.journal, at: fixture.journalURL)
  claimed.rebootSent = true
  try claimed.write(to: fixture.journalURL)
  let pending = try await pipeline.handoffToOutdoorPostflight(expected: claimed, at: fixture.journalURL)
  let before = try Data(contentsOf: fixture.journalURL)

  fixture.runtimeBranch = "wrong-branch"
  let runner = ResumePostflightRunner(fixture: fixture)
  await #expect(throws: ApplyPipelineError.self) {
    _ = try await ApplyPipeline(processRunner: runner).waitForStaticInstalledIdentity(
      preflight: RuntimeDeploymentPreflight(
        repositoryRoot: fixture.repository,
        git: GitDeploymentPreflight(
          branch: fixture.journal.branch,
          localHead: fixture.toolingHead,
          originHead: fixture.toolingHead,
          upstream: "origin/chauffeur-exp01"
        ),
        profile: fixture.journal.profile,
        mapdRecoveryOutcome: .clean,
        release: try fixture.release.validated(),
        tileSet: nil,
        journal: pending,
        journalURL: fixture.journalURL
      ),
      targetHead: fixture.toolingHead,
      identity: TuneDeploymentIdentity(tune: fixture.tune),
      expectedActiveTileSetID: nil,
      timeout: 1,
      pollInterval: 0.001
    )
  }
  #expect(try Data(contentsOf: fixture.journalURL) == before)
  #expect(!(await runner.requests).contains {
    ($0.arguments.last ?? "").contains(TiciProductionRollbackCommandBuilder.resultMarker)
  })
}

@Test func rollbackVerificationRejectsActiveTileWhenRecordedPriorWasNil() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousTileSetID = nil
  fixture.journal.targetTileSetID = nil
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: String(repeating: "e", count: 64)
  )
  do {
    _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
      context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
      tilesWereTouched: false,
      timeout: 1,
      pollInterval: 0.001
    )
    Issue.record("Expected unexpected active tile identity to fail rollback verification")
  } catch {}
  #expect(await runner.runtimeReadCount == 1)
}

@Test func rollbackVerificationRejectsOnroadAndLookaheadEndBracketStraddles() async throws {
  for (endOnroad, endLookahead) in [(true, false), (false, true)] {
    let fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let runner = FreshProcessRollbackRecoveryRunner(
      journal: fixture.journal,
      runtimeEndOnroad: endOnroad,
      runtimeEndLookahead: endLookahead
    )
    do {
      _ = try await ApplyPipeline(processRunner: runner).waitForRollbackVerification(
        context: ProductionRollbackContext(repositoryRoot: fixture.repository, journal: fixture.journal, journalURL: fixture.journalURL),
        tilesWereTouched: false,
        timeout: 1,
        pollInterval: 0.001
      )
      Issue.record("Expected unstable rollback end bracket to fail")
    } catch {}
    #expect(await runner.runtimeReadCount == 1)
  }
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

  let succeeded = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).recoverPendingRollback(
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
  #expect(requests.contains { $0.arguments.joined(separator: " ").contains("sudo reboot") })
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.rollbackPreRebootBootID != nil)
}

@Test func watchdogCycleBeforeDeploymentRebootIntentStillForcesRollbackReboot() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  fixture.journal.previousBootID = "11111111-1111-4111-8111-111111111111"
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let watchdogBoot = "33333333-3333-4333-8333-333333333333"
  let finalBoot = "44444444-4444-4444-8444-444444444444"
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeBootIDs: [watchdogBoot, watchdogBoot, watchdogBoot, watchdogBoot, finalBoot]
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  )
  guard case .rolledBack = result else {
    Issue.record("watchdog-cycle recovery did not settle rolledBack")
    return
  }
  #expect((await runner.requests).contains {
    $0.arguments.joined(separator: " ").contains("sudo reboot")
  })
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).rollbackPreRebootBootID == watchdogBoot)
}

@Test func legacyMissingDeploymentBaselineStillForcesProvenRollbackReboot() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  fixture.journal.previousBootID = nil
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let rollbackBoot = "55555555-5555-4555-8555-555555555555"
  let finalBoot = "66666666-6666-4666-8666-666666666666"
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeBootIDs: [rollbackBoot, rollbackBoot, rollbackBoot, rollbackBoot, finalBoot]
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  )
  guard case .rolledBack = result else {
    Issue.record("legacy missing-baseline recovery did not settle rolledBack")
    return
  }
  #expect((await runner.requests).contains {
    $0.arguments.joined(separator: " ").contains("sudo reboot")
  })
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).rollbackPreRebootBootID == rollbackBoot)
}

@Test func terminalResumeMismatchOffersSelectedAbortAndRollbackRecovery() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  fixture.runtimeDirty = true
  let events = ApplyEventCollector()
  let resumed = await ApplyPipeline(processRunner: ResumePostflightRunner(fixture: fixture))
    .resumePendingPostflight(
      ResumePostflightRequest(
        tune: fixture.tune,
        repositoryRoot: fixture.repository,
        mapdReleaseManifestURL: fixture.releaseURL,
        journalURL: fixture.journalURL,
        timeout: 1,
        pollInterval: 0.001
      )
    ) { await events.append($0) }
  #expect(!resumed)
  #expect((await events.events).contains { event in
    guard case let .step(step) = event else { return false }
    return step.id == 3 && step.text.contains(AbortPendingDeploymentAction.label)
  })
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .awaitingOutdoorPostflight)

  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)
  let aborted = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).abortPendingDeployment(
    AbortPendingDeploymentRequest(
      repositoryRoot: fixture.repository,
      journalURL: fixture.journalURL
    )
  ) { _ in }

  #expect(aborted)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rolledBack)
  #expect((await runner.requests).contains {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  })
}

@Test func abortRejectsReleasedTunerPeerBeforeRollbackMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  var agedForeign = fixture.journal
  agedForeign.deploymentID = UUID()
  agedForeign.createdAt = Date().addingTimeInterval(-1_200).ISO8601Format()
  agedForeign.targetHead = nil
  agedForeign.rebootSent = false
  agedForeign.completed = false
  agedForeign.resolution = nil
  let agedForeignURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(agedForeign.deploymentID.uuidString).json")
  try agedForeign.write(to: agedForeignURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    peerTunerPresent: true
  )

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  #expect(FileManager.default.fileExists(atPath: agedForeignURL.path))
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("rollback --root") == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func abortRejectsForeignUnresolvedJournalBeforeRollbackMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  var foreign = fixture.journal
  foreign.deploymentID = UUID()
  foreign.createdAt = Date().addingTimeInterval(1).ISO8601Format()
  let foreignURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(foreign.deploymentID.uuidString).json")
  try foreign.write(to: foreignURL)
  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  #expect(FileManager.default.fileExists(atPath: foreignURL.path))
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("rollback --root") == true })
}

@Test func abortPrunesAgedTargetlessLegacyForeignThenRecoversSelectedB174() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = nil
  fixture.journal.deploymentPreRebootBootID = nil
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)

  var agedForeign = fixture.journal
  agedForeign.deploymentID = UUID()
  agedForeign.createdAt = Date().addingTimeInterval(-1_200).ISO8601Format()
  agedForeign.targetHead = nil
  agedForeign.rebootSent = false
  agedForeign.completed = false
  agedForeign.resolution = nil
  let agedForeignURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(agedForeign.deploymentID.uuidString).json")
  try agedForeign.write(to: agedForeignURL)
  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)

  let succeeded = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(succeeded)
  #expect(!FileManager.default.fileExists(atPath: agedForeignURL.path))
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rolledBack)
}

@Test func abortRetainsAndBlocksOnFreshTargetlessLegacyForeign() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  var freshForeign = fixture.journal
  freshForeign.deploymentID = UUID()
  freshForeign.createdAt = Date().ISO8601Format()
  freshForeign.targetHead = nil
  freshForeign.rebootSent = false
  freshForeign.completed = false
  freshForeign.resolution = nil
  let freshForeignURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(freshForeign.deploymentID.uuidString).json")
  try freshForeign.write(to: freshForeignURL)
  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(FileManager.default.fileExists(atPath: freshForeignURL.path))
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
}

@Test func selectedTargetlessLegacyJournalIsNeverPrunedByExclusionCheckpoint() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.createdAt = Date().addingTimeInterval(-1_200).ISO8601Format()
  fixture.journal.targetHead = nil
  fixture.journal.rebootSent = false
  fixture.journal.completed = false
  fixture.journal.resolution = nil
  try fixture.journal.write(to: fixture.journalURL)

  let owner = try DeploymentRollbackJournal.acquireProductionOwnerLock(
    directory: fixture.journalURL.deletingLastPathComponent()
  )
  defer { owner.unlock() }
  try DeploymentRollbackJournal.removeAbandonedPreflightReservations(
    directory: fixture.journalURL.deletingLastPathComponent(),
    excluding: fixture.journalURL,
    excludingJournal: fixture.journal
  )

  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
}

@Test func recoverRejectsReleasedTunerPeerBeforeRollbackMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    peerTunerPresent: true
  )

  let succeeded = await ApplyPipeline(processRunner: runner).recoverPendingRollback(
    RollbackRecoveryRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("rollback --root") == true })
}

@Test func recoverRejectsForeignUnresolvedJournalBeforeRollbackMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  try fixture.journal.write(to: fixture.journalURL)
  var foreign = fixture.journal
  foreign.deploymentID = UUID()
  foreign.createdAt = Date().addingTimeInterval(1).ISO8601Format()
  let foreignURL = fixture.journalURL.deletingLastPathComponent()
    .appendingPathComponent("\(foreign.deploymentID.uuidString).json")
  try foreign.write(to: foreignURL)
  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)

  let succeeded = await ApplyPipeline(processRunner: runner).recoverPendingRollback(
    RollbackRecoveryRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("rollback --root") == true })
}

@Test func abortAllowsB174AtHostProvenCompletionCompatibleSuccessor() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let compatibleSuccessor = String(repeating: "c", count: 40)
  let toolingHead = String(repeating: "d", count: 40)
  fixture.journal.resolution = nil
  fixture.journal.deploymentPreRebootBootID = nil
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeHeads: Array(repeating: compatibleSuccessor, count: 4),
    hostToolingHead: toolingHead,
    gitChangedPaths: [
      ".codex/skills/vtsc-tuner-app/references/changelog.md",
      "tools/vtsc_tuner_mac/Sources/VTSCTunerCore/ApplyPipeline.swift",
    ]
  )

  let succeeded = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .rolledBack)
  let requests = await runner.requests
  #expect(requests.contains {
    $0.arguments == ["merge-base", "--is-ancestor", fixture.deployedTargetHead, compatibleSuccessor]
  })
  #expect(requests.contains {
    $0.arguments.last?.contains("expected_current_head='\(compatibleSuccessor)'") == true
  })
}

@Test func abortRejectsProductionChangingSuccessorAndWrongHeadWithoutRemoteMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let productionSuccessor = String(repeating: "c", count: 40)
  fixture.journal.resolution = .awaitingOutdoorPostflight
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeHeads: [productionSuccessor],
    hostToolingHead: String(repeating: "d", count: 40),
    gitChangedPaths: ["selfdrive/controls/lib/longitudinal_lead_helpers.py"]
  )

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(repositoryRoot: fixture.repository, journalURL: fixture.journalURL)
  ) { _ in }

  #expect(!succeeded)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == fixture.journal)
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("rollback --root") == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func tileRollbackFailureStopsGitParamsAndMapdRestoration() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.targetTileSetID = String(repeating: "f", count: 64)
  fixture.journal.previousTileSetID = String(repeating: "e", count: 64)
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: fixture.journal.targetTileSetID,
    tileRollbackFails: true
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: true
  )

  guard case let .rollbackFailed(detail) = result else {
    Issue.record("tile precondition failure did not retain rollback failure")
    return
  }
  #expect(detail.contains("before Git/Params/mapd restoration"))
  let requests = await runner.requests
  #expect(requests.contains { $0.arguments.last?.contains("rollback --root") == true })
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func legacyDirectTileActivationIdentityIsDurableBeforeContinuation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  let legacy = "legacy-0123456789abcdef"
  fixture.journal.completed = true
  fixture.journal.resolution = .mutationInProgress
  fixture.journal.previousTileSetID = nil
  fixture.journal.targetTileSetID = target
  try fixture.journal.write(to: fixture.journalURL)

  let recorded = try await ApplyPipeline().durablyRecordResolvedPreviousTileIdentity(
    expected: fixture.journal,
    activatedTileSetID: target,
    resolvedPreviousTileSetID: legacy,
    at: fixture.journalURL
  )

  #expect(recorded.previousTileSetID == nil)
  #expect(recorded.resolvedPreviousTileSetID == legacy)
  #expect(recorded.effectivePreviousTileSetID == legacy)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == recorded)
}

@Test func sameTargetNoSwitchPersistsExplicitOutcomeWithSamePreviousIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  fixture.journal.completed = true
  fixture.journal.resolution = .mutationInProgress
  fixture.journal.previousTileSetID = target
  fixture.journal.resolvedPreviousTileSetID = nil
  fixture.journal.targetTileSetID = target
  try fixture.journal.write(to: fixture.journalURL)
  let before = try Data(contentsOf: fixture.journalURL)

  let reconciled = try await ApplyPipeline().reconcileTileActivationJournal(
    expected: fixture.journal,
    activation: TiciTileActivationResult(
      tileSetID: target,
      previousTileSetID: nil,
      targetAlreadyActive: true,
      commandOutput: "target already active"
    ),
    at: fixture.journalURL
  )

  #expect(reconciled.hasSameDeploymentIdentity(as: fixture.journal))
  #expect(reconciled.effectivePreviousTileSetID == target)
  #expect(reconciled.tileActivationOutcome == .notSwitched)
  #expect(try Data(contentsOf: fixture.journalURL) != before)

  var incoherent = fixture.journal
  incoherent.previousTileSetID = nil
  try incoherent.write(to: fixture.journalURL)
  await #expect(throws: ApplyPipelineError.self) {
    _ = try await ApplyPipeline().reconcileTileActivationJournal(
      expected: incoherent,
      activation: TiciTileActivationResult(
        tileSetID: target,
        previousTileSetID: nil,
        targetAlreadyActive: true,
        commandOutput: "target already active"
      ),
      at: fixture.journalURL
    )
  }
}

@Test func sameTargetNoSwitchLaterFailureRollsBackSourceAndReplaysAsTerminalSuccess() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  fixture.journal.completed = true
  fixture.journal.resolution = .mutationInProgress
  fixture.journal.previousTileSetID = target
  fixture.journal.targetTileSetID = target
  fixture.journal.tileActivationOutcome = nil
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let reconciled = try await ApplyPipeline().reconcileTileActivationJournal(
    expected: fixture.journal,
    activation: TiciTileActivationResult(
      tileSetID: target,
      previousTileSetID: nil,
      targetAlreadyActive: true,
      commandOutput: "canonical target already active without pointer mutation"
    ),
    at: fixture.journalURL
  )
  var rollbackJournal = reconciled
  rollbackJournal.resolution = .rollbackInProgress
  try rollbackJournal.write(to: fixture.journalURL)
  fixture.journal = rollbackJournal
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: rollbackJournal,
    runtimeActiveTileSetID: target,
    tileActivationNeverStarted: true
  )
  let pipeline = ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  )

  guard case .rolledBack = await pipeline.rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  ) else {
    Issue.record("same-target verified no-switch did not complete source/Params/mapd rollback")
    return
  }
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.effectiveResolution == .rolledBack)
  #expect(settled.tileActivationOutcome == .notSwitched)
  #expect(settled.effectivePreviousTileSetID == target)
  let firstRequests = await runner.requests.count
  #expect((await runner.requests).contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect((await runner.requests).contains { $0.arguments.last?.contains("sudo reboot") == true })

  guard case .rolledBack = await pipeline.rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  ) else {
    Issue.record("fresh same-target rollback retry did not return terminal rolledBack")
    return
  }
  #expect(await runner.requests.count == firstRequests)
}

@Test func beforeJournalDirectIdentityNoSwitchCompletesFullRollbackAndPostflight() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let previous = String(repeating: "c", count: 64)
  let target = String(repeating: "f", count: 64)
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.previousTileSetID = previous
  fixture.journal.targetTileSetID = target
  fixture.journal.tileActivationOutcome = nil
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: previous,
    tileActivationNeverStarted: true
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: false
  )

  guard case .rolledBack = result else {
    Issue.record("before-journal direct identity did not complete coherent rollback: \(result)")
    return
  }
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.effectiveResolution == .rolledBack)
  #expect(settled.effectivePreviousTileSetID == previous)
  #expect(settled.tileActivationOutcome == .notSwitched)
  #expect((await runner.requests).contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect((await runner.requests).contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test(arguments: [false, true])
func legacyDirectTileActivationRollbackAndInterruptedRetrySettle(
  tileActivationAlreadyRolledBack: Bool
) async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  let legacy = "legacy-0123456789abcdef"
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.previousTileSetID = nil
  fixture.journal.resolvedPreviousTileSetID = nil
  fixture.journal.targetTileSetID = target
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: legacy,
    resolvedLegacyTileSetID: legacy,
    tileActivationAlreadyRolledBack: tileActivationAlreadyRolledBack
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: true
  )

  guard case .rolledBack = result else {
    Issue.record("legacy direct-tree rollback did not settle after helper identity resolution: \(result)")
    return
  }
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.effectiveResolution == .rolledBack)
  #expect(settled.previousTileSetID == nil)
  #expect(settled.resolvedPreviousTileSetID == legacy)
  #expect((await runner.requests).contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func preservedCanonicalLegacyIdentityReconcilesAfterActivationReplyWasLost() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  let preservedCanonical = String(repeating: "c", count: 64)
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.previousTileSetID = nil
  fixture.journal.resolvedPreviousTileSetID = nil
  fixture.journal.targetTileSetID = target
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: preservedCanonical,
    resolvedLegacyTileSetID: preservedCanonical
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: true
  )

  guard case .rolledBack = result else {
    Issue.record("preserved canonical legacy identity did not reconcile: \(result)")
    return
  }
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.resolvedPreviousTileSetID == preservedCanonical)
  #expect(settled.effectiveResolution == .rolledBack)
  #expect((await runner.requests).contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func arbitraryUnboundPreviousTileIdentityStopsFreshRecoveryBeforeLaterMutation() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let target = String(repeating: "f", count: 64)
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.previousTileSetID = nil
  fixture.journal.resolvedPreviousTileSetID = nil
  fixture.journal.targetTileSetID = target
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    runtimeActiveTileSetID: target,
    resolvedLegacyTileSetID: String(repeating: "c", count: 64),
    resolvedTileProvenanceTargetID: String(repeating: "d", count: 64)
  )

  let result = await ApplyPipeline(processRunner: runner)
    .rollbackProductionDeploymentIfJournalPending(
      preflight: try resumeRuntimePreflight(fixture),
      tilesActivated: true
    )

  guard case let .rollbackFailed(detail) = result else {
    Issue.record("unbound previous tile identity did not fail closed")
    return
  }
  #expect(detail.contains("before Git/Params/mapd restoration"))
  let requests = await runner.requests
  #expect(!requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(!requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).resolvedPreviousTileSetID == nil)
}

@Test func preActivationTileFailureAllowsSourceRollbackAndKeepsNilPreviousIdentity() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .rollbackInProgress
  fixture.journal.previousTileSetID = nil
  fixture.journal.resolvedPreviousTileSetID = nil
  fixture.journal.targetTileSetID = String(repeating: "f", count: 64)
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let runner = FreshProcessRollbackRecoveryRunner(
    journal: fixture.journal,
    tileActivationNeverStarted: true
  )

  let result = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  ).rollbackProductionDeploymentIfJournalPending(
    preflight: try resumeRuntimePreflight(fixture),
    tilesActivated: true
  )

  guard case .rolledBack = result else {
    Issue.record("pre-activation tile failure did not permit coherent rollback: \(result)")
    return
  }
  let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
  #expect(settled.effectivePreviousTileSetID == nil)
  #expect(settled.effectiveResolution == .rolledBack)
  let requests = await runner.requests
  #expect(requests.contains { $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true })
  #expect(requests.contains { $0.arguments.last?.contains("sudo reboot") == true })
}

@Test func abortRejectsWrongJournalSelectionBeforeAnyRemoteRequest() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.completed = true
  fixture.journal.resolution = .completed
  try fixture.journal.write(to: fixture.journalURL)
  let runner = RollbackClaimRecoveryRunner(failRemoteRollback: false)

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(
      repositoryRoot: fixture.repository,
      journalURL: fixture.journalURL
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(await runner.requests.isEmpty)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .completed)
}

@Test func abortSafetyFailureLeavesAwaitingJournalAndIssuesZeroRollbackMutation() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let before = try Data(contentsOf: fixture.journalURL)
  let runner = UnsafeAbortRunner()

  let succeeded = await ApplyPipeline(processRunner: runner).abortPendingDeployment(
    AbortPendingDeploymentRequest(
      repositoryRoot: fixture.repository,
      journalURL: fixture.journalURL
    )
  ) { _ in }

  #expect(!succeeded)
  #expect(try Data(contentsOf: fixture.journalURL) == before)
  #expect(!(await runner.requests).contains {
    let command = $0.arguments.last ?? ""
    return command.contains(TiciProductionRollbackCommandBuilder.resultMarker) ||
      command.contains(" rollback --root ") || command.contains("sudo reboot")
  })
}

@Test func concurrentCompletionAndAbortSerializeThroughGlobalThenJournalLocks() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  let concurrentFixture = fixture
  let held = try DeploymentRollbackJournal.acquireProductionOwnerLock(
    directory: concurrentFixture.journalURL.deletingLastPathComponent()
  )
  let resumeRunner = ResumePostflightRunner(fixture: concurrentFixture)
  let abortRunner = FreshProcessRollbackRecoveryRunner(journal: concurrentFixture.journal)

  let resumeTask = Task {
    await ApplyPipeline(processRunner: resumeRunner).resumePendingPostflight(
      ResumePostflightRequest(
        tune: concurrentFixture.tune,
        repositoryRoot: concurrentFixture.repository,
        mapdReleaseManifestURL: concurrentFixture.releaseURL,
        journalURL: concurrentFixture.journalURL,
        timeout: 2,
        pollInterval: 0.001
      )
    ) { _ in }
  }
  let abortTask = Task {
    await ApplyPipeline(
      processRunner: abortRunner,
      rebootInitialDelayNanoseconds: 0,
      rebootPollDelayNanoseconds: 1
    ).abortPendingDeployment(
      AbortPendingDeploymentRequest(
        repositoryRoot: concurrentFixture.repository,
        journalURL: concurrentFixture.journalURL
      )
    ) { _ in }
  }
  try await Task.sleep(for: .milliseconds(100))
  held.unlock()

  let resumeSucceeded = await resumeTask.value
  let abortSucceeded = await abortTask.value
  let settled = try DeploymentRollbackJournal.load(from: concurrentFixture.journalURL)
  let rollbackRequests = (await abortRunner.requests).filter {
    $0.arguments.last?.contains(TiciProductionRollbackCommandBuilder.resultMarker) == true
  }
  switch settled.effectiveResolution {
  case .completed:
    #expect(resumeSucceeded)
    #expect(!abortSucceeded)
    #expect(rollbackRequests.isEmpty)
  case .rolledBack:
    #expect(!resumeSucceeded)
    #expect(abortSucceeded)
    #expect(rollbackRequests.count == 1)
  default:
    Issue.record("completion/abort race did not settle exactly once: \(settled.effectiveResolution)")
  }
}

@Test func preRenameDeploymentRebootIntentFailureStillRebootsSuccessfulRollback() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  fixture.journal.previousCachedMapdPath = ""
  fixture.journal.previousCachedMapdSHA256 = nil
  try fixture.journal.write(to: fixture.journalURL)
  var inMemoryPreflight = try resumeRuntimePreflight(fixture)
  inMemoryPreflight.journal.rebootSent = true
  let failingWriterPipeline = ApplyPipeline(
    processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false)
  ) { _, url in
    throw DeploymentRollbackJournalError.couldNotWrite(url, "injected pre-rename failure")
  }
  await #expect(throws: DeploymentRollbackJournalError.couldNotWrite(
    fixture.journalURL,
    "injected pre-rename failure"
  )) {
    try await failingWriterPipeline.durablyRecordRebootIntent(
      inMemoryPreflight.journal,
      at: fixture.journalURL
    )
  }
  #expect(!(try DeploymentRollbackJournal.load(from: fixture.journalURL).rebootSent))

  let runner = FreshProcessRollbackRecoveryRunner(journal: fixture.journal)
  guard case .rolledBack = await ApplyPipeline(
    processRunner: runner,
    rebootInitialDelayNanoseconds: 0,
    rebootPollDelayNanoseconds: 1
  )
    .rollbackProductionDeploymentIfJournalPending(
      preflight: inMemoryPreflight,
      tilesActivated: false
    )
  else { Issue.record("durable rebootSent=false rollback should succeed"); return }
  #expect((await runner.requests).contains {
    $0.arguments.joined(separator: " ").contains("sudo reboot")
  })
}

@Test func postRenameIndeterminateRebootIntentIsRetriedToADurableExactReadback() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  try fixture.journal.write(to: fixture.journalURL)
  var intended = fixture.journal
  intended.rebootSent = true
  let recorder = JournalWriterAttemptRecorder()
  let pipeline = ApplyPipeline(
    processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false)
  ) { journal, url in
    try journal.write(to: url)
    if recorder.recordAttempt() == 1 {
      throw DeploymentRollbackJournalError.committedButNotDurable(url, "injected post-rename failure")
    }
  }

  try await pipeline.durablyRecordRebootIntent(intended, at: fixture.journalURL)

  #expect(recorder.attemptCount == 2)
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL) == intended)
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

@Test func globalProductionOwnerSerializesDifferentWouldBeJournals() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-global-production-lock-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let first = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: root)
  let expectedLockURL = root.standardizedFileURL.appendingPathComponent(".production-owner.lock")
  #expect(throws: DeploymentRollbackJournalError.productionOwnerLocked(expectedLockURL)) {
    try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: root)
  }
  // Distinct per-journal locks do not conflict; the global owner is therefore
  // the required cross-journal serialization layer.
  let a = root.appendingPathComponent("a.json")
  let b = root.appendingPathComponent("b.json")
  let lockA = try DeploymentRollbackJournal.acquireCompletionLock(for: a)
  let lockB = try DeploymentRollbackJournal.acquireCompletionLock(for: b)
  lockA.unlock()
  lockB.unlock()
  first.unlock()
  let next = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: root)
  next.unlock()
}

@Test func abandonedTargetlessPreflightIsRemovedButTargetBearingLegacyStateIsRecoverable() throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let directory = fixture.journalURL.deletingLastPathComponent()
  fixture.journal.rebootSent = false
  fixture.journal.targetHead = nil
  fixture.journal.completed = false
  fixture.journal.resolution = nil
  fixture.journal.createdAt = Date(timeIntervalSinceNow: -3_600).ISO8601Format()
  try fixture.journal.write(to: fixture.journalURL)
  let targetedURL = directory.appendingPathComponent("target-bearing.json")
  var targeted = fixture.journal
  targeted.deploymentID = UUID()
  targeted.targetHead = String(repeating: "e", count: 40)
  try targeted.write(to: targetedURL)
  let freshURL = directory.appendingPathComponent("fresh-targetless.json")
  var fresh = fixture.journal
  fresh.deploymentID = UUID()
  fresh.createdAt = Date().ISO8601Format()
  try fresh.write(to: freshURL)

  let owner = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  defer { owner.unlock() }
  try DeploymentRollbackJournal.removeAbandonedPreflightReservations(directory: directory)

  #expect(!FileManager.default.fileExists(atPath: fixture.journalURL.path))
  #expect(FileManager.default.fileExists(atPath: targetedURL.path))
  #expect(FileManager.default.fileExists(atPath: freshURL.path))
  #expect(try DeploymentRollbackJournal.loadRecoverableRollback(from: targetedURL).journal == targeted)
  let unresolved = try DeploymentRollbackJournal.unresolvedProductionJournals(directory: directory)
  #expect(unresolved.contains { $0.url == freshURL.standardizedFileURL })
}

@Test func customJournalDirectoryOwnsScanCreationLockAndMutationClaimWithoutDefaultLeak() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let custom = fixture.root.appendingPathComponent("authoritative-journals", isDirectory: true)
  let defaultDirectory = try DeploymentRollbackJournal.defaultDirectory()
  let defaultBefore = try jsonFileNames(in: defaultDirectory)
  let runner = ProductionPreflightRunner(head: fixture.toolingHead)
  let pipeline = ApplyPipeline(processRunner: runner)

  let preflight = try await pipeline.productionPreflight(ApplyRequest(
    action: .pullOnTici,
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    preferredTiciProfile: "commaAdb",
    mapdReleaseManifestURL: fixture.releaseURL,
    rollbackJournalDirectoryURL: custom,
    verificationRequests: []
  ))
  defer { preflight.productionOwnerLock?.unlock() }

  #expect(preflight.journalURL.deletingLastPathComponent() == custom.standardizedFileURL)
  #expect(preflight.journalURL.lastPathComponent == "\(preflight.journal.deploymentID.uuidString).json")
  #expect(try jsonFileNames(in: custom) == [preflight.journalURL.lastPathComponent])
  let journalLock = try DeploymentRollbackJournal.acquireCompletionLock(for: preflight.journalURL)
  defer { journalLock.unlock() }
  let claimed = try await pipeline.claimProductionMutation(
    expected: preflight.journal,
    at: preflight.journalURL
  )
  #expect(claimed.effectiveResolution == .mutationInProgress)
  #expect(try jsonFileNames(in: defaultDirectory) == defaultBefore)
}

@Test func twoProductionPreflightsSerializeBeforeJournalCreationAndSecondIssuesNoRemoteRequest() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let directory = fixture.root.appendingPathComponent("concurrent-journals", isDirectory: true)
  let firstRunner = BlockingProductionPreflightRunner(head: fixture.toolingHead)
  let secondRunner = ProductionPreflightRunner(head: fixture.toolingHead)
  let request = ApplyRequest(
    action: .pullOnTici,
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    preferredTiciProfile: "commaAdb",
    mapdReleaseManifestURL: fixture.releaseURL,
    rollbackJournalDirectoryURL: directory,
    verificationRequests: []
  )
  let firstTask = Task {
    try await ApplyPipeline(processRunner: firstRunner).productionPreflight(request)
  }
  await firstRunner.waitUntilGlobalOwnerReachedRemoteBoundary()

  await #expect(throws: DeploymentRollbackJournalError.productionOwnerLocked(
    directory.standardizedFileURL.appendingPathComponent(".production-owner.lock").standardizedFileURL
  )) {
    _ = try await ApplyPipeline(processRunner: secondRunner).productionPreflight(request)
  }
  #expect(!(await secondRunner.requests).contains { $0.executableURL == ApplyPipeline.sshURL })

  await firstRunner.releaseRemoteBoundary()
  let first = try await firstTask.value
  defer { first.productionOwnerLock?.unlock() }
  #expect(try jsonFileNames(in: directory).count == 1)
}

@Test func lateUnresolvedJournalBlocksMutationClaimUnderGlobalOwner() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .preflightReserved
  try fixture.journal.write(to: fixture.journalURL)
  let directory = fixture.journalURL.deletingLastPathComponent()
  let owner = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  var preflight = try resumeRuntimePreflight(fixture)
  preflight.productionOwnerLock = owner

  var late = fixture.journal
  late.deploymentID = UUID()
  late.createdAt = Date().ISO8601Format()
  let lateURL = directory.appendingPathComponent("\(late.deploymentID.uuidString).json")
  try late.write(to: lateURL)

  let runner = ProductionPreflightRunner(head: fixture.toolingHead)
  await #expect(throws: ApplyPipelineError.self) {
    try await ApplyPipeline(
      processRunner: runner,
      adbURL: nil,
      journalWriter: { journal, url in try journal.write(to: url) },
      currentProcessID: 100
    ).validateExclusiveProductionMutationClaim(preflight)
  }
  #expect(try DeploymentRollbackJournal.load(from: fixture.journalURL).effectiveResolution == .preflightReserved)
  owner.unlock()
}

@Test func earlyHostFailureDurablyRemovesItsOwnTargetlessPreflightBeforeUnlock() async throws {
  let fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  let directory = fixture.root.appendingPathComponent("early-failure-journals", isDirectory: true)
  let runner = ProductionPreflightRunner(head: fixture.toolingHead)
  let succeeded = await ApplyPipeline(processRunner: runner).apply(ApplyRequest(
    action: .pullOnTici,
    tune: fixture.tune,
    repositoryRoot: fixture.repository,
    tuneURL: fixture.root, // writing a tune over an existing directory must fail
    preferredTiciProfile: "commaAdb",
    mapdReleaseManifestURL: fixture.releaseURL,
    rollbackJournalDirectoryURL: directory,
    verificationRequests: []
  )) { _ in }

  #expect(!succeeded)
  #expect(try jsonFileNames(in: directory).isEmpty)
  let nextOwner = try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
  nextOwner.unlock()
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
  fixture.runtimeHead = fixture.toolingHead
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
  #expect(requests.contains { $0.executableURL == ApplyPipeline.sshURL })
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

@Test func resumePostflightRejectsProductionChangingDeviceSuccessor() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.runtimeHead = fixture.toolingHead
  fixture.changedPaths = ["selfdrive/controls/lib/longitudinal_lead_helpers.py"]
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
  for state in ["awaitingPostflight", "rollbackFailed", "mutationInProgress"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    switch state {
    case "awaitingPostflight":
      fixture.journal.completed = false
      fixture.journal.rebootSent = true
      fixture.journal.resolution = .awaitingPostflight
    case "mutationInProgress":
      fixture.journal.completed = true
      fixture.journal.rebootSent = false
      fixture.journal.resolution = .mutationInProgress
    default:
      fixture.journal.completed = true
      fixture.journal.resolution = .rollbackFailed
    }
    try fixture.journal.write(to: fixture.journalURL)
    let tuneURL = fixture.root.appendingPathComponent("must-not-save.json")
    let runner = RollbackClaimRecoveryRunner(failRemoteRollback: false)
    let succeeded = await ApplyPipeline(processRunner: runner).apply(ApplyRequest(
      action: .pullOnTici,
      tune: fixture.tune,
      repositoryRoot: fixture.repository,
      tuneURL: tuneURL,
      rollbackJournalDirectoryURL: fixture.journalURL.deletingLastPathComponent()
    )) { _ in }
    #expect(!succeeded)
    #expect(!FileManager.default.fileExists(atPath: tuneURL.path))
    #expect(!(await runner.requests).contains { request in
      request.executableURL == ApplyPipeline.sshURL ||
        request.executableURL == ApplyPipeline.rsyncURL ||
        (request.executableURL == ApplyPipeline.gitURL &&
          ["fetch", "merge", "commit", "push", "reset"].contains(request.arguments.first ?? ""))
    })
  }
}

@Test func mutationClaimIsLegacySafeAndFreshProcessRecoverableAcrossEveryMutationBoundary() async throws {
  for boundary in ["before-git-ff", "after-git-ff", "after-mapd", "after-params", "after-tiles"] {
    var fixture = try resumePostflightFixture(validGPS: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    fixture.journal.rebootSent = false
    fixture.journal.completed = true
    fixture.journal.resolution = .preflightReserved
    fixture.journal.previousCachedMapdPath = ""
    fixture.journal.previousCachedMapdSHA256 = nil
    try fixture.journal.write(to: fixture.journalURL)
    let claimed = try await ApplyPipeline(processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false))
      .claimProductionMutation(expected: fixture.journal, at: fixture.journalURL)
    #expect(claimed.effectiveResolution == .mutationInProgress)
    #expect(claimed.completed)
    let legacy = try JSONDecoder().decode(
      LegacySchemaOnePendingReader.self,
      from: Data(contentsOf: fixture.journalURL)
    )
    #expect(!legacy.isPendingPostflight)
    #expect(try DeploymentRollbackJournal.loadRecoverableRollback(from: fixture.journalURL).journal == claimed)

    // Every remote mutation boundary deliberately retains the same durable
    // mutationInProgress authority. A newly launched pipeline must therefore
    // discover and settle it through guarded rollback, regardless of which
    // mutating command the original process completed before crashing.
    let recoveryRunner = FreshProcessRollbackRecoveryRunner(journal: claimed)
    let recovered = await ApplyPipeline(
      processRunner: recoveryRunner,
      rebootInitialDelayNanoseconds: 0,
      rebootPollDelayNanoseconds: 1
    ).recoverPendingRollback(
      RollbackRecoveryRequest(
        repositoryRoot: fixture.repository,
        journalURL: fixture.journalURL
      )
    ) { _ in }
    #expect(recovered, "fresh-process recovery failed at \(boundary)")
    let settled = try DeploymentRollbackJournal.load(from: fixture.journalURL)
    #expect(settled.effectiveResolution == .rolledBack, "wrong resolution at \(boundary)")
    #expect(settled.completed)
  }
}

@Test func originalDeploymentHandoffCannotCompleteFromRawProfileAndGPSAlone() async throws {
  var fixture = try resumePostflightFixture(validGPS: true)
  defer { try? FileManager.default.removeItem(at: fixture.root) }
  fixture.journal.rebootSent = false
  fixture.journal.completed = true
  fixture.journal.resolution = .preflightReserved
  try fixture.journal.write(to: fixture.journalURL)
  let pipeline = ApplyPipeline(processRunner: RollbackClaimRecoveryRunner(failRemoteRollback: false))
  var claimed = try await pipeline.claimProductionMutation(expected: fixture.journal, at: fixture.journalURL)
  claimed.rebootSent = true
  try claimed.write(to: fixture.journalURL)
  let handedOff = try await pipeline.handoffToOutdoorPostflight(expected: claimed, at: fixture.journalURL)
  #expect(handedOff.effectiveResolution == .awaitingOutdoorPostflight)
  #expect(!handedOff.completed)
  #expect(handedOff.completedAt == nil)
  #expect(try DeploymentRollbackJournal.loadPendingPostflight(from: fixture.journalURL).journal == handedOff)
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

private final class JournalWriterAttemptRecorder: @unchecked Sendable {
  private let lock = NSLock()
  private var attempts = 0

  var attemptCount: Int {
    lock.withLock { attempts }
  }

  func recordAttempt() -> Int {
    lock.withLock {
      attempts += 1
      return attempts
    }
  }
}

private func jsonFileNames(in directory: URL) throws -> [String] {
  guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
  return try FileManager.default.contentsOfDirectory(
    at: directory,
    includingPropertiesForKeys: nil,
    options: [.skipsHiddenFiles]
  ).filter { $0.pathExtension == "json" }.map(\.lastPathComponent).sorted()
}

private actor ProductionPreflightRunner: ProcessRunning {
  let head: String
  let processList: String
  private(set) var requests: [ProcessRequest] = []

  init(head: String, processList: String = "") {
    self.head = head
    self.processList = processList
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == ApplyPipeline.processListURL { return success(processList) }
    if request.executableURL == ApplyPipeline.gitURL {
      switch request.arguments {
      case ["branch", "--show-current"]: return success("chauffeur-exp01\n")
      case ["status", "--porcelain"]: return success("")
      case ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
        return success("origin/chauffeur-exp01\n")
      case ["rev-parse", "HEAD"]: return success(head + "\n")
      case ["ls-remote", "--heads", "origin", "refs/heads/chauffeur-exp01"]:
        return success("\(head)\trefs/heads/chauffeur-exp01\n")
      default: return success("")
      }
    }
    if request.executableURL == ApplyPipeline.networkSetupURL {
      return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unavailable")
    }
    if request.executableURL == ApplyPipeline.sshURL {
      let command = request.arguments.last ?? ""
      if command == "true" { return success("") }
      if command.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker) {
        return success("\(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker)\tclean\n")
      }
      return success(snapshotWire())
    }
    return success("")
  }

  private func snapshotWire() -> String {
    TiciSnapshotWireCodec.encode(.init(rawValues: [
      .bootID: Data("11111111-1111-4111-8111-111111111111".utf8),
      .branch: Data("chauffeur-exp01".utf8),
      .head: Data(head.utf8),
      .dirty: Data("0".utf8),
      .isOffroad: Data("1".utf8),
      .isOnroad: Data("0".utf8),
      .mapLookaheadEnabled: Data("0".utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: .checkoutFallback,
        bands: []
      ).utf8),
      .activeMapdSHA256: Data(String(repeating: "c", count: 64).utf8),
      .tileManifest: Data(),
      .tileTopology: Data("direct-unidentified".utf8),
      .mapdCacheListing: Data("".utf8),
      .activeMapdBuildInfo: Data("{}".utf8),
      .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
      .managerRunning: Data("1".utf8),
      .mapdRunning: Data("1".utf8),
      .runtimeEndIsOffroad: Data("1".utf8),
      .runtimeEndIsOnroad: Data("0".utf8),
      .runtimeEndMapLookaheadEnabled: Data("0".utf8),
    ])) + "\n"
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }
}

private actor OldFirstGapRunner: ProcessRunning {
  let baselineHead: String
  let driftedHead: String
  let journalDirectory: URL
  private(set) var reservationWasPublishedBeforeFirstPeerScan = false
  private(set) var staticSnapshotReadCount = 0
  private var processListReads = 0

  init(baselineHead: String, driftedHead: String, journalDirectory: URL) {
    self.baselineHead = baselineHead
    self.driftedHead = driftedHead
    self.journalDirectory = journalDirectory
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    if request.executableURL == ApplyPipeline.processListURL {
      processListReads += 1
      if processListReads == 1 {
        let candidates = try FileManager.default.contentsOfDirectory(
          at: journalDirectory,
          includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "json" }
        if let ownURL = candidates.first,
           let own = try? DeploymentRollbackJournal.load(from: ownURL),
           own.effectiveResolution == .preflightReserved {
          reservationWasPublishedBeforeFirstPeerScan = true
          var completedOld = own
          completedOld.deploymentID = UUID()
          completedOld.targetHead = baselineHead
          completedOld.completed = true
          completedOld.resolution = .completed
          let oldURL = journalDirectory.appendingPathComponent("\(completedOld.deploymentID.uuidString).json")
          try completedOld.write(to: oldURL)
        }
      }
      return success("")
    }
    if request.executableURL == ApplyPipeline.gitURL {
      switch request.arguments {
      case ["branch", "--show-current"]: return success("chauffeur-exp01\n")
      case ["status", "--porcelain"]: return success("")
      case ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
        return success("origin/chauffeur-exp01\n")
      case ["rev-parse", "HEAD"]: return success(baselineHead + "\n")
      case ["ls-remote", "--heads", "origin", "refs/heads/chauffeur-exp01"]:
        return success("\(baselineHead)\trefs/heads/chauffeur-exp01\n")
      default: return success("")
      }
    }
    if request.executableURL == ApplyPipeline.networkSetupURL {
      return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unavailable")
    }
    if request.executableURL == ApplyPipeline.sshURL {
      let command = request.arguments.last ?? ""
      if command == "true" { return success("") }
      if command.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker) {
        return success("\(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker)\tclean\n")
      }
      staticSnapshotReadCount += 1
      let head = staticSnapshotReadCount == 1 ? baselineHead : driftedHead
      return success(staticWire(head: head))
    }
    return success("")
  }

  private func staticWire(head: String) -> String {
    TiciSnapshotWireCodec.encode(.init(rawValues: [
      .bootID: Data("11111111-1111-4111-8111-111111111111".utf8),
      .branch: Data("chauffeur-exp01".utf8),
      .head: Data(head.utf8),
      .dirty: Data("0".utf8),
      .isOffroad: Data("1".utf8),
      .isOnroad: Data("0".utf8),
      .mapLookaheadEnabled: Data("0".utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: .checkoutFallback,
        bands: []
      ).utf8),
      .activeMapdSHA256: Data(String(repeating: "c", count: 64).utf8),
      .tileManifest: Data(),
      .tileTopology: Data("direct-unidentified".utf8),
      .mapdCacheListing: Data("".utf8),
      .activeMapdBuildInfo: Data("{}".utf8),
      .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
      .managerRunning: Data("1".utf8),
      .mapdRunning: Data("1".utf8),
      .runtimeEndIsOffroad: Data("1".utf8),
      .runtimeEndIsOnroad: Data("0".utf8),
      .runtimeEndMapLookaheadEnabled: Data("0".utf8),
    ])) + "\n"
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }
}

private actor BlockingProductionPreflightRunner: ProcessRunning {
  let head: String
  private var reached = false
  private var released = false
  private var reachedWaiters: [CheckedContinuation<Void, Never>] = []
  private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

  init(head: String) { self.head = head }

  func waitUntilGlobalOwnerReachedRemoteBoundary() async {
    if reached { return }
    await withCheckedContinuation { reachedWaiters.append($0) }
  }

  func releaseRemoteBoundary() {
    released = true
    let waiters = releaseWaiters
    releaseWaiters.removeAll()
    waiters.forEach { $0.resume() }
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    if request.executableURL == ApplyPipeline.processListURL { return success("") }
    if request.executableURL == ApplyPipeline.gitURL {
      switch request.arguments {
      case ["branch", "--show-current"]: return success("chauffeur-exp01\n")
      case ["status", "--porcelain"]: return success("")
      case ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
        return success("origin/chauffeur-exp01\n")
      case ["rev-parse", "HEAD"]: return success(head + "\n")
      case ["ls-remote", "--heads", "origin", "refs/heads/chauffeur-exp01"]:
        return success("\(head)\trefs/heads/chauffeur-exp01\n")
      default: return success("")
      }
    }
    if request.executableURL == ApplyPipeline.networkSetupURL {
      return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unavailable")
    }
    if request.executableURL == ApplyPipeline.sshURL {
      let command = request.arguments.last ?? ""
      if command == "true" {
        reached = true
        let waiters = reachedWaiters
        reachedWaiters.removeAll()
        waiters.forEach { $0.resume() }
        if !released {
          await withCheckedContinuation { releaseWaiters.append($0) }
        }
        return success("")
      }
      if command.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker) {
        return success("\(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker)\tclean\n")
      }
      return success(TiciSnapshotWireCodec.encode(.init(rawValues: [
        .bootID: Data("11111111-1111-4111-8111-111111111111".utf8),
        .branch: Data("chauffeur-exp01".utf8),
        .head: Data(head.utf8),
        .dirty: Data("0".utf8),
        .isOffroad: Data("1".utf8),
        .isOnroad: Data("0".utf8),
        .mapLookaheadEnabled: Data("0".utf8),
        .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
          parameters: .checkoutFallback,
          bands: []
        ).utf8),
        .activeMapdSHA256: Data(String(repeating: "c", count: 64).utf8),
        .tileManifest: Data(),
        .tileTopology: Data("direct-unidentified".utf8),
        .mapdCacheListing: Data("".utf8),
        .activeMapdBuildInfo: Data("{}".utf8),
        .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
        .managerRunning: Data("1".utf8),
        .mapdRunning: Data("1".utf8),
        .runtimeEndIsOffroad: Data("1".utf8),
        .runtimeEndIsOnroad: Data("0".utf8),
        .runtimeEndMapLookaheadEnabled: Data("0".utf8),
      ])) + "\n")
    }
    return success("")
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }
}

private struct LegacySchemaOnePendingReader: Decodable {
  var schema: Int
  var rebootSent: Bool
  var completed: Bool

  var isPendingPostflight: Bool { schema == 1 && rebootSent && !completed }
}

private struct Released5caDeploymentJournal: Decodable {
  enum Resolution: String, Decodable {
    case awaitingPostflight
    case completed
    case rollbackInProgress
    case rolledBack
    case rollbackFailed
  }

  var completed: Bool
  var resolution: Resolution?

  var effectiveResolution: Resolution {
    resolution ?? (completed ? .completed : .awaitingPostflight)
  }

  var blocksProductionPreflight: Bool {
    effectiveResolution == .rollbackInProgress ||
      effectiveResolution == .rollbackFailed ||
      (effectiveResolution == .rolledBack && !completed)
  }
}

private actor FreshProcessRollbackRecoveryRunner: ProcessRunning {
  let journal: DeploymentRollbackJournal
  let runtimeBranches: [String]
  let runtimeTransportFailures: Int
  let runtimeActiveTileSetID: String?
  let runtimeEndOnroad: Bool
  let runtimeEndLookahead: Bool
  let runtimeProcessReady: [Bool]
  let runtimeBootIDs: [String?]
  let runtimeBuildEvidenceAvailable: Bool
  let runtimeHeads: [String]
  let hostToolingHead: String
  let gitChangedPaths: [String]
  let peerTunerPresent: Bool
  let tileRollbackFails: Bool
  let resolvedLegacyTileSetID: String?
  let resolvedTileProvenanceTargetID: String?
  let tileActivationAlreadyRolledBack: Bool
  let tileActivationNeverStarted: Bool
  private(set) var requests: [ProcessRequest] = []
  private(set) var runtimeReadCount = 0
  private var rollbackMutationObserved = false
  private var rebootObserved = false

  init(
    journal: DeploymentRollbackJournal,
    runtimeBranches: [String] = [],
    runtimeTransportFailures: Int = 0,
    runtimeActiveTileSetID: String? = nil,
    runtimeEndOnroad: Bool = false,
    runtimeEndLookahead: Bool = false,
    runtimeProcessReady: [Bool] = [],
    runtimeBootIDs: [String?] = [],
    runtimeBuildEvidenceAvailable: Bool = true,
    runtimeHeads: [String] = [],
    hostToolingHead: String? = nil,
    gitChangedPaths: [String] = [],
    peerTunerPresent: Bool = false,
    tileRollbackFails: Bool = false,
    resolvedLegacyTileSetID: String? = nil,
    resolvedTileProvenanceTargetID: String? = nil,
    tileActivationAlreadyRolledBack: Bool = false,
    tileActivationNeverStarted: Bool = false
  ) {
    self.journal = journal
    self.runtimeBranches = runtimeBranches
    self.runtimeTransportFailures = runtimeTransportFailures
    self.runtimeActiveTileSetID = runtimeActiveTileSetID
    self.runtimeEndOnroad = runtimeEndOnroad
    self.runtimeEndLookahead = runtimeEndLookahead
    self.runtimeProcessReady = runtimeProcessReady
    self.runtimeBootIDs = runtimeBootIDs
    self.runtimeBuildEvidenceAvailable = runtimeBuildEvidenceAvailable
    self.runtimeHeads = runtimeHeads
    self.hostToolingHead = hostToolingHead ?? journal.targetHead ?? journal.previousHead
    self.gitChangedPaths = gitChangedPaths
    self.peerTunerPresent = peerTunerPresent
    self.tileRollbackFails = tileRollbackFails
    self.resolvedLegacyTileSetID = resolvedLegacyTileSetID
    self.resolvedTileProvenanceTargetID = resolvedTileProvenanceTargetID
    self.tileActivationAlreadyRolledBack = tileActivationAlreadyRolledBack
    self.tileActivationNeverStarted = tileActivationNeverStarted
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if request.executableURL == ApplyPipeline.processListURL {
      let peer = peerTunerPresent
        ? "99999 /Applications/VTSC Tuner.app/Contents/MacOS/VTSCTuner\n"
        : ""
      return success("\(ProcessInfo.processInfo.processIdentifier) /tmp/test-runner\n" + peer)
    }
    if request.executableURL == ApplyPipeline.gitURL {
      switch request.arguments {
      case ["branch", "--show-current"]:
        return success(journal.branch + "\n")
      case ["status", "--porcelain"]:
        return success("")
      case ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
        return success("origin/\(journal.branch)\n")
      case ["rev-parse", "HEAD"]:
        return success(hostToolingHead + "\n")
      case ["ls-remote", "--heads", "origin", "refs/heads/\(journal.branch)"]:
        return success("\(hostToolingHead)\trefs/heads/\(journal.branch)\n")
      default:
        if request.arguments.starts(with: ["merge-base", "--is-ancestor"]) {
          return success("")
        }
        if request.arguments.starts(with: ["diff", "--no-ext-diff", "--name-only", "-z"]) {
          return success(gitChangedPaths.joined(separator: "\0") + (gitChangedPaths.isEmpty ? "" : "\0"))
        }
        return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unexpected recovery Git request")
      }
    }
    if request.executableURL == ApplyPipeline.sshURL, command == "true" {
      return success("")
    }
    if request.executableURL == ApplyPipeline.rsyncURL {
      return success("")
    }
    if command.contains("rollback --root") {
      if tileRollbackFails {
        return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "injected tile precondition failure")
      }
      if tileActivationNeverStarted {
        var payload: [String: Any] = [
          "operation": "rollback",
          "tile_activation_not_switched": true,
        ]
        if let previous = journal.effectivePreviousTileSetID {
          payload["active_tile_set_id"] = previous
          payload["previous_tile_set_id"] = previous
        }
        return success(String(decoding: try JSONSerialization.data(withJSONObject: payload), as: UTF8.self) + "\n")
      }
      if let resolvedLegacyTileSetID {
        let target = resolvedTileProvenanceTargetID ?? journal.targetTileSetID ?? ""
        let payload: [String: Any] = tileActivationAlreadyRolledBack ? [
          "operation": "rollback",
          "tile_activation_not_observed": true,
          "active_tile_set_id": resolvedLegacyTileSetID,
          "previous_tile_set_id": resolvedLegacyTileSetID,
          "previous_tile_set_provenance": "legacy-migration-v1",
          "previous_tile_set_target_id": target,
        ] : [
          "operation": "rollback",
          "rolled_back_tile_set_id": resolvedLegacyTileSetID,
          "previous_tile_set_id": resolvedLegacyTileSetID,
          "previous_tile_set_provenance": "legacy-migration-v1",
          "previous_tile_set_target_id": target,
        ]
        return success(String(decoding: try JSONSerialization.data(withJSONObject: payload), as: UTF8.self) + "\n")
      }
      return success(String(decoding: try JSONSerialization.data(withJSONObject: [
        "operation": "rollback",
        "tile_activation_not_observed": true,
        "active_tile_set_id": journal.previousTileSetID ?? "",
      ]), as: UTF8.self) + "\n")
    }
    if command.contains(TiciProductionRollbackCommandBuilder.resultMarker) {
      rollbackMutationObserved = true
      return success(
        "\(TiciProductionRollbackCommandBuilder.resultMarker)\t" +
          Data(journal.previousHead.utf8).base64EncodedString() + "\n"
      )
    }
    if command.contains("sudo reboot") {
      rebootObserved = true
      return success("")
    }
    if command.contains("params_root=/data/params/d") {
      runtimeReadCount += 1
      if runtimeReadCount <= runtimeTransportFailures {
        return ProcessResult(terminationStatus: 255, standardOutput: "", standardError: "transport unavailable")
      }
      return success(runtimeWire(readIndex: runtimeReadCount - runtimeTransportFailures))
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

  private func runtimeWire(readIndex: Int) -> String {
    let branch = runtimeBranches.indices.contains(readIndex - 1)
      ? runtimeBranches[readIndex - 1] : journal.branch
    var fields: [TiciSnapshotWireField: Data] = [
      .branch: Data(branch.utf8),
      .head: Data((rollbackMutationObserved
        ? journal.previousHead
        : (runtimeHeads.indices.contains(readIndex - 1) ? runtimeHeads[readIndex - 1] : journal.previousHead)).utf8),
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
      .managerRunning: Data(((runtimeProcessReady.indices.contains(readIndex - 1) && !runtimeProcessReady[readIndex - 1]) ? "0" : "1").utf8),
      .mapdRunning: Data(((runtimeProcessReady.indices.contains(readIndex - 1) && !runtimeProcessReady[readIndex - 1]) ? "0" : "1").utf8),
      .remoteEpochMilliseconds: Data("1800000000000".utf8),
      .liveMapDataControllerStatus: Data("0|0|0|0|0".utf8),
      .runtimeEndIsOffroad: Data((runtimeEndOnroad ? "0" : "1").utf8),
      .runtimeEndIsOnroad: Data((runtimeEndOnroad ? "1" : "0").utf8),
      .runtimeEndMapLookaheadEnabled: Data((runtimeEndLookahead ? "1" : "0").utf8),
    ]
    if runtimeBuildEvidenceAvailable {
      fields[.activeMapdBuildInfo] = Data("{}".utf8)
      fields[.activeMapdELFHeader] = Data([0x7f, 0x45, 0x4c, 0x46, 2, 1])
    }
    let bootID: String?
    if runtimeBootIDs.indices.contains(readIndex - 1) {
      bootID = runtimeBootIDs[readIndex - 1]
    } else if !runtimeBootIDs.isEmpty {
      bootID = runtimeBootIDs[runtimeBootIDs.count - 1]
    } else {
      bootID = rollbackMutationObserved && !rebootObserved
        ? "11111111-1111-4111-8111-111111111111"
        : "22222222-2222-4222-8222-222222222222"
    }
    if let bootID { fields[.bootID] = Data(bootID.utf8) }
    if let release = journal.previousMapdReleaseVersion {
      fields[.mapdReleaseVersion] = Data(release.utf8)
    }
    if let version = journal.previousMapdVersion {
      fields[.mapdVersion] = Data(version.utf8)
    }
    if let tileSetID = runtimeActiveTileSetID ?? resolvedLegacyTileSetID ?? journal.effectivePreviousTileSetID {
      fields[.tileManifest] = try! JSONSerialization.data(withJSONObject: ["tile_set_id": tileSetID])
      fields[.tileTopology] = Data("canonical:\(tileSetID)".utf8)
    } else {
      fields[.tileManifest] = Data()
      fields[.tileTopology] = Data("direct-unidentified".utf8)
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

private actor UnsafeAbortRunner: ProcessRunning {
  private(set) var requests: [ProcessRequest] = []

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    let command = request.arguments.last ?? ""
    if request.executableURL == ApplyPipeline.sshURL, command == "true" {
      return ProcessResult(terminationStatus: 0, standardOutput: "", standardError: "")
    }
    if request.executableURL == ApplyPipeline.sshURL, command.contains("params_root=/data/params/d") {
      let wire = TiciSnapshotWireCodec.encode(.init(rawValues: [
        .branch: Data("chauffeur-exp01".utf8),
        .head: Data(String(repeating: "b", count: 40).utf8),
        .dirty: Data("0".utf8),
        .isOffroad: Data("0".utf8),
        .isOnroad: Data("1".utf8),
        .mapLookaheadEnabled: Data("0".utf8),
        .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
          parameters: .checkoutFallback,
          bands: []
        ).utf8),
        .activeMapdSHA256: Data(String(repeating: "c", count: 64).utf8),
        .tileManifest: Data(),
        .tileTopology: Data("direct-unidentified".utf8),
        .mapdCacheListing: Data("".utf8),
      ])) + "\n"
      return ProcessResult(terminationStatus: 0, standardOutput: wire, standardError: "")
    }
    return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unexpected mutation")
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
      .bootID: Data("11111111-1111-4111-8111-111111111111".utf8),
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
      .tileManifest: Data(),
      .tileTopology: Data("direct-unidentified".utf8),
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
  var managerRunningByRead: [Bool]
  var bootIDsByRead: [String?]
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
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-test",
    deploymentPreRebootBootID: "11111111-1111-4111-8111-111111111111",
    rollbackPreRebootBootID: "11111111-1111-4111-8111-111111111111"
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
    runtimeHead: deployedTargetHead,
    runtimeBranch: "chauffeur-exp01",
    runtimeDirty: false,
    managerRunningByRead: [],
    bootIDsByRead: [],
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
    repositoryRoot: fixture.repository,
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
        if request.arguments.starts(with: ["merge-base", "--is-ancestor"]) {
          return success("")
        }
        if request.arguments.starts(with: ["diff", "--no-ext-diff", "--name-only", "-z"]) {
          return success(fixture.changedPaths.joined(separator: "\0") + "\0")
        }
        return failure("unexpected git request: \(request.arguments)")
      }
    }
    if request.executableURL == ApplyPipeline.sshURL {
      if request.arguments.last == "true" { return success("") }
      if request.arguments.last?.contains("active_mapd_build_info") == true {
        let staticRead = request.arguments.last?.contains("remote_epoch_milliseconds") != true
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
        return success(runtimeWire(readIndex: successfulRuntimeReadCount, forceOffroad: staticRead))
      }
      return failure("unexpected ssh request")
    }
    return failure("unexpected executable: \(request.executableURL.path)")
  }

  private func runtimeWire(readIndex: Int, forceOffroad: Bool = false) -> String {
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
    let stableOnroadPhase = !forceOffroad && (controllerPhase || waitingOnroadPhase)
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
      .tileManifest: Data(),
      .tileTopology: Data("direct-unidentified".utf8),
      .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
        parameters: fixture.tune.params,
        bands: fixture.tune.bands
      ).utf8),
      .mapdCacheListing: Data("\(cachePath)\t\(fixture.release.sha256)\n".utf8),
      .activeMapdBuildInfo: try! JSONEncoder().encode(buildInfo),
      .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
      .managerRunning: Data(((fixture.managerRunningByRead.indices.contains(readIndex - 1)
        ? fixture.managerRunningByRead[readIndex - 1] : true) ? "1" : "0").utf8),
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
    let bootID: String?
    if fixture.bootIDsByRead.indices.contains(readIndex - 1) {
      bootID = fixture.bootIDsByRead[readIndex - 1]
    } else if !fixture.bootIDsByRead.isEmpty {
      bootID = fixture.bootIDsByRead[fixture.bootIDsByRead.count - 1]
    } else {
      bootID = "22222222-2222-4222-8222-222222222222"
    }
    if let bootID { fields[.bootID] = Data(bootID.utf8) }
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
