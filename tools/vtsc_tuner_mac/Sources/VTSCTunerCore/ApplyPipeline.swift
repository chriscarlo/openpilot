import Foundation

public enum ApplyAction: CaseIterable, Equatable, Sendable {
  case local
  case commit
  case push
  case pullOnTici
  case rebuildTilesAndReboot

  public var label: String {
    switch self {
    case .local: "Apply locally"
    case .commit: "Apply + commit"
    case .push: "Apply + commit + push"
    case .pullOnTici: "Apply + deploy runtime (keep current tiles)"
    case .rebuildTilesAndReboot: "Apply + deploy runtime + build canonical tiles"
    }
  }

  public var description: String {
    switch self {
    case .local:
      "Save the tune and update the VTSC physics constants and Q-curve in the selected Chauffeur checkout."
    case .commit:
      "Apply locally, then create one git commit on the current branch."
    case .push:
      "Apply and commit, then push the current branch to its configured upstream."
    case .pullOnTici:
      "Apply, verify, commit, and push, then deploy the exact runtime release to the reachable tici and reboot once without replacing its active tile set."
    case .rebuildTilesAndReboot:
      "Preflight mapd.json, generate and fully validate a durable canonical tile set, then deploy the exact runtime and atomically activate those tiles before one reboot."
    }
  }
}

public enum ApplyStepStatus: Equatable, Sendable {
  case running
  case succeeded
  case failed
}

public struct ApplyStepEvent: Equatable, Sendable {
  public var id: UInt32
  public var status: ApplyStepStatus
  public var text: String
  public var detail: String

  public init(id: UInt32, status: ApplyStepStatus, text: String, detail: String = "") {
    self.id = id
    self.status = status
    self.text = text
    self.detail = detail
  }
}

public enum ApplyEvent: Equatable, Sendable {
  case step(ApplyStepEvent)
  case finished(success: Bool)
}

public typealias ApplyProgressHandler = @Sendable (ApplyEvent) async -> Void

public struct ApplyRequest: Sendable {
  public var action: ApplyAction
  public var tune: Tune
  public var repositoryRoot: URL
  public var tuneURL: URL?
  public var mapdConfigURL: URL?
  public var preferredTiciProfile: String?
  public var expectedBranch: String
  public var mapdReleaseManifestURL: URL?
  public var tileSetArtifactURL: URL?
  public var tileSetsRootURL: URL?
  public var rollbackJournalDirectoryURL: URL?
  public var tileDecoderURL: URL?
  public var verificationRequests: [ProcessRequest]?
  public var requireNonemptyWholeCurveProfile: Bool

  public init(
    action: ApplyAction,
    tune: Tune,
    repositoryRoot: URL,
    tuneURL: URL? = nil,
    mapdConfigURL: URL? = nil,
    preferredTiciProfile: String? = nil,
    expectedBranch: String = "chauffeur-exp01",
    mapdReleaseManifestURL: URL? = nil,
    tileSetArtifactURL: URL? = nil,
    tileSetsRootURL: URL? = nil,
    rollbackJournalDirectoryURL: URL? = nil,
    tileDecoderURL: URL? = nil,
    verificationRequests: [ProcessRequest]? = nil,
    requireNonemptyWholeCurveProfile: Bool = true
  ) {
    self.action = action
    self.tune = tune
    self.repositoryRoot = repositoryRoot
    self.tuneURL = tuneURL
    self.mapdConfigURL = mapdConfigURL
    self.preferredTiciProfile = preferredTiciProfile
    self.expectedBranch = expectedBranch
    self.mapdReleaseManifestURL = mapdReleaseManifestURL
    self.tileSetArtifactURL = tileSetArtifactURL
    self.tileSetsRootURL = tileSetsRootURL
    self.rollbackJournalDirectoryURL = rollbackJournalDirectoryURL
    self.tileDecoderURL = tileDecoderURL
    self.verificationRequests = verificationRequests
    self.requireNonemptyWholeCurveProfile = requireNonemptyWholeCurveProfile
  }
}

public struct ResumePostflightAction: Sendable {
  public static let label = "Resume Pending Outdoor Postflight"
  public static let description =
    "Load the existing rebooted deployment journal and perform only read-only identity, safety, and real-GPS whole-curve checks. This never applies, commits, deploys, changes Params, or reboots."
}

public struct ResumePostflightRequest: Sendable {
  public var tune: Tune
  public var repositoryRoot: URL
  public var expectedBranch: String
  public var mapdReleaseManifestURL: URL?
  public var journalURL: URL?
  public var timeout: TimeInterval
  var pollInterval: TimeInterval

  public init(
    tune: Tune,
    repositoryRoot: URL,
    expectedBranch: String = "chauffeur-exp01",
    mapdReleaseManifestURL: URL? = nil,
    journalURL: URL? = nil,
    timeout: TimeInterval = 120,
    pollInterval: TimeInterval = 2
  ) {
    self.tune = tune
    self.repositoryRoot = repositoryRoot
    self.expectedBranch = expectedBranch
    self.mapdReleaseManifestURL = mapdReleaseManifestURL
    self.journalURL = journalURL
    self.timeout = timeout
    self.pollInterval = pollInterval
  }
}

public struct RollbackRecoveryAction: Sendable {
  public static let label = "Recover Interrupted Production Rollback"
  public static let description =
    "Load one interrupted or failed rollback journal, recheck the exact recorded identity and fresh parked/kill-switch state, then idempotently finish rollback. This action may restore source, Params, mapd, and tiles and reboot only as part of that recorded rollback."
}

public struct RollbackRecoveryRequest: Sendable {
  public var repositoryRoot: URL
  public var journalURL: URL?

  public init(repositoryRoot: URL, journalURL: URL? = nil) {
    self.repositoryRoot = repositoryRoot
    self.journalURL = journalURL
  }
}

public struct AbortPendingDeploymentAction: Sendable {
  public static let label = "Abort and Roll Back Pending Deployment"
  public static let description =
    "Explicitly abandon one selected rebooted deployment that cannot pass postflight. This rechecks parked/offroad safety, restores only its recorded prior Git, Params, mapd/cache, and tile identity, and reboots only when that journal proves its deployment reboot was sent."
}

public struct AbortPendingDeploymentRequest: Sendable {
  public var repositoryRoot: URL
  public var journalURL: URL

  public init(repositoryRoot: URL, journalURL: URL) {
    self.repositoryRoot = repositoryRoot
    self.journalURL = journalURL.standardizedFileURL
  }
}

private struct ResumedPostflightObservation: Sendable {
  var controllerReady: TiciDeploymentPostflight
  var offroad: TiciDeploymentPostflight
}

private enum ResumedPostflightWaitState: LocalizedError, Sendable {
  case controllerEvidence(String)
  case offroadTransition
  case transport(String)

  var errorDescription: String? {
    switch self {
    case let .controllerEvidence(reason): reason
    case .offroadTransition:
      "Profile acceptance was proven. Turn ignition off; waiting for a stable IsOffroad=1 / IsOnroad=0 observation."
    case let .transport(reason):
      "Tici transport/runtime is not ready yet: \(reason)"
    }
  }
}

private enum RuntimeStartupWaitState: LocalizedError, Sendable {
  case processesNotReady(manager: Bool, mapd: Bool)

  var errorDescription: String? {
    switch self {
    case let .processesNotReady(manager, mapd):
      "post-reboot runtime is still starting: manager_running=\(manager ? 1 : 0) mapd_running=\(mapd ? 1 : 0)"
    }
  }
}

private enum BootTransitionWaitState: LocalizedError, Sendable {
  case identityPending(previous: String, current: String?)

  var errorDescription: String? {
    switch self {
    case let .identityPending(previous, current):
      "post-reboot boot identity has not changed: previous=\(previous) current=\(current ?? "missing")"
    }
  }
}

enum ProductionRollbackResolution: Sendable {
  case alreadyCompleted(String)
  case rolledBack(String)
  case rollbackFailed(String)
}

enum ProductionTilePlan: Equatable, Sendable {
  case unchanged
  case validateExplicit(URL)
  case generateCanonical
}

public enum ApplyPipelineError: LocalizedError, Sendable {
  case commandFailed(String, Int32, String)
  case noReachableTici(String?)
  case invalidProfile(String)
  case noCachedRegions
  case unexpectedRegionListing(String)
  case unexpectedDiskOutput(String)
  case insufficientDisk(freeMB: UInt64, requiredMB: UInt64)
  case missingGeneratedTiles(URL)
  case emptyGeneratedTiles(URL)
  case invalidBranch(String)
  case repositoryDirty(String)
  case upstreamMismatch(expected: String, actual: String)
  case commitMismatch(context: String, expected: String, actual: String)
  case ticiNotOffroad
  case mapLookaheadMustRemainDisabled
  case invalidDeploymentOutput(String)
  case postflightMismatch(String)
  case unresolvedProductionRollbacks([URL])

  public var errorDescription: String? {
    switch self {
    case let .commandFailed(context, status, output):
      "\(context) exited \(status).\n\(output)"
    case let .noReachableTici(adbDetail):
      [
        "No SSH deployment channel could be opened through commaHome, commaCar, or commaAdb.",
        adbDetail.map { "USB / ADB: \($0)" },
      ].compactMap { $0 }.joined(separator: "\n")
    case let .invalidProfile(profile):
      "The SSH profile name contains unsupported characters: \(profile)"
    case .noCachedRegions:
      "The tici has no cached map regions. Download a region in the offroad UI first."
    case let .unexpectedRegionListing(output):
      "Could not parse the tici's cached-region listing: \(output)"
    case let .unexpectedDiskOutput(output):
      "Could not parse free disk space from: \(output)"
    case let .insufficientDisk(freeMB, requiredMB):
      "The tici has \(freeMB) MB free; this rebuild needs about \(requiredMB) MB."
    case let .missingGeneratedTiles(url):
      "mapd exited successfully but did not create \(url.path)."
    case let .emptyGeneratedTiles(url):
      "mapd only created empty placeholder tiles under \(url.path). Verify the PBF was prepared with locations on ways."
    case let .invalidBranch(branch):
      "The deployment branch is invalid: \(branch)"
    case let .repositoryDirty(output):
      "Production deployment requires a clean checkout before tune mutation.\n\(output)"
    case let .upstreamMismatch(expected, actual):
      "The branch upstream is \(actual), expected \(expected)."
    case let .commitMismatch(context, expected, actual):
      "\(context) is \(actual), expected exact commit \(expected)."
    case .ticiNotOffroad:
      "Refusing to deploy because the tici did not report IsOffroad=1."
    case .mapLookaheadMustRemainDisabled:
      "Refusing to deploy while MTSCLookaheadEnabled is on. Leave the road-test kill switch off during staging."
    case let .invalidDeploymentOutput(output):
      "Could not decode deployment verification output: \(output)"
    case let .postflightMismatch(reason):
      "Tici postflight verification failed: \(reason)"
    case let .unresolvedProductionRollbacks(urls):
      "Resolve the recorded production transaction before starting another car-facing deployment. Use Resume or Abort for a rebooted awaiting journal, Recover Interrupted Production Rollback for a mutation/rollback journal, or wait ten minutes and retry if a fresh targetless preflight may belong to another app instance: \(urls.map(\.lastPathComponent).joined(separator: ", "))."
    }
  }
}

struct ADBDevice: Equatable, Sendable {
  var serial: String
  var state: String
  var details: String

  static func parseList(_ output: String) -> [ADBDevice] {
    output.components(separatedBy: .newlines).compactMap { line in
      let fields = line.split(whereSeparator: \Character.isWhitespace)
      guard fields.count >= 2, fields[0] != "List", fields[0] != "*" else { return nil }
      return ADBDevice(
        serial: String(fields[0]),
        state: String(fields[1]),
        details: fields.dropFirst(2).joined(separator: " ")
      )
    }
  }
}

enum ADBDeviceSelection: Equatable, Sendable {
  case selected(String)
  case unavailable(String)
  case ambiguous([String])

  static func select(from output: String) -> ADBDeviceSelection {
    let devices = ADBDevice.parseList(output)
    guard !devices.isEmpty else {
      return .unavailable("No device is listed by adb.")
    }
    guard devices.count == 1 else {
      return .ambiguous(devices.map { "\($0.serial)=\($0.state)" }.sorted())
    }
    let device = devices[0]
    guard device.state == "device" else {
      return .unavailable("No authorized device is ready (\(device.serial)=\(device.state)).")
    }
    return .selected(device.serial)
  }
}

enum ADBExecutableLocator {
  static func candidateURLs(environment: [String: String]) -> [URL] {
    var paths: [String] = []
    if let home = environment["HOME"], !home.isEmpty {
      paths += [
        "\(home)/.local/bin/adb",
        "\(home)/Library/Android/sdk/platform-tools/adb",
      ]
    }
    if let path = environment["PATH"] {
      paths += path.split(separator: ":").map { "\($0)/adb" }
    }
    paths += [
      "/opt/homebrew/bin/adb",
      "/usr/local/bin/adb",
    ]

    var seen: Set<String> = []
    return paths.compactMap { path in
      let url = URL(fileURLWithPath: path).standardizedFileURL
      guard seen.insert(url.path).inserted else { return nil }
      return url
    }
  }

  static func resolve(
    environment: [String: String] = ProcessInfo.processInfo.environment,
    fileManager: FileManager = .default
  ) -> URL? {
    candidateURLs(environment: environment).first { fileManager.isExecutableFile(atPath: $0.path) }
  }
}

enum MapdCommandBuilder {
  static func generationArguments(parameters: SigmoidParameters, region: RegionBox) -> [String] {
    let locale = Locale(identifier: "en_US_POSIX")
    return [
      "--generate",
      "--minlat=\(region.minLatitude)",
      "--minlon=\(region.minLongitude)",
      "--maxlat=\(region.minLatitude + 2)",
      "--maxlon=\(region.minLongitude + 2)",
      String(format: "--phys-a=%.6f", locale: locale, parameters.a),
      String(format: "--phys-b=%.6f", locale: locale, parameters.b),
      String(format: "--phys-c=%.6f", locale: locale, parameters.c),
      String(format: "--phys-d=%.6f", locale: locale, parameters.d),
      String(format: "--phys-min-lat=%.4f", locale: locale, parameters.minLat),
      String(format: "--phys-max-lat=%.4f", locale: locale, parameters.maxLat),
    ]
  }
}

enum TiciPhysicsCommandBuilder {
  /// The device-side operation is deliberately shell-only. The Mac Swift
  /// process validates/renderers the exact six values and the tici only
  /// performs the small atomic Params transaction it cannot perform locally.
  static func synchronizeAndVerifyCommand(parameters: SigmoidParameters) throws -> String {
    try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: parameters)
  }
}

public actor ApplyPipeline {
  public static let gitURL = URL(fileURLWithPath: "/usr/bin/git")
  public static let sshURL = URL(fileURLWithPath: "/usr/bin/ssh")
  public static let rsyncURL = URL(fileURLWithPath: "/usr/bin/rsync")
  public static let networkSetupURL = URL(fileURLWithPath: "/usr/sbin/networksetup")
  static let processListURL = URL(fileURLWithPath: "/bin/ps")

  private let processRunner: any ProcessRunning
  private let adbURL: URL?
  private let journalWriter: @Sendable (DeploymentRollbackJournal, URL) throws -> Void
  private let rebootInitialDelayNanoseconds: UInt64
  private let rebootPollDelayNanoseconds: UInt64
  private let currentProcessID: Int32
  private var lastADBTransportDetail: String?

  public init() {
    processRunner = SystemProcessRunner()
    adbURL = ADBExecutableLocator.resolve()
    journalWriter = { journal, url in try journal.write(to: url) }
    rebootInitialDelayNanoseconds = 15_000_000_000
    rebootPollDelayNanoseconds = 5_000_000_000
    currentProcessID = getpid()
  }

  public init(processRunner: any ProcessRunning) {
    self.processRunner = processRunner
    adbURL = nil
    journalWriter = { journal, url in try journal.write(to: url) }
    rebootInitialDelayNanoseconds = 15_000_000_000
    rebootPollDelayNanoseconds = 5_000_000_000
    currentProcessID = getpid()
  }

  public init(processRunner: any ProcessRunning, adbURL: URL?) {
    self.processRunner = processRunner
    self.adbURL = adbURL
    journalWriter = { journal, url in try journal.write(to: url) }
    rebootInitialDelayNanoseconds = 15_000_000_000
    rebootPollDelayNanoseconds = 5_000_000_000
    currentProcessID = getpid()
  }

  init(
    processRunner: any ProcessRunning,
    rebootInitialDelayNanoseconds: UInt64,
    rebootPollDelayNanoseconds: UInt64
  ) {
    self.processRunner = processRunner
    adbURL = nil
    journalWriter = { journal, url in try journal.write(to: url) }
    self.rebootInitialDelayNanoseconds = rebootInitialDelayNanoseconds
    self.rebootPollDelayNanoseconds = rebootPollDelayNanoseconds
    currentProcessID = getpid()
  }

  init(
    processRunner: any ProcessRunning,
    adbURL: URL? = nil,
    rebootInitialDelayNanoseconds: UInt64 = 15_000_000_000,
    rebootPollDelayNanoseconds: UInt64 = 5_000_000_000,
    journalWriter: @escaping @Sendable (DeploymentRollbackJournal, URL) throws -> Void,
    currentProcessID: Int32 = getpid()
  ) {
    self.processRunner = processRunner
    self.adbURL = adbURL
    self.rebootInitialDelayNanoseconds = rebootInitialDelayNanoseconds
    self.rebootPollDelayNanoseconds = rebootPollDelayNanoseconds
    self.journalWriter = journalWriter
    self.currentProcessID = currentProcessID
  }

  static func productionTilePlan(for request: ApplyRequest) -> ProductionTilePlan {
    if let tileSetArtifactURL = request.tileSetArtifactURL {
      return .validateExplicit(tileSetArtifactURL.standardizedFileURL)
    }
    return request.action == .rebuildTilesAndReboot ? .generateCanonical : .unchanged
  }

  /// Convenience bridge for SwiftUI. Cancelling or discarding the stream stops
  /// the current subprocess and prevents subsequent regions from starting.
  public nonisolated func events(for request: ApplyRequest) -> AsyncStream<ApplyEvent> {
    AsyncStream { continuation in
      let worker = Task {
        _ = await self.apply(request) { event in
          continuation.yield(event)
        }
        continuation.finish()
      }
      continuation.onTermination = { @Sendable _ in worker.cancel() }
    }
  }

  /// Continue only the final proof for a deployment that already rebooted.
  /// The worker performs no source, Git, tici, mapd, tile, Param, or reboot
  /// mutation; its sole write is the final atomic `completed=true` journal
  /// replacement after every current identity and profile check succeeds.
  public nonisolated func resumePostflightEvents(
    for request: ResumePostflightRequest
  ) -> AsyncStream<ApplyEvent> {
    AsyncStream { continuation in
      let worker = Task {
        _ = await self.resumePendingPostflight(request) { event in
          continuation.yield(event)
        }
        continuation.finish()
      }
      continuation.onTermination = { @Sendable _ in worker.cancel() }
    }
  }

  @discardableResult
  public func resumePendingPostflight(
    _ request: ResumePostflightRequest,
    progress: @escaping ApplyProgressHandler
  ) async -> Bool {
    await emit(.running, id: 0, text: "Loading the existing pending deployment journal…", progress: progress)
    do {
      try Task.checkCancellation()
      try RepositoryLocator.validate(request.repositoryRoot)
      try validateExactSourceTune(
        repositoryRoot: request.repositoryRoot,
        tune: request.tune
      )
      guard request.expectedBranch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
            !request.expectedBranch.contains(".."), !request.expectedBranch.hasPrefix("/")
      else { throw ApplyPipelineError.invalidBranch(request.expectedBranch) }

      let loaded = try DeploymentRollbackJournal.loadPendingPostflight(from: request.journalURL)
      var journal = loaded.journal
      try validateProfile(journal.profile)
      guard journal.branch == request.expectedBranch else {
        throw ApplyPipelineError.invalidBranch(journal.branch)
      }
      let git = try await gitDeploymentPreflight(
        repositoryRoot: request.repositoryRoot,
        expectedBranch: request.expectedBranch
      )
      let gitIdentity = try await validateResumePostflightGitIdentity(
        journal: journal,
        git: git,
        repositoryRoot: request.repositoryRoot
      )
      let release = try MapdReleaseArtifact.load(from: request.mapdReleaseManifestURL).validated()
      guard await probeProfile(journal.profile) else {
        throw ApplyPipelineError.noReachableTici(lastADBTransportDetail)
      }
      let profile = journal.profile
      let preflight = RuntimeDeploymentPreflight(
        git: git,
        profile: profile,
        mapdRecoveryOutcome: .clean,
        release: release,
        tileSet: nil,
        journal: journal,
        journalURL: loaded.url
      )
      await emit(
        .succeeded,
        id: 0,
        text: "Pending deployment target and exact clean host-only tooling head match",
        detail: [
          "journal=\(loaded.url.path)",
          "deployed_target_head=\(gitIdentity.deployedTargetHead)",
          "tooling_head=\(gitIdentity.toolingHead)",
          "host_only_paths=\(gitIdentity.hostOnlyPaths.joined(separator: ","))",
          "profile=\(profile)",
        ].joined(separator: "\n"),
        progress: progress
      )

      await emit(
        .running,
        id: 1,
        text: "Proving the profile would be accepted when Map Lookahead is later enabled, then waiting for the clean offroad transition…",
        progress: progress
      )
      let identity = TuneDeploymentIdentity(tune: request.tune)
      let observation = try await waitForResumedProductionPostflight(
        preflight: preflight,
        deployedTargetHead: gitIdentity.deployedTargetHead,
        identity: identity,
        timeout: request.timeout,
        pollInterval: request.pollInterval,
        progress: progress
      )
      await emit(
        .succeeded,
        id: 1,
        text: "Profile acceptance and the subsequent clean offroad identity both passed",
        detail: "profile_points=\(observation.controllerReady.profilePointCount) profile_events=\(observation.controllerReady.profileEventCount)",
        progress: progress
      )

      try Task.checkCancellation()
      // Revalidate the on-disk journal immediately before its only permitted
      // mutation so an external replacement cannot be finalized accidentally.
      // Canonical lock order is global production owner, then journal.
      let productionOwnerLock = try await acquireGlobalProductionOwnerLock(for: loaded.url)
      defer { productionOwnerLock.unlock() }
      let completionLock = try await acquireJournalResolutionLock(for: loaded.url)
      defer { completionLock.unlock() }
      let current = try DeploymentRollbackJournal.load(from: loaded.url)
      guard current.hasSameDeploymentIdentity(as: journal) else {
        throw ApplyPipelineError.postflightMismatch("pending deployment journal changed during read-only verification")
      }
      try current.validateResolutionConsistency()
      let completionAlreadyWon: Bool
      switch current.effectiveResolution {
      case .awaitingPostflight, .awaitingOutdoorPostflight:
        completionAlreadyWon = false
      case .completed:
        guard current.completed else {
          throw ApplyPipelineError.postflightMismatch("completed resolution lacks its legacy completion sentinel")
        }
        completionAlreadyWon = true
      case .preflightReserved, .mutationInProgress, .rollbackInProgress, .rolledBack, .rollbackFailed:
        throw ApplyPipelineError.postflightMismatch(
          "deployment rollback already owns or settled this journal; completion is forbidden"
        )
      }
      try validateExactSourceTune(repositoryRoot: request.repositoryRoot, tune: request.tune)
      let finalGit = try await gitDeploymentPreflight(
        repositoryRoot: request.repositoryRoot,
        expectedBranch: request.expectedBranch
      )
      guard finalGit == git,
            finalGit.localHead == gitIdentity.toolingHead else {
        throw ApplyPipelineError.postflightMismatch("clean local/origin tooling identity changed during postflight")
      }
      let finalReadback = try await readTiciRuntimePostflight(
        profile: preflight.profile,
        context: "final read-only pending postflight safety check"
      )
      let finalPostflight = makeProductionPostflight(
        finalReadback,
        preflight: preflight,
        identity: identity
      )
      try validateResumedOffroadCompletion(
        finalPostflight,
        controllerEvidence: observation.controllerReady,
        preflight: preflight,
        targetHead: gitIdentity.deployedTargetHead,
        identity: identity,
        expectedActiveTileSetID: preflight.journal.targetTileSetID ?? preflight.journal.previousTileSetID
      )
      await emit(
        .running,
        id: 2,
        text: "Final safety checks passed; atomically completing the existing journal…",
        detail: "This is the non-cancellable commit point. The app will report the exact journal readback result.",
        progress: progress
      )
      try Task.checkCancellation()
      if completionAlreadyWon {
        await emit(
          .succeeded,
          id: 2,
          text: "The identical deployment completion was already recorded",
          detail: "journal=\(loaded.url.path)\nNo journal rewrite or deployment mutation was performed.",
          progress: progress
        )
        return await finish(true, progress: progress)
      }
      journal.completed = true
      journal.completedAt = Date().ISO8601Format()
      journal.completedToolingHead = gitIdentity.toolingHead
      journal.completionHostOnlyPaths = gitIdentity.hostOnlyPaths
      journal.resolution = .completed
      // No cancellation check or suspension is permitted after this commit
      // point. A late cancellation request must resolve from the exact atomic
      // readback below instead of claiming that the journal stayed pending.
      do {
        try journalWriter(journal, loaded.url)
      } catch let durabilityError as DeploymentRollbackJournalError {
        guard case .committedButNotDurable = durabilityError else { throw durabilityError }
        guard try DeploymentRollbackJournal.load(from: loaded.url) == journal else { throw durabilityError }
      }
      guard try DeploymentRollbackJournal.load(from: loaded.url) == journal else {
        throw ApplyPipelineError.postflightMismatch("completed deployment journal did not read back exactly")
      }
      await emit(
        .succeeded,
        id: 2,
        text: "The existing deployment journal is now complete",
        detail: "journal=\(loaded.url.path)\nNo Git, mapd, tile, Param, or reboot mutation was performed.",
        progress: progress
      )
      return await finish(true, progress: progress)
    } catch {
      await emitFailure(
        id: 2,
        text: "Pending postflight remains incomplete; no deployment mutation or reboot was attempted",
        error: error,
        progress: progress
      )
      if isTerminalResumePostflightFailure(error) {
        await emit(
          .failed,
          id: 3,
          text: "This identity mismatch is persistent; Abort and Roll Back Pending Deployment is available",
          detail: "Rollback is never automatic. Review and confirm the exact selected deployment journal from the Apply menu.",
          progress: progress
        )
      }
      return await finish(false, progress: progress)
    }
  }

  @discardableResult
  public func recoverPendingRollback(
    _ request: RollbackRecoveryRequest,
    progress: @escaping ApplyProgressHandler
  ) async -> Bool {
    await emit(.running, id: 0, text: "Loading the recorded interrupted rollback identity…", progress: progress)
    do {
      try Task.checkCancellation()
      try RepositoryLocator.validate(request.repositoryRoot)
      let loaded = try DeploymentRollbackJournal.loadRecoverableRollback(from: request.journalURL)
      try validateProfile(loaded.journal.profile)
      guard loaded.journal.branch.range(
        of: #"^[A-Za-z0-9._/-]+$"#,
        options: .regularExpression
      ) != nil,
      !loaded.journal.branch.contains(".."),
      !loaded.journal.branch.hasPrefix("/")
      else { throw ApplyPipelineError.invalidBranch(loaded.journal.branch) }
      guard await probeProfile(loaded.journal.profile) else {
        throw ApplyPipelineError.noReachableTici(lastADBTransportDetail)
      }
      await emit(
        .succeeded,
        id: 0,
        text: "Exact rollback journal and tici transport are available",
        detail: [
          "journal=\(loaded.url.path)",
          "resolution=\(loaded.journal.effectiveResolution.rawValue)",
          "profile=\(loaded.journal.profile)",
          "previous_head=\(loaded.journal.previousHead)",
          "target_head=\(loaded.journal.targetHead ?? "")",
          "previous_mapd_sha256=\(loaded.journal.previousActiveMapdSHA256)",
          "previous_tile_set_id=\(loaded.journal.previousTileSetID ?? "")",
          "target_tile_set_id=\(loaded.journal.targetTileSetID ?? "")",
        ].joined(separator: "\n"),
        progress: progress
      )
      await emit(
        .running,
        id: 1,
        text: "Taking guarded rollback ownership and verifying fresh parked safety…",
        progress: progress
      )
      let productionOwnerLock = try await acquireGlobalProductionOwnerLock(for: loaded.url)
      defer { productionOwnerLock.unlock() }
      let resolution = await rollbackProductionDeploymentIfJournalPending(
        context: ProductionRollbackContext(journal: loaded.journal, journalURL: loaded.url)
      )
      switch resolution {
      case let .alreadyCompleted(detail):
        await emit(.succeeded, id: 1, text: "Deployment was already completed; rollback recovery was not run", detail: detail, progress: progress)
        return await finish(true, progress: progress)
      case let .rolledBack(detail):
        await emit(.succeeded, id: 1, text: "Recorded production rollback recovered and verified", detail: detail, progress: progress)
        return await finish(true, progress: progress)
      case let .rollbackFailed(detail):
        throw ApplyPipelineError.postflightMismatch(detail)
      }
    } catch {
      await emitFailure(
        id: 1,
        text: "Rollback recovery remains unresolved; journal retained for another guarded attempt",
        error: error,
        progress: progress
      )
      return await finish(false, progress: progress)
    }
  }

  /// Explicitly abandon one selected rebooted deployment which cannot satisfy
  /// postflight. This never runs automatically. Lock order is global owner,
  /// then the journal resolution lock acquired by the shared rollback path.
  @discardableResult
  public func abortPendingDeployment(
    _ request: AbortPendingDeploymentRequest,
    progress: @escaping ApplyProgressHandler
  ) async -> Bool {
    await emit(.running, id: 0, text: "Loading the selected pending deployment identity…", progress: progress)
    do {
      try Task.checkCancellation()
      try RepositoryLocator.validate(request.repositoryRoot)
      let loaded = try DeploymentRollbackJournal.loadPendingPostflight(from: request.journalURL)
      try validateProfile(loaded.journal.profile)
      await emit(
        .succeeded,
        id: 0,
        text: "Selected rebooted deployment is eligible for explicit abort",
        detail: [
          "journal=\(loaded.url.path)",
          "deployment_id=\(loaded.journal.deploymentID.uuidString)",
          "target_head=\(loaded.journal.targetHead ?? "")",
          "created_at=\(loaded.journal.createdAt)",
        ].joined(separator: "\n"),
        progress: progress
      )
      await emit(
        .running,
        id: 1,
        text: "Taking global and journal ownership, then restoring the recorded prior deployment…",
        progress: progress
      )
      let productionOwnerLock = try await acquireGlobalProductionOwnerLock(for: loaded.url)
      defer { productionOwnerLock.unlock() }
      let current = try DeploymentRollbackJournal.load(from: loaded.url)
      guard current == loaded.journal else {
        throw ApplyPipelineError.postflightMismatch(
          "selected pending deployment changed before abort ownership was acquired"
        )
      }
      let resolution = await rollbackProductionDeploymentIfJournalPending(
        context: ProductionRollbackContext(journal: current, journalURL: loaded.url)
      )
      switch resolution {
      case let .alreadyCompleted(detail):
        await emit(.succeeded, id: 1, text: "Deployment completed before abort ownership was acquired", detail: detail, progress: progress)
        return await finish(true, progress: progress)
      case let .rolledBack(detail):
        await emit(.succeeded, id: 1, text: "Pending deployment was rolled back and verified", detail: detail, progress: progress)
        return await finish(true, progress: progress)
      case let .rollbackFailed(detail):
        throw ApplyPipelineError.postflightMismatch(detail)
      }
    } catch {
      await emitFailure(
        id: 1,
        text: "Pending deployment was not aborted; its journal remains available for a guarded retry",
        error: error,
        progress: progress
      )
      return await finish(false, progress: progress)
    }
  }

  @discardableResult
  public func apply(
    _ request: ApplyRequest,
    progress: @escaping ApplyProgressHandler
  ) async -> Bool {
    // Every car-facing prerequisite is read and verified before tune/source,
    // Git, Params, binary, or tile mutation. Runtime-only deployment leaves
    // tiles unchanged; the rebuild action must produce or receive one exact,
    // fully validated canonical tile artifact during this preflight.
    let runtimePreflight: RuntimeDeploymentPreflight?
    if request.action == .pullOnTici || request.action == .rebuildTilesAndReboot {
      await emit(.running, id: 0, text: "Preflighting the complete production deployment…", progress: progress)
      do {
        runtimePreflight = try await productionPreflight(request)
        let tileDetail = runtimePreflight?.tileSet.map {
          "tile_set=\($0.manifest.tileSetID)\ntile_artifact=\($0.rootURL.path)"
        }
          ?? "tile_set=unchanged (runtime profile does not require replacement)"
        await emit(
          .succeeded,
          id: 0,
          text: runtimePreflight!.mapdRecoveryOutcome == .recovered
            ? "Recovered an interrupted mapd update; production deployment preflight passed"
            : "Production deployment preflight passed without mutation",
          detail: [
            "branch=\(runtimePreflight!.git.branch)",
            "head=\(runtimePreflight!.git.localHead)",
            "tici=\(runtimePreflight!.profile) offroad",
            "mapd_recovery=\(runtimePreflight!.mapdRecoveryOutcome.rawValue)",
            "release=\(runtimePreflight!.release.artifact.releaseID)",
            "mapd_sha256=\(runtimePreflight!.release.artifact.sha256)",
            tileDetail,
          ].joined(separator: "\n"),
          progress: progress
        )
      } catch {
        await emitFailure(
          id: 0,
          text: "Production deployment preflight failed before mutation",
          error: error,
          progress: progress
        )
        return await finish(false, progress: progress)
      }
    } else {
      runtimePreflight = nil
    }
    // Idempotent unlock is also performed at the exact durable handoff inside
    // deployProductionRuntime. This defer covers every earlier host-side exit
    // (save/patch/test/commit/push) after production preflight reserved the
    // global transaction.
    defer {
      if let preflight = runtimePreflight {
        try? DeploymentRollbackJournal.removeOwnedPreflightReservation(
          matching: preflight.journal,
          at: preflight.journalURL
        )
        preflight.productionOwnerLock?.unlock()
      }
    }

    await emit(.running, id: 1, text: "Saving the tune file…", progress: progress)
    let tuneURL: URL
    do {
      tuneURL = try request.tuneURL ?? TuneStore.defaultTuneURL()
      try TuneStore.save(request.tune, to: tuneURL)
      await emit(
        .succeeded,
        id: 1,
        text: "Saved the tune file",
        detail: "→ \(tuneURL.path)",
        progress: progress
      )
    } catch {
      await emitFailure(id: 1, text: "Couldn't save the tune file", error: error, progress: progress)
      return await finish(false, progress: progress)
    }

    await emit(.running, id: 2, text: "Updating the openpilot source files…", progress: progress)
    let touched: [URL]
    do {
      try RepositoryLocator.validate(request.repositoryRoot)
      touched = try SourcePatcher.apply(
        repositoryRoot: request.repositoryRoot,
        parameters: request.tune.params,
        bands: request.tune.bands
      )
      await emit(
        .succeeded,
        id: 2,
        text: touched.isEmpty
          ? "Source already matched the tune — nothing to rewrite"
          : "Updated the openpilot source files",
        detail: touched.map(\.path).joined(separator: "\n"),
        progress: progress
      )
    } catch {
      await emitFailure(id: 2, text: "Couldn't update the openpilot source files", error: error, progress: progress)
      return await finish(false, progress: progress)
    }

    guard request.action != .local else { return await finish(true, progress: progress) }

    if runtimePreflight != nil {
      await emit(.running, id: 25, text: "Running the complete host verification gate…", progress: progress)
      do {
        let requests = request.verificationRequests ?? ProductionVerificationSuite.requests(
          repositoryRoot: request.repositoryRoot
        )
        var details: [String] = []
        for verification in requests {
          let label = ([verification.executableURL.lastPathComponent] + verification.arguments).joined(separator: " ")
          let result = try await checked(verification, context: label)
          details.append("PASS \(label)\n\(result.combinedOutput)")
        }
        await emit(
          .succeeded,
          id: 25,
          text: "Complete host verification gate passed",
          detail: details.joined(separator: "\n"),
          progress: progress
        )
      } catch {
        await emitFailure(id: 25, text: "Host verification failed before commit or deployment", error: error, progress: progress)
        return await finish(false, progress: progress)
      }
    }

    await emit(
      .running,
      id: 3,
      text: touched.isEmpty ? "Checking for changes to commit…" : "Committing the change to the local repo…",
      progress: progress
    )
    do {
      let detail: String
      if touched.isEmpty {
        detail = ""
      } else {
        detail = try await gitCommit(repositoryRoot: request.repositoryRoot, touched: touched)
      }
      await emit(
        .succeeded,
        id: 3,
        text: touched.isEmpty
          ? "Nothing to commit — tune is already on this branch"
          : "Committed the change to the local repo",
        detail: detail,
        progress: progress
      )
    } catch {
      await emitFailure(id: 3, text: "Couldn't commit the change", error: error, progress: progress)
      return await finish(false, progress: progress)
    }

    guard request.action != .commit else { return await finish(true, progress: progress) }

    await emit(.running, id: 4, text: "Pushing the current branch to the remote…", progress: progress)
    let pushedHead: String
    do {
      let pushed = try await pushExact(
        repositoryRoot: request.repositoryRoot,
        expectedBranch: request.expectedBranch
      )
      pushedHead = pushed.head
      await emit(
        .succeeded,
        id: 4,
        text: "Pushed and verified the exact origin commit",
        detail: pushed.detail,
        progress: progress
      )
    } catch {
      await emitFailure(id: 4, text: "Couldn't push to the remote", error: error, progress: progress)
      return await finish(false, progress: progress)
    }

    guard request.action != .push else { return await finish(true, progress: progress) }

    guard var deployment = runtimePreflight else {
      await emit(
        .failed,
        id: 5,
        text: "Production deployment preflight state was lost",
        detail: "No car-facing mutation was attempted.",
        progress: progress
      )
      return await finish(false, progress: progress)
    }
    deployment.journal.targetHead = pushedHead
    do { try journalWriter(deployment.journal, deployment.journalURL) }
    catch {
      await emitFailure(id: 5, text: "Could not persist the rollback journal", error: error, progress: progress)
      return await finish(false, progress: progress)
    }
    return await deployProductionRuntime(
      request: request,
      preflight: deployment,
      targetHead: pushedHead,
      progress: progress
    )
  }

  func productionPreflight(_ request: ApplyRequest) async throws -> RuntimeDeploymentPreflight {
    try RepositoryLocator.validate(request.repositoryRoot)
    _ = try SourcePatcher.readParameters(from: request.repositoryRoot)
    _ = try SourcePatcher.makePatch(
      physicsText: String(
        contentsOf: request.repositoryRoot.appendingPathComponent(RepositoryLocator.physicsRelativePath),
        encoding: .utf8
      ),
      qCurveText: String(
        contentsOf: request.repositoryRoot.appendingPathComponent(RepositoryLocator.qCurveRelativePath),
        encoding: .utf8
      ),
      paramsDefaultsText: String(
        contentsOf: request.repositoryRoot.appendingPathComponent(RepositoryLocator.paramsDefaultsRelativePath),
        encoding: .utf8
      ),
      physicsPanelText: String(
        contentsOf: request.repositoryRoot.appendingPathComponent(RepositoryLocator.physicsPanelRelativePath),
        encoding: .utf8
      ),
      parameters: request.tune.params,
      bands: request.tune.bands
    )
    guard request.expectedBranch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
          !request.expectedBranch.contains(".."), !request.expectedBranch.hasPrefix("/")
    else { throw ApplyPipelineError.invalidBranch(request.expectedBranch) }

    // Reserve the one production transaction before any host preflight work
    // that could overlap an already-running released app. The new lifecycle
    // value is intentionally unknown to that reader, so it fails closed even
    // though it does not know this version's global flock.
    let requestedJournalDirectory = try request.rollbackJournalDirectoryURL?.standardizedFileURL ??
      DeploymentRollbackJournal.defaultDirectory()
    let productionOwnerLock = try DeploymentRollbackJournal.acquireProductionOwnerLock(
      directory: requestedJournalDirectory
    )
    let journalDirectory = productionOwnerLock.directoryURL
    let deploymentID = UUID()
    let createdAt = Date().ISO8601Format()
    let rollbackPath = "/data/media/0/osm/binaries/mapd-rollback-\(deploymentID.uuidString.lowercased())"
    let journalURL = journalDirectory.appendingPathComponent("\(deploymentID.uuidString).json")
    let reservation = DeploymentRollbackJournal(
      deploymentID: deploymentID,
      createdAt: createdAt,
      profile: request.preferredTiciProfile ?? "",
      branch: request.expectedBranch,
      previousHead: "",
      previousPhysicsParams: [:],
      previousQCurveSHA256: "",
      previousMapdReleaseVersion: nil,
      previousMapdVersion: nil,
      previousActiveMapdSHA256: "",
      previousCachedMapdPath: "",
      mapdRollbackPath: rollbackPath,
      completed: true,
      resolution: .preflightReserved
    )
    var reservationPublished = false
    var handedReservationToRuntime = false
    defer {
      if !handedReservationToRuntime {
        if reservationPublished {
          try? DeploymentRollbackJournal.removeOwnedPreflightReservation(
            matching: reservation,
            at: journalURL
          )
        }
        productionOwnerLock.unlock()
      }
    }

    // Publishing this released-reader-unknown lifecycle is the first durable
    // journal-namespace action under the global owner. A released app which
    // scans after this point fails closed even though it does not honor the
    // current owner's flock.
    try reservation.write(to: journalURL)
    reservationPublished = true
    try await requireNoOtherTunerProcess()
    try DeploymentRollbackJournal.removeAbandonedPreflightReservations(
      directory: journalDirectory,
      excluding: journalURL
    )
    let unresolved = try DeploymentRollbackJournal.unresolvedProductionJournals(
      directory: journalDirectory
    ).filter { $0.url != journalURL.standardizedFileURL }
    guard unresolved.isEmpty else {
      throw ApplyPipelineError.unresolvedProductionRollbacks(unresolved.map(\.url))
    }

    let git = try await gitDeploymentPreflight(
      repositoryRoot: request.repositoryRoot,
      expectedBranch: request.expectedBranch
    )
    let release = try MapdReleaseArtifact.load(from: request.mapdReleaseManifestURL).validated()
    let tuneIdentity = TuneDeploymentIdentity(tune: request.tune)

    // Resolve every host-side tile input that does not depend on the current
    // car region list before contacting the car. An explicit artifact remains
    // a supported deterministic/test injection.
    let explicitTileSet: CanonicalTileSetArtifact?
    let rebuildConfig: MapdConfig?
    let decoderURL: URL?
    switch Self.productionTilePlan(for: request) {
    case let .validateExplicit(tileSetArtifactURL):
      let artifact = try CanonicalTileSetArtifact.load(from: tileSetArtifactURL)
      let resolvedDecoderURL = try resolveTileDecoderURL(request)
      explicitTileSet = try await validateTileSetArtifact(
        artifact,
        release: release,
        tuneIdentity: tuneIdentity,
        decoderURL: resolvedDecoderURL
      )
      rebuildConfig = nil
      decoderURL = resolvedDecoderURL
    case .generateCanonical:
      let config = try MapdConfig.load(from: request.mapdConfigURL).validated()
      try config.validateDarwinBinary()
      explicitTileSet = nil
      rebuildConfig = config
      decoderURL = try resolveTileDecoderURL(request)
    case .unchanged:
      explicitTileSet = nil
      rebuildConfig = nil
      decoderURL = nil
    }

    let profile = try await pickTiciProfile(preferred: request.preferredTiciProfile)
    let recovery = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          profile,
          TiciMapdReleaseTransactionCommandBuilder.recoveryCommand(),
        ],
        timeout: 60
      ),
      context: "recover an interrupted tici mapd transaction"
    )
    let mapdRecoveryOutcome: TiciMapdReleaseRecoveryOutcome
    do {
      mapdRecoveryOutcome = try TiciMapdReleaseTransactionCommandBuilder.decodeRecoveryOutcome(
        recovery.standardOutput
      )
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(recovery.standardOutput)\n\(error.localizedDescription)"
      )
    }
    let baselineRead = try await readTiciStaticPostflight(
      profile: profile,
      context: "read-only tici deployment preflight",
      timeout: 60
    )
    let snapshot = baselineRead.deployment.snapshot
    try await validateTiciPreflight(snapshot, git: git, repositoryRoot: request.repositoryRoot)
    guard baselineRead.runtimeEndIsOffroad, !baselineRead.runtimeEndIsOnroad,
          !baselineRead.runtimeEndMapLookaheadEnabled,
          baselineRead.managerRunning, baselineRead.mapdRunning,
          isELF64LittleEndianARM64(baselineRead.activeMapdELFHeader),
          snapshot.bootID != nil else {
      throw ApplyPipelineError.postflightMismatch(
        "tici pre-mutation baseline lacks stable parked state, boot identity, or manager/mapd build evidence"
      )
    }

    let tileSet: CanonicalTileSetArtifact?
    if let explicitTileSet {
      tileSet = explicitTileSet
    } else if let rebuildConfig, let decoderURL {
      let regions: [RegionBox]
      if let configuredRegions = rebuildConfig.regionsOverride {
        regions = normalizedRegions(configuredRegions)
      } else {
        regions = try await discoverRegions(profile: profile)
      }
      guard !regions.isEmpty else { throw ApplyPipelineError.noCachedRegions }
      tileSet = try await generateCanonicalTileSet(
        request: request,
        config: rebuildConfig,
        regions: regions,
        release: release,
        tuneIdentity: tuneIdentity,
        decoderURL: decoderURL
      )
    } else {
      tileSet = nil
    }

    if let tileSet {
      let freeMB = try await ticiFreeMegabytes(profile: profile)
      let requiredMB = requiredTiciTileDeploymentMegabytes(tileSet.manifest.totalBytes)
      guard freeMB >= requiredMB else {
        throw ApplyPipelineError.insufficientDisk(freeMB: freeMB, requiredMB: requiredMB)
      }
    }

    let journal = DeploymentRollbackJournal(
      deploymentID: deploymentID,
      createdAt: createdAt,
      profile: profile,
      branch: git.branch,
      previousBootID: snapshot.bootID,
      previousHead: snapshot.head,
      previousPhysicsParams: snapshot.physicsParams,
      previousQCurveSHA256: snapshot.qCurveSHA256,
      previousMapdReleaseVersion: snapshot.mapdReleaseVersion,
      previousMapdVersion: snapshot.mapdVersion,
      previousActiveMapdSHA256: snapshot.activeMapdSHA256,
      previousActiveMapdBuildInfoSHA256: TuneDeploymentIdentity.sha256Hex(
        baselineRead.activeMapdBuildInfo
      ),
      previousCachedMapdPath: snapshot.cachedMapdPath,
      previousCachedMapdSHA256: snapshot.cachedMapdSHA256,
      mapdRollbackPath: rollbackPath,
      previousTileSetID: snapshot.activeTileSetID,
      targetTileSetID: tileSet?.manifest.tileSetID,
      completed: true,
      resolution: .preflightReserved
    )
    try journal.write(to: journalURL)
    let runtimePreflight = RuntimeDeploymentPreflight(
      git: git,
      profile: profile,
      mapdRecoveryOutcome: mapdRecoveryOutcome,
      release: release,
      tileSet: tileSet,
      journal: journal,
      journalURL: journalURL,
      productionOwnerLock: productionOwnerLock
    )
    handedReservationToRuntime = true
    return runtimePreflight
  }

  private func validateExactSourceTune(repositoryRoot: URL, tune: Tune) throws {
    let physicsURL = repositoryRoot.appendingPathComponent(RepositoryLocator.physicsRelativePath)
    let qCurveURL = repositoryRoot.appendingPathComponent(RepositoryLocator.qCurveRelativePath)
    let paramsDefaultsURL = repositoryRoot.appendingPathComponent(RepositoryLocator.paramsDefaultsRelativePath)
    let physicsPanelURL = repositoryRoot.appendingPathComponent(RepositoryLocator.physicsPanelRelativePath)
    let physicsText = try String(contentsOf: physicsURL, encoding: .utf8)
    let qCurveText = try String(contentsOf: qCurveURL, encoding: .utf8)
    let paramsDefaultsText = try String(contentsOf: paramsDefaultsURL, encoding: .utf8)
    let physicsPanelText = try String(contentsOf: physicsPanelURL, encoding: .utf8)
    let current = try SourcePatcher.readParameters(from: repositoryRoot)
    guard current == VTSCMath.sourceRoundedParameters(tune.params) else {
      throw ApplyPipelineError.postflightMismatch("checked-in VTSC physics authorities differ from the pending tune")
    }
    let patch = try SourcePatcher.makePatch(
      physicsText: physicsText,
      qCurveText: qCurveText,
      paramsDefaultsText: paramsDefaultsText,
      physicsPanelText: physicsPanelText,
      parameters: tune.params,
      bands: tune.bands
    )
    guard patch == SourcePatchResult(
      physicsText: physicsText,
      qCurveText: qCurveText,
      paramsDefaultsText: paramsDefaultsText,
      physicsPanelText: physicsPanelText
    ) else {
      throw ApplyPipelineError.postflightMismatch("checked-in VTSC physics/Q source differs from the pending tune")
    }
  }

  func validateResumePostflightGitIdentity(
    journal: DeploymentRollbackJournal,
    git: GitDeploymentPreflight,
    repositoryRoot: URL
  ) async throws -> ResumePostflightGitIdentity {
    try journal.validatePendingPostflight()
    guard let deployedTargetHead = journal.targetHead else {
      throw DeploymentRollbackJournalError.invalidTargetHead("")
    }
    guard journal.branch == git.branch else { throw ApplyPipelineError.invalidBranch(journal.branch) }
    guard git.localHead == git.originHead else {
      throw ApplyPipelineError.commitMismatch(
        context: "resume local/origin tooling head",
        expected: git.originHead,
        actual: git.localHead
      )
    }
    guard deployedTargetHead != git.localHead else {
      return ResumePostflightGitIdentity(
        deployedTargetHead: deployedTargetHead,
        toolingHead: git.localHead,
        hostOnlyPaths: []
      )
    }

    let relationship = try await processRunner.run(ProcessRequest(
      executableURL: Self.gitURL,
      arguments: ["merge-base", "--is-ancestor", deployedTargetHead, git.localHead],
      currentDirectoryURL: repositoryRoot,
      timeout: 30
    ))
    guard relationship.terminationStatus == 0 else {
      if relationship.terminationStatus == 1 {
        throw ApplyPipelineError.commitMismatch(
          context: "pending deployment target is not an ancestor of the tooling head",
          expected: deployedTargetHead,
          actual: git.localHead
        )
      }
      throw ApplyPipelineError.invalidDeploymentOutput(
        "could not verify pending deployment/tooling ancestry: \(relationship.combinedOutput)"
      )
    }

    let changed = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: [
          "diff", "--no-ext-diff", "--name-only", "-z",
          "\(deployedTargetHead)..\(git.localHead)", "--",
        ],
        currentDirectoryURL: repositoryRoot,
        timeout: 30
      ),
      context: "prove pending deployment follow-up is host-only tooling"
    )
    let paths = changed.standardOutput.split(separator: "\0", omittingEmptySubsequences: true).map(String.init)
    guard !paths.isEmpty else {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "the tooling head differs from the deployed target but Git reported no changed paths"
      )
    }
    let allowedPrefixes = [
      "tools/vtsc_tuner_mac/",
      ".codex/skills/vtsc-tuner-app/",
    ]
    guard paths.allSatisfy({ path in
      !path.hasPrefix("/") && !path.contains("..") && allowedPrefixes.contains { path.hasPrefix($0) }
    }) else {
      throw ApplyPipelineError.postflightMismatch(
        "commits after the immutable deployed target are not host-only VTSC tuner changes: \(paths.joined(separator: ", "))"
      )
    }
    return ResumePostflightGitIdentity(
      deployedTargetHead: deployedTargetHead,
      toolingHead: git.localHead,
      hostOnlyPaths: paths.sorted()
    )
  }

  private func resolveTileDecoderURL(_ request: ApplyRequest) throws -> URL {
    let url = try request.tileDecoderURL ?? MapTileHelperLocator.bundledURL(
      currentDirectoryURL: request.repositoryRoot.appendingPathComponent(
        "tools/vtsc_tuner_mac",
        isDirectory: true
      )
    )
    guard FileManager.default.isExecutableFile(atPath: url.path) else {
      throw MapTileDecoderError.missingHelper(url)
    }
    return url
  }

  private func validateTileSetArtifact(
    _ artifact: CanonicalTileSetArtifact,
    release: ValidatedMapdReleaseArtifact,
    tuneIdentity: TuneDeploymentIdentity,
    decoderURL: URL
  ) async throws -> CanonicalTileSetArtifact {
    guard artifact.manifest.mapdReleaseID == release.artifact.releaseID,
          artifact.manifest.mapdBuildID == release.artifact.buildID,
          artifact.manifest.mapdBinarySHA256 == release.artifact.sha256,
          artifact.manifest.estimatorVersion == release.artifact.estimatorVersion
    else {
      throw ApplyPipelineError.postflightMismatch("tile manifest does not match the exact mapd release")
    }
    guard artifact.manifest.tuneIdentitySHA256 == tuneIdentity.identitySHA256,
          artifact.manifest.tileSigmoidHash == tuneIdentity.tileSigmoidHash
    else {
      throw ApplyPipelineError.postflightMismatch("tile manifest does not match the exact source-rounded tune")
    }
    return try await TileSetValidator(
      decoder: MapTileHelperDecoder(helperURL: decoderURL, processRunner: processRunner)
    ).validate(artifact)
  }

  func generateCanonicalTileSet(
    request: ApplyRequest,
    config: MapdConfig,
    regions: [RegionBox],
    release: ValidatedMapdReleaseArtifact,
    tuneIdentity: TuneDeploymentIdentity,
    decoderURL: URL
  ) async throws -> CanonicalTileSetArtifact {
    let stagingURL = try prepareMapdStaging(pbfURL: config.pbfURL)
    defer { try? FileManager.default.removeItem(at: stagingURL) }
    for region in regions {
      _ = try await generateRegion(
        config: config,
        stagingURL: stagingURL,
        region: region,
        parameters: request.tune.params
      )
    }
    let sourceOfflineURL = stagingURL.appendingPathComponent("offline", isDirectory: true)
    let pbfAttributes = try FileManager.default.attributesOfItem(atPath: config.pbfURL.path)
    guard let pbfDate = pbfAttributes[.modificationDate] as? Date else {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "The prepared PBF has no readable modification date: \(config.pbfURL.path)"
      )
    }
    let setsRootURL = try request.tileSetsRootURL ?? TuneStore.canonicalTileSetsURL()
    let builder = CanonicalTileSetBuilder(
      decoder: MapTileHelperDecoder(helperURL: decoderURL, processRunner: processRunner)
    )
    return try await builder.build(CanonicalTileSetBuildRequest(
      sourceOfflineURL: sourceOfflineURL,
      setsRootURL: setsRootURL,
      release: release,
      tuneIdentity: tuneIdentity,
      tileSigmoidHash: tuneIdentity.tileSigmoidHash,
      pbfURL: config.pbfURL,
      pbfDate: ISO8601DateFormatter().string(from: pbfDate),
      regions: regions
    ))
  }

  private func requiredTiciTileDeploymentMegabytes(_ bytes: UInt64) -> UInt64 {
    let artifactMB = (bytes + 1_048_575) / 1_048_576
    // First activation may need both an immutable copy of the legacy active
    // tree and the new generation. Keep additional headroom for rsync partials
    // and the transaction/manifest files.
    return artifactMB * 2 + 256
  }

  private func gitDeploymentPreflight(
    repositoryRoot: URL,
    expectedBranch: String
  ) async throws -> GitDeploymentPreflight {
    let branch = try await gitOutput(["branch", "--show-current"], repositoryRoot: repositoryRoot)
    guard branch == expectedBranch else { throw ApplyPipelineError.invalidBranch(branch) }
    let status = try await gitOutput(["status", "--porcelain"], repositoryRoot: repositoryRoot)
    guard status.isEmpty else { throw ApplyPipelineError.repositoryDirty(status) }
    let upstream = try await gitOutput(
      ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
      repositoryRoot: repositoryRoot
    )
    let expectedUpstream = "origin/\(expectedBranch)"
    guard upstream == expectedUpstream else {
      throw ApplyPipelineError.upstreamMismatch(expected: expectedUpstream, actual: upstream)
    }
    let localHead = try await gitOutput(["rev-parse", "HEAD"], repositoryRoot: repositoryRoot)
    let remote = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: ["ls-remote", "--heads", "origin", "refs/heads/\(expectedBranch)"],
        currentDirectoryURL: repositoryRoot,
        timeout: 60
      ),
      context: "read exact origin branch head"
    )
    guard let originHead = remote.standardOutput.split(whereSeparator: \Character.isWhitespace).first.map(String.init),
          originHead.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
    else {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "origin did not return an exact head for refs/heads/\(expectedBranch): \(remote.standardOutput)"
      )
    }
    guard localHead == originHead else {
      throw ApplyPipelineError.commitMismatch(context: "local/origin preflight", expected: originHead, actual: localHead)
    }
    return GitDeploymentPreflight(branch: branch, localHead: localHead, originHead: originHead, upstream: upstream)
  }

  func validateTiciPreflight(
    _ snapshot: TiciDeploymentSnapshot,
    git: GitDeploymentPreflight,
    repositoryRoot: URL
  ) async throws {
    guard snapshot.isOffroad, !snapshot.isOnroad else { throw ApplyPipelineError.ticiNotOffroad }
    guard !snapshot.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    guard snapshot.branch == git.branch else { throw ApplyPipelineError.invalidBranch(snapshot.branch) }
    guard !snapshot.dirty else { throw ApplyPipelineError.repositoryDirty("tici checkout is dirty") }
    guard snapshot.head.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil else {
      throw ApplyPipelineError.invalidDeploymentOutput("tici preflight returned an invalid Git head: \(snapshot.head)")
    }
    if snapshot.head != git.localHead {
      let relationship = try await processRunner.run(ProcessRequest(
        executableURL: Self.gitURL,
        arguments: ["merge-base", "--is-ancestor", snapshot.head, git.localHead],
        currentDirectoryURL: repositoryRoot,
        timeout: 30
      ))
      switch relationship.terminationStatus {
      case 0:
        break
      case 1:
        throw ApplyPipelineError.commitMismatch(
          context: "tici preflight commit is not a fast-forward ancestor of the exact target",
          expected: git.localHead,
          actual: snapshot.head
        )
      default:
        throw ApplyPipelineError.invalidDeploymentOutput(
          "could not verify the tici fast-forward relationship: \(relationship.combinedOutput)"
        )
      }
    }
  }

  private func pushExact(
    repositoryRoot: URL,
    expectedBranch: String
  ) async throws -> (head: String, detail: String) {
    let branch = try await gitOutput(["branch", "--show-current"], repositoryRoot: repositoryRoot)
    guard branch == expectedBranch else { throw ApplyPipelineError.invalidBranch(branch) }
    let head = try await gitOutput(["rev-parse", "HEAD"], repositoryRoot: repositoryRoot)
    let push = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: ["push", "origin", "HEAD:refs/heads/\(expectedBranch)"],
        currentDirectoryURL: repositoryRoot,
        timeout: 120
      ),
      context: "push exact \(expectedBranch)"
    )
    let remote = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: ["ls-remote", "--heads", "origin", "refs/heads/\(expectedBranch)"],
        currentDirectoryURL: repositoryRoot,
        timeout: 60
      ),
      context: "verify origin branch head"
    )
    guard let remoteHead = remote.standardOutput.split(whereSeparator: \Character.isWhitespace).first.map(String.init),
          remoteHead == head
    else {
      throw ApplyPipelineError.commitMismatch(
        context: "origin post-push",
        expected: head,
        actual: remote.standardOutput.trimmingCharacters(in: .whitespacesAndNewlines)
      )
    }
    return (head, [push.combinedOutput, "origin=\(remoteHead)"].filter { !$0.isEmpty }.joined(separator: "\n"))
  }

  private func deployProductionRuntime(
    request: ApplyRequest,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    progress: @escaping ApplyProgressHandler
  ) async -> Bool {
    var deployment = preflight
    guard let productionOwnerLock = deployment.productionOwnerLock else {
      await emit(
        .failed,
        id: 5,
        text: "Production deployment lost its global transaction owner",
        detail: "No tici mutation was attempted.",
        progress: progress
      )
      return await finish(false, progress: progress)
    }
    var remoteMutationStarted = false
    var journalResolutionLock: DeploymentRollbackJournalCompletionLock?
    defer {
      journalResolutionLock?.unlock()
      productionOwnerLock.unlock()
    }
    let identity = TuneDeploymentIdentity(tune: request.tune)
    do {
      await emit(.running, id: 5, text: "Reconfirming offroad/kill-switch safety immediately before mutation…", progress: progress)
      let snapshot = try await readTiciDeploymentSnapshot(
        profile: deployment.profile,
        context: "final read-only tici safety check",
        timeout: 60
      ).snapshot
      try await validateTiciPreflight(
        snapshot,
        git: deployment.git,
        repositoryRoot: request.repositoryRoot
      )
      await emit(.succeeded, id: 5, text: "Tici is still parked/offroad with Map Lookahead disabled", progress: progress)

      try await validateExclusiveProductionMutationClaim(deployment)
      let ownerLock = try DeploymentRollbackJournal.acquireCompletionLock(for: deployment.journalURL)
      journalResolutionLock = ownerLock
      deployment.journal = try claimProductionMutation(
        expected: deployment.journal,
        at: deployment.journalURL
      )
      // In-process error handling may now roll back immediately; crash
      // recovery does not depend on this flag because mutationInProgress is
      // already fully durable and old-reader-safe.
      remoteMutationStarted = true

      await emit(.running, id: 6, text: "Fast-forwarding the tici to the exact pushed commit…", progress: progress)
      let fastForward = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            deployment.profile,
            try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
              branch: deployment.git.branch,
              head: targetHead
            ),
          ],
          timeout: 180
        ),
        context: "exact tici fast-forward"
      )
      await emit(.succeeded, id: 6, text: "Tici now matches the exact origin commit", detail: fastForward.combinedOutput, progress: progress)

      await emit(.running, id: 7, text: "Installing and verifying the exact ARM64 mapd release and persistent cache…", progress: progress)
      let releaseDetail = try await stageAndInstallMapdRelease(preflight: deployment)
      await emit(.succeeded, id: 7, text: "Exact mapd release and persistent cache are staged", detail: releaseDetail, progress: progress)

      await emit(.running, id: 8, text: "Writing/read-back verifying all six physics Params and exact Q data…", progress: progress)
      let physics = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            deployment.profile,
            try TiciPhysicsCommandBuilder.synchronizeAndVerifyCommand(parameters: request.tune.params),
          ],
          timeout: 60
        ),
        context: "synchronize exact VTSC physics Params"
      )
      let qReadback = try await readTiciDeploymentSnapshot(
        profile: deployment.profile,
        context: "verify exact VTSC Q curve",
        timeout: 60
      )
      guard qReadback.qCurve.sha256 == identity.qCurveSHA256,
            qReadback.qCurve.enabled == identity.qCurveEnabled,
            qReadback.qCurve.pointCount == identity.qCurvePointCount
      else {
        throw ApplyPipelineError.postflightMismatch("Q curve differs from the exact deployed source")
      }
      let qCurveDetail = [
        "q_curve_sha256=\(qReadback.qCurve.sha256)",
        "q_curve_enabled=\(qReadback.qCurve.enabled)",
        "q_curve_point_count=\(qReadback.qCurve.pointCount)",
      ].joined(separator: "\\n")
      await emit(
        .succeeded,
        id: 8,
        text: "All six physics Params and Q data match the source-rounded tune",
        detail: [physics.combinedOutput, qCurveDetail].filter { !$0.isEmpty }.joined(separator: "\n"),
        progress: progress
      )

      if let tileSet = deployment.tileSet {
        await emit(.running, id: 9, text: "Staging, fully verifying, and atomically activating the requested tile set…", progress: progress)
        let service = TiciTileSetDeploymentService(processRunner: processRunner)
        let staged = try await service.stageAndVerify(artifact: tileSet, profile: deployment.profile)
        // Activation can complete remotely even if SSH disconnects before the
        // result arrives. Recovery inspects the durable on-device transaction.
        let activated = try await service.activate(staged)
        await emit(
          .succeeded,
          id: 9,
          text: "Requested tile set activated atomically; prior set retained",
          detail: activated.commandOutput,
          progress: progress
        )
      } else {
        await emit(
          .succeeded,
          id: 9,
          text: "No tile replacement requested or required for the runtime whole-curve profile",
          detail: "The active tile tree was not transferred to or modified.",
          progress: progress
        )
      }

      await emit(.running, id: 10, text: "Rebooting once after every artifact is ready…", progress: progress)
      let preRebootSnapshot = try await freshParkedSafetySnapshot(profile: deployment.profile)
      guard let preRebootBootID = preRebootSnapshot.bootID else {
        throw ApplyPipelineError.postflightMismatch("tici boot identity is missing before deployment reboot")
      }
      deployment.journal.rebootSent = true
      deployment.journal.deploymentPreRebootBootID = preRebootBootID
      try durablyRecordRebootIntent(deployment.journal, at: deployment.journalURL)
      try await sendReboot(profile: deployment.profile)
      deployment.journal = try handoffToOutdoorPostflight(
        expected: deployment.journal,
        at: deployment.journalURL
      )
      journalResolutionLock?.unlock()
      journalResolutionLock = nil
      productionOwnerLock.unlock()
      await emit(.succeeded, id: 10, text: "Single deployment reboot sent", progress: progress)

      await emit(
        .running,
        id: 11,
        text: "Waiting for the tici and verifying the installed static identity…",
        progress: progress
      )
      do {
        try await waitForTici(
          profile: deployment.profile,
          timeout: 300,
          initialDelayNanoseconds: rebootInitialDelayNanoseconds,
          pollDelayNanoseconds: rebootPollDelayNanoseconds
        )
        _ = try await waitForStaticInstalledIdentity(
          preflight: deployment,
          targetHead: targetHead,
          identity: identity,
          expectedActiveTileSetID: deployment.tileSet?.manifest.tileSetID ?? deployment.journal.previousTileSetID,
          timeout: 300,
          pollInterval: max(0.001, Double(rebootPollDelayNanoseconds) / 1_000_000_000)
        )
      } catch {
        await emitFailure(
          id: 11,
          text: "Deployment reboot/static identity could not be proven; outdoor postflight remains pending",
          error: error,
          progress: progress
        )
        return await finish(false, progress: progress)
      }
      await emit(
        .succeeded,
        id: 11,
        text: "Deployment installed; outdoor controller-ready proof remains pending",
        detail: [
          "journal=\(deployment.journalURL.path)",
          "deployed_target_head=\(targetHead)",
          "Use Resume Pending Outdoor Postflight after a real-GPS drive.",
          "Only that two-phase controller-ready then stable-offroad proof can complete this deployment.",
        ].joined(separator: "\n"),
        progress: progress
      )
      return await finish(true, progress: progress)
    } catch {
      if remoteMutationStarted {
        let resolution = await rollbackProductionDeploymentIfJournalPending(
          context: ProductionRollbackContext(preflight: deployment),
          ownerLock: journalResolutionLock
        )
        switch resolution {
        case let .alreadyCompleted(detail):
          await emit(
            .succeeded,
            id: 12,
            text: "Deployment completion was already certified by another VTSC Tuner instance",
            detail: detail,
            progress: progress
          )
          return await finish(true, progress: progress)
        case let .rolledBack(detail):
          await emitFailure(id: 12, text: "Production deployment failed", error: error, progress: progress)
          await emit(.succeeded, id: 13, text: "Tici rollback completed", detail: detail, progress: progress)
        case let .rollbackFailed(detail):
          await emitFailure(id: 12, text: "Production deployment failed", error: error, progress: progress)
          await emit(.failed, id: 13, text: "Tici rollback needs manual attention", detail: detail, progress: progress)
        }
      } else {
        await emitFailure(id: 12, text: "Production deployment failed before remote mutation", error: error, progress: progress)
      }
      return await finish(false, progress: progress)
    }
  }

  func validateExclusiveProductionMutationClaim(_ deployment: RuntimeDeploymentPreflight) async throws {
    guard let productionOwnerLock = deployment.productionOwnerLock else {
      throw ApplyPipelineError.postflightMismatch("production deployment lost its global transaction owner")
    }
    try await requireNoOtherTunerProcess()
    let unresolved = try DeploymentRollbackJournal.unresolvedProductionJournals(
      directory: productionOwnerLock.directoryURL
    ).filter { $0.url != deployment.journalURL.standardizedFileURL }
    guard unresolved.isEmpty else {
      throw ApplyPipelineError.unresolvedProductionRollbacks(unresolved.map(\.url))
    }
    let current = try DeploymentRollbackJournal.load(from: deployment.journalURL)
    guard current == deployment.journal,
          current.effectiveResolution == .preflightReserved,
          current.completed,
          !current.rebootSent else {
      throw ApplyPipelineError.postflightMismatch("production reservation changed before mutation claim")
    }
    let baselineRead = try await readTiciStaticPostflight(
      profile: deployment.profile,
      context: "revalidate exact tici pre-mutation baseline",
      timeout: 60
    )
    try validateExactPreMutationBaseline(baselineRead, journal: current)
  }

  private func validateExactPreMutationBaseline(
    _ readback: TiciStaticDeploymentPostflightRead,
    journal: DeploymentRollbackJournal
  ) throws {
    let snapshot = readback.deployment.snapshot
    guard snapshot.isOffroad, !snapshot.isOnroad, !snapshot.mapLookaheadEnabled,
          readback.runtimeEndIsOffroad, !readback.runtimeEndIsOnroad,
          !readback.runtimeEndMapLookaheadEnabled else {
      throw ApplyPipelineError.postflightMismatch("tici parked state changed before mutation claim")
    }
    guard !snapshot.dirty,
          snapshot.branch == journal.branch,
          snapshot.bootID == journal.previousBootID,
          snapshot.head == journal.previousHead,
          snapshot.physicsParams == journal.previousPhysicsParams,
          readback.deployment.qCurve.sha256 == journal.previousQCurveSHA256,
          snapshot.mapdReleaseVersion == journal.previousMapdReleaseVersion,
          snapshot.mapdVersion == journal.previousMapdVersion,
          snapshot.activeMapdSHA256 == journal.previousActiveMapdSHA256,
          snapshot.cachedMapdPath == journal.previousCachedMapdPath,
          snapshot.cachedMapdSHA256 == journal.previousCachedMapdSHA256,
          snapshot.activeTileSetID == journal.previousTileSetID,
          journal.previousActiveMapdBuildInfoSHA256 ==
            TuneDeploymentIdentity.sha256Hex(readback.activeMapdBuildInfo),
          isELF64LittleEndianARM64(readback.activeMapdELFHeader),
          readback.managerRunning,
          readback.mapdRunning else {
      throw ApplyPipelineError.postflightMismatch(
        "tici source/tune/mapd/tile/boot baseline changed before mutation claim"
      )
    }
  }

  private func stageAndInstallMapdRelease(preflight: RuntimeDeploymentPreflight) async throws -> String {
    let stagedPath = "/data/media/0/osm/binaries/.mapd-release-\(preflight.journal.deploymentID.uuidString.lowercased()).partial"
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          preflight.profile,
          TiciParkedMutationGate.guardedCommand(
            "mkdir -p /data/media/0/osm/binaries && rm -f \(stagedPath)",
            refusalMessage: "refusing mapd staging unless tici is exactly offroad and Map Lookahead is disabled"
          ),
        ],
        timeout: 30
      ),
      context: "prepare mapd release staging"
    )
    let transfer = try await checked(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-a", "--partial",
          "--rsync-path", TiciParkedMutationGate.gatedRsyncPath(
            refusalMessage: "refusing mapd transfer unless tici is exactly offroad and Map Lookahead is disabled"
          ),
          "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          preflight.release.artifact.binaryURL.path,
          "\(preflight.profile):\(stagedPath)",
        ],
        timeout: 600
      ),
      context: "stage exact mapd release"
    )
    let probe = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          preflight.profile,
          try TiciMapdReleaseTransactionCommandBuilder.probeCommand(
            release: preflight.release,
            stagedPath: stagedPath
          ),
        ],
        timeout: 60
      ),
      context: "run read-only mapd release identity probe"
    )
    do {
      _ = try TiciMapdReleaseTransactionCommandBuilder.validateBuildInfo(
        TiciMapdReleaseTransactionCommandBuilder.decodeProbe(probe.standardOutput),
        matches: preflight.release
      )
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(probe.standardOutput)\n\(error.localizedDescription)"
      )
    }
    let install = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          preflight.profile,
          try TiciMapdReleaseTransactionCommandBuilder.installCommand(
            release: preflight.release,
            stagedPath: stagedPath,
            rollbackPath: preflight.journal.mapdRollbackPath
          ),
        ],
        timeout: 180
      ),
      context: "install and verify exact mapd release"
    )
    do {
      let result = try TiciMapdReleaseTransactionCommandBuilder.decodeResult(install.standardOutput)
      _ = try TiciMapdReleaseTransactionCommandBuilder.validate(
        result,
        matches: preflight.release,
        rollbackPath: preflight.journal.mapdRollbackPath
      )
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(install.standardOutput)\n\(error.localizedDescription)"
      )
    }
    return [transfer.combinedOutput, probe.combinedOutput, install.combinedOutput]
      .filter { !$0.isEmpty }
      .joined(separator: "\n")
  }

  private func waitForResumedProductionPostflight(
    preflight: RuntimeDeploymentPreflight,
    deployedTargetHead: String,
    identity: TuneDeploymentIdentity,
    timeout: TimeInterval,
    pollInterval: TimeInterval,
    progress: @escaping ApplyProgressHandler
  ) async throws -> ResumedPostflightObservation {
    let clock = ContinuousClock()
    let phaseTimeout = Duration.seconds(max(0, timeout))
    var phaseDeadline = clock.now.advanced(by: phaseTimeout)
    var attemptedCurrentPhase = false
    var lastWait: ResumedPostflightWaitState = .controllerEvidence(
      "Waiting for fresh real-GPS profile and liveMapDataSP controller-readiness evidence."
    )
    var controllerEvidence: TiciDeploymentPostflight?
    while true {
      try Task.checkCancellation()
      if attemptedCurrentPhase, clock.now >= phaseDeadline { throw lastWait }
      let readback: TiciRuntimePostflightRead
      do {
        let remaining = phaseDeadline - clock.now
        let parts = remaining.components
        let remainingSeconds = max(
          0.001,
          Double(parts.seconds) + Double(parts.attoseconds) / 1_000_000_000_000_000_000
        )
        readback = try await readTiciRuntimePostflight(
          profile: preflight.profile,
          context: "resume pending tici deployment postflight",
          timeout: min(30, remainingSeconds)
        )
      } catch {
        guard isRetryableResumedTransportError(error) else { throw error }
        lastWait = .transport(error.localizedDescription)
        attemptedCurrentPhase = true
        guard clock.now < phaseDeadline else { throw lastWait }
        try await clock.sleep(for: .seconds(max(0.001, pollInterval)))
        continue
      }
      let postflight = makeProductionPostflight(
        readback,
        preflight: preflight,
        identity: identity
      )
      if let controllerEvidence {
        do {
          try validateResumedOffroadCompletion(
            postflight,
            controllerEvidence: controllerEvidence,
            preflight: preflight,
            targetHead: deployedTargetHead,
            identity: identity,
            expectedActiveTileSetID: preflight.journal.targetTileSetID ?? preflight.journal.previousTileSetID
          )
          return ResumedPostflightObservation(controllerReady: controllerEvidence, offroad: postflight)
        } catch let wait as ResumedPostflightWaitState {
          lastWait = wait
        }
      } else {
        do {
          try validateResumedControllerEvidence(
            postflight,
            preflight: preflight,
            targetHead: deployedTargetHead,
            identity: identity,
            expectedActiveTileSetID: preflight.journal.targetTileSetID ?? preflight.journal.previousTileSetID
          )
          // This evidence is deliberately in-memory only. A later offroad
          // liveMapDataSP.valid=false cannot erase the controller-ready state
          // observed while ignition was still on.
          controllerEvidence = postflight
          phaseDeadline = clock.now.advanced(by: phaseTimeout)
          attemptedCurrentPhase = false
          lastWait = .offroadTransition
          await emit(
            .running,
            id: 1,
            text: "Profile acceptance proven — turn ignition off now",
            detail: "With Map Lookahead disabled, this proves the profile would be accepted by the controller when later enabled. The proof is retained only in memory while the app waits for the clean offroad transition.",
            progress: progress
          )
          // Phase 2 receives its own full grace window. Poll it immediately
          // once before sleeping so a just-completed transition is not lost.
          continue
        } catch let wait as ResumedPostflightWaitState {
          lastWait = wait
        }
      }
      attemptedCurrentPhase = true
      guard clock.now < phaseDeadline else { throw lastWait }
      try await clock.sleep(for: .seconds(max(0.001, pollInterval)))
    }
  }

  private func readTiciRuntimePostflight(
    profile: String,
    context: String,
    timeout: TimeInterval = 30
  ) async throws -> TiciRuntimePostflightRead {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          profile,
          TiciSnapshotWireCommandBuilder.inspectionCommand(includeRuntimePostflight: true),
        ],
        timeout: max(0.001, min(30, timeout))
      ),
      context: context
    )
    do {
      return try TiciDeploymentSnapshotDecoder.decodeRuntimePostflight(result.standardOutput)
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(result.standardOutput)\n\(error.localizedDescription)"
      )
    }
  }

  private func readTiciStaticPostflight(
    profile: String,
    context: String,
    timeout: TimeInterval = 30
  ) async throws -> TiciStaticDeploymentPostflightRead {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          profile,
          TiciSnapshotWireCommandBuilder.inspectionCommand(includeStaticPostflight: true),
        ],
        timeout: max(0.001, min(30, timeout))
      ),
      context: context
    )
    do {
      return try TiciDeploymentSnapshotDecoder.decodeStaticPostflight(result.standardOutput)
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(result.standardOutput)\n\(error.localizedDescription)"
      )
    }
  }

  private func readTiciRollbackStaticPostflight(
    profile: String,
    context: String,
    timeout: TimeInterval = 30
  ) async throws -> TiciRollbackStaticPostflightRead {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          profile,
          TiciSnapshotWireCommandBuilder.inspectionCommand(
            includeStaticPostflight: true,
            includeMapdBuildIdentity: false
          ),
        ],
        timeout: max(0.001, min(30, timeout))
      ),
      context: context
    )
    do {
      return try TiciDeploymentSnapshotDecoder.decodeRollbackStaticPostflight(result.standardOutput)
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(result.standardOutput)\n\(error.localizedDescription)"
      )
    }
  }

  private func isRetryableResumedTransportError(_ error: Error) -> Bool {
    if let runnerError = error as? ProcessRunnerError,
       case .timedOut = runnerError { return true }
    guard let pipelineError = error as? ApplyPipelineError else { return false }
    return switch pipelineError {
    case let .commandFailed(_, status, _):
      status == 255
    case .noReachableTici:
      true
    default:
      false
    }
  }

  private func isRetryablePostRebootStartupError(_ error: Error) -> Bool {
    error is RuntimeStartupWaitState || error is BootTransitionWaitState ||
      isRetryableResumedTransportError(error)
  }

  private func requireBootTransition(
    current: String?,
    previous: String?,
    missingPreviousDetail: String
  ) throws {
    guard let previous else {
      throw ApplyPipelineError.postflightMismatch(missingPreviousDetail)
    }
    guard TiciBootIdentity.isValid(previous) else {
      throw ApplyPipelineError.postflightMismatch(
        "journal pre-reboot boot identity is invalid; reboot transition cannot be certified"
      )
    }
    guard let current, current != previous else {
      throw BootTransitionWaitState.identityPending(previous: previous, current: current)
    }
  }

  private func requireResumedDeploymentBootTransition(
    current: String?,
    journal: DeploymentRollbackJournal
  ) throws {
    // B174-style schema-1 journals predate boot identity capture and use the
    // legacy awaitingPostflight lifecycle. Their current supervised onroad
    // controller evidence remains resumable without fabricating history. New
    // awaitingOutdoorPostflight journals never receive this exception.
    if journal.effectiveResolution == .awaitingPostflight,
       journal.deploymentPreRebootBootID == nil {
      return
    }
    try requireBootTransition(
      current: current,
      previous: journal.deploymentPreRebootBootID,
      missingPreviousDetail: "This new outdoor-postflight journal has no durable deployment pre-reboot boot identity and cannot be completed safely. Abort and roll back, then redeploy with the current tuner."
    )
  }

  private func requireNoOtherTunerProcess() async throws {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.processListURL,
        arguments: ["-axo", "pid=,command="],
        timeout: 10
      ),
      context: "inspect running VTSC Tuner processes"
    )
    let peers = Self.otherTunerProcessLines(
      result.standardOutput,
      excluding: currentProcessID
    )
    guard peers.isEmpty else {
      throw ApplyPipelineError.postflightMismatch(
        "Another VTSC Tuner process is running. Close every other VTSC Tuner window before a car-facing deployment.\n\(peers.joined(separator: "\n"))"
      )
    }
  }

  static func otherTunerProcessLines(_ processList: String, excluding currentPID: Int32) -> [String] {
    processList.components(separatedBy: .newlines).compactMap { rawLine in
      let line = rawLine.trimmingCharacters(in: .whitespaces)
      guard !line.isEmpty else { return nil }
      let fields = line.split(maxSplits: 1, whereSeparator: \Character.isWhitespace)
      guard fields.count == 2, let pid = Int32(fields[0]), pid != currentPID else { return nil }
      let command = String(fields[1])
      let executableToken = command.split(whereSeparator: \Character.isWhitespace).first.map(String.init) ?? ""
      let exactBundleExecutable = URL(fileURLWithPath: executableToken).lastPathComponent == "VTSCTuner" ||
        command.contains("/VTSC Tuner.app/Contents/MacOS/VTSCTuner")
      guard exactBundleExecutable else { return nil }
      return "pid=\(pid) \(command)"
    }
  }

  private func isTerminalResumePostflightFailure(_ error: Error) -> Bool {
    if error is CancellationError || error is ResumedPostflightWaitState { return false }
    if let journalError = error as? DeploymentRollbackJournalError {
      switch journalError {
      case .productionOwnerLocked, .completionLocked:
        return false
      default:
        break
      }
    }
    return !isRetryableResumedTransportError(error)
  }

  private func acquireJournalResolutionLock(
    for journalURL: URL,
    timeout: Duration = .seconds(180)
  ) async throws -> DeploymentRollbackJournalCompletionLock {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while true {
      try Task.checkCancellation()
      do {
        return try DeploymentRollbackJournal.acquireCompletionLock(for: journalURL)
      } catch DeploymentRollbackJournalError.completionLocked {
        guard clock.now < deadline else { throw DeploymentRollbackJournalError.completionLocked(journalURL) }
        try await clock.sleep(for: .milliseconds(10))
      }
    }
  }

  private func acquireGlobalProductionOwnerLock(
    for journalURL: URL,
    timeout: Duration = .seconds(180)
  ) async throws -> ProductionDeploymentOwnerLock {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    let directory = journalURL.standardizedFileURL.deletingLastPathComponent()
    while true {
      try Task.checkCancellation()
      do {
        return try DeploymentRollbackJournal.acquireProductionOwnerLock(directory: directory)
      } catch DeploymentRollbackJournalError.productionOwnerLocked {
        guard clock.now < deadline else {
          throw DeploymentRollbackJournalError.productionOwnerLocked(
            directory.appendingPathComponent(".production-owner.lock")
          )
        }
        try await clock.sleep(for: .milliseconds(10))
      }
    }
  }

  private func makeProductionPostflight(
    _ readback: TiciRuntimePostflightRead,
    preflight: RuntimeDeploymentPreflight,
    identity: TuneDeploymentIdentity
  ) -> TiciDeploymentPostflight {
    let snapshot = readback.deployment.snapshot
    let physics = Dictionary(uniqueKeysWithValues: identity.physics.map { ($0.paramKey, $0.value) })
    let physicsMatches = physics.allSatisfy { snapshot.physicsParams[$0.key] ?? nil == $0.value }
    let release = preflight.release.artifact
    let buildInfo = try? JSONDecoder().decode(TiciMapdReleaseBuildInfo.self, from: readback.activeMapdBuildInfo)
    let buildInfoMatches = buildInfo?.releaseID == release.releaseID &&
      buildInfo?.buildID == release.buildID &&
      buildInfo?.estimatorVersion == release.estimatorVersion &&
      buildInfo?.capabilities.contains(release.capability) == true
    let activeELFARM64 = isELF64LittleEndianARM64(readback.activeMapdELFHeader)
    let profile = TiciWholeCurvePostflightValidator.inspect(
      profileData: readback.wholeCurveProfile,
      gpsData: readback.lastGPSPosition,
      now: Date(timeIntervalSince1970: TimeInterval(readback.remoteEpochMilliseconds) / 1_000)
    )
    return TiciDeploymentPostflight(
      bootID: snapshot.bootID,
      isOffroad: snapshot.isOffroad,
      isOnroad: snapshot.isOnroad,
      runtimeEndIsOffroad: readback.runtimeEndIsOffroad,
      runtimeEndIsOnroad: readback.runtimeEndIsOnroad,
      runtimeEndMapLookaheadEnabled: readback.runtimeEndMapLookaheadEnabled,
      mapLookaheadEnabled: snapshot.mapLookaheadEnabled,
      branch: snapshot.branch,
      head: snapshot.head,
      dirty: snapshot.dirty,
      physicsMatches: physicsMatches,
      qCurveSHA256: readback.deployment.qCurve.sha256,
      qCurveEnabled: readback.deployment.qCurve.enabled,
      qCurvePointCount: readback.deployment.qCurve.pointCount,
      mapdReleaseVersion: snapshot.mapdReleaseVersion ?? "",
      mapdVersion: snapshot.mapdVersion ?? "",
      activeMapdSHA256: snapshot.activeMapdSHA256,
      cachedMapdSHA256: snapshot.cachedMapdSHA256 ?? "",
      managerRunning: readback.managerRunning,
      mapdRunning: readback.mapdRunning,
      // An exact active digest is a stronger marker check than scanning a
      // moving executable, and the local immutable artifact was marker- and
      // build-info-validated before it was transferred.
      capabilityPresent: snapshot.activeMapdSHA256 == release.sha256,
      activeELFARM64: activeELFARM64,
      buildInfoMatches: buildInfoMatches,
      profileEstimatorVersion: profile.estimatorVersion,
      profileRouteFingerprint: profile.routeFingerprint,
      profileSigmoidHash: profile.sigmoidHash,
      profileFresh: profile.fresh,
      profileValuesFinite: profile.valuesFinite,
      gpsStatus: profile.gpsStatus,
      profileValidationStatus: profile.validationStatus,
      profilePointCount: profile.pointCount,
      profileEventCount: profile.eventCount,
      liveMapDataUpdated: readback.liveMapDataControllerStatus.updated,
      liveMapDataValid: readback.liveMapDataControllerStatus.valid,
      liveMapDataLogMonoTimeNs: readback.liveMapDataControllerStatus.logMonoTimeNs,
      liveMapDataSampleMonoTimeNs: readback.liveMapDataControllerStatus.sampleMonoTimeNs,
      roadGeometryValid: readback.liveMapDataControllerStatus.roadGeometryValid,
      activeTileSetID: snapshot.activeTileSetID
    )
  }

  private func makeStaticProductionPostflight(
    _ readback: TiciStaticDeploymentPostflightRead,
    preflight: RuntimeDeploymentPreflight,
    identity: TuneDeploymentIdentity
  ) -> TiciDeploymentPostflight {
    let snapshot = readback.deployment.snapshot
    let physics = Dictionary(uniqueKeysWithValues: identity.physics.map { ($0.paramKey, $0.value) })
    let physicsMatches = physics.allSatisfy { snapshot.physicsParams[$0.key] ?? nil == $0.value }
    let release = preflight.release.artifact
    let buildInfo = try? JSONDecoder().decode(TiciMapdReleaseBuildInfo.self, from: readback.activeMapdBuildInfo)
    let buildInfoMatches = buildInfo?.releaseID == release.releaseID &&
      buildInfo?.buildID == release.buildID &&
      buildInfo?.estimatorVersion == release.estimatorVersion &&
      buildInfo?.capabilities.contains(release.capability) == true
    return TiciDeploymentPostflight(
      bootID: snapshot.bootID,
      isOffroad: snapshot.isOffroad,
      isOnroad: snapshot.isOnroad,
      runtimeEndIsOffroad: readback.runtimeEndIsOffroad,
      runtimeEndIsOnroad: readback.runtimeEndIsOnroad,
      runtimeEndMapLookaheadEnabled: readback.runtimeEndMapLookaheadEnabled,
      mapLookaheadEnabled: snapshot.mapLookaheadEnabled,
      branch: snapshot.branch,
      head: snapshot.head,
      dirty: snapshot.dirty,
      physicsMatches: physicsMatches,
      qCurveSHA256: readback.deployment.qCurve.sha256,
      qCurveEnabled: readback.deployment.qCurve.enabled,
      qCurvePointCount: readback.deployment.qCurve.pointCount,
      mapdReleaseVersion: snapshot.mapdReleaseVersion ?? "",
      mapdVersion: snapshot.mapdVersion ?? "",
      activeMapdSHA256: snapshot.activeMapdSHA256,
      cachedMapdSHA256: snapshot.cachedMapdSHA256 ?? "",
      managerRunning: readback.managerRunning,
      mapdRunning: readback.mapdRunning,
      capabilityPresent: snapshot.activeMapdSHA256 == release.sha256,
      activeELFARM64: isELF64LittleEndianARM64(readback.activeMapdELFHeader),
      buildInfoMatches: buildInfoMatches,
      profileEstimatorVersion: "",
      profileRouteFingerprint: "",
      profileSigmoidHash: "",
      profileFresh: false,
      profileValuesFinite: false,
      gpsStatus: "not_checked",
      profileValidationStatus: "not_checked",
      profilePointCount: 0,
      profileEventCount: 0,
      liveMapDataUpdated: false,
      liveMapDataValid: false,
      liveMapDataLogMonoTimeNs: 0,
      liveMapDataSampleMonoTimeNs: 0,
      roadGeometryValid: false,
      activeTileSetID: snapshot.activeTileSetID
    )
  }

  private func isELF64LittleEndianARM64(_ header: Data) -> Bool {
    guard header.count >= 20 else { return false }
    return Array(header.prefix(6)) == [0x7f, 0x45, 0x4c, 0x46, 2, 1] &&
      header[18] == 183 && header[19] == 0
  }

  private func validatePostflight(
    _ result: TiciDeploymentPostflight,
    requireNonemptyWholeCurveProfile: Bool,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?
  ) throws {
    guard result.isOffroad, !result.isOnroad,
          result.runtimeEndIsOffroad, !result.runtimeEndIsOnroad else {
      throw ApplyPipelineError.ticiNotOffroad
    }
    guard !result.mapLookaheadEnabled, !result.runtimeEndMapLookaheadEnabled else {
      throw ApplyPipelineError.mapLookaheadMustRemainDisabled
    }
    guard result.branch == preflight.journal.branch else {
      throw ApplyPipelineError.invalidBranch(result.branch)
    }
    guard !result.dirty else {
      throw ApplyPipelineError.repositoryDirty("tici checkout is dirty during postflight")
    }
    guard result.head == targetHead else {
      throw ApplyPipelineError.commitMismatch(context: "tici postflight", expected: targetHead, actual: result.head)
    }
    guard result.physicsMatches else { throw ApplyPipelineError.postflightMismatch("physics Params differ from source") }
    guard result.qCurveSHA256 == identity.qCurveSHA256,
          result.qCurveEnabled == identity.qCurveEnabled,
          result.qCurvePointCount == identity.qCurvePointCount
    else { throw ApplyPipelineError.postflightMismatch("Q curve differs from the exact deployed source") }
    let release = preflight.release.artifact
    guard result.mapdReleaseVersion == release.releaseID, result.mapdVersion == release.releaseID else {
      throw ApplyPipelineError.postflightMismatch("MapdVersion/MapdReleaseVersion do not match \(release.releaseID)")
    }
    guard result.activeMapdSHA256 == release.sha256, result.cachedMapdSHA256 == release.sha256 else {
      throw ApplyPipelineError.postflightMismatch("active and persistent mapd digests are not exact")
    }
    guard result.managerRunning, result.mapdRunning, result.capabilityPresent,
          result.activeELFARM64, result.buildInfoMatches else {
      throw ApplyPipelineError.postflightMismatch(
        "manager/mapd is not running with the exact ARM64 build identity/capability"
      )
    }
    if requireNonemptyWholeCurveProfile {
      guard result.profileValidationStatus != "profile_pending",
            result.gpsStatus == "valid"
      else {
        throw ApplyPipelineError.postflightMismatch(
          "whole-curve profile pending real GPS/profile: gps=\(result.gpsStatus) profile=\(result.profileValidationStatus)"
        )
      }
    }
    guard result.profileEstimatorVersion == release.estimatorVersion else {
      throw ApplyPipelineError.postflightMismatch("whole-curve estimator version is missing or mismatched")
    }
    guard result.profileSigmoidHash == identity.tileSigmoidHash else {
      throw ApplyPipelineError.postflightMismatch(
        "whole-curve profile sigmoid hash is \(result.profileSigmoidHash), expected \(identity.tileSigmoidHash)"
      )
    }
    if requireNonemptyWholeCurveProfile {
      guard result.profilePointCount >= 3,
            result.profileFresh,
            result.profileValuesFinite,
            result.profileRouteFingerprint.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil
      else {
        throw ApplyPipelineError.postflightMismatch(
          "whole-curve profile is missing, stale, malformed, or not near real GPS: \(result.profileValidationStatus)"
        )
      }
    }
    guard result.activeTileSetID == expectedActiveTileSetID else {
      throw ApplyPipelineError.postflightMismatch(
        "active tile-set identity is \(result.activeTileSetID ?? "<none>"), expected \(expectedActiveTileSetID ?? "<none>")"
      )
    }
  }

  private func validateStaticInstalledIdentity(
    _ result: TiciDeploymentPostflight,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?
  ) throws {
    guard result.isOffroad, !result.isOnroad,
          result.runtimeEndIsOffroad, !result.runtimeEndIsOnroad else {
      throw ApplyPipelineError.ticiNotOffroad
    }
    guard !result.mapLookaheadEnabled, !result.runtimeEndMapLookaheadEnabled else {
      throw ApplyPipelineError.mapLookaheadMustRemainDisabled
    }
    guard result.branch == preflight.journal.branch else { throw ApplyPipelineError.invalidBranch(result.branch) }
    guard !result.dirty else { throw ApplyPipelineError.repositoryDirty("tici checkout is dirty after reboot") }
    guard result.head == targetHead else {
      throw ApplyPipelineError.commitMismatch(context: "tici static post-reboot", expected: targetHead, actual: result.head)
    }
    guard result.physicsMatches else { throw ApplyPipelineError.postflightMismatch("physics Params differ from source") }
    guard result.qCurveSHA256 == identity.qCurveSHA256,
          result.qCurveEnabled == identity.qCurveEnabled,
          result.qCurvePointCount == identity.qCurvePointCount else {
      throw ApplyPipelineError.postflightMismatch("Q curve differs from the exact deployed source")
    }
    let release = preflight.release.artifact
    guard result.mapdReleaseVersion == release.releaseID, result.mapdVersion == release.releaseID else {
      throw ApplyPipelineError.postflightMismatch("MapdVersion/MapdReleaseVersion do not match \(release.releaseID)")
    }
    guard result.activeMapdSHA256 == release.sha256, result.cachedMapdSHA256 == release.sha256 else {
      throw ApplyPipelineError.postflightMismatch("active and persistent mapd digests are not exact")
    }
    guard result.capabilityPresent, result.activeELFARM64, result.buildInfoMatches else {
      throw ApplyPipelineError.postflightMismatch("native mapd build identity/capability is not exact")
    }
    guard result.activeTileSetID == expectedActiveTileSetID else {
      throw ApplyPipelineError.postflightMismatch("active tile-set identity changed after reboot")
    }
    try requireBootTransition(
      current: result.bootID,
      previous: preflight.journal.deploymentPreRebootBootID,
      missingPreviousDetail: "This deployment journal predates durable boot-transition proof. Use Abort and Roll Back Pending Deployment, then redeploy with the current tuner."
    )
    guard result.managerRunning, result.mapdRunning else {
      throw RuntimeStartupWaitState.processesNotReady(
        manager: result.managerRunning,
        mapd: result.mapdRunning
      )
    }
  }

  func waitForStaticInstalledIdentity(
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?,
    timeout: TimeInterval,
    pollInterval: TimeInterval = 5
  ) async throws -> TiciDeploymentPostflight {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(timeout))
    var lastError: Error = RuntimeStartupWaitState.processesNotReady(manager: false, mapd: false)
    repeat {
      try Task.checkCancellation()
      do {
        let readback = try await readTiciStaticPostflight(
          profile: preflight.profile,
          context: "verify static post-reboot installation identity"
        )
        let postflight = makeStaticProductionPostflight(readback, preflight: preflight, identity: identity)
        try validateStaticInstalledIdentity(
          postflight,
          preflight: preflight,
          targetHead: targetHead,
          identity: identity,
          expectedActiveTileSetID: expectedActiveTileSetID
        )
        return postflight
      } catch {
        guard isRetryablePostRebootStartupError(error) else { throw error }
        lastError = error
      }
      guard clock.now < deadline else { break }
      try await Task.sleep(for: .seconds(max(0.001, pollInterval)))
    } while clock.now < deadline
    throw lastError
  }

  private func validateResumedStaticIdentity(
    _ result: TiciDeploymentPostflight,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?
  ) throws {
    guard !result.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    guard !result.runtimeEndMapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    guard result.branch == preflight.journal.branch else { throw ApplyPipelineError.invalidBranch(result.branch) }
    guard !result.dirty else {
      throw ApplyPipelineError.repositoryDirty("tici checkout is dirty during postflight")
    }
    guard result.head == targetHead else {
      throw ApplyPipelineError.commitMismatch(context: "tici postflight", expected: targetHead, actual: result.head)
    }
    guard result.physicsMatches else { throw ApplyPipelineError.postflightMismatch("physics Params differ from source") }
    guard result.qCurveSHA256 == identity.qCurveSHA256,
          result.qCurveEnabled == identity.qCurveEnabled,
          result.qCurvePointCount == identity.qCurvePointCount
    else { throw ApplyPipelineError.postflightMismatch("Q curve differs from the exact deployed source") }
    let release = preflight.release.artifact
    guard result.mapdReleaseVersion == release.releaseID, result.mapdVersion == release.releaseID else {
      throw ApplyPipelineError.postflightMismatch("MapdVersion/MapdReleaseVersion do not match \(release.releaseID)")
    }
    guard result.activeMapdSHA256 == release.sha256, result.cachedMapdSHA256 == release.sha256 else {
      throw ApplyPipelineError.postflightMismatch("active and persistent mapd digests are not exact")
    }
    guard result.managerRunning, result.mapdRunning, result.capabilityPresent,
          result.activeELFARM64, result.buildInfoMatches else {
      throw ApplyPipelineError.postflightMismatch(
        "manager/mapd is not running with the exact ARM64 build identity/capability"
      )
    }
    guard result.activeTileSetID == expectedActiveTileSetID else {
      throw ApplyPipelineError.postflightMismatch("active tile-set identity changed during resumed postflight")
    }
    try requireResumedDeploymentBootTransition(
      current: result.bootID,
      journal: preflight.journal
    )
  }

  private func validateResumedControllerEvidence(
    _ result: TiciDeploymentPostflight,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?
  ) throws {
    try validateResumedStaticIdentity(
      result,
      preflight: preflight,
      targetHead: targetHead,
      identity: identity,
      expectedActiveTileSetID: expectedActiveTileSetID
    )
    let stableOnroad = result.isOnroad && !result.isOffroad &&
      result.runtimeEndIsOnroad && !result.runtimeEndIsOffroad
    let stableOffroad = result.isOffroad && !result.isOnroad &&
      result.runtimeEndIsOffroad && !result.runtimeEndIsOnroad
    guard stableOnroad || stableOffroad else {
      throw ResumedPostflightWaitState.controllerEvidence(
        "road state changed across the snapshot bracket; waiting for a stable onroad observation"
      )
    }
    guard result.profileValidationStatus != "profile_pending", result.gpsStatus == "valid" else {
      throw ResumedPostflightWaitState.controllerEvidence(
        "whole-curve profile pending real GPS/profile: gps=\(result.gpsStatus) profile=\(result.profileValidationStatus)"
      )
    }
    guard stableOnroad else {
      throw ResumedPostflightWaitState.controllerEvidence(
        "No controller-ready evidence was captured. Start Verify while parked with ignition still on after a normal GPS/profile-producing drive."
      )
    }
    guard result.profileEstimatorVersion == preflight.release.artifact.estimatorVersion else {
      throw ApplyPipelineError.postflightMismatch("whole-curve estimator version is missing or mismatched")
    }
    guard result.profileSigmoidHash == identity.tileSigmoidHash else {
      throw ApplyPipelineError.postflightMismatch(
        "whole-curve profile sigmoid hash is \(result.profileSigmoidHash), expected \(identity.tileSigmoidHash)"
      )
    }
    guard result.profilePointCount >= 3,
          result.profileFresh,
          result.profileValuesFinite,
          result.profileRouteFingerprint.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil
    else {
      throw ApplyPipelineError.postflightMismatch(
        "whole-curve profile is missing, stale, malformed, or not near real GPS: \(result.profileValidationStatus)"
      )
    }
    guard result.liveMapDataUpdated,
          result.liveMapDataValid,
          result.liveMapDataLogMonoTimeNs > 0,
          result.roadGeometryValid
    else {
      throw ResumedPostflightWaitState.controllerEvidence(
        "liveMapDataSP is not newly updated, valid, and roadGeometryValid=true; waiting to prove this profile would be accepted when Map Lookahead is later enabled"
      )
    }
    guard result.liveMapDataSampleMonoTimeNs >= result.liveMapDataLogMonoTimeNs,
          result.liveMapDataSampleMonoTimeNs - result.liveMapDataLogMonoTimeNs <= 1_500_000_000
    else {
      throw ResumedPostflightWaitState.controllerEvidence(
        "liveMapDataSP is stale or from the future relative to the bounded controller probe; waiting for a fresh message"
      )
    }
  }

  private func validateResumedOffroadCompletion(
    _ result: TiciDeploymentPostflight,
    controllerEvidence: TiciDeploymentPostflight,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    expectedActiveTileSetID: String?
  ) throws {
    try validateResumedStaticIdentity(
      result,
      preflight: preflight,
      targetHead: targetHead,
      identity: identity,
      expectedActiveTileSetID: expectedActiveTileSetID
    )
    let stableOnroad = result.isOnroad && !result.isOffroad &&
      result.runtimeEndIsOnroad && !result.runtimeEndIsOffroad
    let stableOffroad = result.isOffroad && !result.isOnroad &&
      result.runtimeEndIsOffroad && !result.runtimeEndIsOnroad
    guard stableOnroad || stableOffroad else {
      throw ResumedPostflightWaitState.offroadTransition
    }
    guard stableOffroad else { throw ResumedPostflightWaitState.offroadTransition }
    try validatePostflight(
      result,
      requireNonemptyWholeCurveProfile: true,
      preflight: preflight,
      targetHead: targetHead,
      identity: identity,
      expectedActiveTileSetID: expectedActiveTileSetID
    )
    guard result.profileRouteFingerprint == controllerEvidence.profileRouteFingerprint,
          result.profilePointCount == controllerEvidence.profilePointCount,
          result.profileEventCount == controllerEvidence.profileEventCount
    else {
      throw ApplyPipelineError.postflightMismatch(
        "whole-curve profile identity changed between controller-ready and offroad observations"
      )
    }
  }

  func claimProductionMutation(
    expected: DeploymentRollbackJournal,
    at url: URL
  ) throws -> DeploymentRollbackJournal {
    let current = try DeploymentRollbackJournal.load(from: url)
    guard current.hasSameDeploymentIdentity(as: expected),
          current.effectiveResolution == .preflightReserved,
          current.completed,
          !current.rebootSent
    else {
      throw ApplyPipelineError.postflightMismatch(
        "production journal is not an untouched pre-mutation deployment"
      )
    }
    var claimed = current
    claimed.completed = true
    claimed.resolution = .mutationInProgress
    try durablyWriteLifecycleBarrier(claimed, at: url)
    return claimed
  }

  func handoffToOutdoorPostflight(
    expected: DeploymentRollbackJournal,
    at url: URL
  ) throws -> DeploymentRollbackJournal {
    let current = try DeploymentRollbackJournal.load(from: url)
    guard current.hasSameDeploymentIdentity(as: expected),
          current.effectiveResolution == .mutationInProgress,
          current.completed,
          current.rebootSent
    else {
      throw ApplyPipelineError.postflightMismatch(
        "production journal is not a rebooted mutation awaiting postflight handoff"
      )
    }
    var awaiting = current
    awaiting.completed = false
    awaiting.resolution = .awaitingOutdoorPostflight
    try durablyWriteLifecycleBarrier(awaiting, at: url)
    return awaiting
  }

  /// Own rollback under the same stable journal transaction lock used by
  /// resume completion. A stale original app instance must never roll back a
  /// deployment that another instance has already certified as complete.
  func rollbackProductionDeploymentIfJournalPending(
    preflight: RuntimeDeploymentPreflight,
    tilesActivated _: Bool
  ) async -> ProductionRollbackResolution {
    let productionOwnerLock: ProductionDeploymentOwnerLock
    do {
      productionOwnerLock = try DeploymentRollbackJournal.acquireProductionOwnerLock(
        directory: preflight.journalURL.deletingLastPathComponent()
      )
    } catch {
      return .rollbackFailed(
        "rollback mutation was not attempted because global production ownership could not be acquired: \(error.localizedDescription)"
      )
    }
    defer { productionOwnerLock.unlock() }
    return await rollbackProductionDeploymentIfJournalPending(
      context: ProductionRollbackContext(preflight: preflight),
      ownerLock: nil
    )
  }

  private func rollbackProductionDeploymentIfJournalPending(
    context: ProductionRollbackContext,
    ownerLock: DeploymentRollbackJournalCompletionLock? = nil
  ) async -> ProductionRollbackResolution {
    let claimLock: DeploymentRollbackJournalCompletionLock
    let acquiredHere = ownerLock == nil
    if let ownerLock {
      claimLock = ownerLock
    } else {
      do {
        claimLock = try DeploymentRollbackJournal.acquireCompletionLock(for: context.journalURL)
      } catch {
        return .rollbackFailed(
          "rollback mutation was not attempted because journal transaction ownership could not be acquired: \(error.localizedDescription)\n" +
            "journal retained at \(context.journalURL.path)"
        )
      }
    }
    // Rollback ownership is intentionally held through the complete device
    // mutation, verification, and terminal journal settlement. A process
    // crash releases flock and permits orphan recovery; a live owner excludes
    // every other app instance from concurrent rollback or completion.
    defer { if acquiredHere { claimLock.unlock() } }

    let loadedCurrent: DeploymentRollbackJournal
    do {
      loadedCurrent = try DeploymentRollbackJournal.load(from: context.journalURL)
    } catch {
      return .rollbackFailed(
        "rollback mutation was not attempted because the owned journal could not be reloaded: \(error.localizedDescription)\n" +
          "journal retained at \(context.journalURL.path)"
      )
    }
    guard loadedCurrent.hasSameDeploymentIdentity(as: context.journal) else {
      return .rollbackFailed(
        "rollback mutation was not attempted because the journal deployment identity changed\n" +
          "journal retained at \(context.journalURL.path)"
      )
    }
    var current = loadedCurrent
    if current.resolution != nil, !current.completed {
      switch current.effectiveResolution {
      case .completed:
        return .rollbackFailed(
          "rollback mutation was not attempted because a corrupt completed resolution lacks its proof sentinel"
        )
      case .preflightReserved, .mutationInProgress, .rollbackInProgress, .rolledBack, .rollbackFailed:
        // Never promote a torn rolledBack/rollbackFailed state directly. Move
        // it back to rollbackInProgress and replay full device restoration and
        // verification before any terminal rollback resolution is trusted.
        current.completed = true
        current.resolution = .rollbackInProgress
        do {
          try journalWriter(current, context.journalURL)
          guard try DeploymentRollbackJournal.load(from: context.journalURL) == current else {
            throw ApplyPipelineError.postflightMismatch(
              "normalized rollback recovery claim did not read back exactly"
            )
          }
        } catch {
          return .rollbackFailed(
            "rollback mutation was not attempted because journal recovery normalization failed: \(error.localizedDescription)"
          )
        }
      case .awaitingPostflight, .awaitingOutdoorPostflight:
        break
      }
    }
    do {
      try current.validateResolutionConsistency()
    } catch {
      return .rollbackFailed(
        "rollback mutation was not attempted because journal resolution is inconsistent: \(error.localizedDescription)"
      )
    }
    switch current.effectiveResolution {
    case .completed:
      return .alreadyCompleted(
        "rollback skipped: this exact deployment journal was already completed by another VTSC Tuner instance"
      )
    case .rolledBack:
      return .rolledBack("rollback already completed for this exact deployment journal")
    case .rollbackInProgress:
      // Holding the nonblocking resolution lock proves that no live owner is
      // still performing the rollback. Recover the orphaned claim by rerunning
      // the idempotent rollback path after a fresh parked-state gate.
      break
    case .mutationInProgress:
      // The original deployer durably claimed mutation ownership before its
      // first device write. If that owner died, acquiring this lock proves the
      // claim is orphaned and must be converted to rollback ownership.
      break
    case .rollbackFailed:
      // An explicit new invocation is the recovery action. Reacquiring the
      // resolution lock proves the previous owner is gone; after the fresh
      // parked gate below, the idempotent rollback may be retried.
      break
    case .preflightReserved:
      return .rollbackFailed(
        "rollback mutation was not attempted because a preflight reservation has no durable mutation identity"
      )
    case .awaitingPostflight, .awaitingOutdoorPostflight:
      break
    }

    do {
      _ = try await freshParkedSafetySnapshot(profile: context.profile)
    } catch {
      return .rollbackFailed(
        "rollback mutation was not attempted because fresh parked-state verification failed: \(error.localizedDescription)\n" +
          "journal retained at \(context.journalURL.path)"
      )
    }

    if current.effectiveResolution != .rollbackInProgress {
      var claimed = current
      // The legacy completed flag is deliberately true for every rollback
      // resolution. An older schema-1 app therefore rejects the journal as
      // non-pending even though it does not understand the new resolution key.
      claimed.completed = true
      claimed.resolution = .rollbackInProgress
      do {
        try journalWriter(claimed, context.journalURL)
        guard try DeploymentRollbackJournal.load(from: context.journalURL) == claimed else {
          throw ApplyPipelineError.postflightMismatch("rollback claim did not read back exactly")
        }
      } catch {
        return .rollbackFailed(
          "rollback mutation was not attempted because its durable journal claim failed: \(error.localizedDescription)"
        )
      }
    }

    let durableClaim: DeploymentRollbackJournal
    do {
      durableClaim = try DeploymentRollbackJournal.load(from: context.journalURL)
      guard durableClaim.hasSameDeploymentIdentity(as: current),
            durableClaim.effectiveResolution == .rollbackInProgress,
            durableClaim.completed
      else {
        throw ApplyPipelineError.postflightMismatch(
          "durable rollback claim changed before device restoration"
        )
      }
    } catch {
      return .rollbackFailed(
        "rollback mutation was not attempted because its durable lifecycle state could not be reloaded: \(error.localizedDescription)"
      )
    }
    let durableContext = ProductionRollbackContext(
      journal: durableClaim,
      journalURL: context.journalURL
    )
    let rollbackResult = await rollbackProductionDeployment(
      context: durableContext
    )
    do {
      var settled = try DeploymentRollbackJournal.load(from: context.journalURL)
      guard settled.hasSameDeploymentIdentity(as: durableClaim),
            settled.rebootSent == durableClaim.rebootSent,
            settled.effectiveResolution == .rollbackInProgress
      else {
        throw ApplyPipelineError.postflightMismatch(
          "rollback journal claim changed before terminal settlement"
        )
      }
      settled.resolution = rollbackResult.success ? .rolledBack : .rollbackFailed
      settled.completed = true
      try journalWriter(settled, context.journalURL)
      guard try DeploymentRollbackJournal.load(from: context.journalURL) == settled else {
        throw ApplyPipelineError.postflightMismatch("settled rollback journal did not read back exactly")
      }
    } catch {
      return .rollbackFailed(
        rollbackResult.detail + "\nrollback terminal journal settlement failed: \(error.localizedDescription)"
      )
    }
    return rollbackResult.success
      ? .rolledBack(rollbackResult.detail)
      : .rollbackFailed(rollbackResult.detail)
  }

  /// Persist the reboot lifecycle intent before the reboot command can run.
  /// A post-rename directory-sync failure is visible but not crash-durable, so
  /// retry the exact same transition until one write confirms the directory
  /// barrier. A pre-rename failure returns immediately and leaves the durable
  /// journal's rebootSent=false authority intact for rollback.
  func durablyRecordRebootIntent(
    _ journal: DeploymentRollbackJournal,
    at url: URL
  ) throws {
    guard journal.rebootSent,
          journal.deploymentPreRebootBootID.map(TiciBootIdentity.isValid) == true else {
      throw ApplyPipelineError.postflightMismatch(
        "deployment reboot intent requires a durable pre-reboot boot identity"
      )
    }
    try durablyWriteLifecycleBarrier(journal, at: url)
  }

  private func durablyRecordRollbackRebootIdentity(
    expected: DeploymentRollbackJournal,
    bootID: String,
    at url: URL
  ) throws -> DeploymentRollbackJournal {
    guard TiciBootIdentity.isValid(bootID) else {
      throw ApplyPipelineError.postflightMismatch("rollback pre-reboot boot identity is invalid")
    }
    var current = try DeploymentRollbackJournal.load(from: url)
    guard current.hasSameDeploymentIdentity(as: expected),
          current.effectiveResolution == .rollbackInProgress,
          current.completed else {
      throw ApplyPipelineError.postflightMismatch(
        "rollback journal changed before pre-reboot boot identity could be recorded"
      )
    }
    current.rollbackPreRebootBootID = bootID
    try durablyWriteLifecycleBarrier(current, at: url)
    return current
  }

  private func durablyWriteLifecycleBarrier(
    _ journal: DeploymentRollbackJournal,
    at url: URL
  ) throws {
    var lastIndeterminate: DeploymentRollbackJournalError?
    for _ in 0..<3 {
      do {
        try journalWriter(journal, url)
        guard try DeploymentRollbackJournal.load(from: url) == journal else {
          throw ApplyPipelineError.postflightMismatch("reboot intent did not read back exactly")
        }
        return
      } catch let durabilityError as DeploymentRollbackJournalError {
        guard case .committedButNotDurable = durabilityError else { throw durabilityError }
        guard try DeploymentRollbackJournal.load(from: url) == journal else { throw durabilityError }
        lastIndeterminate = durabilityError
      }
    }
    throw lastIndeterminate ?? ApplyPipelineError.postflightMismatch(
      "reboot intent durability was not established"
    )
  }

  func rollbackProductionDeployment(
    preflight: RuntimeDeploymentPreflight,
    tilesActivated _: Bool
  ) async -> (success: Bool, detail: String) {
    let productionOwnerLock: ProductionDeploymentOwnerLock
    do {
      productionOwnerLock = try DeploymentRollbackJournal.acquireProductionOwnerLock(
        directory: preflight.journalURL.deletingLastPathComponent()
      )
    } catch {
      return (false, "rollback mutation was not attempted because global production ownership could not be acquired: \(error.localizedDescription)")
    }
    defer { productionOwnerLock.unlock() }
    return await rollbackProductionDeployment(
      context: ProductionRollbackContext(preflight: preflight)
    )
  }

  private func rollbackProductionDeployment(
    context: ProductionRollbackContext
  ) async -> (success: Bool, detail: String) {
    var details: [String] = []
    var succeeded = true
    do {
      _ = try await freshParkedSafetySnapshot(profile: context.profile)
      details.append("fresh pre-rollback parked-state gate passed")
    } catch {
      return (
        false,
        "rollback mutation was not attempted because fresh parked-state verification failed: \(error.localizedDescription)\n" +
          "journal retained at \(context.journalURL.path)"
      )
    }
    if context.journal.targetTileSetID != nil {
      do {
        try await TiciTileSetDeploymentService(processRunner: processRunner).rollback(
          profile: context.profile,
          expectedActivatedTileSetID: context.journal.targetTileSetID,
          expectedRestoredTileSetID: context.journal.previousTileSetID
        )
        details.append("tile activation transaction recovered and prior set restored when needed")
      } catch {
        succeeded = false
        details.append("tile rollback failed: \(error.localizedDescription)")
      }
    }
    do {
      let result = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            context.profile,
            try TiciProductionRollbackCommandBuilder.command(journal: context.journal),
          ],
          timeout: 180
        ),
        context: "restore tici source, Params, and mapd"
      )
      let restoredHead = try TiciProductionRollbackCommandBuilder.restoredHead(from: result.standardOutput)
      guard restoredHead == context.journal.previousHead else {
        throw ApplyPipelineError.commitMismatch(
          context: "tici rollback",
          expected: context.journal.previousHead,
          actual: restoredHead
        )
      }
      details.append(result.combinedOutput)
    } catch {
      succeeded = false
      details.append("source/Params/mapd rollback failed: \(error.localizedDescription)")
    }
    if succeeded {
      do {
        let preRebootSnapshot = try await freshParkedSafetySnapshot(profile: context.profile)
        guard let preRebootBootID = preRebootSnapshot.bootID else {
          throw ApplyPipelineError.postflightMismatch("tici boot identity is missing before rollback reboot")
        }
        let rebootJournal = try durablyRecordRollbackRebootIdentity(
          expected: context.journal,
          bootID: preRebootBootID,
          at: context.journalURL
        )
        let verificationContext = ProductionRollbackContext(
          journal: rebootJournal,
          journalURL: context.journalURL
        )
        details.append("fresh pre-reboot parked-state gate passed")
        try await sendReboot(profile: context.profile)
        details.append("rollback reboot sent")
        try await waitForTici(
          profile: context.profile,
          timeout: 300,
          initialDelayNanoseconds: rebootInitialDelayNanoseconds,
          pollDelayNanoseconds: rebootPollDelayNanoseconds
        )
        let verification = try await waitForRollbackVerification(
          context: verificationContext,
          tilesWereTouched: context.journal.targetTileSetID != nil,
          timeout: 900
        )
        details.append("post-rollback source/Params/binary/cache/tiles/runtime verification passed")
        details.append(verification)
      } catch {
        succeeded = false
        details.append("rollback reboot or post-rollback verification failed: \(error.localizedDescription)")
      }
    } else {
      details.append("rollback reboot suppressed because rollback mutation did not complete")
    }
    details.append("journal=\(context.journalURL.path)")
    return (succeeded, details.filter { !$0.isEmpty }.joined(separator: "\n"))
  }

  func waitForRollbackVerification(
    context: ProductionRollbackContext,
    tilesWereTouched: Bool,
    timeout: TimeInterval,
    pollInterval: TimeInterval = 5
  ) async throws -> String {
    let deadline = Date().addingTimeInterval(timeout)
    var lastError: Error = ApplyPipelineError.postflightMismatch("rollback verification did not run")
    repeat {
      try Task.checkCancellation()
      do {
        let readback = try await readTiciRollbackStaticPostflight(
          profile: context.profile,
          context: "verify complete post-rollback identity"
        )
        try validateRollbackReadback(
          readback,
          journal: context.journal,
          tilesWereTouched: tilesWereTouched
        )
        return [
          "rollback_verified=true",
          "head=\(readback.deployment.snapshot.head)",
          "active_tile_set_id=\(readback.deployment.snapshot.activeTileSetID ?? "")",
        ].joined(separator: "\n")
      } catch {
        guard isRetryablePostRebootStartupError(error) else { throw error }
        lastError = error
      }
      if Date() < deadline { try await Task.sleep(for: .seconds(max(0.001, pollInterval))) }
    } while Date() < deadline
    throw lastError
  }

  private func validateRollbackReadback(
    _ readback: TiciRollbackStaticPostflightRead,
    journal: DeploymentRollbackJournal,
    tilesWereTouched: Bool
  ) throws {
    let snapshot = readback.deployment.snapshot
    guard snapshot.isOffroad, !snapshot.isOnroad, !snapshot.mapLookaheadEnabled,
          readback.runtimeEndIsOffroad, !readback.runtimeEndIsOnroad,
          !readback.runtimeEndMapLookaheadEnabled else {
      throw ApplyPipelineError.postflightMismatch("rollback parked/offroad/kill-switch state differs")
    }
    guard snapshot.branch == journal.branch else {
      throw ApplyPipelineError.invalidBranch(snapshot.branch)
    }
    guard snapshot.head == journal.previousHead else {
      throw ApplyPipelineError.commitMismatch(
        context: "tici rollback verification",
        expected: journal.previousHead,
        actual: snapshot.head
      )
    }
    guard !snapshot.dirty else { throw ApplyPipelineError.repositoryDirty("tici checkout is dirty after rollback") }
    for (key, expected) in journal.previousPhysicsParams {
      guard (snapshot.physicsParams[key] ?? nil) == expected else {
        throw ApplyPipelineError.postflightMismatch("rollback physics Param differs: \(key)")
      }
    }
    guard readback.deployment.qCurve.sha256 == journal.previousQCurveSHA256 else {
      throw ApplyPipelineError.postflightMismatch("rollback Q curve differs from the recorded source")
    }
    guard snapshot.mapdReleaseVersion == journal.previousMapdReleaseVersion,
          snapshot.mapdVersion == journal.previousMapdVersion else {
      throw ApplyPipelineError.postflightMismatch("rollback MapdVersion/MapdReleaseVersion differ")
    }
    guard snapshot.activeMapdSHA256 == journal.previousActiveMapdSHA256 else {
      throw ApplyPipelineError.postflightMismatch("rollback active mapd digest differs")
    }
    guard snapshot.cachedMapdPath == journal.previousCachedMapdPath else {
      throw ApplyPipelineError.postflightMismatch("rollback persistent mapd cache path differs")
    }
    if let expectedCacheSHA = journal.previousCachedMapdSHA256 {
      guard snapshot.cachedMapdSHA256 == expectedCacheSHA else {
        throw ApplyPipelineError.postflightMismatch("rollback persistent mapd cache digest differs")
      }
    }
    guard snapshot.activeTileSetID == journal.previousTileSetID else {
      throw ApplyPipelineError.postflightMismatch("rollback tile-set identity differs")
    }
    if tilesWereTouched, let targetTileSetID = journal.targetTileSetID,
       snapshot.activeTileSetID == targetTileSetID {
      throw ApplyPipelineError.postflightMismatch("rolled-back tile set is still active")
    }
    try requireBootTransition(
      current: snapshot.bootID,
      previous: journal.rollbackPreRebootBootID,
      missingPreviousDetail: "This rollback journal has no durable pre-reboot boot identity. Recovery cannot certify a reboot; retry the explicit Recover Interrupted Production Rollback action."
    )
    guard readback.managerRunning, readback.mapdRunning else {
      throw RuntimeStartupWaitState.processesNotReady(
        manager: readback.managerRunning,
        mapd: readback.mapdRunning
      )
    }
  }

  private func freshParkedSafetySnapshot(profile: String) async throws -> TiciDeploymentSnapshot {
    let snapshot = try await readTiciDeploymentSnapshot(
      profile: profile,
      context: "fresh parked-state verification",
      timeout: 20,
      connectTimeout: 5
    ).snapshot
    guard snapshot.isOffroad, !snapshot.isOnroad else { throw ApplyPipelineError.ticiNotOffroad }
    guard !snapshot.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    return snapshot
  }

  private func readTiciDeploymentSnapshot(
    profile: String,
    context: String,
    timeout: TimeInterval,
    connectTimeout: Int = 10
  ) async throws -> TiciDeploymentSnapshotRead {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: connectTimeout) + [
          profile,
          TiciSnapshotWireCommandBuilder.inspectionCommand(),
        ],
        timeout: timeout
      ),
      context: context
    )
    do {
      return try TiciDeploymentSnapshotDecoder.decodeRead(result.standardOutput)
    } catch {
      throw ApplyPipelineError.invalidDeploymentOutput(
        "\(result.standardOutput)\n\(error.localizedDescription)"
      )
    }
  }

  private func gitOutput(_ arguments: [String], repositoryRoot: URL) async throws -> String {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: arguments,
        currentDirectoryURL: repositoryRoot,
        timeout: 120
      ),
      context: "git \(arguments.joined(separator: " "))"
    )
    return result.standardOutput.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  private func gitCommit(repositoryRoot: URL, touched: [URL]) async throws -> String {
    let rootPath = repositoryRoot.standardizedFileURL.path + "/"
    for url in touched {
      let path = url.standardizedFileURL.path
      let relative = path.hasPrefix(rootPath) ? String(path.dropFirst(rootPath.count)) : path
      _ = try await checked(
        ProcessRequest(
          executableURL: Self.gitURL,
          arguments: ["add", "--", relative],
          currentDirectoryURL: repositoryRoot,
          timeout: 30
        ),
        context: "git add \(relative)"
      )
    }
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.gitURL,
        arguments: ["commit", "-m", "vtsc: tune sigmoid via vtsc_tuner"],
        currentDirectoryURL: repositoryRoot,
        timeout: 60
      ),
      context: "git commit"
    )
    return result.combinedOutput
  }

  private func checked(_ request: ProcessRequest, context: String) async throws -> ProcessResult {
    try Task.checkCancellation()
    let result = try await processRunner.run(request)
    guard result.succeeded else {
      throw ApplyPipelineError.commandFailed(context, result.terminationStatus, result.combinedOutput)
    }
    return result
  }

  func pickTiciProfile(preferred: String?) async throws -> String {
    if let preferred { try validateProfile(preferred) }
    lastADBTransportDetail = nil
    let ssid = await detectSSID()
    let defaults = ssid?.localizedCaseInsensitiveContains("comma") == true
      ? ["commaCar", "commaHome", "commaAdb"]
      : ["commaHome", "commaCar", "commaAdb"]
    var profiles = preferred.map { [$0] } ?? []
    for profile in defaults where !profiles.contains(profile) { profiles.append(profile) }
    for profile in profiles where await probeProfile(profile) { return profile }
    throw ApplyPipelineError.noReachableTici(lastADBTransportDetail)
  }

  private func validateProfile(_ profile: String) throws {
    let range = profile.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression)
    guard range != nil else { throw ApplyPipelineError.invalidProfile(profile) }
  }

  private func detectSSID() async -> String? {
    guard let ports = try? await processRunner.run(
      ProcessRequest(executableURL: Self.networkSetupURL, arguments: ["-listallhardwareports"], timeout: 5)
    ), ports.succeeded else { return nil }
    let lines = ports.standardOutput.components(separatedBy: .newlines)
    var wifiDevice: String?
    for index in lines.indices where lines[index].localizedCaseInsensitiveContains("Hardware Port: Wi-Fi") {
      guard lines.indices.contains(index + 1), let separator = lines[index + 1].firstIndex(of: ":") else { continue }
      wifiDevice = String(lines[index + 1][lines[index + 1].index(after: separator)...])
        .trimmingCharacters(in: .whitespaces)
      break
    }
    guard let wifiDevice,
          let result = try? await processRunner.run(
            ProcessRequest(
              executableURL: Self.networkSetupURL,
              arguments: ["-getairportnetwork", wifiDevice],
              timeout: 5
            )
          ), result.succeeded,
          let separator = result.standardOutput.firstIndex(of: ":")
    else { return nil }
    let ssid = String(result.standardOutput[result.standardOutput.index(after: separator)...])
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return ssid.isEmpty ? nil : ssid
  }

  func probeProfile(_ profile: String) async -> Bool {
    if await sshProbe(profile)?.succeeded == true { return true }
    guard profile == "commaAdb" else { return false }
    return await restoreADBForwardAndProbeSSH()
  }

  private func sshProbe(_ profile: String) async -> ProcessResult? {
    try? await processRunner.run(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 2, batchMode: true) + [profile, "true"],
        timeout: 5
      )
    )
  }

  private func restoreADBForwardAndProbeSSH() async -> Bool {
    guard let adbURL else {
      lastADBTransportDetail = "adb was not found in the Mac app's standard executable locations."
      return false
    }

    let listing: ProcessResult
    do {
      listing = try await processRunner.run(
        ProcessRequest(executableURL: adbURL, arguments: ["devices", "-l"], timeout: 10)
      )
    } catch {
      lastADBTransportDetail = "Could not run \(adbURL.path): \(error.localizedDescription)"
      return false
    }
    guard listing.succeeded else {
      lastADBTransportDetail = "adb devices failed: \(listing.combinedOutput)"
      return false
    }

    let serial: String
    switch ADBDeviceSelection.select(from: listing.standardOutput) {
    case let .selected(selectedSerial):
      serial = selectedSerial
    case let .unavailable(detail):
      lastADBTransportDetail = detail
      return false
    case let .ambiguous(serials):
      lastADBTransportDetail = "Multiple ADB devices are listed (\(serials.joined(separator: ", "))). Disconnect the extras and retry."
      return false
    }

    let forward: ProcessResult
    do {
      forward = try await processRunner.run(
        ProcessRequest(
          executableURL: adbURL,
          arguments: ["-s", serial, "forward", "tcp:2222", "tcp:22"],
          timeout: 10
        )
      )
    } catch {
      lastADBTransportDetail = "Device \(serial) is authorized, but the SSH port forward could not be started: \(error.localizedDescription)"
      return false
    }
    guard forward.succeeded else {
      lastADBTransportDetail = "Device \(serial) is authorized, but adb could not forward tcp:2222 to tcp:22: \(forward.combinedOutput)"
      return false
    }

    let retry = await sshProbe("commaAdb")
    guard retry?.succeeded == true else {
      let sshOutput = retry?.combinedOutput ?? ""
      let sshDetail = sshOutput.isEmpty ? "" : " \(sshOutput)"
      lastADBTransportDetail = "Device \(serial) is authorized and the port forward was created, but an SSH channel through commaAdb still could not be opened.\(sshDetail)"
      return false
    }
    lastADBTransportDetail = "Connected to \(serial) through an app-created tcp:2222 to tcp:22 forward."
    return true
  }

  private func sendReboot(profile: String) async throws {
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 5) + [
          profile,
          TiciRebootCommandBuilder.command(),
        ],
        timeout: 10
      ),
      context: "ssh reboot"
    )
  }

  func waitForTici(
    profile: String,
    timeout: TimeInterval,
    initialDelayNanoseconds: UInt64 = 15_000_000_000,
    pollDelayNanoseconds: UInt64 = 5_000_000_000
  ) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    try await Task.sleep(nanoseconds: initialDelayNanoseconds)
    while Date() < deadline {
      try Task.checkCancellation()
      if await probeProfile(profile) { return }
      try await Task.sleep(nanoseconds: pollDelayNanoseconds)
    }
    let adbDetail = lastADBTransportDetail.map { "\nUSB / ADB: \($0)" } ?? ""
    throw ApplyPipelineError.commandFailed(
      "wait for tici",
      -1,
      "The tici did not return within \(Int(timeout)) seconds.\(adbDetail)"
    )
  }

  private func discoverRegions(profile: String) async throws -> [RegionBox] {
    let command = #"for d in /data/media/0/osm/offline/*/; do lat=$(basename "$d"); for sd in "$d"*/; do [ -d "$sd" ] && echo "$lat $(basename "$sd")"; done; done 2>/dev/null"#
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [profile, command],
        timeout: 30
      ),
      context: "discover cached map regions"
    )
    var regions: [RegionBox] = []
    for line in result.standardOutput.components(separatedBy: .newlines) where !line.isEmpty {
      let fields = line.split(whereSeparator: \Character.isWhitespace)
      guard fields.count == 2, let latitude = Int(fields[0]), let longitude = Int(fields[1]) else {
        throw ApplyPipelineError.unexpectedRegionListing(line)
      }
      let region = RegionBox(minLatitude: latitude, minLongitude: longitude)
      try region.validate()
      regions.append(region)
    }
    return normalizedRegions(regions)
  }

  private func normalizedRegions(_ regions: [RegionBox]) -> [RegionBox] {
    Array(Set(regions)).sorted {
      ($0.minLatitude, $0.minLongitude) < ($1.minLatitude, $1.minLongitude)
    }
  }

  private func formatRegions(_ regions: [RegionBox]) -> String {
    regions.map { "(\($0.minLatitude)°, \($0.minLongitude)°)" }.joined(separator: "  ")
  }

  private func ticiFreeMegabytes(profile: String) async throws -> UInt64 {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [profile, "df -BM /data/media/0 | tail -1"],
        timeout: 20
      ),
      context: "read tici disk usage"
    )
    let fields = result.standardOutput.split(whereSeparator: \Character.isWhitespace)
    guard fields.count >= 4, let value = UInt64(fields[3].trimmingCharacters(in: CharacterSet(charactersIn: "M"))) else {
      throw ApplyPipelineError.unexpectedDiskOutput(result.standardOutput)
    }
    return value
  }

  private func prepareMapdStaging(pbfURL: URL) throws -> URL {
    let stagingURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("vtsc-tuner-mapd-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: stagingURL, withIntermediateDirectories: true)
    do {
      try FileManager.default.createSymbolicLink(
        at: stagingURL.appendingPathComponent("map.osm.pbf"),
        withDestinationURL: pbfURL
      )
    } catch {
      try? FileManager.default.removeItem(at: stagingURL)
      throw error
    }
    return stagingURL
  }

  private func generateRegion(
    config: MapdConfig,
    stagingURL: URL,
    region: RegionBox,
    parameters: SigmoidParameters
  ) async throws -> String {
    let regionURL = generatedRegionURL(stagingURL: stagingURL, region: region)
    try? FileManager.default.removeItem(at: regionURL)
    let arguments = MapdCommandBuilder.generationArguments(parameters: parameters, region: region)
    let result = try await checked(
      ProcessRequest(
        executableURL: config.effectiveBinaryURL,
        arguments: arguments,
        currentDirectoryURL: stagingURL,
        timeout: 600
      ),
      context: "mapd --generate"
    )
    guard FileManager.default.fileExists(atPath: regionURL.path) else {
      throw ApplyPipelineError.missingGeneratedTiles(regionURL)
    }
    let metrics = generatedTileMetrics(at: regionURL)
    guard metrics.fileCount > 0, metrics.maximumBytes > 64 else {
      throw ApplyPipelineError.emptyGeneratedTiles(regionURL)
    }
    return [
      "\(metrics.fileCount) tile file(s), \(metrics.totalBytes) bytes under \(regionURL.path)",
      result.combinedOutput,
    ].filter { !$0.isEmpty }.joined(separator: "\n")
  }

  private func generatedRegionURL(stagingURL: URL, region: RegionBox) -> URL {
    stagingURL
      .appendingPathComponent("offline", isDirectory: true)
      .appendingPathComponent(String(region.minLatitude), isDirectory: true)
      .appendingPathComponent(String(region.minLongitude), isDirectory: true)
  }

  private func generatedTileMetrics(at regionURL: URL) -> (fileCount: Int, totalBytes: UInt64, maximumBytes: UInt64) {
    guard let enumerator = FileManager.default.enumerator(
      at: regionURL,
      includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
    ) else { return (0, 0, 0) }
    var fileCount = 0
    var totalBytes: UInt64 = 0
    var maximumBytes: UInt64 = 0
    for case let url as URL in enumerator {
      guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
            values.isRegularFile == true
      else { continue }
      let bytes = UInt64(values.fileSize ?? 0)
      fileCount += 1
      totalBytes += bytes
      maximumBytes = max(maximumBytes, bytes)
    }
    return (fileCount, totalBytes, maximumBytes)
  }

  private func sshOptions(connectTimeout: Int, batchMode: Bool = true) -> [String] {
    var options = [
      "-o", "ConnectTimeout=\(connectTimeout)",
      "-o", "StrictHostKeyChecking=accept-new",
    ]
    if batchMode { options += ["-o", "BatchMode=yes"] }
    return options
  }

  private func emit(
    _ status: ApplyStepStatus,
    id: UInt32,
    text: String,
    detail: String = "",
    progress: ApplyProgressHandler
  ) async {
    await progress(.step(ApplyStepEvent(id: id, status: status, text: text, detail: detail)))
  }

  private func emitFailure(
    id: UInt32,
    text: String,
    error: Error,
    progress: ApplyProgressHandler
  ) async {
    await emit(.failed, id: id, text: text, detail: error.localizedDescription, progress: progress)
  }

  private func finish(_ success: Bool, progress: ApplyProgressHandler) async -> Bool {
    await progress(.finished(success: success))
    return success
  }
}
