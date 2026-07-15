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
    self.tileDecoderURL = tileDecoderURL
    self.verificationRequests = verificationRequests
    self.requireNonemptyWholeCurveProfile = requireNonemptyWholeCurveProfile
  }
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
  static func synchronizeAndVerifyCommand(parameters: SigmoidParameters) -> String {
    let values = Dictionary(
      uniqueKeysWithValues: VTSCPhysicsAuthority.entries(for: parameters).map {
        ($0.moduleName, $0.formattedValue)
      }
    )
    let amplitude = values["PHYSICS_A"]!
    let steepness = values["PHYSICS_B"]!
    let center = values["PHYSICS_C"]!
    let baseline = values["PHYSICS_D"]!
    let minLat = values["PHYSICS_MIN_LAT_ACCEL"]!
    let maxLat = values["PHYSICS_MAX_LAT_ACCEL"]!
    let invocation = [
      "PYTHONPATH=/data/openpilot \(TiciDeploymentCommandBuilder.ticiPython) tools/vtsc/apply_physics_params.py",
      "--amplitude \(amplitude)",
      "--steepness \(steepness)",
      "--center \(center)",
      "--baseline \(baseline)",
      "--min-lat \(minLat)",
      "--max-lat \(maxLat)",
    ].joined(separator: " ")
    return "cd /data/openpilot && \(invocation)"
  }
}

public actor ApplyPipeline {
  public static let gitURL = URL(fileURLWithPath: "/usr/bin/git")
  public static let sshURL = URL(fileURLWithPath: "/usr/bin/ssh")
  public static let rsyncURL = URL(fileURLWithPath: "/usr/bin/rsync")
  public static let networkSetupURL = URL(fileURLWithPath: "/usr/sbin/networksetup")

  private let processRunner: any ProcessRunning
  private let adbURL: URL?
  private var lastADBTransportDetail: String?

  public init() {
    processRunner = SystemProcessRunner()
    adbURL = ADBExecutableLocator.resolve()
  }

  public init(processRunner: any ProcessRunning) {
    self.processRunner = processRunner
    adbURL = nil
  }

  public init(processRunner: any ProcessRunning, adbURL: URL?) {
    self.processRunner = processRunner
    self.adbURL = adbURL
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
          text: "Production deployment preflight passed without mutation",
          detail: [
            "branch=\(runtimePreflight!.git.branch)",
            "head=\(runtimePreflight!.git.localHead)",
            "tici=\(runtimePreflight!.profile) offroad",
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
    do { try deployment.journal.write(to: deployment.journalURL) }
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

  private func productionPreflight(_ request: ApplyRequest) async throws -> RuntimeDeploymentPreflight {
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
    let inspection = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [profile, TiciDeploymentCommandBuilder.preflightInspectionCommand()],
        timeout: 60
      ),
      context: "read-only tici deployment preflight"
    )
    let snapshot = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
      TiciDeploymentSnapshot.self,
      output: inspection.standardOutput
    )
    try validateTiciPreflight(snapshot, git: git)

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

    let deploymentID = UUID()
    let rollbackPath = "/data/media/0/osm/binaries/mapd-rollback-\(deploymentID.uuidString.lowercased())"
    let journal = DeploymentRollbackJournal(
      deploymentID: deploymentID,
      profile: profile,
      branch: git.branch,
      previousHead: snapshot.head,
      previousPhysicsParams: snapshot.physicsParams,
      previousQCurveSHA256: snapshot.qCurveSHA256,
      previousMapdReleaseVersion: snapshot.mapdReleaseVersion,
      previousMapdVersion: snapshot.mapdVersion,
      previousActiveMapdSHA256: snapshot.activeMapdSHA256,
      previousCachedMapdPath: snapshot.cachedMapdPath,
      previousCachedMapdSHA256: snapshot.cachedMapdSHA256,
      mapdRollbackPath: rollbackPath,
      previousTileSetID: snapshot.activeTileSetID,
      targetTileSetID: tileSet?.manifest.tileSetID
    )
    let journalURL = try journal.write()
    return RuntimeDeploymentPreflight(
      git: git,
      profile: profile,
      release: release,
      tileSet: tileSet,
      journal: journal,
      journalURL: journalURL
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

  private func validateTiciPreflight(
    _ snapshot: TiciDeploymentSnapshot,
    git: GitDeploymentPreflight
  ) throws {
    guard snapshot.isOffroad, !snapshot.isOnroad else { throw ApplyPipelineError.ticiNotOffroad }
    guard !snapshot.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    guard snapshot.branch == git.branch else { throw ApplyPipelineError.invalidBranch(snapshot.branch) }
    guard !snapshot.dirty else { throw ApplyPipelineError.repositoryDirty("tici checkout is dirty") }
    guard snapshot.head == git.localHead else {
      throw ApplyPipelineError.commitMismatch(context: "dev/origin/tici preflight", expected: git.localHead, actual: snapshot.head)
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
    var remoteMutationStarted = false
    var tilesMayHaveActivated = false
    let identity = TuneDeploymentIdentity(tune: request.tune)
    do {
      await emit(.running, id: 5, text: "Reconfirming offroad/kill-switch safety immediately before mutation…", progress: progress)
      let recheck = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            deployment.profile, TiciDeploymentCommandBuilder.preflightInspectionCommand(),
          ],
          timeout: 60
        ),
        context: "final read-only tici safety check"
      )
      let snapshot = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
        TiciDeploymentSnapshot.self,
        output: recheck.standardOutput
      )
      try validateTiciPreflight(snapshot, git: deployment.git)
      await emit(.succeeded, id: 5, text: "Tici is still parked/offroad with Map Lookahead disabled", progress: progress)

      await emit(.running, id: 6, text: "Fast-forwarding the tici to the exact pushed commit…", progress: progress)
      // The SSH result is not proof that the remote command did not run. Mark
      // the transaction dirty before issuing the first mutating command so a
      // transport failure cannot suppress rollback.
      remoteMutationStarted = true
      let fastForward = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            deployment.profile,
            TiciDeploymentCommandBuilder.exactFastForwardCommand(branch: deployment.git.branch, head: targetHead),
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
            TiciPhysicsCommandBuilder.synchronizeAndVerifyCommand(parameters: request.tune.params),
          ],
          timeout: 60
        ),
        context: "synchronize exact VTSC physics Params"
      )
      let qCurve = try await checked(
        ProcessRequest(
          executableURL: Self.sshURL,
          arguments: sshOptions(connectTimeout: 10) + [
            deployment.profile,
            TiciDeploymentCommandBuilder.qCurveVerificationCommand(identity: identity),
          ],
          timeout: 60
        ),
        context: "verify exact VTSC Q curve"
      )
      await emit(
        .succeeded,
        id: 8,
        text: "All six physics Params and Q data match the source-rounded tune",
        detail: [physics.combinedOutput, qCurve.combinedOutput].filter { !$0.isEmpty }.joined(separator: "\n"),
        progress: progress
      )

      if let tileSet = deployment.tileSet {
        await emit(.running, id: 9, text: "Staging, fully verifying, and atomically activating the requested tile set…", progress: progress)
        let service = TiciTileSetDeploymentService(processRunner: processRunner)
        let staged = try await service.stageAndVerify(artifact: tileSet, profile: deployment.profile)
        // Activation can complete remotely even if SSH disconnects before the
        // result arrives. Recovery inspects the durable on-device transaction.
        tilesMayHaveActivated = true
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
      _ = try await freshParkedSafetySnapshot(profile: deployment.profile)
      deployment.journal.rebootSent = true
      try deployment.journal.write(to: deployment.journalURL)
      try await sendReboot(profile: deployment.profile)
      await emit(.succeeded, id: 10, text: "Single deployment reboot sent", progress: progress)

      await emit(.running, id: 11, text: "Waiting for the tici and verifying the complete postflight identity…", progress: progress)
      try await waitForTici(profile: deployment.profile, timeout: 300)
      let postflight = try await waitForProductionPostflight(
        request: request,
        preflight: deployment,
        targetHead: targetHead,
        identity: identity,
        timeout: 900
      )
      deployment.journal.completed = true
      try deployment.journal.write(to: deployment.journalURL)
      await emit(
        .succeeded,
        id: 11,
        text: "Postflight passed: exact commit/release/tune/profile identity is running",
        detail: "journal=\(deployment.journalURL.path)\nprofile_points=\(postflight.profilePointCount) profile_events=\(postflight.profileEventCount)",
        progress: progress
      )
      return await finish(true, progress: progress)
    } catch {
      await emitFailure(id: 12, text: "Production deployment failed; starting coherent rollback", error: error, progress: progress)
      if remoteMutationStarted {
        let rollbackDetail = await rollbackProductionDeployment(
          preflight: deployment,
          tilesActivated: tilesMayHaveActivated
        )
        await emit(
          rollbackDetail.success ? .succeeded : .failed,
          id: 13,
          text: rollbackDetail.success ? "Tici rollback completed" : "Tici rollback needs manual attention",
          detail: rollbackDetail.detail,
          progress: progress
        )
      }
      return await finish(false, progress: progress)
    }
  }

  private func stageAndInstallMapdRelease(preflight: RuntimeDeploymentPreflight) async throws -> String {
    let stagedPath = "/data/media/0/osm/binaries/.mapd-release-\(preflight.journal.deploymentID.uuidString.lowercased()).partial"
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [preflight.profile, "mkdir -p /data/media/0/osm/binaries && rm -f \(stagedPath)"],
        timeout: 30
      ),
      context: "prepare mapd release staging"
    )
    let transfer = try await checked(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-a", "--partial",
          "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          preflight.release.artifact.binaryURL.path,
          "\(preflight.profile):\(stagedPath)",
        ],
        timeout: 600
      ),
      context: "stage exact mapd release"
    )
    let install = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 10) + [
          preflight.profile,
          TiciDeploymentCommandBuilder.installReleaseCommand(
            release: preflight.release,
            stagedPath: stagedPath,
            rollbackPath: preflight.journal.mapdRollbackPath
          ),
        ],
        timeout: 180
      ),
      context: "install and verify exact mapd release"
    )
    guard install.standardOutput.contains(preflight.release.artifact.releaseID),
          install.standardOutput.contains(preflight.release.artifact.sha256)
    else { throw ApplyPipelineError.invalidDeploymentOutput(install.standardOutput) }
    return [transfer.combinedOutput, install.combinedOutput].filter { !$0.isEmpty }.joined(separator: "\n")
  }

  private func waitForProductionPostflight(
    request: ApplyRequest,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity,
    timeout: TimeInterval
  ) async throws -> TiciDeploymentPostflight {
    let deadline = Date().addingTimeInterval(timeout)
    var lastError: Error = ApplyPipelineError.postflightMismatch("postflight did not run")
    repeat {
      try Task.checkCancellation()
      do {
        let result = try await checked(
          ProcessRequest(
            executableURL: Self.sshURL,
            arguments: sshOptions(connectTimeout: 10) + [
              preflight.profile,
              TiciDeploymentCommandBuilder.postflightInspectionCommand(
                head: targetHead,
                release: preflight.release,
                identity: identity,
                expectedTileSetID: preflight.tileSet?.manifest.tileSetID
              ),
            ],
            timeout: 30
          ),
          context: "complete tici deployment postflight"
        )
        let postflight = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
          TiciDeploymentPostflight.self,
          output: result.standardOutput
        )
        try validatePostflight(
          postflight,
          request: request,
          preflight: preflight,
          targetHead: targetHead,
          identity: identity
        )
        return postflight
      } catch {
        lastError = error
      }
      if Date() < deadline { try await Task.sleep(for: .seconds(2)) }
    } while Date() < deadline
    throw lastError
  }

  private func validatePostflight(
    _ result: TiciDeploymentPostflight,
    request: ApplyRequest,
    preflight: RuntimeDeploymentPreflight,
    targetHead: String,
    identity: TuneDeploymentIdentity
  ) throws {
    guard result.isOffroad, !result.isOnroad else { throw ApplyPipelineError.ticiNotOffroad }
    guard !result.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
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
    guard result.mapdRunning, result.capabilityPresent, result.activeELFARM64, result.buildInfoMatches else {
      throw ApplyPipelineError.postflightMismatch("native mapd is not running with the exact ARM64 build identity/capability")
    }
    guard result.profileEstimatorVersion == release.estimatorVersion else {
      throw ApplyPipelineError.postflightMismatch("whole-curve estimator version is missing or mismatched")
    }
    if request.requireNonemptyWholeCurveProfile {
      guard result.gpsStatus == "valid" else {
        throw ApplyPipelineError.postflightMismatch("whole-curve profile pending real GPS: \(result.gpsStatus)")
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
    }
    if let expectedTileSetID = preflight.tileSet?.manifest.tileSetID {
      guard result.activeTileSetID == expectedTileSetID else {
        throw ApplyPipelineError.postflightMismatch("active tile-set identity is not \(expectedTileSetID)")
      }
    } else if result.activeTileSetID != preflight.journal.previousTileSetID {
      throw ApplyPipelineError.postflightMismatch("tile identity changed even though no replacement was requested")
    }
  }

  func rollbackProductionDeployment(
    preflight: RuntimeDeploymentPreflight,
    tilesActivated: Bool
  ) async -> (success: Bool, detail: String) {
    var details: [String] = []
    var succeeded = true
    do {
      _ = try await freshParkedSafetySnapshot(profile: preflight.profile)
      details.append("fresh pre-rollback parked-state gate passed")
    } catch {
      return (
        false,
        "rollback mutation was not attempted because fresh parked-state verification failed: \(error.localizedDescription)\n" +
          "journal retained at \(preflight.journalURL.path)"
      )
    }
    if tilesActivated {
      do {
        try await TiciTileSetDeploymentService(processRunner: processRunner).rollback(
          profile: preflight.profile,
          expectedActivatedTileSetID: preflight.tileSet?.manifest.tileSetID
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
            preflight.profile,
            TiciDeploymentCommandBuilder.rollbackCommand(
              journal: preflight.journal,
              rollbackMapdPath: preflight.journal.mapdRollbackPath
            ),
          ],
          timeout: 180
        ),
        context: "restore tici source, Params, and mapd"
      )
      details.append(result.combinedOutput)
    } catch {
      succeeded = false
      details.append("source/Params/mapd rollback failed: \(error.localizedDescription)")
    }
    if succeeded {
      do {
        _ = try await freshParkedSafetySnapshot(profile: preflight.profile)
        details.append("fresh pre-reboot parked-state gate passed")
        try await sendReboot(profile: preflight.profile)
        details.append("rollback reboot sent")
        try await waitForTici(profile: preflight.profile, timeout: 300)
        let verification = try await waitForRollbackVerification(
          preflight: preflight,
          tilesWereTouched: tilesActivated,
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
    details.append("journal=\(preflight.journalURL.path)")
    return (succeeded, details.filter { !$0.isEmpty }.joined(separator: "\n"))
  }

  private func waitForRollbackVerification(
    preflight: RuntimeDeploymentPreflight,
    tilesWereTouched: Bool,
    timeout: TimeInterval
  ) async throws -> String {
    let deadline = Date().addingTimeInterval(timeout)
    var lastError: Error = ApplyPipelineError.postflightMismatch("rollback verification did not run")
    repeat {
      try Task.checkCancellation()
      do {
        let result = try await checked(
          ProcessRequest(
            executableURL: Self.sshURL,
            arguments: sshOptions(connectTimeout: 10) + [
              preflight.profile,
              TiciDeploymentCommandBuilder.rollbackVerificationCommand(
                journal: preflight.journal,
                tilesWereTouched: tilesWereTouched
              ),
            ],
            timeout: 30
          ),
          context: "verify complete post-rollback identity"
        )
        guard result.standardOutput.contains("\"rollback_verified\": true") else {
          throw ApplyPipelineError.invalidDeploymentOutput(result.standardOutput)
        }
        return result.combinedOutput
      } catch {
        lastError = error
      }
      if Date() < deadline { try await Task.sleep(for: .seconds(5)) }
    } while Date() < deadline
    throw lastError
  }

  private func freshParkedSafetySnapshot(profile: String) async throws -> TiciDeploymentSnapshot {
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: sshOptions(connectTimeout: 5) + [
          profile,
          TiciDeploymentCommandBuilder.preflightInspectionCommand(),
        ],
        timeout: 20
      ),
      context: "fresh parked-state verification"
    )
    let snapshot = try TiciDeploymentCommandBuilder.decodeLastJSONLine(
      TiciDeploymentSnapshot.self,
      output: result.standardOutput
    )
    guard snapshot.isOffroad, !snapshot.isOnroad else { throw ApplyPipelineError.ticiNotOffroad }
    guard !snapshot.mapLookaheadEnabled else { throw ApplyPipelineError.mapLookaheadMustRemainDisabled }
    return snapshot
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
          "nohup sudo reboot >/dev/null 2>&1 </dev/null &",
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
