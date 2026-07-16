import Foundation

struct GitDeploymentPreflight: Equatable, Sendable {
  var branch: String
  var localHead: String
  var originHead: String
  var upstream: String
}

struct ResumePostflightGitIdentity: Equatable, Sendable {
  var deployedTargetHead: String
  var toolingHead: String
  var hostOnlyPaths: [String]
}

struct TiciDeploymentSnapshot: Codable, Equatable, Sendable {
  var isOffroad: Bool
  var isOnroad: Bool
  var mapLookaheadEnabled: Bool
  var branch: String
  var head: String
  var dirty: Bool
  var physicsParams: [String: String?]
  var qCurveSHA256: String
  var mapdReleaseVersion: String?
  var mapdVersion: String?
  var activeMapdSHA256: String
  var cachedMapdPath: String
  var cachedMapdSHA256: String?
  var activeTileSetID: String?

  enum CodingKeys: String, CodingKey {
    case isOffroad = "is_offroad"
    case isOnroad = "is_onroad"
    case mapLookaheadEnabled = "map_lookahead_enabled"
    case branch, head, dirty
    case physicsParams = "physics_params"
    case qCurveSHA256 = "q_curve_sha256"
    case mapdReleaseVersion = "mapd_release_version"
    case mapdVersion = "mapd_version"
    case activeMapdSHA256 = "active_mapd_sha256"
    case cachedMapdPath = "cached_mapd_path"
    case cachedMapdSHA256 = "cached_mapd_sha256"
    case activeTileSetID = "active_tile_set_id"
  }
}

struct TiciDeploymentPostflight: Codable, Equatable, Sendable {
  var isOffroad: Bool
  var isOnroad: Bool
  var runtimeEndIsOffroad: Bool
  var runtimeEndIsOnroad: Bool
  var runtimeEndMapLookaheadEnabled: Bool
  var mapLookaheadEnabled: Bool
  var branch: String
  var head: String
  var dirty: Bool
  var physicsMatches: Bool
  var qCurveSHA256: String
  var qCurveEnabled: Bool
  var qCurvePointCount: Int
  var mapdReleaseVersion: String
  var mapdVersion: String
  var activeMapdSHA256: String
  var cachedMapdSHA256: String
  var managerRunning: Bool
  var mapdRunning: Bool
  var capabilityPresent: Bool
  var activeELFARM64: Bool
  var buildInfoMatches: Bool
  var profileEstimatorVersion: String
  var profileRouteFingerprint: String
  var profileSigmoidHash: String
  var profileFresh: Bool
  var profileValuesFinite: Bool
  var gpsStatus: String
  var profileValidationStatus: String
  var profilePointCount: Int
  var profileEventCount: Int
  var liveMapDataUpdated: Bool
  var liveMapDataValid: Bool
  var liveMapDataLogMonoTimeNs: UInt64
  var liveMapDataSampleMonoTimeNs: UInt64
  var roadGeometryValid: Bool
  var activeTileSetID: String?

  enum CodingKeys: String, CodingKey {
    case isOffroad = "is_offroad"
    case isOnroad = "is_onroad"
    case runtimeEndIsOffroad = "runtime_end_is_offroad"
    case runtimeEndIsOnroad = "runtime_end_is_onroad"
    case runtimeEndMapLookaheadEnabled = "runtime_end_map_lookahead_enabled"
    case mapLookaheadEnabled = "map_lookahead_enabled"
    case branch, head, dirty
    case physicsMatches = "physics_matches"
    case qCurveSHA256 = "q_curve_sha256"
    case qCurveEnabled = "q_curve_enabled"
    case qCurvePointCount = "q_curve_point_count"
    case mapdReleaseVersion = "mapd_release_version"
    case mapdVersion = "mapd_version"
    case activeMapdSHA256 = "active_mapd_sha256"
    case cachedMapdSHA256 = "cached_mapd_sha256"
    case managerRunning = "manager_running"
    case mapdRunning = "mapd_running"
    case capabilityPresent = "capability_present"
    case activeELFARM64 = "active_elf_arm64"
    case buildInfoMatches = "build_info_matches"
    case profileEstimatorVersion = "profile_estimator_version"
    case profileRouteFingerprint = "profile_route_fingerprint"
    case profileSigmoidHash = "profile_sigmoid_hash"
    case profileFresh = "profile_fresh"
    case profileValuesFinite = "profile_values_finite"
    case gpsStatus = "gps_status"
    case profileValidationStatus = "profile_validation_status"
    case profilePointCount = "profile_point_count"
    case profileEventCount = "profile_event_count"
    case liveMapDataUpdated = "live_map_data_updated"
    case liveMapDataValid = "live_map_data_valid"
    case liveMapDataLogMonoTimeNs = "live_map_data_log_mono_time_ns"
    case liveMapDataSampleMonoTimeNs = "live_map_data_sample_mono_time_ns"
    case roadGeometryValid = "road_geometry_valid"
    case activeTileSetID = "active_tile_set_id"
  }
}

struct RuntimeDeploymentPreflight: Sendable {
  var git: GitDeploymentPreflight
  var profile: String
  var mapdRecoveryOutcome: TiciMapdReleaseRecoveryOutcome
  var release: ValidatedMapdReleaseArtifact
  var tileSet: CanonicalTileSetArtifact?
  var journal: DeploymentRollbackJournal
  var journalURL: URL
  /// Held only for a live production transaction. Resume reconstruction and
  /// test fixtures leave this nil and acquire the global owner explicitly at
  /// the terminal action boundary.
  var productionOwnerLock: ProductionDeploymentOwnerLock? = nil
}

struct ProductionRollbackContext: Sendable {
  var profile: String
  var journal: DeploymentRollbackJournal
  var journalURL: URL

  init(preflight: RuntimeDeploymentPreflight) {
    profile = preflight.profile
    journal = preflight.journal
    journalURL = preflight.journalURL
  }

  init(journal: DeploymentRollbackJournal, journalURL: URL) {
    profile = journal.profile
    self.journal = journal
    self.journalURL = journalURL
  }
}

enum ProductionVerificationSuite {
  static func requests(repositoryRoot: URL) -> [ProcessRequest] {
    let app = repositoryRoot.appendingPathComponent("tools/vtsc_tuner_mac", isDirectory: true)
    let mapd = repositoryRoot.appendingPathComponent("mapd_repo/openpilot-mapd", isDirectory: true)
    let python = repositoryRoot.appendingPathComponent(".venv/bin/python", isDirectory: false)
    let pinnedGo = app.appendingPathComponent(".build-tools/go-1.26.5-darwin-arm64/bin/go", isDirectory: false)
    let goEnvironment = [
      "GOTOOLCHAIN": "local",
      "CGO_ENABLED": "0",
      "GOMODCACHE": app.appendingPathComponent(".build-tools/go-mod-cache", isDirectory: true).path,
      "GOCACHE": app.appendingPathComponent(".build-tools/go-build-cache", isDirectory: true).path,
    ]
    return [
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/bin/bash"),
        arguments: ["scripts/build_tile_decoder.sh", "--test"],
        currentDirectoryURL: app,
        timeout: 600
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/bin/bash"),
        arguments: ["scripts/build_tile_transaction_helper.sh", "--test"],
        currentDirectoryURL: app,
        timeout: 600
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/usr/bin/swift"),
        arguments: ["test"],
        currentDirectoryURL: app,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/bin/bash"),
        arguments: ["scripts/build_app.sh"],
        currentDirectoryURL: app,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/usr/bin/codesign"),
        arguments: ["--verify", "--deep", "--strict", "--verbose=2", "dist/VTSC Tuner.app"],
        currentDirectoryURL: app,
        timeout: 120
      ),
      ProcessRequest(
        executableURL: pinnedGo,
        arguments: ["test", "./..."],
        currentDirectoryURL: mapd,
        environment: goEnvironment,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: python,
        arguments: ["-m", "pytest", "-q", "sunnypilot/mapd/tests"],
        currentDirectoryURL: repositoryRoot,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: python,
        arguments: [
          "-m", "pytest", "--noconftest", "-o", "addopts=",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_whole_curve_profile.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_response_model.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_map_timing.py",
        ],
        currentDirectoryURL: repositoryRoot,
        timeout: 1_800
      ),
      ProcessRequest(
        executableURL: ApplyPipeline.gitURL,
        arguments: ["diff", "--check"],
        currentDirectoryURL: repositoryRoot,
        timeout: 60
      ),
    ]
  }
}
