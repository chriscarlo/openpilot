import Foundation

public struct TiciStagedTileSet: Equatable, Sendable {
  public var profile: String
  public var tileSetID: String
  public var remoteStagingRoot: String
  public var transactionHelperPath: String

  public init(
    profile: String,
    tileSetID: String,
    remoteStagingRoot: String,
    transactionHelperPath: String = ""
  ) {
    self.profile = profile
    self.tileSetID = tileSetID
    self.remoteStagingRoot = remoteStagingRoot
    self.transactionHelperPath = transactionHelperPath
  }
}

public struct TiciTileActivationResult: Equatable, Sendable {
  public var tileSetID: String
  public var commandOutput: String

  public init(tileSetID: String, commandOutput: String) {
    self.tileSetID = tileSetID
    self.commandOutput = commandOutput
  }
}

enum TileTransactionRecoveryDecision: Equatable, Sendable {
  case restartBeforeSwitch
  case finishAfterSwitch
  case alreadyRolledBack
  case performRollback
  case manualRecoveryRequired
}

enum TileTransactionRecovery {
  static func activation(
    pendingID: String,
    expectedID: String,
    activeTarget: String,
    newTarget: String,
    previousTarget: String
  ) -> TileTransactionRecoveryDecision {
    guard pendingID == expectedID else { return .manualRecoveryRequired }
    if activeTarget == newTarget { return .finishAfterSwitch }
    if activeTarget.isEmpty || activeTarget == previousTarget { return .restartBeforeSwitch }
    return .manualRecoveryRequired
  }

  static func rollback(
    activeTarget: String,
    previousTarget: String,
    recordedActiveTarget: String,
    recordedPreviousTarget: String
  ) -> TileTransactionRecoveryDecision {
    if activeTarget == recordedPreviousTarget, previousTarget == recordedActiveTarget {
      return .alreadyRolledBack
    }
    if activeTarget == recordedActiveTarget, previousTarget == recordedPreviousTarget {
      return .performRollback
    }
    return .manualRecoveryRequired
  }
}

/// Swift owns identity, helper integrity, result parsing, and safety policy.
/// The static tici helper only supplies the Linux rename-exchange primitive
/// plus its durable on-device tile journal.
public struct TiciTileSetDeploymentService: Sendable {
  public static let sshURL = URL(fileURLWithPath: "/usr/bin/ssh")
  public static let rsyncURL = URL(fileURLWithPath: "/usr/bin/rsync")
  public static let remoteRoot = "/data/media/0/osm"
  public static let activeOfflinePath = "/data/media/0/osm/offline"
  public static let previousOfflinePath = "/data/media/0/osm/offline.previous"
  public static let generationRoot = "/data/media/0/osm/tile-generations"
  public static let embeddedManifestName = ".tileset-manifest.json"
  public static let transactionPath = "/data/media/0/osm/.tileset-transaction.json"
  public static let helperDirectory = "/data/media/0/osm/binaries"

  private let processRunner: any ProcessRunning
  private let transactionHelperURL: URL?

  public init(
    processRunner: any ProcessRunning = SystemProcessRunner(),
    transactionHelperURL: URL? = nil
  ) {
    self.processRunner = processRunner
    self.transactionHelperURL = transactionHelperURL
  }

  public func stageAndVerify(
    artifact: CanonicalTileSetArtifact,
    profile: String
  ) async throws -> TiciStagedTileSet {
    try Self.validateProfile(profile)
    let manifest = try artifact.manifest.validatedMetadata()
    let stagingRoot = Self.stagingRoot(tileSetID: manifest.tileSetID)
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          "rm -rf \(Self.shellQuote(stagingRoot)) && mkdir -p \(Self.shellQuote(stagingRoot + "/offline"))",
        ],
        timeout: 30
      ),
      context: "prepare remote tile staging"
    )

    let tileRsync = ProcessRequest(
      executableURL: Self.rsyncURL,
      arguments: [
        "-a", "--delete", "--partial-dir=.rsync-partial",
        "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
        artifact.offlineURL.path + "/",
        "\(profile):\(stagingRoot)/offline/",
      ],
      timeout: 3_600
    )
    guard !tileRsync.arguments.contains(where: { $0 == "\(profile):\(Self.activeOfflinePath)/" }) else {
      throw TiciTileSetDeploymentError.activePathTransferForbidden
    }
    _ = try await checked(tileRsync, context: "stage canonical tile files")
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-a",
          "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          artifact.manifestURL.path,
          "\(profile):\(stagingRoot)/manifest.json",
        ],
        timeout: 120
      ),
      context: "stage tile manifest"
    )

    let helperPath = try await stageTransactionHelper(profile: profile)
    let verified = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          try Self.remoteManifestVerificationCommand(
            helperPath: helperPath,
            stagingRoot: stagingRoot,
            tileSetID: manifest.tileSetID
          ),
        ],
        timeout: max(300, TimeInterval(manifest.fileCount) * 2)
      ),
      context: "verify staged tile manifest"
    )
    let result = try Self.decodeHelperResult(verified.standardOutput)
    guard result.operation == "verify",
          result.tileSetID == manifest.tileSetID,
          result.fileCount == manifest.fileCount,
          result.totalBytes == manifest.totalBytes
    else {
      throw TiciTileSetDeploymentError.verificationIdentityMissing(manifest.tileSetID)
    }
    return TiciStagedTileSet(
      profile: profile,
      tileSetID: manifest.tileSetID,
      remoteStagingRoot: stagingRoot,
      transactionHelperPath: helperPath
    )
  }

  public func activate(_ staged: TiciStagedTileSet) async throws -> TiciTileActivationResult {
    try Self.validateProfile(staged.profile)
    try Self.validateTileSetID(staged.tileSetID)
    guard staged.remoteStagingRoot == Self.stagingRoot(tileSetID: staged.tileSetID) else {
      throw TiciTileSetDeploymentError.invalidStagingRoot(staged.remoteStagingRoot)
    }
    let helperPath: String
    if staged.transactionHelperPath.isEmpty {
      helperPath = try await stageTransactionHelper(profile: staged.profile)
    } else {
      try Self.validateHelperPath(staged.transactionHelperPath)
      helperPath = staged.transactionHelperPath
    }
    let command = try Self.atomicActivationCommand(
      helperPath: helperPath,
      stagingRoot: staged.remoteStagingRoot,
      tileSetID: staged.tileSetID
    )
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [staged.profile, command],
        timeout: 180
      ),
      context: "atomically activate tile set"
    )
    let decoded = try Self.decodeHelperResult(result.standardOutput)
    guard decoded.operation == "activate", decoded.activatedTileSetID == staged.tileSetID else {
      throw TiciTileSetDeploymentError.activationIdentityMissing(staged.tileSetID)
    }
    return TiciTileActivationResult(tileSetID: staged.tileSetID, commandOutput: result.combinedOutput)
  }

  public func rollback(profile: String, expectedActivatedTileSetID: String? = nil) async throws {
    try Self.validateProfile(profile)
    if let expectedActivatedTileSetID { try Self.validateTileSetID(expectedActivatedTileSetID) }
    let helperPath = try await stageTransactionHelper(profile: profile)
    let command = try Self.atomicRollbackCommand(
      helperPath: helperPath,
      expectedActivatedTileSetID: expectedActivatedTileSetID
    )
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [profile, command],
        timeout: 180
      ),
      context: "roll back active tile set"
    )
    let decoded = try Self.decodeHelperResult(result.standardOutput)
    guard decoded.operation == "rollback" else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
    }
    if let expectedActivatedTileSetID,
       decoded.tileActivationNotObserved == true {
      throw TiciTileSetDeploymentError.activationIdentityMissing(expectedActivatedTileSetID)
    }
    guard decoded.rolledBackTileSetID != nil || decoded.tileActivationNotSwitched == true else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
    }
  }

  public static func stagingRoot(tileSetID: String) -> String {
    "\(remoteRoot)/.tileset-\(tileSetID).partial"
  }

  static func remoteManifestVerificationCommand(
    helperPath: String,
    stagingRoot: String,
    tileSetID: String
  ) throws -> String {
    try validateHelperPath(helperPath)
    try validateTileSetID(tileSetID)
    guard stagingRoot == Self.stagingRoot(tileSetID: tileSetID) else {
      throw TiciTileSetDeploymentError.invalidStagingRoot(stagingRoot)
    }
    return "\(shellQuote(helperPath)) verify --root \(shellQuote(stagingRoot)) --tile-set-id \(shellQuote(tileSetID))"
  }

  static func atomicActivationCommand(
    helperPath: String,
    stagingRoot: String,
    tileSetID: String,
    injectedFailurePoint: String? = nil
  ) throws -> String {
    try validateHelperPath(helperPath)
    try validateTileSetID(tileSetID)
    guard stagingRoot == Self.stagingRoot(tileSetID: tileSetID) else {
      throw TiciTileSetDeploymentError.invalidStagingRoot(stagingRoot)
    }
    return """
    \(parkedMutationPreamble())
    exec \(shellQuote(helperPath)) activate --root \(shellQuote(remoteRoot)) --stage \(shellQuote(stagingRoot)) --tile-set-id \(shellQuote(tileSetID))\(try injectionArgument(injectedFailurePoint))
    """
  }

  static func atomicRollbackCommand(
    helperPath: String,
    expectedActivatedTileSetID: String? = nil,
    injectedFailurePoint: String? = nil
  ) throws -> String {
    try validateHelperPath(helperPath)
    if let expectedActivatedTileSetID { try validateTileSetID(expectedActivatedTileSetID) }
    let expected = expectedActivatedTileSetID.map {
      " --expected-tile-set-id \(shellQuote($0))"
    } ?? ""
    return """
    \(parkedMutationPreamble())
    exec \(shellQuote(helperPath)) rollback --root \(shellQuote(remoteRoot))\(expected)\(try injectionArgument(injectedFailurePoint))
    """
  }

  private func stageTransactionHelper(profile: String) async throws -> String {
    let helperURL = try resolvedTransactionHelperURL()
    let digest = try FileSHA256.hex(helperURL)
    guard digest.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidHelperOutput("local helper SHA-256 is malformed")
    }
    let remotePath = Self.helperRemotePath(sha256: digest)
    let partialPath = remotePath + ".partial"
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          "mkdir -p \(Self.shellQuote(Self.helperDirectory)) && rm -f \(Self.shellQuote(partialPath))",
        ],
        timeout: 30
      ),
      context: "prepare tile transaction helper"
    )
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-a", "--partial",
          "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          helperURL.path,
          "\(profile):\(partialPath)",
        ],
        timeout: 300
      ),
      context: "stage tile transaction helper"
    )
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          Self.installHelperCommand(
            partialPath: partialPath,
            destinationPath: remotePath,
            sha256: digest
          ),
        ],
        timeout: 60
      ),
      context: "verify tile transaction helper"
    )
    return remotePath
  }

  private func resolvedTransactionHelperURL() throws -> URL {
    let url = try transactionHelperURL ?? TiciTileTransactionHelperLocator.bundledURL()
    guard FileManager.default.isExecutableFile(atPath: url.path) else {
      throw TiciTileTransactionHelperError.missingHelper(url)
    }
    return url.standardizedFileURL
  }

  private static func helperRemotePath(sha256: String) -> String {
    "\(helperDirectory)/vtsc-tile-transaction-\(sha256.prefix(16))"
  }

  private static func installHelperCommand(
    partialPath: String,
    destinationPath: String,
    sha256: String
  ) -> String {
    """
    set -eu
    partial=\(shellQuote(partialPath))
    destination=\(shellQuote(destinationPath))
    expected=\(shellQuote(sha256))
    [ -f "$partial" ] || { printf '%s\\n' 'tile transaction helper staging file is missing' >&2; exit 1; }
    actual="$(sha256sum "$partial" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || { printf '%s\\n' 'tile transaction helper digest mismatch' >&2; exit 1; }
    chmod 755 "$partial"
    sync -f "$partial"
    mv -f "$partial" "$destination"
    sync -f "$destination"
    sync -f \(shellQuote(helperDirectory)) || sync
    [ "$(sha256sum "$destination" | awk '{print $1}')" = "$expected" ] || {
      printf '%s\\n' 'tile transaction helper read-back digest mismatch' >&2
      exit 1
    }
    """
  }

  private static func parkedMutationPreamble() -> String {
    """
    set -eu
    params_dir='/data/params/d'
    [ -d "$params_dir" ] || { printf '%s\\n' 'Params directory is missing' >&2; exit 1; }
    offroad="$(cat "$params_dir/IsOffroad" 2>/dev/null || true)"
    onroad="$(cat "$params_dir/IsOnroad" 2>/dev/null || true)"
    lookahead="$(cat "$params_dir/MTSCLookaheadEnabled" 2>/dev/null || true)"
    if [ "$offroad" != '1' ] || [ "$onroad" = '1' ] || [ "$lookahead" = '1' ]; then
      printf '%s\\n' 'refusing tile mutation unless tici is offroad and Map Lookahead is disabled' >&2
      exit 1
    fi
    """
  }

  private static func decodeHelperResult(_ output: String) throws -> HelperResult {
    let records = output.split(whereSeparator: \.isNewline)
    guard records.count == 1, let record = records.first else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(output)
    }
    do {
      return try JSONDecoder().decode(HelperResult.self, from: Data(record.utf8))
    } catch {
      throw TiciTileSetDeploymentError.invalidHelperOutput("\(output)\n\(error.localizedDescription)")
    }
  }

  private func checked(_ request: ProcessRequest, context: String) async throws -> ProcessResult {
    try Task.checkCancellation()
    let result = try await processRunner.run(request)
    guard result.succeeded else {
      throw TiciTileSetDeploymentError.commandFailed(context, result.terminationStatus, result.combinedOutput)
    }
    return result
  }

  private static func validateProfile(_ profile: String) throws {
    guard profile.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidProfile(profile)
    }
  }

  private static func validateTileSetID(_ tileSetID: String) throws {
    guard tileSetID.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidTileSetID(tileSetID)
    }
  }

  private static func validateHelperPath(_ path: String) throws {
    guard path.range(
      of: #"^/data/media/0/osm/binaries/vtsc-tile-transaction-[0-9a-f]{16}$"#,
      options: .regularExpression
    ) != nil else {
      throw TiciTileSetDeploymentError.invalidHelperPath(path)
    }
  }

  private static func injectionArgument(_ value: String?) throws -> String {
    guard let value else { return "" }
    guard value.range(of: #"^[a-z_]{1,64}$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidHelperOutput("unsafe tile failure injection")
    }
    return " --inject-failure \(shellQuote(value))"
  }

  private static func sshOptions(connectTimeout: Int) -> [String] {
    [
      "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=\(connectTimeout)",
      "-o", "StrictHostKeyChecking=accept-new",
    ]
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }

  private struct HelperResult: Decodable {
    var operation: String
    var tileSetID: String?
    var activatedTileSetID: String?
    var rolledBackTileSetID: String?
    var activeTileSetID: String?
    var fileCount: Int?
    var totalBytes: UInt64?
    var tileActivationNotSwitched: Bool?
    var tileActivationNotObserved: Bool?

    enum CodingKeys: String, CodingKey {
      case operation
      case tileSetID = "tile_set_id"
      case activatedTileSetID = "activated_tile_set_id"
      case rolledBackTileSetID = "rolled_back_tile_set_id"
      case activeTileSetID = "active_tile_set_id"
      case fileCount = "file_count"
      case totalBytes = "total_bytes"
      case tileActivationNotSwitched = "tile_activation_not_switched"
      case tileActivationNotObserved = "tile_activation_not_observed"
    }
  }
}

public enum TiciTileSetDeploymentError: LocalizedError, Equatable, Sendable {
  case invalidProfile(String)
  case invalidStagingRoot(String)
  case invalidTileSetID(String)
  case invalidHelperPath(String)
  case activePathTransferForbidden
  case commandFailed(String, Int32, String)
  case verificationIdentityMissing(String)
  case activationIdentityMissing(String)
  case invalidHelperOutput(String)

  public var errorDescription: String? {
    switch self {
    case let .invalidProfile(profile): "Invalid SSH profile: \(profile)"
    case let .invalidStagingRoot(path): "Invalid remote tile staging root: \(path)"
    case let .invalidTileSetID(tileSetID): "Invalid canonical tile-set ID: \(tileSetID)"
    case let .invalidHelperPath(path): "Invalid tici tile transaction helper path: \(path)"
    case .activePathTransferForbidden: "Refusing to transfer files directly into the active tile directory."
    case let .commandFailed(context, status, output): "\(context) exited \(status): \(output)"
    case let .verificationIdentityMissing(identity): "Remote verification did not confirm tile set \(identity)."
    case let .activationIdentityMissing(identity): "Atomic activation did not confirm tile set \(identity)."
    case let .invalidHelperOutput(output): "The tici tile transaction helper returned invalid output: \(output)"
    }
  }
}
