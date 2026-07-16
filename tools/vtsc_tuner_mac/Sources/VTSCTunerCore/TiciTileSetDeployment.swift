import Foundation

public enum TiciTileActivationOutcome: String, Codable, Equatable, Sendable {
  case switched
  case notSwitched
}

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
  public var previousTileSetID: String?
  public var targetAlreadyActive: Bool
  public var commandOutput: String

  public init(
    tileSetID: String,
    previousTileSetID: String?,
    targetAlreadyActive: Bool = false,
    commandOutput: String
  ) {
    self.tileSetID = tileSetID
    self.previousTileSetID = previousTileSetID
    self.targetAlreadyActive = targetAlreadyActive
    self.commandOutput = commandOutput
  }
}

public struct TiciTileRollbackResult: Equatable, Sendable {
  public var restoredTileSetID: String?
  public var activationOutcome: TiciTileActivationOutcome

  public init(restoredTileSetID: String?, activationOutcome: TiciTileActivationOutcome) {
    self.restoredTileSetID = restoredTileSetID
    self.activationOutcome = activationOutcome
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
  static let legacyMigrationProvenance = "legacy-migration-v1"
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
          TiciParkedMutationGate.guardedCommand(
            "rm -rf \(Self.shellQuote(stagingRoot)) && mkdir -p \(Self.shellQuote(stagingRoot + "/offline"))",
            refusalMessage: "refusing tile staging unless tici is exactly offroad and Map Lookahead is disabled"
          ),
        ],
        timeout: 30
      ),
      context: "prepare remote tile staging"
    )

    let tileRsync = ProcessRequest(
      executableURL: Self.rsyncURL,
      arguments: [
        "-a", "--delete", "--partial-dir=.rsync-partial",
        "--rsync-path", TiciParkedMutationGate.gatedRsyncPath(
          refusalMessage: "refusing tile-data staging unless tici is exactly offroad and Map Lookahead is disabled"
        ),
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
          "--rsync-path", TiciParkedMutationGate.gatedRsyncPath(
            refusalMessage: "refusing tile-manifest staging unless tici is exactly offroad and Map Lookahead is disabled"
          ),
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

  public func activate(
    _ staged: TiciStagedTileSet,
    expectedCurrentTileSetID: String?
  ) async throws -> TiciTileActivationResult {
    try Self.validateProfile(staged.profile)
    try Self.validateTileSetID(staged.tileSetID)
    if let expectedCurrentTileSetID { try Self.validateStoredTileSetID(expectedCurrentTileSetID) }
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
      tileSetID: staged.tileSetID,
      expectedCurrentTileSetID: expectedCurrentTileSetID
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
    if decoded.targetAlreadyActive == true {
      guard expectedCurrentTileSetID == staged.tileSetID,
            decoded.tileActivationNotSwitched == true,
            decoded.activeTileSetID == staged.tileSetID,
            decoded.previousTileSetID == staged.tileSetID,
            decoded.previousTileSetProvenance == nil,
            decoded.previousTileSetTargetID == nil
      else {
        throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
      }
      return TiciTileActivationResult(
        tileSetID: staged.tileSetID,
        previousTileSetID: nil,
        targetAlreadyActive: true,
        commandOutput: result.combinedOutput
      )
    }
    guard let previousTileSetID = decoded.previousTileSetID else {
      throw TiciTileSetDeploymentError.activationIdentityMissing(staged.tileSetID)
    }
    guard decoded.tileActivationNotSwitched != true else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
    }
    try Self.validateStoredTileSetID(previousTileSetID)
    guard previousTileSetID != staged.tileSetID else {
      throw TiciTileSetDeploymentError.activationIdentityMissing(previousTileSetID)
    }
    return TiciTileActivationResult(
      tileSetID: staged.tileSetID,
      previousTileSetID: previousTileSetID,
      targetAlreadyActive: false,
      commandOutput: result.combinedOutput
    )
  }

  public func rollback(
    profile: String,
    expectedActivatedTileSetID: String? = nil,
    expectedRestoredTileSetID: String? = nil,
    expectedGitBranch: String,
    expectedGitHead: String
  ) async throws -> String? {
    try await rollbackWithOutcome(
      profile: profile,
      expectedActivatedTileSetID: expectedActivatedTileSetID,
      expectedRestoredTileSetID: expectedRestoredTileSetID,
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
    ).restoredTileSetID
  }

  public func rollbackWithOutcome(
    profile: String,
    expectedActivatedTileSetID: String? = nil,
    expectedRestoredTileSetID: String? = nil,
    expectedGitBranch: String,
    expectedGitHead: String
  ) async throws -> TiciTileRollbackResult {
    try Self.validateProfile(profile)
    if let expectedActivatedTileSetID { try Self.validateTileSetID(expectedActivatedTileSetID) }
    if let expectedRestoredTileSetID { try Self.validateStoredTileSetID(expectedRestoredTileSetID) }
    try Self.validateGitBinding(branch: expectedGitBranch, head: expectedGitHead)
    let helperPath = try await stageTransactionHelper(
      profile: profile,
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
    )
    let command = try Self.atomicRollbackCommand(
      helperPath: helperPath,
      expectedActivatedTileSetID: expectedActivatedTileSetID,
      expectedRestoredTileSetID: expectedRestoredTileSetID,
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
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
      if let expectedRestoredTileSetID {
        guard decoded.activeTileSetID == expectedRestoredTileSetID else {
          throw TiciTileSetDeploymentError.activationIdentityMissing(expectedActivatedTileSetID)
        }
        return TiciTileRollbackResult(
          restoredTileSetID: expectedRestoredTileSetID,
          activationOutcome: .switched
        )
      }
      guard let resolved = decoded.previousTileSetID else {
        throw TiciTileSetDeploymentError.activationIdentityMissing(expectedActivatedTileSetID)
      }
      try Self.validateStoredTileSetID(resolved)
      guard decoded.activeTileSetID == resolved,
            decoded.previousTileSetProvenance == Self.legacyMigrationProvenance,
            decoded.previousTileSetTargetID == expectedActivatedTileSetID
      else {
        throw TiciTileSetDeploymentError.activationIdentityMissing(resolved)
      }
      return TiciTileRollbackResult(restoredTileSetID: resolved, activationOutcome: .switched)
    }
    if let expectedRestoredTileSetID, decoded.tileActivationNotSwitched != true {
      guard decoded.rolledBackTileSetID == expectedRestoredTileSetID else {
        throw TiciTileSetDeploymentError.activationIdentityMissing(expectedRestoredTileSetID)
      }
    }
    guard decoded.rolledBackTileSetID != nil || decoded.tileActivationNotSwitched == true else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
    }
    if decoded.tileActivationNotSwitched == true {
      guard decoded.rolledBackTileSetID == nil,
            decoded.previousTileSetProvenance == nil,
            decoded.previousTileSetTargetID == nil
      else {
        throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
      }
      if let expectedRestoredTileSetID {
        guard decoded.previousTileSetID == expectedRestoredTileSetID,
              decoded.activeTileSetID == expectedRestoredTileSetID else {
          throw TiciTileSetDeploymentError.activationIdentityMissing(expectedRestoredTileSetID)
        }
      } else if decoded.previousTileSetID != nil || decoded.activeTileSetID != nil {
        throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
      }
      return TiciTileRollbackResult(
        restoredTileSetID: expectedRestoredTileSetID,
        activationOutcome: .notSwitched
      )
    }
    guard let restored = decoded.rolledBackTileSetID else {
      throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
    }
    if let expectedRestoredTileSetID {
      guard restored == expectedRestoredTileSetID else {
        throw TiciTileSetDeploymentError.activationIdentityMissing(expectedRestoredTileSetID)
      }
    } else {
      guard let expectedActivatedTileSetID else {
        throw TiciTileSetDeploymentError.invalidHelperOutput(result.standardOutput)
      }
      try Self.validateStoredTileSetID(restored)
      guard decoded.previousTileSetID == restored,
            decoded.previousTileSetProvenance == Self.legacyMigrationProvenance,
            decoded.previousTileSetTargetID == expectedActivatedTileSetID
      else {
        throw TiciTileSetDeploymentError.activationIdentityMissing(restored)
      }
    }
    return TiciTileRollbackResult(restoredTileSetID: restored, activationOutcome: .switched)
  }

  func proveDurableNoSwitch(
    profile: String,
    expectedGitBranch: String,
    expectedGitHead: String
  ) async throws -> TiciDeploymentSnapshot {
    try Self.validateProfile(profile)
    try Self.validateGitBinding(branch: expectedGitBranch, head: expectedGitHead)
    let command = try Self.lockedNoSwitchInspectionCommand(
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
    )
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [profile, command],
        timeout: 60
      ),
      context: "prove unchanged no-switch tile topology"
    )
    return try TiciDeploymentSnapshotDecoder.decodeRead(result.standardOutput).snapshot
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
    expectedCurrentTileSetID: String? = nil,
    injectedFailurePoint: String? = nil
  ) throws -> String {
    try validateHelperPath(helperPath)
    try validateTileSetID(tileSetID)
    guard stagingRoot == Self.stagingRoot(tileSetID: tileSetID) else {
      throw TiciTileSetDeploymentError.invalidStagingRoot(stagingRoot)
    }
    if let expectedCurrentTileSetID { try validateStoredTileSetID(expectedCurrentTileSetID) }
    let expectedCurrent = expectedCurrentTileSetID.map {
      " --expected-current-tile-set-id \(shellQuote($0))"
    } ?? ""
    return """
    \(parkedMutationPreamble())
    exec \(shellQuote(helperPath)) activate --root \(shellQuote(remoteRoot)) --params-dir \(shellQuote(TiciParkedMutationGate.defaultParamsDirectory)) --stage \(shellQuote(stagingRoot)) --tile-set-id \(shellQuote(tileSetID))\(expectedCurrent)\(try injectionArgument(injectedFailurePoint))
    """
  }

  static func atomicRollbackCommand(
    helperPath: String,
    expectedActivatedTileSetID: String? = nil,
    expectedRestoredTileSetID: String? = nil,
    expectedGitBranch: String,
    expectedGitHead: String,
    injectedFailurePoint: String? = nil
  ) throws -> String {
    try validateHelperPath(helperPath)
    if let expectedActivatedTileSetID { try validateTileSetID(expectedActivatedTileSetID) }
    if let expectedRestoredTileSetID { try validateStoredTileSetID(expectedRestoredTileSetID) }
    try validateGitBinding(branch: expectedGitBranch, head: expectedGitHead)
    let expected = expectedActivatedTileSetID.map {
      " --expected-tile-set-id \(shellQuote($0))"
    } ?? ""
    let expectedPrevious = expectedRestoredTileSetID.map {
      " --expected-previous-tile-set-id \(shellQuote($0))"
    } ?? ""
    return """
    \(parkedMutationPreamble())
    repo='/data/openpilot'
    expected_git_branch=\(shellQuote(expectedGitBranch))
    expected_git_head=\(shellQuote(expectedGitHead))
    [ "$(git -C "$repo" branch --show-current)" = "$expected_git_branch" ] || { printf '%s\n' 'tile rollback branch is not the host-proven identity' >&2; exit 1; }
    [ "$(git -C "$repo" rev-parse HEAD)" = "$expected_git_head" ] || { printf '%s\n' 'tile rollback head is not the host-proven identity' >&2; exit 1; }
    [ -z "$(git -C "$repo" status --porcelain)" ] || { printf '%s\n' 'tile rollback checkout is dirty' >&2; exit 1; }
    exec \(shellQuote(helperPath)) rollback --root \(shellQuote(remoteRoot)) --params-dir \(shellQuote(TiciParkedMutationGate.defaultParamsDirectory)) --repo-root "$repo" --expected-git-branch "$expected_git_branch" --expected-git-head "$expected_git_head"\(expected)\(expectedPrevious)\(try injectionArgument(injectedFailurePoint))
    """
  }

  static func lockedNoSwitchInspectionCommand(
    expectedGitBranch: String,
    expectedGitHead: String
  ) throws -> String {
    try validateGitBinding(branch: expectedGitBranch, head: expectedGitHead)
    return """
    set -eu
    params_dir='/data/params/d'
    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing no-switch proof unless tici is exactly offroad and Map Lookahead is disabled"
    ))
    exec 8>\(shellQuote(remoteRoot + "/.tileset-transaction.lock"))
    flock -x 8
    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing no-switch proof after lock unless tici is exactly offroad and Map Lookahead is disabled"
    ))
    repo='/data/openpilot'
    branch=$(git -C "$repo" branch --show-current) || exit 1
    [ "$branch" = \(shellQuote(expectedGitBranch)) ] || exit 1
    head=$(git -C "$repo" rev-parse HEAD) || exit 1
    [ "$head" = \(shellQuote(expectedGitHead)) ] || exit 1
    status=$(git -C "$repo" status --porcelain) || exit 1
    [ -z "$status" ] || exit 1
    \(TiciSnapshotWireCommandBuilder.inspectionCommand())
    """
  }

  private func stageTransactionHelper(
    profile: String,
    expectedGitBranch: String? = nil,
    expectedGitHead: String? = nil
  ) async throws -> String {
    if expectedGitBranch != nil || expectedGitHead != nil {
      guard let expectedGitBranch, let expectedGitHead else {
        throw TiciTileSetDeploymentError.invalidHelperOutput("incomplete rollback Git binding")
      }
      try Self.validateGitBinding(branch: expectedGitBranch, head: expectedGitHead)
    }
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
          Self.helperStagingCommand(
            "mkdir -p \(Self.shellQuote(Self.helperDirectory)) && rm -f \(Self.shellQuote(partialPath))",
            expectedGitBranch: expectedGitBranch,
            expectedGitHead: expectedGitHead
          ),
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
          "--rsync-path", Self.helperRsyncPath(
            expectedGitBranch: expectedGitBranch,
            expectedGitHead: expectedGitHead
          ),
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
            sha256: digest,
            expectedGitBranch: expectedGitBranch,
            expectedGitHead: expectedGitHead
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
    sha256: String,
    expectedGitBranch: String? = nil,
    expectedGitHead: String? = nil
  ) -> String {
    helperStagingCommand(
    """
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
    """,
    expectedGitBranch: expectedGitBranch,
    expectedGitHead: expectedGitHead
    )
  }

  private static func helperStagingCommand(
    _ mutation: String,
    expectedGitBranch: String?,
    expectedGitHead: String?
  ) -> String {
    let identity = rollbackGitIdentityFragment(
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
    )
    return TiciParkedMutationGate.guardedCommand(
      identity + mutation,
      refusalMessage: "refusing tile-helper staging unless tici is exactly offroad and Map Lookahead is disabled"
    )
  }

  private static func helperRsyncPath(
    expectedGitBranch: String?,
    expectedGitHead: String?
  ) -> String {
    let body = """
    params_dir=\(shellQuote(TiciParkedMutationGate.defaultParamsDirectory))
    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing tile-helper transfer unless tici is exactly offroad and Map Lookahead is disabled"
    ))
    \(rollbackGitIdentityFragment(
      expectedGitBranch: expectedGitBranch,
      expectedGitHead: expectedGitHead
    ))
    exec rsync "$@"
    """
    return "sh -c \(shellQuote(body)) sh"
  }

  private static func rollbackGitIdentityFragment(
    expectedGitBranch: String?,
    expectedGitHead: String?
  ) -> String {
    guard let expectedGitBranch, let expectedGitHead else { return "" }
    return """
    repo='/data/openpilot'
    [ "$(git -C "$repo" branch --show-current)" = \(shellQuote(expectedGitBranch)) ] || { printf '%s\n' 'tile-helper staging branch is not the host-proven identity' >&2; exit 1; }
    [ "$(git -C "$repo" rev-parse HEAD)" = \(shellQuote(expectedGitHead)) ] || { printf '%s\n' 'tile-helper staging head is not the host-proven identity' >&2; exit 1; }
    [ -z "$(git -C "$repo" status --porcelain)" ] || { printf '%s\n' 'tile-helper staging checkout is dirty' >&2; exit 1; }
    """
  }

  private static func parkedMutationPreamble() -> String {
    """
    set -eu
    params_dir='/data/params/d'
    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing tile mutation unless tici is exactly offroad and Map Lookahead is disabled"
    ))
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

  private static func validateStoredTileSetID(_ tileSetID: String) throws {
    if tileSetID.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil { return }
    try validateLegacyTileSetID(tileSetID)
  }

  private static func validateLegacyTileSetID(_ tileSetID: String) throws {
    guard tileSetID.range(of: #"^legacy-[0-9a-f]{16}$"#, options: .regularExpression) != nil else {
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

  private static func validateGitBinding(branch: String, head: String) throws {
    guard branch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
          !branch.contains(".."), !branch.hasPrefix("/"),
          head.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidHelperOutput("unsafe rollback Git identity")
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
    var previousTileSetID: String?
    var previousTileSetProvenance: String?
    var previousTileSetTargetID: String?
    var targetAlreadyActive: Bool?
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
      case previousTileSetID = "previous_tile_set_id"
      case previousTileSetProvenance = "previous_tile_set_provenance"
      case previousTileSetTargetID = "previous_tile_set_target_id"
      case targetAlreadyActive = "target_already_active"
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
