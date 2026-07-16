import CryptoKit
import Darwin
import Foundation

@_silgen_name("flock")
private func vtscSystemFlock(_ fileDescriptor: Int32, _ operation: Int32) -> Int32

/// The exact source-rounded tune identity used by deployment and postflight.
/// It deliberately hashes the emitted Python representation rather than the
/// higher-precision editor values.
public struct TuneDeploymentIdentity: Codable, Equatable, Sendable {
  public struct PhysicsValue: Codable, Equatable, Sendable {
    public var paramKey: String
    public var value: String

    public init(paramKey: String, value: String) {
      self.paramKey = paramKey
      self.value = value
    }
  }

  public var physics: [PhysicsValue]
  public var qCurveEnabled: Bool
  public var qCurvePointCount: Int
  public var qCurveSHA256: String
  public var identitySHA256: String

  public var tileSigmoidHash: String {
    let entries = physics.map(\.value)
    guard entries.count == 6 else { return "" }
    let canonical = entries.joined(separator: "|") + "|70.00"
    return String(Self.sha256Hex(Data(canonical.utf8)).prefix(12))
  }

  public init(tune: Tune) {
    physics = VTSCPhysicsAuthority.entries(for: tune.params).map {
      PhysicsValue(paramKey: $0.paramKey, value: $0.formattedValue)
    }
    let enabledBands = tune.bands.filter(\.enabled)
    qCurveEnabled = !enabledBands.isEmpty
    let qSource = Self.canonicalQCurveSource(parameters: tune.params, bands: enabledBands)
    qCurvePointCount = enabledBands.isEmpty ? 0 : 256
    qCurveSHA256 = Self.sha256Hex(Data(qSource.utf8))
    identitySHA256 = ""
    identitySHA256 = Self.sha256Hex(Data(canonicalText.utf8))
  }

  public var canonicalText: String {
    let physicsText = physics.map { "\($0.paramKey)=\($0.value)" }.joined(separator: "\n")
    return [
      "vtsc-tune-deployment-v1",
      physicsText,
      "q_enabled=\(qCurveEnabled ? 1 : 0)",
      "q_count=\(qCurvePointCount)",
      "q_sha256=\(qCurveSHA256)",
    ].joined(separator: "\n") + "\n"
  }

  public static func canonicalQCurveSource(parameters: SigmoidParameters, bands: [EQBand]) -> String {
    let enabledBands = bands.filter(\.enabled)
    guard !enabledBands.isEmpty else {
      return "Q_CURVE_ENABLED = False\nQ_CURVE_POINTS: list[tuple[float, float]] = []\n"
    }
    let exported = VTSCMath.sourceRoundedParameters(parameters)
    let points = VTSCMath.qCurvePoints(parameters: exported, bands: enabledBands, count: 256)
    var lines = ["Q_CURVE_ENABLED = True", "Q_CURVE_POINTS: list[tuple[float, float]] = ["]
    lines += points.map { String(format: "  (%.6e, %.4f),", $0.kappa, $0.speedMultiplier) }
    lines.append("]")
    return lines.joined(separator: "\n") + "\n"
  }

  public static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}

public struct DeploymentRollbackJournal: Codable, Equatable, Sendable {
  public static let schemaVersion = 1

  public enum Resolution: String, Codable, Equatable, Sendable {
    case awaitingPostflight
    case preflightReserved
    case mutationInProgress
    case awaitingOutdoorPostflight
    case completed
    case rollbackInProgress
    case rolledBack
    case rollbackFailed
  }

  public var schema: Int
  public var deploymentID: UUID
  public var createdAt: String
  public var profile: String
  public var branch: String
  public var previousBootID: String?
  public var previousHead: String
  public var targetHead: String?
  public var previousPhysicsParams: [String: String?]
  public var previousQCurveSHA256: String
  public var previousMapdReleaseVersion: String?
  public var previousMapdVersion: String?
  public var previousActiveMapdSHA256: String
  public var previousActiveMapdBuildInfoSHA256: String?
  public var previousCachedMapdPath: String
  public var previousCachedMapdSHA256: String?
  public var mapdRollbackPath: String
  public var previousTileSetID: String?
  public var targetTileSetID: String?
  /// Helper-resolved immutable identity for a legacy direct `offline/` tree.
  /// This is lifecycle evidence, not part of the original deployment
  /// identity, and is durably filled before any post-activation continuation.
  public var resolvedPreviousTileSetID: String?
  /// Exact verified helper outcome. This distinguishes a true same-target
  /// no-switch from pathological equal-ID mutation and survives reboot/fresh
  /// rollback recovery without changing immutable deployment identity.
  public var tileActivationOutcome: TiciTileActivationOutcome?
  /// Exact Linux boot identity captured and durably synced immediately before
  /// the deployment reboot. Optional only for schema-1 compatibility; a
  /// missing value can never certify the immediate post-reboot install.
  public var deploymentPreRebootBootID: String?
  /// Exact Linux boot identity captured and durably synced immediately before
  /// a rollback reboot. A rebooted rollback cannot settle without it.
  public var rollbackPreRebootBootID: String?
  public var rebootSent: Bool
  public var completed: Bool
  public var completedAt: String?
  public var completedToolingHead: String?
  /// Exact tici Git identity certified at completion. This may be the
  /// immutable target or one explicitly host-proven completion-compatible
  /// successor; it is never inferred from the host tooling HEAD.
  public var completedDeviceHead: String?
  public var completionHostOnlyPaths: [String]?
  /// Added compatibly to schema 1. A legacy journal without this key resolves
  /// from `completed`, so B174 remains byte-for-byte awaiting postflight until
  /// a terminal owner actually settles it.
  public var resolution: Resolution?

  public var effectiveResolution: Resolution {
    resolution ?? (completed ? .completed : .awaitingPostflight)
  }

  public var isAwaitingOutdoorPostflight: Bool {
    effectiveResolution == .awaitingPostflight ||
      effectiveResolution == .awaitingOutdoorPostflight
  }

  public var effectivePreviousTileSetID: String? {
    resolvedPreviousTileSetID ?? previousTileSetID
  }

  public func hasSameDeploymentIdentity(as other: Self) -> Bool {
    schema == other.schema &&
      deploymentID == other.deploymentID &&
      createdAt == other.createdAt &&
      profile == other.profile &&
      branch == other.branch &&
      previousBootID == other.previousBootID &&
      previousHead == other.previousHead &&
      targetHead == other.targetHead &&
      previousPhysicsParams == other.previousPhysicsParams &&
      previousQCurveSHA256 == other.previousQCurveSHA256 &&
      previousMapdReleaseVersion == other.previousMapdReleaseVersion &&
      previousMapdVersion == other.previousMapdVersion &&
      previousActiveMapdSHA256 == other.previousActiveMapdSHA256 &&
      previousActiveMapdBuildInfoSHA256 == other.previousActiveMapdBuildInfoSHA256 &&
      previousCachedMapdPath == other.previousCachedMapdPath &&
      previousCachedMapdSHA256 == other.previousCachedMapdSHA256 &&
      mapdRollbackPath == other.mapdRollbackPath &&
      previousTileSetID == other.previousTileSetID &&
      targetTileSetID == other.targetTileSetID
  }

  public func validateResolutionConsistency() throws {
    switch effectiveResolution {
    case .awaitingPostflight, .awaitingOutdoorPostflight:
      guard !completed else {
        throw DeploymentRollbackJournalError.inconsistentResolution(
          "\(effectiveResolution.rawValue) requires completed=false"
        )
      }
    case .preflightReserved, .mutationInProgress, .completed, .rollbackInProgress, .rolledBack, .rollbackFailed:
      guard completed else {
        throw DeploymentRollbackJournalError.inconsistentResolution(
          "\(effectiveResolution.rawValue) requires the legacy completed=true sentinel"
        )
      }
    }
  }

  public init(
    deploymentID: UUID = UUID(),
    createdAt: String = Date().ISO8601Format(),
    profile: String,
    branch: String,
    previousBootID: String? = nil,
    previousHead: String,
    targetHead: String? = nil,
    previousPhysicsParams: [String: String?],
    previousQCurveSHA256: String,
    previousMapdReleaseVersion: String?,
    previousMapdVersion: String?,
    previousActiveMapdSHA256: String,
    previousActiveMapdBuildInfoSHA256: String? = nil,
    previousCachedMapdPath: String,
    previousCachedMapdSHA256: String? = nil,
    mapdRollbackPath: String,
    previousTileSetID: String? = nil,
    targetTileSetID: String? = nil,
    resolvedPreviousTileSetID: String? = nil,
    tileActivationOutcome: TiciTileActivationOutcome? = nil,
    deploymentPreRebootBootID: String? = nil,
    rollbackPreRebootBootID: String? = nil,
    rebootSent: Bool = false,
    completed: Bool = false,
    completedAt: String? = nil,
    completedToolingHead: String? = nil,
    completedDeviceHead: String? = nil,
    completionHostOnlyPaths: [String]? = nil,
    resolution: Resolution? = nil
  ) {
    schema = Self.schemaVersion
    self.deploymentID = deploymentID
    self.createdAt = createdAt
    self.profile = profile
    self.branch = branch
    self.previousBootID = previousBootID
    self.previousHead = previousHead
    self.targetHead = targetHead
    self.previousPhysicsParams = previousPhysicsParams
    self.previousQCurveSHA256 = previousQCurveSHA256
    self.previousMapdReleaseVersion = previousMapdReleaseVersion
    self.previousMapdVersion = previousMapdVersion
    self.previousActiveMapdSHA256 = previousActiveMapdSHA256
    self.previousActiveMapdBuildInfoSHA256 = previousActiveMapdBuildInfoSHA256
    self.previousCachedMapdPath = previousCachedMapdPath
    self.previousCachedMapdSHA256 = previousCachedMapdSHA256
    self.mapdRollbackPath = mapdRollbackPath
    self.previousTileSetID = previousTileSetID
    self.targetTileSetID = targetTileSetID
    self.resolvedPreviousTileSetID = resolvedPreviousTileSetID
    self.tileActivationOutcome = tileActivationOutcome
    self.deploymentPreRebootBootID = deploymentPreRebootBootID
    self.rollbackPreRebootBootID = rollbackPreRebootBootID
    self.rebootSent = rebootSent
    self.completed = completed
    self.completedAt = completedAt
    self.completedToolingHead = completedToolingHead
    self.completedDeviceHead = completedDeviceHead
    self.completionHostOnlyPaths = completionHostOnlyPaths
    self.resolution = resolution
  }

  public static func defaultDirectory(fileManager: FileManager = .default) throws -> URL {
    try TuneStore.applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("deployment-journals", isDirectory: true)
  }

  public static func load(
    from url: URL,
    fileManager: FileManager = .default
  ) throws -> Self {
    let standardizedURL = url.standardizedFileURL
    guard fileManager.fileExists(atPath: standardizedURL.path) else {
      throw DeploymentRollbackJournalError.couldNotLoad(standardizedURL, "file does not exist")
    }
    let journal: Self
    do {
      journal = try JSONDecoder().decode(Self.self, from: Data(contentsOf: standardizedURL))
    } catch {
      throw DeploymentRollbackJournalError.couldNotLoad(standardizedURL, error.localizedDescription)
    }
    guard journal.schema == Self.schemaVersion else {
      throw DeploymentRollbackJournalError.unsupportedSchema(journal.schema)
    }
    return journal
  }

  public static func loadPendingPostflight(
    from explicitURL: URL? = nil,
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> (journal: Self, url: URL) {
    if let explicitURL {
      let url = explicitURL.standardizedFileURL
      let journal = try load(from: url, fileManager: fileManager)
      try journal.validatePendingPostflight()
      return (journal, url)
    }

    let directory = try explicitDirectory?.standardizedFileURL ?? defaultDirectory(fileManager: fileManager)
    let urls: [URL]
    do {
      urls = try fileManager.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      ).filter { $0.pathExtension.lowercased() == "json" }.sorted { $0.path < $1.path }
    } catch {
      throw DeploymentRollbackJournalError.couldNotList(directory, error.localizedDescription)
    }

    var candidates: [(Self, URL)] = []
    for url in urls {
      let journal = try load(from: url, fileManager: fileManager)
      guard journal.isAwaitingOutdoorPostflight, journal.rebootSent else { continue }
      try journal.validatePendingPostflight()
      candidates.append((journal, url.standardizedFileURL))
    }
    guard !candidates.isEmpty else { throw DeploymentRollbackJournalError.noPendingPostflight }
    guard candidates.count == 1 else {
      throw DeploymentRollbackJournalError.multiplePendingPostflights(candidates.map { $0.1 })
    }
    return (candidates[0].0, candidates[0].1)
  }

  public static func loadRecoverableRollback(
    from explicitURL: URL? = nil,
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> (journal: Self, url: URL) {
    let recoverable = try recoverableRollbacks(
      from: explicitURL,
      directory: explicitDirectory,
      fileManager: fileManager
    )
    guard !recoverable.isEmpty else { throw DeploymentRollbackJournalError.noRecoverableRollback }
    guard recoverable.count == 1 else {
      throw DeploymentRollbackJournalError.multipleRecoverableRollbacks(recoverable.map(\.url))
    }
    return (recoverable[0].journal, recoverable[0].url)
  }

  public struct RecoverableRollback: Identifiable, Equatable, Sendable {
    public var journal: DeploymentRollbackJournal
    public var url: URL
    public var id: URL { url }
  }

  public static func pendingPostflights(
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> [RecoverableRollback] {
    try journalCandidates(directory: explicitDirectory, fileManager: fileManager).compactMap { journal, url in
      guard journal.isAwaitingOutdoorPostflight,
            journal.rebootSent,
            !journal.completed else { return nil }
      try journal.validatePendingPostflight()
      return RecoverableRollback(journal: journal, url: url)
    }
  }

  public static func recoverableRollbacks(
    from explicitURL: URL? = nil,
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> [RecoverableRollback] {
    let candidates = try journalCandidates(
      from: explicitURL,
      directory: explicitDirectory,
      fileManager: fileManager
    )
    let recoverable = candidates.filter {
      $0.0.effectiveResolution == .rollbackInProgress ||
        $0.0.effectiveResolution == .mutationInProgress ||
        $0.0.effectiveResolution == .rollbackFailed ||
        ($0.0.effectiveResolution == .awaitingPostflight &&
          !$0.0.rebootSent && $0.0.targetHead != nil) ||
        ($0.0.effectiveResolution == .rolledBack && !$0.0.completed)
    }
    return try recoverable.map { journal, url in
      guard journal.targetHead?.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
      else { throw DeploymentRollbackJournalError.notRecoverableRollback }
      return RecoverableRollback(journal: journal, url: url)
    }
  }

  /// Every journal that represents a deployment which must be resolved before
  /// another car-facing transaction can safely begin. A rebooted awaiting
  /// journal belongs to outdoor postflight; rollback states belong to guarded
  /// rollback recovery.
  public static func unresolvedProductionJournals(
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> [RecoverableRollback] {
    try journalCandidates(directory: explicitDirectory, fileManager: fileManager).compactMap { journal, url in
      let unresolvedAwaiting = journal.isAwaitingOutdoorPostflight && journal.rebootSent
      let unresolvedRollback = journal.effectiveResolution == .rollbackInProgress ||
        journal.effectiveResolution == .preflightReserved ||
        journal.effectiveResolution == .mutationInProgress ||
        journal.effectiveResolution == .rollbackFailed ||
        (journal.effectiveResolution == .awaitingPostflight &&
          !journal.rebootSent && journal.targetHead == nil) ||
        (journal.effectiveResolution == .awaitingPostflight &&
          !journal.rebootSent && journal.targetHead != nil) ||
        (journal.effectiveResolution == .rolledBack && !journal.completed)
      guard unresolvedAwaiting || unresolvedRollback else { return nil }
      return RecoverableRollback(journal: journal, url: url)
    }
  }

  private static func journalCandidates(
    from explicitURL: URL? = nil,
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> [(Self, URL)] {
    if let explicitURL {
      let url = explicitURL.standardizedFileURL
      return [(try load(from: url, fileManager: fileManager), url)]
    }
    let directory = try explicitDirectory?.standardizedFileURL ?? defaultDirectory(fileManager: fileManager)
    guard fileManager.fileExists(atPath: directory.path) else { return [] }
    let urls = try fileManager.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    ).filter { $0.pathExtension.lowercased() == "json" }.sorted { $0.path < $1.path }
    return try urls.map { (try load(from: $0, fileManager: fileManager), $0.standardizedFileURL) }
  }

  public func validatePendingPostflight() throws {
    guard schema == Self.schemaVersion else {
      throw DeploymentRollbackJournalError.unsupportedSchema(schema)
    }
    try validateResolutionConsistency()
    guard isAwaitingOutdoorPostflight, !completed, rebootSent else {
      throw DeploymentRollbackJournalError.notAwaitingPostflight
    }
    guard let targetHead,
          targetHead.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
    else { throw DeploymentRollbackJournalError.invalidTargetHead(targetHead ?? "") }
  }

  public static func acquireCompletionLock(
    for journalURL: URL
  ) throws -> DeploymentRollbackJournalCompletionLock {
    try DeploymentRollbackJournalCompletionLock(journalURL: journalURL.standardizedFileURL)
  }

  /// Acquire the one process-wide production mutation owner for this journal
  /// namespace. Every path which can mutate the tici takes this lock before a
  /// per-journal resolution lock; this global -> journal ordering is the only
  /// permitted lock order.
  public static func acquireProductionOwnerLock(
    directory explicitDirectory: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> ProductionDeploymentOwnerLock {
    let requested = try explicitDirectory?.standardizedFileURL ?? defaultDirectory(fileManager: fileManager)
    try ensureDurableDirectory(requested, fileManager: fileManager)
    let authoritative = requested.resolvingSymlinksInPath().standardizedFileURL
    return try ProductionDeploymentOwnerLock(directory: authoritative)
  }

  /// Remove only states which prove that the current transaction never
  /// reached a remote mutation. A target-bearing legacy awaiting journal is
  /// deliberately retained and routed through guarded rollback because old
  /// app builds lacked the durable mutation claim.
  public static func removeAbandonedPreflightReservations(
    directory explicitDirectory: URL? = nil,
    excluding excludedURL: URL? = nil,
    excludingJournal excludedJournal: Self? = nil,
    fileManager: FileManager = .default,
    olderThan minimumAge: TimeInterval = 600,
    now: Date = Date()
  ) throws {
    let directory = try explicitDirectory?.standardizedFileURL ?? defaultDirectory(fileManager: fileManager)
    guard fileManager.fileExists(atPath: directory.path) else { return }
    let excluded = excludedURL?.standardizedFileURL
    for (journal, url) in try journalCandidates(directory: directory, fileManager: fileManager) {
      if url.standardizedFileURL == excluded { continue }
      if let excludedJournal, journal.hasSameDeploymentIdentity(as: excludedJournal) { continue }
      // The caller owns the canonical global production flock. Acquiring that
      // lock proves no prior new-version deployment process remains alive, so
      // either targetless or fully populated preflightReserved is safe to
      // delete: claimProductionMutation was the durable boundary before any
      // remote mutation was authorized.
      let orphanedCurrentPreflight = journal.effectiveResolution == .preflightReserved &&
        journal.completed && !journal.rebootSent
      if orphanedCurrentPreflight {
        try durablyRemoveJournal(at: url, fileManager: fileManager)
        continue
      }
      let targetlessLegacyPreflight = journal.effectiveResolution == .awaitingPostflight &&
        !journal.completed && !journal.rebootSent && journal.targetHead == nil
      guard targetlessLegacyPreflight,
            let created = ISO8601DateFormatter().date(from: journal.createdAt),
            now.timeIntervalSince(created) >= minimumAge else { continue }
      try durablyRemoveJournal(at: url, fileManager: fileManager)
    }
  }

  public static func removeOwnedPreflightReservation(
    matching expected: Self,
    at url: URL,
    fileManager: FileManager = .default
  ) throws {
    let current = try load(from: url, fileManager: fileManager)
    guard current == expected,
          current.effectiveResolution == .preflightReserved,
          current.completed,
          !current.rebootSent else { return }
    try durablyRemoveJournal(at: url, fileManager: fileManager)
  }

  private static func durablyRemoveJournal(at url: URL, fileManager: FileManager) throws {
    let directory = url.deletingLastPathComponent()
    do {
      try fileManager.removeItem(at: url)
      let directoryFD = Darwin.open(directory.path, O_RDONLY)
      guard directoryFD >= 0 else {
        throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
      }
      defer { Darwin.close(directoryFD) }
      guard Darwin.fsync(directoryFD) == 0 else {
        throw DeploymentRollbackJournalError.couldNotWrite(
          url,
          "journal directory fsync after preflight removal failed: \(String(cString: strerror(errno)))"
        )
      }
    } catch let error as DeploymentRollbackJournalError {
      throw error
    } catch {
      throw DeploymentRollbackJournalError.couldNotWrite(url, error.localizedDescription)
    }
  }

  @discardableResult
  public func write(
    to explicitURL: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> URL {
    let url: URL
    if let explicitURL {
      url = explicitURL.standardizedFileURL
    } else {
      url = try Self.defaultDirectory(fileManager: fileManager)
        .appendingPathComponent("\(deploymentID.uuidString).json")
    }
    let directory = url.deletingLastPathComponent()
    try Self.ensureDurableDirectory(directory, fileManager: fileManager)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(self)
    try Self.durablyReplace(data, at: url, fileManager: fileManager)
    return url
  }

  /// Make each transaction transition durable before any following device
  /// mutation: sync a same-directory sibling, atomically rename it, then sync
  /// the containing directory so crash recovery can trust the visible state.
  static func durablyReplace(
    _ data: Data,
    at url: URL,
    fileManager: FileManager = .default,
    directorySync: (Int32) -> Int32 = { Darwin.fsync($0) }
  ) throws {
    let directory = url.deletingLastPathComponent()
    let temporaryURL = directory.appendingPathComponent(
      ".\(url.lastPathComponent).\(UUID().uuidString.lowercased()).durable-tmp"
    )
    var temporaryFD: Int32 = -1
    var directoryFD: Int32 = -1
    defer {
      if temporaryFD >= 0 { Darwin.close(temporaryFD) }
      if directoryFD >= 0 { Darwin.close(directoryFD) }
      try? fileManager.removeItem(at: temporaryURL)
    }

    temporaryFD = Darwin.open(
      temporaryURL.path,
      O_WRONLY | O_CREAT | O_EXCL,
      S_IRUSR | S_IWUSR
    )
    guard temporaryFD >= 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
    }
    try data.withUnsafeBytes { rawBuffer in
      guard let base = rawBuffer.baseAddress else { return }
      var written = 0
      while written < rawBuffer.count {
        let count = Darwin.write(
          temporaryFD,
          base.advanced(by: written),
          rawBuffer.count - written
        )
        guard count > 0 else {
          throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
        }
        written += count
      }
    }
    guard Darwin.fsync(temporaryFD) == 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
    }
    guard Darwin.fcntl(temporaryFD, F_FULLFSYNC) == 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(
        url,
        "F_FULLFSYNC failed: \(String(cString: strerror(errno)))"
      )
    }
    guard Darwin.close(temporaryFD) == 0 else {
      temporaryFD = -1
      throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
    }
    temporaryFD = -1
    guard Darwin.rename(temporaryURL.path, url.path) == 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(url, String(cString: strerror(errno)))
    }
    directoryFD = Darwin.open(directory.path, O_RDONLY)
    guard directoryFD >= 0 else {
      throw DeploymentRollbackJournalError.committedButNotDurable(
        url,
        "could not open containing directory for fsync: \(String(cString: strerror(errno))); exact_visible=\(((try? Data(contentsOf: url)) == data) ? 1 : 0)"
      )
    }
    var directorySyncSucceeded = false
    var directorySyncDetail = ""
    for _ in 0..<3 {
      if directorySync(directoryFD) == 0 {
        directorySyncSucceeded = true
        break
      }
      directorySyncDetail = String(cString: strerror(errno))
    }
    guard directorySyncSucceeded else {
      // rename(2) has already made the new journal visible, but readback alone
      // cannot prove crash durability. Rollback callers must treat this as a
      // hard barrier before any device mutation.
      throw DeploymentRollbackJournalError.committedButNotDurable(
        url,
        "directory fsync failed after three attempts: \(directorySyncDetail); exact_visible=\(((try? Data(contentsOf: url)) == data) ? 1 : 0)"
      )
    }
  }

  /// Persist directory entries created for a journal before the journal can
  /// become the durable recovery authority for a following device mutation.
  private static func ensureDurableDirectory(
    _ directory: URL,
    fileManager: FileManager
  ) throws {
    var missing: [URL] = []
    var cursor = directory.standardizedFileURL
    while !fileManager.fileExists(atPath: cursor.path) {
      missing.append(cursor)
      let parent = cursor.deletingLastPathComponent()
      guard parent.path != cursor.path else {
        throw DeploymentRollbackJournalError.couldNotWrite(
          directory,
          "could not find an existing parent directory"
        )
      }
      cursor = parent
    }
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    for created in missing.reversed() {
      try syncDirectoryEntry(created, journalURL: directory)
    }
  }

  private static func syncDirectoryEntry(_ createdDirectory: URL, journalURL: URL) throws {
    let parent = createdDirectory.deletingLastPathComponent()
    let fd = Darwin.open(parent.path, O_RDONLY)
    guard fd >= 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(journalURL, String(cString: strerror(errno)))
    }
    defer { Darwin.close(fd) }
    guard Darwin.fsync(fd) == 0 else {
      throw DeploymentRollbackJournalError.couldNotWrite(
        journalURL,
        "parent directory fsync failed: \(String(cString: strerror(errno)))"
      )
    }
  }
}

public enum DeploymentRollbackJournalError: LocalizedError, Equatable, Sendable {
  case couldNotLoad(URL, String)
  case couldNotList(URL, String)
  case unsupportedSchema(Int)
  case noPendingPostflight
  case multiplePendingPostflights([URL])
  case notAwaitingPostflight
  case invalidTargetHead(String)
  case productionOwnerLocked(URL)
  case completionLocked(URL)
  case couldNotLock(URL, String)
  case inconsistentResolution(String)
  case couldNotWrite(URL, String)
  case committedButNotDurable(URL, String)
  case noRecoverableRollback
  case multipleRecoverableRollbacks([URL])
  case notRecoverableRollback

  public var errorDescription: String? {
    switch self {
    case let .couldNotLoad(url, reason):
      "Could not load the deployment journal at \(url.path): \(reason)"
    case let .couldNotList(url, reason):
      "Could not inspect deployment journals at \(url.path): \(reason)"
    case let .unsupportedSchema(schema):
      "The deployment journal uses unsupported schema \(schema); expected schema \(DeploymentRollbackJournal.schemaVersion)."
    case .noPendingPostflight:
      "No rebooted, incomplete deployment journal is waiting for outdoor postflight."
    case let .multiplePendingPostflights(urls):
      "More than one deployment is waiting for postflight; refusing to choose between: \(urls.map(\.lastPathComponent).joined(separator: ", "))."
    case .notAwaitingPostflight:
      "The selected deployment journal is not an incomplete deployment that has already rebooted."
    case let .invalidTargetHead(head):
      "The pending deployment journal has an invalid exact target head: \(head)."
    case let .productionOwnerLocked(url):
      "Another VTSC Tuner instance owns the global production transaction at \(url.path). No car-facing mutation was attempted."
    case let .completionLocked(url):
      "Another VTSC Tuner instance already owns the deployment journal transaction at \(url.path)."
    case let .couldNotLock(url, reason):
      "Could not lock the deployment journal at \(url.path): \(reason)"
    case let .inconsistentResolution(reason):
      "The deployment journal resolution is inconsistent: \(reason)."
    case let .couldNotWrite(url, reason):
      "Could not durably write the deployment journal at \(url.path): \(reason)"
    case let .committedButNotDurable(url, reason):
      "The deployment journal transition is visible but crash durability is unconfirmed at \(url.path): \(reason)"
    case .noRecoverableRollback:
      "No interrupted or failed production rollback is waiting for guarded recovery."
    case let .multipleRecoverableRollbacks(urls):
      "More than one rollback requires recovery; refusing to choose between: \(urls.map(\.lastPathComponent).joined(separator: ", "))."
    case .notRecoverableRollback:
      "The selected journal does not contain a complete recoverable deployment identity."
    }
  }
}

/// One stable advisory lock per authoritative deployment-journal namespace.
/// Every device-mutating production path acquires this lock before any
/// per-journal resolution lock. It is retained from atomic scan/reservation
/// through durable post-reboot handoff or rollback terminal settlement.
public final class ProductionDeploymentOwnerLock: @unchecked Sendable {
  public let directoryURL: URL
  public let lockURL: URL
  private var fileDescriptor: Int32

  fileprivate init(directory: URL) throws {
    directoryURL = directory.standardizedFileURL
    lockURL = directoryURL.appendingPathComponent(".production-owner.lock").standardizedFileURL
    fileDescriptor = Darwin.open(lockURL.path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
    guard fileDescriptor >= 0 else {
      throw DeploymentRollbackJournalError.couldNotLock(lockURL, String(cString: strerror(errno)))
    }
    guard vtscSystemFlock(fileDescriptor, LOCK_EX | LOCK_NB) == 0 else {
      let errorNumber = errno
      Darwin.close(fileDescriptor)
      fileDescriptor = -1
      if errorNumber == EWOULDBLOCK {
        throw DeploymentRollbackJournalError.productionOwnerLocked(lockURL)
      }
      throw DeploymentRollbackJournalError.couldNotLock(lockURL, String(cString: strerror(errorNumber)))
    }
  }

  public func unlock() {
    guard fileDescriptor >= 0 else { return }
    _ = vtscSystemFlock(fileDescriptor, LOCK_UN)
    Darwin.close(fileDescriptor)
    fileDescriptor = -1
  }

  deinit { unlock() }
}

/// A stable sibling advisory lock shared by every app instance. File presence
/// is harmless; only the kernel-held flock owns the completion-or-rollback
/// transaction. The lock is intentionally nonblocking so a second app fails
/// closed instead of acting on evidence gathered before another owner settled
/// the deployment journal.
public final class DeploymentRollbackJournalCompletionLock: @unchecked Sendable {
  public let lockURL: URL
  private var fileDescriptor: Int32

  fileprivate init(journalURL: URL) throws {
    lockURL = URL(fileURLWithPath: journalURL.path + ".lock").standardizedFileURL
    fileDescriptor = Darwin.open(lockURL.path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
    guard fileDescriptor >= 0 else {
      throw DeploymentRollbackJournalError.couldNotLock(journalURL, String(cString: strerror(errno)))
    }
    guard vtscSystemFlock(fileDescriptor, LOCK_EX | LOCK_NB) == 0 else {
      let errorNumber = errno
      Darwin.close(fileDescriptor)
      fileDescriptor = -1
      if errorNumber == EWOULDBLOCK {
        throw DeploymentRollbackJournalError.completionLocked(journalURL)
      }
      throw DeploymentRollbackJournalError.couldNotLock(journalURL, String(cString: strerror(errorNumber)))
    }
  }

  public func unlock() {
    guard fileDescriptor >= 0 else { return }
    _ = vtscSystemFlock(fileDescriptor, LOCK_UN)
    Darwin.close(fileDescriptor)
    fileDescriptor = -1
  }

  deinit { unlock() }
}
