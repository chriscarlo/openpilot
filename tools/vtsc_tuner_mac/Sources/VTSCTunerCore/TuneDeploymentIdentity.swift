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

  public var schema: Int
  public var deploymentID: UUID
  public var createdAt: String
  public var profile: String
  public var branch: String
  public var previousHead: String
  public var targetHead: String?
  public var previousPhysicsParams: [String: String?]
  public var previousQCurveSHA256: String
  public var previousMapdReleaseVersion: String?
  public var previousMapdVersion: String?
  public var previousActiveMapdSHA256: String
  public var previousCachedMapdPath: String
  public var previousCachedMapdSHA256: String?
  public var mapdRollbackPath: String
  public var previousTileSetID: String?
  public var targetTileSetID: String?
  public var rebootSent: Bool
  public var completed: Bool
  public var completedAt: String?
  public var completedToolingHead: String?
  public var completionHostOnlyPaths: [String]?

  public init(
    deploymentID: UUID = UUID(),
    createdAt: String = Date().ISO8601Format(),
    profile: String,
    branch: String,
    previousHead: String,
    targetHead: String? = nil,
    previousPhysicsParams: [String: String?],
    previousQCurveSHA256: String,
    previousMapdReleaseVersion: String?,
    previousMapdVersion: String?,
    previousActiveMapdSHA256: String,
    previousCachedMapdPath: String,
    previousCachedMapdSHA256: String? = nil,
    mapdRollbackPath: String,
    previousTileSetID: String? = nil,
    targetTileSetID: String? = nil,
    rebootSent: Bool = false,
    completed: Bool = false,
    completedAt: String? = nil,
    completedToolingHead: String? = nil,
    completionHostOnlyPaths: [String]? = nil
  ) {
    schema = Self.schemaVersion
    self.deploymentID = deploymentID
    self.createdAt = createdAt
    self.profile = profile
    self.branch = branch
    self.previousHead = previousHead
    self.targetHead = targetHead
    self.previousPhysicsParams = previousPhysicsParams
    self.previousQCurveSHA256 = previousQCurveSHA256
    self.previousMapdReleaseVersion = previousMapdReleaseVersion
    self.previousMapdVersion = previousMapdVersion
    self.previousActiveMapdSHA256 = previousActiveMapdSHA256
    self.previousCachedMapdPath = previousCachedMapdPath
    self.previousCachedMapdSHA256 = previousCachedMapdSHA256
    self.mapdRollbackPath = mapdRollbackPath
    self.previousTileSetID = previousTileSetID
    self.targetTileSetID = targetTileSetID
    self.rebootSent = rebootSent
    self.completed = completed
    self.completedAt = completedAt
    self.completedToolingHead = completedToolingHead
    self.completionHostOnlyPaths = completionHostOnlyPaths
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
      guard !journal.completed, journal.rebootSent else { continue }
      try journal.validatePendingPostflight()
      candidates.append((journal, url.standardizedFileURL))
    }
    guard !candidates.isEmpty else { throw DeploymentRollbackJournalError.noPendingPostflight }
    guard candidates.count == 1 else {
      throw DeploymentRollbackJournalError.multiplePendingPostflights(candidates.map { $0.1 })
    }
    return (candidates[0].0, candidates[0].1)
  }

  public func validatePendingPostflight() throws {
    guard schema == Self.schemaVersion else {
      throw DeploymentRollbackJournalError.unsupportedSchema(schema)
    }
    guard !completed, rebootSent else {
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
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(self)
    // Foundation writes a sibling temporary file and atomically renames it over
    // the destination. There is never a remove-then-move gap where a pending
    // production journal can disappear between validation and completion.
    try data.write(to: url, options: .atomic)
    return url
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
  case completionLocked(URL)
  case couldNotLock(URL, String)

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
    case let .completionLocked(url):
      "Another VTSC Tuner instance is already completing the deployment journal at \(url.path)."
    case let .couldNotLock(url, reason):
      "Could not lock the deployment journal at \(url.path): \(reason)"
    }
  }
}

/// A stable sibling advisory lock shared by every app instance. File presence
/// is harmless; only the kernel-held flock owns the completion critical
/// section. The lock is intentionally nonblocking so a second app fails closed
/// instead of waiting on stale evidence gathered before another completion.
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
