import CryptoKit
import Foundation

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
    completed: Bool = false
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
  }

  public static func defaultDirectory(fileManager: FileManager = .default) throws -> URL {
    try TuneStore.applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("deployment-journals", isDirectory: true)
  }

  @discardableResult
  public func write(
    to explicitURL: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> URL {
    let directory = try Self.defaultDirectory(fileManager: fileManager)
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = explicitURL ?? directory.appendingPathComponent("\(deploymentID.uuidString).json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(self)
    let temporary = url.deletingLastPathComponent()
      .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
    try data.write(to: temporary, options: .atomic)
    do {
      if fileManager.fileExists(atPath: url.path) { try fileManager.removeItem(at: url) }
      try fileManager.moveItem(at: temporary, to: url)
    } catch {
      try? fileManager.removeItem(at: temporary)
      throw error
    }
    return url
  }
}
