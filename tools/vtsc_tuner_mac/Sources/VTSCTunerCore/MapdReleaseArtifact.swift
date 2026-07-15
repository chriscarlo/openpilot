import CryptoKit
import Foundation

public struct MapdReleaseArtifact: Codable, Equatable, Sendable {
  public static let defaultCapability = "MapWholeCurveProfile:whole-curve-v1"
  public static let defaultEstimatorVersion = "whole-curve-v1"

  public var releaseID: String
  public var buildID: String
  public var estimatorVersion: String
  public var capability: String
  public var binaryURL: URL
  public var sha256: String

  public init(
    releaseID: String,
    buildID: String,
    estimatorVersion: String = Self.defaultEstimatorVersion,
    capability: String = Self.defaultCapability,
    binaryURL: URL,
    sha256: String
  ) {
    self.releaseID = releaseID
    self.buildID = buildID
    self.estimatorVersion = estimatorVersion
    self.capability = capability
    self.binaryURL = binaryURL
    self.sha256 = sha256.lowercased()
  }

  enum CodingKeys: String, CodingKey {
    case releaseID = "release_id"
    case buildID = "build_id"
    case estimatorVersion = "estimator_version"
    case capability
    case binaryURL = "binary_path"
    case sha256
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    releaseID = try container.decode(String.self, forKey: .releaseID)
    buildID = try container.decode(String.self, forKey: .buildID)
    estimatorVersion = try container.decodeIfPresent(String.self, forKey: .estimatorVersion)
      ?? Self.defaultEstimatorVersion
    capability = try container.decodeIfPresent(String.self, forKey: .capability)
      ?? Self.defaultCapability
    binaryURL = URL(fileURLWithPath: try container.decode(String.self, forKey: .binaryURL))
    sha256 = try container.decode(String.self, forKey: .sha256).lowercased()
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(releaseID, forKey: .releaseID)
    try container.encode(buildID, forKey: .buildID)
    try container.encode(estimatorVersion, forKey: .estimatorVersion)
    try container.encode(capability, forKey: .capability)
    try container.encode(binaryURL.path, forKey: .binaryURL)
    try container.encode(sha256, forKey: .sha256)
  }

  public static func defaultManifestURL(fileManager: FileManager = .default) throws -> URL {
    try TuneStore.applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("mapd-release.json", isDirectory: false)
  }

  public static func load(
    from explicitURL: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> Self {
    let url = try explicitURL ?? defaultManifestURL(fileManager: fileManager)
    do {
      return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    } catch {
      throw MapdReleaseArtifactError.couldNotLoad(url, error.localizedDescription)
    }
  }

  public func validated(fileManager: FileManager = .default) throws -> ValidatedMapdReleaseArtifact {
    guard Self.isSafeIdentifier(releaseID) else {
      throw MapdReleaseArtifactError.invalidIdentifier("release_id", releaseID)
    }
    guard Self.isSafeIdentifier(buildID) else {
      throw MapdReleaseArtifactError.invalidIdentifier("build_id", buildID)
    }
    guard Self.isSafeMarker(estimatorVersion) else {
      throw MapdReleaseArtifactError.invalidIdentifier("estimator_version", estimatorVersion)
    }
    guard Self.isSafeMarker(capability) else {
      throw MapdReleaseArtifactError.invalidIdentifier("capability", capability)
    }
    guard sha256.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil else {
      throw MapdReleaseArtifactError.invalidDigest(sha256)
    }

    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: binaryURL.path, isDirectory: &isDirectory),
          !isDirectory.boolValue
    else {
      throw MapdReleaseArtifactError.missingBinary(binaryURL)
    }
    let data = try Data(contentsOf: binaryURL, options: [.mappedIfSafe])
    guard data.count >= 20 else { throw MapdReleaseArtifactError.notELF64ARM64(binaryURL) }
    guard Array(data.prefix(4)) == [0x7f, 0x45, 0x4c, 0x46],
          data[4] == 2, // ELFCLASS64
          data[5] == 1, // little endian
          (UInt16(data[18]) | (UInt16(data[19]) << 8)) == 183 // EM_AARCH64
    else {
      throw MapdReleaseArtifactError.notELF64ARM64(binaryURL)
    }

    let actualDigest = Self.sha256Hex(data)
    guard actualDigest == sha256 else {
      throw MapdReleaseArtifactError.digestMismatch(expected: sha256, actual: actualDigest)
    }
    for marker in ["MapdReleaseID:\(releaseID)", "MapdBuildID:\(buildID)", capability] {
      guard data.range(of: Data(marker.utf8)) != nil else {
        throw MapdReleaseArtifactError.missingMarker(marker)
      }
    }
    return ValidatedMapdReleaseArtifact(
      artifact: self,
      byteCount: UInt64(data.count),
      persistentCacheFileName: [
        "mapd",
        String(Self.sha256Hex(Data(releaseID.utf8)).prefix(16)),
        String(sha256.prefix(16)),
      ].joined(separator: "-")
    )
  }

  public static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  public static func sha256Hex(fileURL: URL) throws -> String {
    try sha256Hex(Data(contentsOf: fileURL, options: [.mappedIfSafe]))
  }

  private static func isSafeIdentifier(_ value: String) -> Bool {
    !value.isEmpty && value.range(of: #"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$"#, options: .regularExpression) != nil
  }

  private static func isSafeMarker(_ value: String) -> Bool {
    !value.isEmpty && value.range(
      of: #"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$"#,
      options: .regularExpression
    ) != nil
  }
}

public struct ValidatedMapdReleaseArtifact: Equatable, Sendable {
  public var artifact: MapdReleaseArtifact
  public var byteCount: UInt64
  public var persistentCacheFileName: String

  public init(artifact: MapdReleaseArtifact, byteCount: UInt64, persistentCacheFileName: String) {
    self.artifact = artifact
    self.byteCount = byteCount
    self.persistentCacheFileName = persistentCacheFileName
  }
}

public enum MapdReleaseArtifactError: LocalizedError, Equatable, Sendable {
  case couldNotLoad(URL, String)
  case invalidIdentifier(String, String)
  case invalidDigest(String)
  case missingBinary(URL)
  case notELF64ARM64(URL)
  case digestMismatch(expected: String, actual: String)
  case missingMarker(String)

  public var errorDescription: String? {
    switch self {
    case let .couldNotLoad(url, reason):
      "Could not load the mapd release manifest at \(url.path): \(reason)"
    case let .invalidIdentifier(field, value):
      "The mapd release \(field) is invalid: \(value)"
    case let .invalidDigest(value):
      "The mapd release SHA-256 is invalid: \(value)"
    case let .missingBinary(url):
      "The ARM64 mapd release binary does not exist: \(url.path)"
    case let .notELF64ARM64(url):
      "The mapd release binary is not a little-endian ELF64 ARM64 executable: \(url.path)"
    case let .digestMismatch(expected, actual):
      "The mapd release binary SHA-256 is \(actual), expected \(expected)."
    case let .missingMarker(marker):
      "The mapd release binary does not advertise required marker \(marker)."
    }
  }
}
