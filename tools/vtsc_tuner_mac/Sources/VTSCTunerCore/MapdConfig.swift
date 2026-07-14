import Foundation

public struct RegionBox: Codable, Hashable, Sendable {
  public var minLatitude: Int
  public var minLongitude: Int

  public init(minLatitude: Int, minLongitude: Int) {
    self.minLatitude = minLatitude
    self.minLongitude = minLongitude
  }

  enum CodingKeys: String, CodingKey {
    case minLatitude = "min_lat"
    case minLongitude = "min_lon"
  }

  public func validate() throws {
    guard (-90 ... 88).contains(minLatitude), (-180 ... 178).contains(minLongitude),
          minLatitude.isMultiple(of: 2), minLongitude.isMultiple(of: 2)
    else {
      throw MapdConfigError.invalidRegion(self)
    }
  }
}

public struct MapdConfig: Codable, Equatable, Sendable {
  public var pbfURL: URL
  public var mapdRepositoryURL: URL
  public var mapdBinaryURL: URL?
  public var regionsOverride: [RegionBox]?

  public init(
    pbfURL: URL,
    mapdRepositoryURL: URL,
    mapdBinaryURL: URL? = nil,
    regionsOverride: [RegionBox]? = nil
  ) {
    self.pbfURL = pbfURL
    self.mapdRepositoryURL = mapdRepositoryURL
    self.mapdBinaryURL = mapdBinaryURL
    self.regionsOverride = regionsOverride
  }

  enum CodingKeys: String, CodingKey {
    case pbfURL = "pbf_path"
    case mapdRepositoryURL = "mapd_repo_path"
    case mapdBinaryURL = "mapd_binary_path"
    case regionsOverride = "regions_override"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    pbfURL = URL(fileURLWithPath: try container.decode(String.self, forKey: .pbfURL))
    mapdRepositoryURL = URL(
      fileURLWithPath: try container.decode(String.self, forKey: .mapdRepositoryURL)
    )
    if let binaryPath = try container.decodeIfPresent(String.self, forKey: .mapdBinaryURL) {
      mapdBinaryURL = URL(fileURLWithPath: binaryPath)
    } else {
      mapdBinaryURL = nil
    }
    regionsOverride = try container.decodeIfPresent([RegionBox].self, forKey: .regionsOverride)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(pbfURL.path, forKey: .pbfURL)
    try container.encode(mapdRepositoryURL.path, forKey: .mapdRepositoryURL)
    try container.encodeIfPresent(mapdBinaryURL?.path, forKey: .mapdBinaryURL)
    try container.encodeIfPresent(regionsOverride, forKey: .regionsOverride)
  }

  public var effectiveBinaryURL: URL {
    mapdBinaryURL ?? mapdRepositoryURL.appendingPathComponent("build/mapd", isDirectory: false)
  }

  public static func load(
    from explicitURL: URL? = nil,
    fileManager: FileManager = .default
  ) throws -> MapdConfig {
    let url = try explicitURL ?? TuneStore.mapdConfigURL(fileManager: fileManager)
    do {
      return try JSONDecoder().decode(MapdConfig.self, from: Data(contentsOf: url))
    } catch {
      throw MapdConfigError.couldNotLoad(url, error.localizedDescription)
    }
  }

  public func validated(fileManager: FileManager = .default) throws -> MapdConfig {
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: pbfURL.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
      throw MapdConfigError.missingPBF(pbfURL)
    }
    let capnpURL = mapdRepositoryURL.appendingPathComponent("offline.capnp", isDirectory: false)
    guard fileManager.fileExists(atPath: capnpURL.path) else {
      throw MapdConfigError.invalidRepository(mapdRepositoryURL)
    }
    for region in regionsOverride ?? [] { try region.validate() }
    return self
  }

  /// Rebuild generation runs this binary directly on the Mac. An ELF binary
  /// emitted by Earthly is therefore rejected even when its executable bit is set.
  public func validateDarwinBinary(fileManager: FileManager = .default) throws {
    let binaryURL = effectiveBinaryURL
    guard fileManager.fileExists(atPath: binaryURL.path),
          fileManager.isExecutableFile(atPath: binaryURL.path)
    else {
      throw MapdConfigError.missingBinary(binaryURL)
    }
    let data = try Data(contentsOf: binaryURL, options: [.mappedIfSafe])
    guard data.count >= 4 else { throw MapdConfigError.notDarwinBinary(binaryURL) }
    let magic = Array(data.prefix(4))
    let machOMagic: Set<[UInt8]> = [
      [0xcf, 0xfa, 0xed, 0xfe], [0xfe, 0xed, 0xfa, 0xcf],
      [0xce, 0xfa, 0xed, 0xfe], [0xfe, 0xed, 0xfa, 0xce],
      [0xca, 0xfe, 0xba, 0xbe], [0xbe, 0xba, 0xfe, 0xca],
      [0xca, 0xfe, 0xba, 0xbf], [0xbf, 0xba, 0xfe, 0xca],
    ]
    guard machOMagic.contains(magic) else { throw MapdConfigError.notDarwinBinary(binaryURL) }
  }
}

public enum MapdConfigError: LocalizedError, Equatable, Sendable {
  case couldNotLoad(URL, String)
  case missingPBF(URL)
  case invalidRepository(URL)
  case missingBinary(URL)
  case notDarwinBinary(URL)
  case invalidRegion(RegionBox)

  public var errorDescription: String? {
    switch self {
    case let .couldNotLoad(url, reason):
      "Could not load \(url.path): \(reason)"
    case let .missingPBF(url):
      "The configured PBF does not exist: \(url.path)"
    case let .invalidRepository(url):
      "The configured mapd checkout has no offline.capnp: \(url.path)"
    case let .missingBinary(url):
      "A native macOS mapd generator is required at \(url.path). Configure mapd_binary_path to a Darwin executable."
    case let .notDarwinBinary(url):
      "The configured mapd generator is not a Darwin Mach-O executable: \(url.path)"
    case let .invalidRegion(region):
      "Region (\(region.minLatitude), \(region.minLongitude)) is not a valid 2-degree map box."
    }
  }
}
