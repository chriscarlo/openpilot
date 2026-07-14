import CryptoKit
import Foundation

public struct TileSetManifest: Codable, Equatable, Sendable {
  public static let schemaVersion = 1
  public static let fileName = "manifest.json"

  public struct FileEntry: Codable, Equatable, Sendable {
    public var path: String
    public var byteCount: UInt64
    public var sha256: String

    public init(path: String, byteCount: UInt64, sha256: String) {
      self.path = path
      self.byteCount = byteCount
      self.sha256 = sha256.lowercased()
    }

    enum CodingKeys: String, CodingKey {
      case path
      case byteCount = "byte_count"
      case sha256
    }
  }

  public var schema: Int
  public var tileSetID: String
  public var createdAt: String
  public var tileSchemaVersion: UInt16
  public var estimatorVersion: String
  public var tuneIdentitySHA256: String
  public var tileSigmoidHash: String
  public var mapdReleaseID: String
  public var mapdBuildID: String
  public var mapdBinarySHA256: String
  public var pbfSHA256: String
  public var pbfDate: String
  public var regions: [RegionBox]
  public var fileCount: Int
  public var totalBytes: UInt64
  public var files: [FileEntry]

  public init(
    tileSetID: String = "",
    createdAt: String = Date().ISO8601Format(),
    tileSchemaVersion: UInt16,
    estimatorVersion: String,
    tuneIdentitySHA256: String,
    tileSigmoidHash: String,
    mapdReleaseID: String,
    mapdBuildID: String,
    mapdBinarySHA256: String,
    pbfSHA256: String,
    pbfDate: String,
    regions: [RegionBox],
    files: [FileEntry]
  ) {
    schema = Self.schemaVersion
    self.tileSetID = tileSetID
    self.createdAt = createdAt
    self.tileSchemaVersion = tileSchemaVersion
    self.estimatorVersion = estimatorVersion
    self.tuneIdentitySHA256 = tuneIdentitySHA256.lowercased()
    self.tileSigmoidHash = tileSigmoidHash.lowercased()
    self.mapdReleaseID = mapdReleaseID
    self.mapdBuildID = mapdBuildID
    self.mapdBinarySHA256 = mapdBinarySHA256.lowercased()
    self.pbfSHA256 = pbfSHA256.lowercased()
    self.pbfDate = pbfDate
    self.regions = Self.normalized(regions)
    self.files = files.sorted { $0.path < $1.path }
    fileCount = files.count
    totalBytes = files.reduce(0) { $0 + $1.byteCount }
    if tileSetID.isEmpty { self.tileSetID = identitySHA256() }
  }

  enum CodingKeys: String, CodingKey {
    case schema
    case tileSetID = "tile_set_id"
    case createdAt = "created_at"
    case tileSchemaVersion = "tile_schema_version"
    case estimatorVersion = "estimator_version"
    case tuneIdentitySHA256 = "tune_identity_sha256"
    case tileSigmoidHash = "tile_sigmoid_hash"
    case mapdReleaseID = "mapd_release_id"
    case mapdBuildID = "mapd_build_id"
    case mapdBinarySHA256 = "mapd_binary_sha256"
    case pbfSHA256 = "pbf_sha256"
    case pbfDate = "pbf_date"
    case regions
    case fileCount = "file_count"
    case totalBytes = "total_bytes"
    case files
  }

  public func identitySHA256() -> String {
    struct Identity: Encodable {
      var schema: Int
      var tileSchemaVersion: UInt16
      var estimatorVersion: String
      var tuneIdentitySHA256: String
      var tileSigmoidHash: String
      var mapdReleaseID: String
      var mapdBuildID: String
      var mapdBinarySHA256: String
      var pbfSHA256: String
      var pbfDate: String
      var regions: [RegionBox]
      var files: [FileEntry]

      enum CodingKeys: String, CodingKey {
        case schema
        case tileSchemaVersion = "tile_schema_version"
        case estimatorVersion = "estimator_version"
        case tuneIdentitySHA256 = "tune_identity_sha256"
        case tileSigmoidHash = "tile_sigmoid_hash"
        case mapdReleaseID = "mapd_release_id"
        case mapdBuildID = "mapd_build_id"
        case mapdBinarySHA256 = "mapd_binary_sha256"
        case pbfSHA256 = "pbf_sha256"
        case pbfDate = "pbf_date"
        case regions, files
      }
    }
    let identity = Identity(
      schema: schema,
      tileSchemaVersion: tileSchemaVersion,
      estimatorVersion: estimatorVersion,
      tuneIdentitySHA256: tuneIdentitySHA256,
      tileSigmoidHash: tileSigmoidHash,
      mapdReleaseID: mapdReleaseID,
      mapdBuildID: mapdBuildID,
      mapdBinarySHA256: mapdBinarySHA256,
      pbfSHA256: pbfSHA256,
      pbfDate: pbfDate,
      regions: Self.normalized(regions),
      files: files.sorted { $0.path < $1.path }
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = (try? encoder.encode(identity)) ?? Data()
    return TuneDeploymentIdentity.sha256Hex(data)
  }

  public func validatedMetadata() throws -> Self {
    guard schema == Self.schemaVersion else { throw TileSetError.unsupportedManifestSchema(schema) }
    guard tileSchemaVersion == 1 else { throw TileSetError.unsupportedTileSchema(tileSchemaVersion) }
    guard !files.isEmpty, fileCount == files.count else {
      throw TileSetError.fileCountMismatch(expected: fileCount, actual: files.count)
    }
    guard totalBytes == files.reduce(0, { $0 + $1.byteCount }) else {
      throw TileSetError.totalBytesMismatch(expected: totalBytes, actual: files.reduce(0, { $0 + $1.byteCount }))
    }
    for digest in [
      tileSetID, tuneIdentitySHA256, mapdBinarySHA256, pbfSHA256,
    ] where !Self.isSHA256(digest) {
      throw TileSetError.invalidDigest(digest)
    }
    guard tileSigmoidHash.range(of: #"^[0-9a-f]{12}$"#, options: .regularExpression) != nil else {
      throw TileSetError.invalidDigest(tileSigmoidHash)
    }
    guard estimatorVersion == MapdReleaseArtifact.defaultEstimatorVersion else {
      throw TileSetError.estimatorMismatch(
        expected: MapdReleaseArtifact.defaultEstimatorVersion,
        actual: estimatorVersion
      )
    }
    guard !mapdReleaseID.isEmpty, !mapdBuildID.isEmpty, !pbfDate.isEmpty else {
      throw TileSetError.invalidMetadata
    }
    let paths = files.map(\.path)
    guard paths == paths.sorted(), Set(paths).count == paths.count else {
      throw TileSetError.nonCanonicalFileOrder
    }
    for file in files {
      guard Self.isSafeRelativeTilePath(file.path) else { throw TileSetError.unsafePath(file.path) }
      guard Self.isSHA256(file.sha256) else { throw TileSetError.invalidDigest(file.sha256) }
      guard file.byteCount > 64 else { throw TileSetError.placeholderTile(file.path, file.byteCount) }
    }
    let normalizedRegions = Self.normalized(regions)
    guard !normalizedRegions.isEmpty, normalizedRegions == regions else { throw TileSetError.nonCanonicalRegions }
    for region in regions { try region.validate() }
    let fileRegions = try files.map { file -> RegionBox in
      let parts = file.path.split(separator: "/")
      guard parts.count >= 4, let latitude = Int(parts[1]), let longitude = Int(parts[2]) else {
        throw TileSetError.unsafePath(file.path)
      }
      let region = RegionBox(minLatitude: latitude, minLongitude: longitude)
      try region.validate()
      return region
    }
    guard Self.normalized(fileRegions) == regions else { throw TileSetError.regionFileMismatch }
    guard identitySHA256() == tileSetID else {
      throw TileSetError.identityMismatch(expected: tileSetID, actual: identitySHA256())
    }
    return self
  }

  public static func load(from artifactRootURL: URL) throws -> Self {
    let manifestURL = artifactRootURL.appendingPathComponent(Self.fileName)
    do {
      return try JSONDecoder().decode(Self.self, from: Data(contentsOf: manifestURL)).validatedMetadata()
    } catch let error as TileSetError {
      throw error
    } catch {
      throw TileSetError.couldNotLoadManifest(manifestURL, error.localizedDescription)
    }
  }

  public func write(to artifactRootURL: URL, fileManager: FileManager = .default) throws {
    _ = try validatedMetadata()
    try fileManager.createDirectory(at: artifactRootURL, withIntermediateDirectories: true)
    let url = artifactRootURL.appendingPathComponent(Self.fileName)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(self).write(to: url, options: .atomic)
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil
  }

  fileprivate static func isSafeRelativeTilePath(_ path: String) -> Bool {
    guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0") else { return false }
    let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
    guard components.count >= 4, components.first == "offline",
          components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
    else { return false }
    return Int(components[1]) != nil && Int(components[2]) != nil
  }

  private static func normalized(_ regions: [RegionBox]) -> [RegionBox] {
    Array(Set(regions)).sorted {
      ($0.minLatitude, $0.minLongitude) < ($1.minLatitude, $1.minLongitude)
    }
  }
}

public struct CanonicalTileSetArtifact: Equatable, Sendable {
  public var rootURL: URL
  public var manifest: TileSetManifest

  public init(rootURL: URL, manifest: TileSetManifest) {
    self.rootURL = rootURL.standardizedFileURL
    self.manifest = manifest
  }

  public var offlineURL: URL { rootURL.appendingPathComponent("offline", isDirectory: true) }
  public var manifestURL: URL { rootURL.appendingPathComponent(TileSetManifest.fileName) }

  public static func load(from rootURL: URL) throws -> Self {
    Self(rootURL: rootURL, manifest: try TileSetManifest.load(from: rootURL))
  }
}

public struct CanonicalTileSetBuildRequest: Sendable {
  public var sourceOfflineURL: URL
  public var setsRootURL: URL
  public var release: ValidatedMapdReleaseArtifact
  public var tuneIdentity: TuneDeploymentIdentity
  public var tileSigmoidHash: String
  public var pbfURL: URL
  public var pbfDate: String
  public var regions: [RegionBox]

  public init(
    sourceOfflineURL: URL,
    setsRootURL: URL,
    release: ValidatedMapdReleaseArtifact,
    tuneIdentity: TuneDeploymentIdentity,
    tileSigmoidHash: String,
    pbfURL: URL,
    pbfDate: String,
    regions: [RegionBox]
  ) {
    self.sourceOfflineURL = sourceOfflineURL
    self.setsRootURL = setsRootURL
    self.release = release
    self.tuneIdentity = tuneIdentity
    self.tileSigmoidHash = tileSigmoidHash.lowercased()
    self.pbfURL = pbfURL
    self.pbfDate = pbfDate
    self.regions = regions
  }
}

public struct TileSetValidator: Sendable {
  private let decoder: any MapTileDecoding
  private let batchSize: Int

  public init(decoder: any MapTileDecoding, batchSize: Int = 64) {
    self.decoder = decoder
    self.batchSize = max(1, batchSize)
  }

  @discardableResult
  public func validate(
    _ artifact: CanonicalTileSetArtifact,
    fileManager: FileManager = .default
  ) async throws -> CanonicalTileSetArtifact {
    let manifest = try artifact.manifest.validatedMetadata()
    let actualFiles = try Self.enumerateFiles(rootURL: artifact.rootURL, fileManager: fileManager)
    let expectedPaths = manifest.files.map(\.path)
    let actualPaths = actualFiles.map(\.relativePath)
    guard expectedPaths == actualPaths else {
      throw TileSetError.fileSetMismatch(expected: expectedPaths, actual: actualPaths)
    }
    let expectedByPath = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.path, $0) })
    for file in actualFiles {
      guard let expected = expectedByPath[file.relativePath] else {
        throw TileSetError.unexpectedFile(file.relativePath)
      }
      guard file.byteCount == expected.byteCount else {
        throw TileSetError.byteCountMismatch(
          path: file.relativePath,
          expected: expected.byteCount,
          actual: file.byteCount
        )
      }
      let digest = try FileSHA256.hex(file.url)
      guard digest == expected.sha256 else {
        throw TileSetError.fileDigestMismatch(path: file.relativePath, expected: expected.sha256, actual: digest)
      }
    }

    for start in stride(from: 0, to: actualFiles.count, by: batchSize) {
      try Task.checkCancellation()
      let end = min(start + batchSize, actualFiles.count)
      let slice = Array(actualFiles[start ..< end])
      let decoded = try await decoder.decode(fileURLs: slice.map(\.url))
      guard decoded.count == slice.count else {
        throw TileSetError.decodeCountMismatch(expected: slice.count, actual: decoded.count)
      }
      for (file, tile) in zip(slice, decoded) {
        try Self.validateDecodedTile(tile, file: file, manifest: manifest)
      }
    }
    return CanonicalTileSetArtifact(rootURL: artifact.rootURL, manifest: manifest)
  }

  fileprivate struct DiscoveredFile: Sendable {
    var url: URL
    var relativePath: String
    var byteCount: UInt64
  }

  fileprivate static func enumerateFiles(rootURL: URL, fileManager: FileManager) throws -> [DiscoveredFile] {
    let offlineURL = rootURL.appendingPathComponent("offline", isDirectory: true)
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: offlineURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw TileSetError.missingOfflineDirectory(offlineURL)
    }
    guard let enumerator = fileManager.enumerator(
      at: offlineURL,
      includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey],
      options: [.skipsPackageDescendants]
    ) else { throw TileSetError.missingOfflineDirectory(offlineURL) }
    let rootPath = rootURL.standardizedFileURL.path + "/"
    var result: [DiscoveredFile] = []
    for case let rawURL as URL in enumerator {
      let url = rawURL.standardizedFileURL
      let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
      let relative = url.path.hasPrefix(rootPath) ? String(url.path.dropFirst(rootPath.count)) : url.path
      if values.isSymbolicLink == true { throw TileSetError.symbolicLink(relative) }
      guard values.isRegularFile == true else { continue }
      guard TileSetManifest.isSafeRelativeTilePath(relative) else { throw TileSetError.unsafePath(relative) }
      result.append(DiscoveredFile(
        url: url,
        relativePath: relative,
        byteCount: UInt64(values.fileSize ?? 0)
      ))
    }
    result.sort { $0.relativePath < $1.relativePath }
    guard !result.isEmpty else { throw TileSetError.emptyTileSet(offlineURL) }
    return result
  }

  private static func validateDecodedTile(
    _ tile: MapTile,
    file: DiscoveredFile,
    manifest: TileSetManifest
  ) throws {
    guard URL(fileURLWithPath: tile.sourcePath).standardizedFileURL == file.url else {
      throw TileSetError.decodedSourceMismatch(path: file.relativePath, source: tile.sourcePath)
    }
    guard tile.schemaVersion == manifest.tileSchemaVersion else {
      throw TileSetError.tileSchemaMismatch(
        path: file.relativePath,
        expected: manifest.tileSchemaVersion,
        actual: tile.schemaVersion
      )
    }
    guard tile.sigmoidHash == manifest.tileSigmoidHash else {
      throw TileSetError.tileSigmoidMismatch(
        path: file.relativePath,
        expected: manifest.tileSigmoidHash,
        actual: tile.sigmoidHash
      )
    }
    guard let parsedBounds = MapTileIndex.parseBounds(filename: file.url.lastPathComponent),
          parsedBounds == tile.bounds
    else { throw TileSetError.tileBoundsMismatch(file.relativePath) }
    try validateBounds(tile.bounds, path: file.relativePath)
    guard tile.overlap.isFinite, tile.overlap >= 0 else { throw TileSetError.nonFiniteTileValue(file.relativePath) }
    for way in tile.ways {
      guard !way.stableID.isEmpty else { throw TileSetError.invalidWay(file.relativePath) }
      try validateBounds(way.bounds, path: file.relativePath)
      for value in [way.maxSpeedMPS, way.maxSpeedForwardMPS, way.maxSpeedBackwardMPS, way.advisorySpeedMPS] {
        guard value.isFinite, value >= 0 else { throw TileSetError.nonFiniteTileValue(file.relativePath) }
      }
      for node in way.nodes {
        guard node.latitude.isFinite, node.longitude.isFinite,
              (-90 ... 90).contains(node.latitude), (-180 ... 180).contains(node.longitude),
              node.bakedSpeedMPS.map({ $0.isFinite && $0 >= 0 }) ?? true
        else { throw TileSetError.nonFiniteTileValue(file.relativePath) }
      }
    }
  }

  private static func validateBounds(_ bounds: MapTileBounds, path: String) throws {
    let values = [bounds.minLatitude, bounds.minLongitude, bounds.maxLatitude, bounds.maxLongitude]
    guard values.allSatisfy(\.isFinite),
          (-90 ... 90).contains(bounds.minLatitude), (-90 ... 90).contains(bounds.maxLatitude),
          (-180 ... 180).contains(bounds.minLongitude), (-180 ... 180).contains(bounds.maxLongitude),
          bounds.minLatitude < bounds.maxLatitude, bounds.minLongitude < bounds.maxLongitude
    else { throw TileSetError.nonFiniteTileValue(path) }
  }
}

public struct CanonicalTileSetBuilder: Sendable {
  private let validator: TileSetValidator

  public init(decoder: any MapTileDecoding, batchSize: Int = 64) {
    validator = TileSetValidator(decoder: decoder, batchSize: batchSize)
  }

  public func build(
    _ request: CanonicalTileSetBuildRequest,
    fileManager: FileManager = .default
  ) async throws -> CanonicalTileSetArtifact {
    guard fileManager.fileExists(atPath: request.pbfURL.path) else { throw TileSetError.missingPBF(request.pbfURL) }
    try fileManager.createDirectory(at: request.setsRootURL, withIntermediateDirectories: true)
    let staging = request.setsRootURL.appendingPathComponent(".building-\(UUID().uuidString)", isDirectory: true)
    let stagingOffline = staging.appendingPathComponent("offline", isDirectory: true)
    try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: staging) }
    try fileManager.copyItem(at: request.sourceOfflineURL, to: stagingOffline)

    let discovered = try TileSetValidator.enumerateFiles(rootURL: staging, fileManager: fileManager)
    let files = try discovered.map {
      TileSetManifest.FileEntry(
        path: $0.relativePath,
        byteCount: $0.byteCount,
        sha256: try FileSHA256.hex($0.url)
      )
    }
    let manifest = TileSetManifest(
      tileSchemaVersion: 1,
      estimatorVersion: request.release.artifact.estimatorVersion,
      tuneIdentitySHA256: request.tuneIdentity.identitySHA256,
      tileSigmoidHash: request.tileSigmoidHash,
      mapdReleaseID: request.release.artifact.releaseID,
      mapdBuildID: request.release.artifact.buildID,
      mapdBinarySHA256: request.release.artifact.sha256,
      pbfSHA256: try FileSHA256.hex(request.pbfURL),
      pbfDate: request.pbfDate,
      regions: request.regions,
      files: files
    )
    try manifest.write(to: staging, fileManager: fileManager)
    let stagedArtifact = CanonicalTileSetArtifact(rootURL: staging, manifest: manifest)
    _ = try await validator.validate(stagedArtifact, fileManager: fileManager)

    let destination = request.setsRootURL.appendingPathComponent(manifest.tileSetID, isDirectory: true)
    if fileManager.fileExists(atPath: destination.path) {
      let existing = try CanonicalTileSetArtifact.load(from: destination)
      return try await validator.validate(existing, fileManager: fileManager)
    }
    try fileManager.moveItem(at: staging, to: destination)
    let artifact = CanonicalTileSetArtifact(rootURL: destination, manifest: manifest)
    return try await validator.validate(artifact, fileManager: fileManager)
  }
}

enum FileSHA256 {
  static func hex(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
      let data = try handle.read(upToCount: 1_048_576) ?? Data()
      if data.isEmpty { break }
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

public enum TileSetError: LocalizedError, Equatable, Sendable {
  case couldNotLoadManifest(URL, String)
  case unsupportedManifestSchema(Int)
  case unsupportedTileSchema(UInt16)
  case invalidMetadata
  case invalidDigest(String)
  case estimatorMismatch(expected: String, actual: String)
  case identityMismatch(expected: String, actual: String)
  case fileCountMismatch(expected: Int, actual: Int)
  case totalBytesMismatch(expected: UInt64, actual: UInt64)
  case nonCanonicalFileOrder
  case nonCanonicalRegions
  case regionFileMismatch
  case unsafePath(String)
  case symbolicLink(String)
  case placeholderTile(String, UInt64)
  case missingOfflineDirectory(URL)
  case emptyTileSet(URL)
  case missingPBF(URL)
  case fileSetMismatch(expected: [String], actual: [String])
  case unexpectedFile(String)
  case byteCountMismatch(path: String, expected: UInt64, actual: UInt64)
  case fileDigestMismatch(path: String, expected: String, actual: String)
  case decodeCountMismatch(expected: Int, actual: Int)
  case decodedSourceMismatch(path: String, source: String)
  case tileSchemaMismatch(path: String, expected: UInt16, actual: UInt16)
  case tileSigmoidMismatch(path: String, expected: String, actual: String)
  case tileBoundsMismatch(String)
  case nonFiniteTileValue(String)
  case invalidWay(String)

  public var errorDescription: String? {
    switch self {
    case let .couldNotLoadManifest(url, reason): "Could not load tile manifest at \(url.path): \(reason)"
    case let .unsupportedManifestSchema(schema): "Unsupported tile manifest schema \(schema)."
    case let .unsupportedTileSchema(schema): "Only production tile schema 1 is deployable; got \(schema)."
    case .invalidMetadata: "The tile manifest is missing required production metadata."
    case let .invalidDigest(value): "The tile manifest contains an invalid SHA-256: \(value)"
    case let .estimatorMismatch(expected, actual): "Tile estimator \(actual) does not match \(expected)."
    case let .identityMismatch(expected, actual): "Tile-set identity is \(actual), expected \(expected)."
    case let .fileCountMismatch(expected, actual): "Tile manifest file count is \(expected), found \(actual)."
    case let .totalBytesMismatch(expected, actual): "Tile manifest byte count is \(expected), found \(actual)."
    case .nonCanonicalFileOrder: "Tile manifest paths must be sorted and unique."
    case .nonCanonicalRegions: "Tile manifest regions must be nonempty, sorted, and unique."
    case .regionFileMismatch: "Tile manifest regions do not exactly match the tile directory tree."
    case let .unsafePath(path): "Unsafe tile path in manifest: \(path)"
    case let .symbolicLink(path): "Tile sets may not contain symbolic links: \(path)"
    case let .placeholderTile(path, count): "Tile \(path) is only \(count) bytes and appears to be a placeholder."
    case let .missingOfflineDirectory(url): "Tile artifact has no offline directory at \(url.path)."
    case let .emptyTileSet(url): "Tile artifact contains no files under \(url.path)."
    case let .missingPBF(url): "Tile provenance PBF does not exist at \(url.path)."
    case let .fileSetMismatch(expected, actual): "Tile file set differs from its manifest. expected=\(expected) actual=\(actual)"
    case let .unexpectedFile(path): "Unexpected tile file: \(path)"
    case let .byteCountMismatch(path, expected, actual): "Tile \(path) is \(actual) bytes, expected \(expected)."
    case let .fileDigestMismatch(path, expected, actual): "Tile \(path) SHA-256 is \(actual), expected \(expected)."
    case let .decodeCountMismatch(expected, actual): "Decoded \(actual) tiles, expected \(expected)."
    case let .decodedSourceMismatch(path, source): "Decoder source mismatch for \(path): \(source)"
    case let .tileSchemaMismatch(path, expected, actual): "Tile \(path) schema is \(actual), expected \(expected)."
    case let .tileSigmoidMismatch(path, expected, actual): "Tile \(path) sigmoid hash is \(actual), expected \(expected)."
    case let .tileBoundsMismatch(path): "Decoded bounds do not match tile filename for \(path)."
    case let .nonFiniteTileValue(path): "Tile \(path) contains invalid or non-finite geometry/speed values."
    case let .invalidWay(path): "Tile \(path) contains a way without a stable identity."
    }
  }
}
