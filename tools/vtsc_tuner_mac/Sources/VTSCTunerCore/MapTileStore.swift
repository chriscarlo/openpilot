import Foundation

public enum MapTileRootError: LocalizedError, Equatable, Sendable {
  case missingDirectory(URL)
  case noTiles(URL)
  case malformedFilename(String)

  public var errorDescription: String? {
    switch self {
    case let .missingDirectory(url):
      "The map tile directory does not exist: \(url.path)"
    case let .noTiles(url):
      "No mapd tile files were found under \(url.path)."
    case let .malformedFilename(name):
      "The map tile filename is invalid: \(name)"
    }
  }
}

public enum MapTileRoot {
  public static func defaultURL(fileManager: FileManager = .default) throws -> URL {
    try TuneStore.applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("map_tiles", isDirectory: true)
      .appendingPathComponent("offline", isDirectory: true)
  }

  public static func resolve(_ selectedURL: URL, fileManager: FileManager = .default) throws -> URL {
    let standardized = selectedURL.standardizedFileURL
    let offline = standardized.lastPathComponent == "offline"
      ? standardized
      : standardized.appendingPathComponent("offline", isDirectory: true)
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: offline.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw MapTileRootError.missingDirectory(offline)
    }
    let index = try MapTileIndex(rootURL: offline, fileManager: fileManager)
    guard !index.entries.isEmpty else { throw MapTileRootError.noTiles(offline) }
    return offline
  }
}

public struct MapTileIndex: Equatable, Sendable {
  public struct Entry: Equatable, Hashable, Sendable {
    public var fileURL: URL
    public var bounds: MapTileBounds

    public init(fileURL: URL, bounds: MapTileBounds) {
      self.fileURL = fileURL
      self.bounds = bounds
    }
  }

  public let rootURL: URL
  public let entries: [Entry]
  public let bounds: MapTileBounds?

  public init(rootURL: URL, fileManager: FileManager = .default) throws {
    self.rootURL = rootURL.standardizedFileURL
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: self.rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw MapTileRootError.missingDirectory(self.rootURL)
    }
    guard let enumerator = fileManager.enumerator(
      at: self.rootURL,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles, .skipsPackageDescendants]
    ) else {
      throw MapTileRootError.noTiles(self.rootURL)
    }
    var discovered: [Entry] = []
    for case let fileURL as URL in enumerator {
      let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
      guard values?.isRegularFile == true,
            let parsed = Self.parseBounds(filename: fileURL.lastPathComponent)
      else { continue }
      discovered.append(Entry(fileURL: fileURL.standardizedFileURL, bounds: parsed))
    }
    discovered.sort { $0.fileURL.path < $1.fileURL.path }
    entries = discovered
    bounds = discovered.dropFirst().reduce(discovered.first?.bounds) { accumulated, entry in
      accumulated?.union(entry.bounds)
    }
  }

  public func entries(
    intersecting visibleBounds: MapTileBounds,
    paddingDegrees: Double = 0
  ) -> [Entry] {
    let query = visibleBounds.expanded(by: max(0, paddingDegrees))
    return entries.filter { $0.bounds.intersects(query) }
  }

  public static func parseBounds(filename: String) -> MapTileBounds? {
    let fields = filename.split(separator: "_", omittingEmptySubsequences: false)
    guard fields.count == 4,
          let minLatitude = Double(fields[0]),
          let minLongitude = Double(fields[1]),
          let maxLatitude = Double(fields[2]),
          let maxLongitude = Double(fields[3]),
          minLatitude >= -90, maxLatitude <= 90, minLongitude >= -180, maxLongitude <= 180,
          minLatitude < maxLatitude, minLongitude < maxLongitude
    else { return nil }
    return MapTileBounds(
      minLatitude: minLatitude,
      minLongitude: minLongitude,
      maxLatitude: maxLatitude,
      maxLongitude: maxLongitude
    )
  }
}

public actor MapTileStore {
  private struct CacheEntry {
    var tile: MapTile
    var lastAccess: UInt64
    var bytes: Int
  }

  public private(set) var index: MapTileIndex
  private let decoder: any MapTileDecoding
  private let maximumDecodedBytes: Int
  private var cache: [URL: CacheEntry] = [:]
  private var accessCounter: UInt64 = 0

  public init(
    rootURL: URL,
    decoder: any MapTileDecoding,
    maximumDecodedBytes: Int = 128 * 1_024 * 1_024
  ) throws {
    index = try MapTileIndex(rootURL: rootURL)
    self.decoder = decoder
    self.maximumDecodedBytes = max(1, maximumDecodedBytes)
  }

  public func refreshIndex() throws {
    index = try MapTileIndex(rootURL: index.rootURL)
    let indexedURLs = Set(index.entries.map(\.fileURL))
    cache = cache.filter { indexedURLs.contains($0.key) }
  }

  public func clearCache() {
    cache.removeAll(keepingCapacity: true)
  }

  public func tiles(
    intersecting visibleBounds: MapTileBounds,
    paddingDegrees: Double = 0
  ) async throws -> [MapTile] {
    let visibleEntries = index.entries(intersecting: visibleBounds, paddingDegrees: paddingDegrees)
    var resolved: [URL: MapTile] = [:]
    var missing: [MapTileIndex.Entry] = []
    for entry in visibleEntries {
      if var cached = cache[entry.fileURL] {
        accessCounter &+= 1
        cached.lastAccess = accessCounter
        cache[entry.fileURL] = cached
        resolved[entry.fileURL] = cached.tile
      } else {
        missing.append(entry)
      }
    }
    var firstDecodeError: (any Error)?
    for entry in missing {
      try Task.checkCancellation()
      do {
        let decoded = try await decoder.decode(fileURLs: [entry.fileURL])
        guard let tile = decoded.first else { throw MapTileDecoderError.emptyOutput }
        accessCounter &+= 1
        cache[entry.fileURL] = CacheEntry(
          tile: tile,
          lastAccess: accessCounter,
          bytes: tile.estimatedDecodedBytes
        )
        resolved[entry.fileURL] = tile
        evictIfNeeded(protectedURLs: Set(visibleEntries.map(\.fileURL)))
      } catch is CancellationError {
        throw CancellationError()
      } catch {
        if firstDecodeError == nil { firstDecodeError = error }
      }
    }
    if resolved.isEmpty, let firstDecodeError {
      throw firstDecodeError
    }
    return visibleEntries.compactMap { resolved[$0.fileURL] }
  }

  private func evictIfNeeded(protectedURLs: Set<URL>) {
    var total = cache.values.reduce(0) { $0 + $1.bytes }
    guard total > maximumDecodedBytes else { return }
    let candidates = cache
      .filter { !protectedURLs.contains($0.key) }
      .sorted { $0.value.lastAccess < $1.value.lastAccess }
    for candidate in candidates where total > maximumDecodedBytes {
      total -= candidate.value.bytes
      cache.removeValue(forKey: candidate.key)
    }
  }
}
