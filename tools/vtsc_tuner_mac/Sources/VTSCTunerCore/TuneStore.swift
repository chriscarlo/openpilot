import Foundation

public enum TuneStore {
  public static func applicationDirectory(fileManager: FileManager = .default) throws -> URL {
    let base = try fileManager.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    return base.appendingPathComponent("vtsc_tuner", isDirectory: true)
  }

  public static func defaultTuneURL(fileManager: FileManager = .default) throws -> URL {
    try applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("current.tune.json", isDirectory: false)
  }

  public static func mapdConfigURL(fileManager: FileManager = .default) throws -> URL {
    try applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("mapd.json", isDirectory: false)
  }

  public static func canonicalTileSetsURL(fileManager: FileManager = .default) throws -> URL {
    try applicationDirectory(fileManager: fileManager)
      .appendingPathComponent("map_tiles", isDirectory: true)
      .appendingPathComponent("sets", isDirectory: true)
  }

  public static func legacyTuneURL(fileManager: FileManager = .default) -> URL {
    fileManager.homeDirectoryForCurrentUser
      .appendingPathComponent(".config/vtsc_tuner/current.tune.json", isDirectory: false)
  }

  public static func load(from explicitURL: URL? = nil, fileManager: FileManager = .default) throws -> Tune {
    let primary = try explicitURL ?? defaultTuneURL(fileManager: fileManager)
    let source: URL
    if fileManager.fileExists(atPath: primary.path) {
      source = primary
    } else if explicitURL == nil, fileManager.fileExists(atPath: legacyTuneURL(fileManager: fileManager).path) {
      source = legacyTuneURL(fileManager: fileManager)
    } else {
      throw CocoaError(.fileNoSuchFile)
    }
    let decoder = JSONDecoder()
    return try decoder.decode(Tune.self, from: Data(contentsOf: source))
  }

  public static func save(_ tune: Tune, to explicitURL: URL? = nil, fileManager: FileManager = .default) throws {
    let destination = try explicitURL ?? defaultTuneURL(fileManager: fileManager)
    try fileManager.createDirectory(
      at: destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(tune)
    try data.write(to: destination, options: .atomic)
  }
}
