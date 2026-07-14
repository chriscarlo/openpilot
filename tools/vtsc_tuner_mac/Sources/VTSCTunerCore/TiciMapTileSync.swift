import Foundation

public struct TiciMapTileSyncResult: Equatable, Sendable {
  public var destinationRootURL: URL
  public var fileCount: Int
  public var commandOutput: String

  public init(destinationRootURL: URL, fileCount: Int, commandOutput: String) {
    self.destinationRootURL = destinationRootURL
    self.fileCount = fileCount
    self.commandOutput = commandOutput
  }
}

public enum TiciMapTileSyncError: LocalizedError, Equatable, Sendable {
  case invalidProfile(String)
  case commandFailed(Int32, String)

  public var errorDescription: String? {
    switch self {
    case let .invalidProfile(profile):
      "The SSH profile name is invalid: \(profile)"
    case let .commandFailed(status, output):
      "Tile sync exited with status \(status): \(output)"
    }
  }
}

/// Explicitly mirrors the tici's current mapd tiles into the tuner's durable
/// Application Support cache. Nothing calls this service automatically.
public struct TiciMapTileSyncService: Sendable {
  public static let rsyncURL = URL(fileURLWithPath: "/usr/bin/rsync")
  private let processRunner: any ProcessRunning

  public init(processRunner: any ProcessRunning = SystemProcessRunner()) {
    self.processRunner = processRunner
  }

  public func sync(
    profile: String,
    destinationRootURL: URL? = nil,
    fileManager: FileManager = .default
  ) async throws -> TiciMapTileSyncResult {
    guard profile.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
      throw TiciMapTileSyncError.invalidProfile(profile)
    }
    let destination = try destinationRootURL ?? MapTileRoot.defaultURL(fileManager: fileManager)
    let parent = destination.deletingLastPathComponent()
    let staging = parent.appendingPathComponent(
      ".\(destination.lastPathComponent).sync-\(UUID().uuidString)",
      isDirectory: true
    )
    let previous = parent.appendingPathComponent(
      "\(destination.lastPathComponent).previous",
      isDirectory: true
    )
    try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: staging) }
    let result = try await processRunner.run(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-av", "--partial",
          "-e", "/usr/bin/ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          "\(profile):/data/media/0/osm/offline/",
          staging.path + "/",
        ],
        timeout: 3_600
      )
    )
    guard result.succeeded else {
      throw TiciMapTileSyncError.commandFailed(result.terminationStatus, result.combinedOutput)
    }
    let index = try MapTileIndex(rootURL: staging, fileManager: fileManager)
    guard !index.entries.isEmpty else { throw MapTileRootError.noTiles(staging) }

    if fileManager.fileExists(atPath: previous.path) {
      try fileManager.removeItem(at: previous)
    }
    let hadDestination = fileManager.fileExists(atPath: destination.path)
    if hadDestination {
      try fileManager.moveItem(at: destination, to: previous)
    }
    do {
      try fileManager.moveItem(at: staging, to: destination)
    } catch {
      if hadDestination, !fileManager.fileExists(atPath: destination.path) {
        try? fileManager.moveItem(at: previous, to: destination)
      }
      throw error
    }
    return TiciMapTileSyncResult(
      destinationRootURL: destination,
      fileCount: try MapTileIndex(rootURL: destination, fileManager: fileManager).entries.count,
      commandOutput: result.combinedOutput
    )
  }
}
