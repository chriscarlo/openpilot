import Foundation

public enum MapTileDecoderError: LocalizedError, Equatable, Sendable {
  case missingHelper(URL)
  case helperFailed(Int32, String)
  case emptyOutput
  case outputCount(expected: Int, actual: Int)
  case sourceMismatch(expected: URL, actual: String)
  case invalidOutput(String)

  public var errorDescription: String? {
    switch self {
    case let .missingHelper(url):
      "The bundled map tile decoder is missing or not executable: \(url.path)"
    case let .helperFailed(status, detail):
      "The map tile decoder exited with status \(status): \(detail)"
    case .emptyOutput:
      "The map tile decoder returned no data."
    case let .outputCount(expected, actual):
      "The map tile decoder returned \(actual) tiles for \(expected) inputs."
    case let .sourceMismatch(expected, actual):
      "The map tile decoder returned \(actual) while decoding \(expected.path)."
    case let .invalidOutput(reason):
      "The map tile decoder returned invalid JSON: \(reason)"
    }
  }
}

public protocol MapTileDecoding: Sendable {
  func decode(fileURLs: [URL]) async throws -> [MapTile]
}

public enum MapTileHelperLocator {
  public static func bundledURL(
    bundle: Bundle = .main,
    environment: [String: String] = ProcessInfo.processInfo.environment,
    currentDirectoryURL: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
    fileManager: FileManager = .default
  ) throws -> URL {
    var candidates: [URL] = []
    if bundle.bundleURL.pathExtension == "app" {
      candidates.append(bundle.bundleURL
        .appendingPathComponent("Contents", isDirectory: true)
        .appendingPathComponent("Helpers", isDirectory: true)
        .appendingPathComponent("vtsc-tile-decoder", isDirectory: false))
    }
    if let override = environment["VTSC_TILE_DECODER"], !override.isEmpty {
      candidates.append(URL(fileURLWithPath: override))
    }
    if let resource = bundle.url(
      forResource: "vtsc-tile-decoder",
      withExtension: nil,
      subdirectory: "Helpers"
    ) {
      candidates.append(resource)
    }
    candidates.append(bundle.bundleURL.appendingPathComponent("vtsc-tile-decoder", isDirectory: false))
    candidates.append(
      currentDirectoryURL
        .appendingPathComponent(".build-tools", isDirectory: true)
        .appendingPathComponent("bin", isDirectory: true)
        .appendingPathComponent("vtsc-tile-decoder", isDirectory: false)
    )

    var ancestor = bundle.executableURL?.deletingLastPathComponent()
    for _ in 0..<8 {
      guard let directory = ancestor else { break }
      candidates.append(
        directory
          .appendingPathComponent(".build-tools", isDirectory: true)
          .appendingPathComponent("bin", isDirectory: true)
          .appendingPathComponent("vtsc-tile-decoder", isDirectory: false)
      )
      let parent = directory.deletingLastPathComponent()
      if parent == directory { break }
      ancestor = parent
    }

    for candidate in candidates {
      let standardized = candidate.standardizedFileURL
      if fileManager.isExecutableFile(atPath: standardized.path) { return standardized }
    }
    throw MapTileDecoderError.missingHelper(candidates.first ?? bundle.bundleURL)
  }
}

public struct MapTileHelperDecoder: MapTileDecoding {
  public let helperURL: URL
  private let processRunner: any ProcessRunning

  public init(
    helperURL: URL,
    processRunner: any ProcessRunning = SystemProcessRunner()
  ) {
    self.helperURL = helperURL
    self.processRunner = processRunner
  }

  public func decode(fileURLs: [URL]) async throws -> [MapTile] {
    guard !fileURLs.isEmpty else { return [] }
    guard FileManager.default.isExecutableFile(atPath: helperURL.path) else {
      throw MapTileDecoderError.missingHelper(helperURL)
    }
    let standardized = fileURLs.map(\.standardizedFileURL)
    let result = try await processRunner.run(
      ProcessRequest(
        executableURL: helperURL,
        arguments: standardized.map(\.path),
        timeout: max(30.0, Double(standardized.count) * 30.0)
      )
    )
    guard result.succeeded else {
      throw MapTileDecoderError.helperFailed(result.terminationStatus, result.combinedOutput)
    }
    let lines = result.standardOutput.split(whereSeparator: \.isNewline)
    guard !lines.isEmpty else { throw MapTileDecoderError.emptyOutput }
    guard lines.count == standardized.count else {
      throw MapTileDecoderError.outputCount(expected: standardized.count, actual: lines.count)
    }
    let decoder = JSONDecoder()
    var tiles: [MapTile] = []
    tiles.reserveCapacity(lines.count)
    for (index, line) in lines.enumerated() {
      do {
        let tile = try decoder.decode(MapTile.self, from: Data(line.utf8))
        let actual = URL(fileURLWithPath: tile.sourcePath).standardizedFileURL
        guard actual == standardized[index] else {
          throw MapTileDecoderError.sourceMismatch(expected: standardized[index], actual: tile.sourcePath)
        }
        tiles.append(tile)
      } catch let error as MapTileDecoderError {
        throw error
      } catch {
        throw MapTileDecoderError.invalidOutput(error.localizedDescription)
      }
    }
    return tiles
  }
}
