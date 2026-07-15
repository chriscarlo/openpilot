import Foundation

enum TiciTileTransactionHelperError: LocalizedError, Equatable, Sendable {
  case missingHelper(URL)

  var errorDescription: String? {
    switch self {
    case let .missingHelper(url):
      "The bundled tici tile transaction helper is missing or not executable: \(url.path)"
    }
  }
}

/// Resolves the small Linux ARM64 helper that owns the one syscall Swift cannot
/// run on the tici: renameat2 with RENAME_EXCHANGE. It is bundled with the Mac
/// app, but a checked-out development build can use the same pinned output.
enum TiciTileTransactionHelperLocator {
  static let helperName = "vtsc-tile-transaction"

  static func bundledURL(
    bundle: Bundle = .main,
    environment: [String: String] = ProcessInfo.processInfo.environment,
    currentDirectoryURL: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
    fileManager: FileManager = .default
  ) throws -> URL {
    var candidates: [URL] = []
    if bundle.bundleURL.pathExtension == "app" {
      candidates.append(bundle.bundleURL
        .appendingPathComponent("Contents", isDirectory: true)
        .appendingPathComponent("Resources", isDirectory: true)
        .appendingPathComponent(helperName, isDirectory: false))
    }
    if let override = environment["VTSC_TILE_TRANSACTION_HELPER"], !override.isEmpty {
      candidates.append(URL(fileURLWithPath: override))
    }
    if let resource = bundle.url(forResource: helperName, withExtension: nil) {
      candidates.append(resource)
    }
    candidates.append(bundle.bundleURL.appendingPathComponent(helperName, isDirectory: false))
    candidates.append(developmentCandidate(in: currentDirectoryURL))

    var ancestor = bundle.executableURL?.deletingLastPathComponent()
    for _ in 0..<8 {
      guard let directory = ancestor else { break }
      candidates.append(developmentCandidate(in: directory))
      let parent = directory.deletingLastPathComponent()
      if parent == directory { break }
      ancestor = parent
    }

    for candidate in candidates {
      let standardized = candidate.standardizedFileURL
      if fileManager.isExecutableFile(atPath: standardized.path) { return standardized }
    }
    throw TiciTileTransactionHelperError.missingHelper(candidates.first ?? bundle.bundleURL)
  }

  private static func developmentCandidate(in directory: URL) -> URL {
    directory
      .appendingPathComponent(".build-tools", isDirectory: true)
      .appendingPathComponent("bin", isDirectory: true)
      .appendingPathComponent(helperName, isDirectory: false)
  }
}
