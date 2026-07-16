import Foundation

enum TiciTileTransactionHelperError: LocalizedError, Equatable, Sendable {
  case disabled
  case missingHelper(URL)

  var errorDescription: String? {
    switch self {
    case .disabled:
      ApplyAction.deviceTileReplacementUnavailableReason
    case let .missingHelper(url):
      "The bundled tici tile transaction helper is missing or not executable: \(url.path)"
    }
  }
}

/// The helper source remains testable, but production lookup is deliberately
/// disabled while on-device canonical tile replacement is quarantined.
enum TiciTileTransactionHelperLocator {
  static let helperName = "vtsc-tile-transaction"

  static func bundledURL(
    bundle: Bundle = .main,
    environment: [String: String] = ProcessInfo.processInfo.environment,
    currentDirectoryURL: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
    fileManager: FileManager = .default
  ) throws -> URL {
    _ = (bundle, environment, currentDirectoryURL, fileManager)
    throw TiciTileTransactionHelperError.disabled
  }
}
