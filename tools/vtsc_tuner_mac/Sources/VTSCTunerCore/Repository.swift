import Foundation

public enum RepositoryError: LocalizedError, Equatable {
  case notFound
  case invalid(URL, String)
  case missingConstant(String)
  case duplicateConstant(String)
  case missingAuthority(String)
  case duplicateAuthority(String)
  case authorityMismatch(String)
  case missingQCurveMarker(String)

  public var errorDescription: String? {
    switch self {
    case .notFound:
      "Could not find the Chauffeur checkout. Choose it from File → Choose Chauffeur Repository."
    case let .invalid(url, reason):
      "\(url.path) is not a usable Chauffeur checkout: \(reason)"
    case let .missingConstant(name):
      "Could not find \(name) in vision_turn_controller.py."
    case let .duplicateConstant(name):
      "Found more than one module-level \(name) in vision_turn_controller.py."
    case let .missingAuthority(name):
      "Could not find the VTSC physics authority \(name)."
    case let .duplicateAuthority(name):
      "Found more than one VTSC physics authority \(name)."
    case let .authorityMismatch(detail):
      "The checked-in VTSC physics authorities disagree: \(detail)"
    case let .missingQCurveMarker(name):
      "Could not find \(name) in vtsc_curve_tuning.py."
    }
  }
}

public enum RepositoryLocator {
  public static let physicsRelativePath = "sunnypilot/selfdrive/controls/lib/vision_turn_controller.py"
  public static let qCurveRelativePath = "sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py"
  public static let paramsDefaultsRelativePath = "common/params_keys.h"
  public static let physicsPanelRelativePath =
    "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vtsc_physics_panel.cc"

  public static func validate(_ root: URL, fileManager: FileManager = .default) throws {
    var isDirectory: ObjCBool = false
    let gitURL = root.appendingPathComponent(".git")
    guard fileManager.fileExists(atPath: gitURL.path, isDirectory: &isDirectory) else {
      throw RepositoryError.invalid(root, "missing .git")
    }
    for relativePath in [
      physicsRelativePath,
      qCurveRelativePath,
      paramsDefaultsRelativePath,
      physicsPanelRelativePath,
    ] {
      guard fileManager.fileExists(atPath: root.appendingPathComponent(relativePath).path) else {
        throw RepositoryError.invalid(root, "missing \(relativePath)")
      }
    }
  }

  public static func detect(
    savedPath: String? = nil,
    environment: [String: String] = ProcessInfo.processInfo.environment,
    fileManager: FileManager = .default,
    bundleURL: URL = Bundle.main.bundleURL
  ) -> URL? {
    var starts: [URL] = []
    if let savedPath, !savedPath.isEmpty { starts.append(URL(fileURLWithPath: savedPath)) }
    if let override = environment["VTSC_REPO_ROOT"] ?? environment["VTSC_TUNER_REPO"], !override.isEmpty {
      starts.append(URL(fileURLWithPath: override))
    }
    starts.append(URL(fileURLWithPath: fileManager.currentDirectoryPath))
    starts.append(bundleURL)
    starts.append(fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Documents/chauffeur"))

    for start in starts {
      var candidate = start.standardizedFileURL
      while candidate.path != "/" {
        if (try? validate(candidate, fileManager: fileManager)) != nil { return candidate }
        candidate.deleteLastPathComponent()
      }
    }
    return nil
  }
}

public struct SourcePatchResult: Equatable, Sendable {
  public var physicsText: String
  public var qCurveText: String
  public var paramsDefaultsText: String
  public var physicsPanelText: String

  public init(
    physicsText: String,
    qCurveText: String,
    paramsDefaultsText: String,
    physicsPanelText: String
  ) {
    self.physicsText = physicsText
    self.qCurveText = qCurveText
    self.paramsDefaultsText = paramsDefaultsText
    self.physicsPanelText = physicsPanelText
  }
}

struct PhysicsAuthorityEntry: Equatable, Sendable {
  var moduleName: String
  var paramKey: String
  var value: Double
  var formattedValue: String
}

enum VTSCPhysicsAuthority {
  static func entries(for parameters: SigmoidParameters) -> [PhysicsAuthorityEntry] {
    let locale = Locale(identifier: "en_US_POSIX")
    return [
      .init(
        moduleName: "PHYSICS_A",
        paramKey: "VisionTurnSpeedControlPhysicsAmplitude",
        value: parameters.a,
        formattedValue: String(format: "%.6f", locale: locale, parameters.a)
      ),
      .init(
        moduleName: "PHYSICS_B",
        paramKey: "VisionTurnSpeedControlPhysicsSteepness",
        value: parameters.b,
        formattedValue: String(format: "%.6f", locale: locale, parameters.b)
      ),
      .init(
        moduleName: "PHYSICS_C",
        paramKey: "VisionTurnSpeedControlPhysicsCenter",
        value: parameters.c,
        formattedValue: String(format: "%.6f", locale: locale, parameters.c)
      ),
      .init(
        moduleName: "PHYSICS_D",
        paramKey: "VisionTurnSpeedControlPhysicsBaseline",
        value: parameters.d,
        formattedValue: String(format: "%.6f", locale: locale, parameters.d)
      ),
      .init(
        moduleName: "PHYSICS_MIN_LAT_ACCEL",
        paramKey: "VisionTurnSpeedControlPhysicsMinLatAccel",
        value: parameters.minLat,
        formattedValue: String(format: "%.4f", locale: locale, parameters.minLat)
      ),
      .init(
        moduleName: "PHYSICS_MAX_LAT_ACCEL",
        paramKey: "VisionTurnSpeedControlPhysicsMaxLatAccel",
        value: parameters.maxLat,
        formattedValue: String(format: "%.4f", locale: locale, parameters.maxLat)
      ),
    ]
  }
}

public enum SourcePatcher {
  private static let constantNames = [
    "PHYSICS_A", "PHYSICS_B", "PHYSICS_C", "PHYSICS_D",
    "PHYSICS_MIN_LAT_ACCEL", "PHYSICS_MAX_LAT_ACCEL",
  ]

  public static func readParameters(from repositoryRoot: URL) throws -> SigmoidParameters {
    let physicsText = try String(
      contentsOf: repositoryRoot.appendingPathComponent(RepositoryLocator.physicsRelativePath),
      encoding: .utf8
    )
    let parameters = try readModuleParameters(from: physicsText)
    let paramsDefaultsText = try String(
      contentsOf: repositoryRoot.appendingPathComponent(RepositoryLocator.paramsDefaultsRelativePath),
      encoding: .utf8
    )
    let physicsPanelText = try String(
      contentsOf: repositoryRoot.appendingPathComponent(RepositoryLocator.physicsPanelRelativePath),
      encoding: .utf8
    )
    try validateAuthority(
      parameters: parameters,
      paramsDefaultsText: paramsDefaultsText,
      physicsPanelText: physicsPanelText
    )
    return parameters
  }

  static func readModuleParameters(from text: String) throws -> SigmoidParameters {
    var values: [String: Double] = [:]
    for name in constantNames {
      let matches = text.split(separator: "\n", omittingEmptySubsequences: false).filter { line in
        line.hasPrefix("\(name) ") || line.hasPrefix("\(name)=")
      }
      guard !matches.isEmpty else { throw RepositoryError.missingConstant(name) }
      guard matches.count == 1 else { throw RepositoryError.duplicateConstant(name) }
      let pieces = matches[0].split(separator: "=", maxSplits: 1)
      guard pieces.count == 2,
            let token = pieces[1].split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "#" }).first,
            let value = Double(token)
      else { throw RepositoryError.missingConstant(name) }
      values[name] = value
    }
    return .init(
      a: values["PHYSICS_A"]!,
      b: values["PHYSICS_B"]!,
      c: values["PHYSICS_C"]!,
      d: values["PHYSICS_D"]!,
      minLat: values["PHYSICS_MIN_LAT_ACCEL"]!,
      maxLat: values["PHYSICS_MAX_LAT_ACCEL"]!
    )
  }

  public static func makePatch(
    physicsText: String,
    qCurveText: String,
    paramsDefaultsText: String,
    physicsPanelText: String,
    parameters: SigmoidParameters,
    bands: [EQBand]
  ) throws -> SourcePatchResult {
    let replacements = VTSCPhysicsAuthority.entries(for: parameters)
    var patchedPhysics = physicsText
    var patchedParamsDefaults = paramsDefaultsText
    var patchedPhysicsPanel = physicsPanelText
    for entry in replacements {
      patchedPhysics = try replaceModuleConstant(
        in: patchedPhysics,
        name: entry.moduleName,
        value: entry.formattedValue
      )
      patchedParamsDefaults = try replaceParamDefault(
        in: patchedParamsDefaults,
        key: entry.paramKey,
        value: entry.formattedValue
      )
      patchedPhysicsPanel = try replacePhysicsPanelLiteral(
        in: patchedPhysicsPanel,
        call: "params.put",
        key: entry.paramKey,
        value: entry.formattedValue
      )
      patchedPhysicsPanel = try replacePhysicsPanelLiteral(
        in: patchedPhysicsPanel,
        call: "ensure",
        key: entry.paramKey,
        value: entry.formattedValue
      )
    }

    let enabledBands = bands.filter(\.enabled)
    let enabledValue = enabledBands.isEmpty ? "False" : "True"
    var qLines = qCurveText.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    guard let enabledIndex = uniqueLineIndex(prefix: "Q_CURVE_ENABLED", in: qLines) else {
      throw RepositoryError.missingQCurveMarker("Q_CURVE_ENABLED")
    }
    guard let pointsIndex = uniqueLineIndex(prefix: "Q_CURVE_POINTS", in: qLines) else {
      throw RepositoryError.missingQCurveMarker("Q_CURVE_POINTS")
    }
    qLines[enabledIndex] = "Q_CURVE_ENABLED = \(enabledValue)"

    var endIndex = pointsIndex
    if !qLines[pointsIndex].trimmingCharacters(in: .whitespaces).hasSuffix("]") {
      while endIndex + 1 < qLines.count {
        endIndex += 1
        if qLines[endIndex].trimmingCharacters(in: .whitespaces).hasSuffix("]") { break }
      }
      guard qLines[endIndex].trimmingCharacters(in: .whitespaces).hasSuffix("]") else {
        throw RepositoryError.missingQCurveMarker("closing ] for Q_CURVE_POINTS")
      }
    }

    var replacement = ["Q_CURVE_POINTS: list[tuple[float, float]] = ["]
    if enabledBands.isEmpty {
      replacement[0] += "]"
    } else {
      let exportedParameters = VTSCMath.sourceRoundedParameters(parameters)
      replacement += VTSCMath.qCurvePoints(
        parameters: exportedParameters,
        bands: enabledBands,
        count: 256
      ).map {
        String(format: "  (%.6e, %.4f),", $0.kappa, $0.speedMultiplier)
      }
      replacement.append("]")
    }
    qLines.replaceSubrange(pointsIndex ... endIndex, with: replacement)
    let trailingNewline = qCurveText.hasSuffix("\n")
    var patchedQCurve = qLines.joined(separator: "\n")
    if trailingNewline, !patchedQCurve.hasSuffix("\n") { patchedQCurve += "\n" }
    return SourcePatchResult(
      physicsText: patchedPhysics,
      qCurveText: patchedQCurve,
      paramsDefaultsText: patchedParamsDefaults,
      physicsPanelText: patchedPhysicsPanel
    )
  }

  public static func apply(
    repositoryRoot: URL,
    parameters: SigmoidParameters,
    bands: [EQBand]
  ) throws -> [URL] {
    let physicsURL = repositoryRoot.appendingPathComponent(RepositoryLocator.physicsRelativePath)
    let qCurveURL = repositoryRoot.appendingPathComponent(RepositoryLocator.qCurveRelativePath)
    let paramsDefaultsURL = repositoryRoot.appendingPathComponent(RepositoryLocator.paramsDefaultsRelativePath)
    let physicsPanelURL = repositoryRoot.appendingPathComponent(RepositoryLocator.physicsPanelRelativePath)
    let physicsText = try String(contentsOf: physicsURL, encoding: .utf8)
    let qCurveText = try String(contentsOf: qCurveURL, encoding: .utf8)
    let paramsDefaultsText = try String(contentsOf: paramsDefaultsURL, encoding: .utf8)
    let physicsPanelText = try String(contentsOf: physicsPanelURL, encoding: .utf8)
    let currentParameters = try readModuleParameters(from: physicsText)
    try validateAuthority(
      parameters: currentParameters,
      paramsDefaultsText: paramsDefaultsText,
      physicsPanelText: physicsPanelText
    )
    let patch = try makePatch(
      physicsText: physicsText,
      qCurveText: qCurveText,
      paramsDefaultsText: paramsDefaultsText,
      physicsPanelText: physicsPanelText,
      parameters: parameters,
      bands: bands
    )
    let outputs: [(url: URL, original: String, patched: String)] = [
      (physicsURL, physicsText, patch.physicsText),
      (qCurveURL, qCurveText, patch.qCurveText),
      (paramsDefaultsURL, paramsDefaultsText, patch.paramsDefaultsText),
      (physicsPanelURL, physicsPanelText, patch.physicsPanelText),
    ]
    let changed = outputs.filter { $0.patched != $0.original }.map(\.url)
    guard !changed.isEmpty else { return [] }

    // Every output is fully validated before any source file is touched.
    var written: [(url: URL, original: String)] = []
    do {
      for output in outputs where output.patched != output.original {
        try Data(output.patched.utf8).write(to: output.url, options: .atomic)
        written.append((output.url, output.original))
      }
    } catch {
      // Best-effort rollback keeps the four source authorities aligned.
      for output in written.reversed() {
        try? Data(output.original.utf8).write(to: output.url, options: .atomic)
      }
      throw error
    }
    return changed
  }

  private static func replaceModuleConstant(in text: String, name: String, value: String) throws -> String {
    var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let indices = lines.indices.filter { index in
      lines[index].hasPrefix("\(name) ") || lines[index].hasPrefix("\(name)=")
    }
    guard !indices.isEmpty else { throw RepositoryError.missingConstant(name) }
    guard indices.count == 1 else { throw RepositoryError.duplicateConstant(name) }
    let index = indices[0]
    let comment = lines[index].firstIndex(of: "#").map { String(lines[index][$0...]) }
    lines[index] = "\(name) = \(value)" + (comment.map { " \($0)" } ?? "")
    return lines.joined(separator: "\n")
  }

  static func validateAuthority(
    parameters: SigmoidParameters,
    paramsDefaultsText: String,
    physicsPanelText: String
  ) throws {
    for entry in VTSCPhysicsAuthority.entries(for: parameters) {
      let declaredDefault = try extractParamDefault(in: paramsDefaultsText, key: entry.paramKey)
      guard declaredDefault == entry.formattedValue else {
        throw RepositoryError.authorityMismatch(
          "\(entry.paramKey) is \(declaredDefault) in params_keys.h but \(entry.formattedValue) in \(entry.moduleName)"
        )
      }
      for call in ["params.put", "ensure"] {
        let panelValue = try extractPhysicsPanelLiteral(
          in: physicsPanelText,
          call: call,
          key: entry.paramKey
        )
        guard panelValue == entry.formattedValue else {
          throw RepositoryError.authorityMismatch(
            "\(call)(\(entry.paramKey)) is \(panelValue) but \(entry.moduleName) is \(entry.formattedValue)"
          )
        }
      }
    }
  }

  private static func extractParamDefault(in text: String, key: String) throws -> String {
    let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let matches = lines.filter { $0.contains("{\"\(key)\",") }
    guard !matches.isEmpty else { throw RepositoryError.missingAuthority("params_keys.h:\(key)") }
    guard matches.count == 1 else { throw RepositoryError.duplicateAuthority("params_keys.h:\(key)") }
    let marker = "FLOAT, \""
    guard let start = matches[0].range(of: marker)?.upperBound,
          let end = matches[0][start...].firstIndex(of: "\"")
    else { throw RepositoryError.missingAuthority("params_keys.h:\(key) default") }
    return String(matches[0][start ..< end])
  }

  private static func replaceParamDefault(in text: String, key: String, value: String) throws -> String {
    var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let indices = lines.indices.filter { lines[$0].contains("{\"\(key)\",") }
    guard !indices.isEmpty else { throw RepositoryError.missingAuthority("params_keys.h:\(key)") }
    guard indices.count == 1 else { throw RepositoryError.duplicateAuthority("params_keys.h:\(key)") }
    let index = indices[0]
    let marker = "FLOAT, \""
    guard let start = lines[index].range(of: marker)?.upperBound,
          let end = lines[index][start...].firstIndex(of: "\"")
    else { throw RepositoryError.missingAuthority("params_keys.h:\(key) default") }
    lines[index].replaceSubrange(start ..< end, with: value)
    return lines.joined(separator: "\n")
  }

  private static func extractPhysicsPanelLiteral(
    in text: String,
    call: String,
    key: String
  ) throws -> String {
    let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let prefix = "\(call)(\"\(key)\", \""
    let matches = lines.filter { $0.contains(prefix) }
    guard !matches.isEmpty else { throw RepositoryError.missingAuthority("vtsc_physics_panel.cc:\(call):\(key)") }
    guard matches.count == 1 else { throw RepositoryError.duplicateAuthority("vtsc_physics_panel.cc:\(call):\(key)") }
    guard let start = matches[0].range(of: prefix)?.upperBound,
          let end = matches[0][start...].firstIndex(of: "\"")
    else { throw RepositoryError.missingAuthority("vtsc_physics_panel.cc:\(call):\(key) value") }
    return String(matches[0][start ..< end])
  }

  private static func replacePhysicsPanelLiteral(
    in text: String,
    call: String,
    key: String,
    value: String
  ) throws -> String {
    var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let prefix = "\(call)(\"\(key)\", \""
    let indices = lines.indices.filter { lines[$0].contains(prefix) }
    guard !indices.isEmpty else { throw RepositoryError.missingAuthority("vtsc_physics_panel.cc:\(call):\(key)") }
    guard indices.count == 1 else { throw RepositoryError.duplicateAuthority("vtsc_physics_panel.cc:\(call):\(key)") }
    let index = indices[0]
    guard let start = lines[index].range(of: prefix)?.upperBound,
          let end = lines[index][start...].firstIndex(of: "\"")
    else { throw RepositoryError.missingAuthority("vtsc_physics_panel.cc:\(call):\(key) value") }
    lines[index].replaceSubrange(start ..< end, with: value)
    return lines.joined(separator: "\n")
  }

  private static func uniqueLineIndex(prefix: String, in lines: [String]) -> Int? {
    let indices = lines.indices.filter { lines[$0].hasPrefix(prefix) }
    return indices.count == 1 ? indices[0] : nil
  }
}
