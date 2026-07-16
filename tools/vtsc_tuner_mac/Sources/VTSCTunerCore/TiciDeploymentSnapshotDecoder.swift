import Foundation

/// The source-relevant subset of `vtsc_curve_tuning.py` that the Python
/// deployer historically hashed on the tici. Keeping this parser on the Mac
/// means a read-only device snapshot has no dependency on openpilot imports.
struct TiciQCurveIdentity: Equatable, Sendable {
  var sha256: String
  var enabled: Bool
  var pointCount: Int
}

struct TiciDeploymentSnapshotRead: Equatable, Sendable {
  var snapshot: TiciDeploymentSnapshot
  var qCurve: TiciQCurveIdentity
  var wire: TiciSnapshotWireSnapshot
}

struct TiciRuntimePostflightRead: Equatable, Sendable {
  var deployment: TiciDeploymentSnapshotRead
  var activeMapdBuildInfo: Data
  var activeMapdELFHeader: Data
  var managerRunning: Bool
  var mapdRunning: Bool
  var remoteEpochMilliseconds: Int64
  var wholeCurveProfile: Data?
  var lastGPSPosition: Data?
  var liveMapDataControllerStatus: TiciLiveMapDataControllerStatus
  var runtimeEndIsOffroad: Bool
  var runtimeEndIsOnroad: Bool
  var runtimeEndMapLookaheadEnabled: Bool
}

struct TiciStaticDeploymentPostflightRead: Equatable, Sendable {
  var deployment: TiciDeploymentSnapshotRead
  var activeMapdBuildInfo: Data
  var activeMapdELFHeader: Data
  var managerRunning: Bool
  var mapdRunning: Bool
  var runtimeEndIsOffroad: Bool
  var runtimeEndIsOnroad: Bool
  var runtimeEndMapLookaheadEnabled: Bool
}

struct TiciRollbackStaticPostflightRead: Equatable, Sendable {
  var deployment: TiciDeploymentSnapshotRead
  var managerRunning: Bool
  var mapdRunning: Bool
  var runtimeEndIsOffroad: Bool
  var runtimeEndIsOnroad: Bool
  var runtimeEndMapLookaheadEnabled: Bool
}

struct TiciLiveMapDataControllerStatus: Equatable, Sendable {
  var updated: Bool
  var valid: Bool
  var logMonoTimeNs: UInt64
  var roadGeometryValid: Bool
  var sampleMonoTimeNs: UInt64
}

enum TiciDeploymentSnapshotDecodeError: LocalizedError, Equatable, Sendable {
  case missingField(TiciSnapshotWireField)
  case invalidBoolean(TiciSnapshotWireField, String)
  case invalidBootID(String)
  case invalidEpochMilliseconds(String)
  case malformedCacheListing(String)
  case malformedTileManifest(String)
  case qCurveMarkerMissing(String)
  case malformedLiveMapDataControllerStatus(String)

  var errorDescription: String? {
    switch self {
    case let .missingField(field):
      "The tici snapshot did not contain \(field.rawValue)."
    case let .invalidBoolean(field, value):
      "The tici snapshot reported invalid \(field.rawValue)=\(value.debugDescription)."
    case let .invalidBootID(value):
      "The tici snapshot reported invalid boot_id=\(value.debugDescription)."
    case let .invalidEpochMilliseconds(value):
      "The tici snapshot reported invalid remote_epoch_milliseconds=\(value.debugDescription)."
    case let .malformedCacheListing(line):
      "The tici mapd cache listing is malformed: \(line)"
    case let .malformedTileManifest(value):
      "The tici active tile manifest/topology is malformed: \(value)"
    case let .qCurveMarkerMissing(marker):
      "The tici Q-curve source is missing \(marker)."
    case let .malformedLiveMapDataControllerStatus(value):
      "The tici liveMapDataSP controller probe is malformed: \(value.debugDescription)."
    }
  }
}

/// Decodes the dependency-free shell snapshot into the same model used by the
/// existing deployment journal. The host owns all interpretation: the tici
/// merely reads files and calculates ordinary SHA-256 values.
enum TiciDeploymentSnapshotDecoder {
  private static let physicsFields: [(String, TiciSnapshotWireField)] = [
    ("VisionTurnSpeedControlPhysicsAmplitude", .physicsAmplitude),
    ("VisionTurnSpeedControlPhysicsSteepness", .physicsSteepness),
    ("VisionTurnSpeedControlPhysicsCenter", .physicsCenter),
    ("VisionTurnSpeedControlPhysicsBaseline", .physicsBaseline),
    ("VisionTurnSpeedControlPhysicsMinLatAccel", .physicsMinLatAccel),
    ("VisionTurnSpeedControlPhysicsMaxLatAccel", .physicsMaxLatAccel),
  ]

  static func decode(_ output: String) throws -> TiciDeploymentSnapshot {
    try decodeRead(output).snapshot
  }

  static func decodeRead(_ output: String) throws -> TiciDeploymentSnapshotRead {
    try decodeRead(TiciSnapshotWireCodec.decode(output))
  }

  static func decodeRead(_ wire: TiciSnapshotWireSnapshot) throws -> TiciDeploymentSnapshotRead {
    let snapshot = try decode(wire)
    return TiciDeploymentSnapshotRead(
      snapshot: snapshot,
      qCurve: try qCurveIdentity(source: requiredData(wire, .qCurveFile)),
      wire: wire
    )
  }

  static func decodeRuntimePostflight(_ output: String) throws -> TiciRuntimePostflightRead {
    let wire = try TiciSnapshotWireCodec.decode(output)
    let deployment = try decodeRead(wire)
    return TiciRuntimePostflightRead(
      deployment: deployment,
      activeMapdBuildInfo: try requiredData(wire, .activeMapdBuildInfo),
      activeMapdELFHeader: try requiredData(wire, .activeMapdELFHeader),
      managerRunning: try requiredBool(wire, .managerRunning),
      mapdRunning: try requiredBool(wire, .mapdRunning),
      remoteEpochMilliseconds: try requiredEpochMilliseconds(wire),
      wholeCurveProfile: firstNonempty(wire[.memoryWholeCurveProfile], wire[.persistentWholeCurveProfile]),
      lastGPSPosition: firstNonempty(wire[.memoryLastGPSPosition], wire[.persistentLastGPSPosition]),
      liveMapDataControllerStatus: try requiredLiveMapDataControllerStatus(wire),
      runtimeEndIsOffroad: try requiredBool(wire, .runtimeEndIsOffroad),
      runtimeEndIsOnroad: try requiredBool(wire, .runtimeEndIsOnroad),
      runtimeEndMapLookaheadEnabled: try requiredBool(wire, .runtimeEndMapLookaheadEnabled)
    )
  }

  static func decodeStaticPostflight(_ output: String) throws -> TiciStaticDeploymentPostflightRead {
    let wire = try TiciSnapshotWireCodec.decode(output)
    return TiciStaticDeploymentPostflightRead(
      deployment: try decodeRead(wire),
      activeMapdBuildInfo: try requiredData(wire, .activeMapdBuildInfo),
      activeMapdELFHeader: try requiredData(wire, .activeMapdELFHeader),
      managerRunning: try requiredBool(wire, .managerRunning),
      mapdRunning: try requiredBool(wire, .mapdRunning),
      runtimeEndIsOffroad: try requiredBool(wire, .runtimeEndIsOffroad),
      runtimeEndIsOnroad: try requiredBool(wire, .runtimeEndIsOnroad),
      runtimeEndMapLookaheadEnabled: try requiredBool(wire, .runtimeEndMapLookaheadEnabled)
    )
  }

  static func decodeRollbackStaticPostflight(_ output: String) throws -> TiciRollbackStaticPostflightRead {
    let wire = try TiciSnapshotWireCodec.decode(output)
    return TiciRollbackStaticPostflightRead(
      deployment: try decodeRead(wire),
      managerRunning: try requiredBool(wire, .managerRunning),
      mapdRunning: try requiredBool(wire, .mapdRunning),
      runtimeEndIsOffroad: try requiredBool(wire, .runtimeEndIsOffroad),
      runtimeEndIsOnroad: try requiredBool(wire, .runtimeEndIsOnroad),
      runtimeEndMapLookaheadEnabled: try requiredBool(wire, .runtimeEndMapLookaheadEnabled)
    )
  }

  static func decode(_ wire: TiciSnapshotWireSnapshot) throws -> TiciDeploymentSnapshot {
    let release = optionalText(wire, .mapdReleaseVersion)
    let version = optionalText(wire, .mapdVersion)
    let cacheEntries = try decodeCacheEntries(optionalText(wire, .mapdCacheListing) ?? "")
    let activeRelease = release ?? version
    let cached = activeRelease.flatMap { selectCachedMapd(releaseID: $0, entries: cacheEntries) }

    let qCurveData = try requiredData(wire, .qCurveFile)
    let qCurve = try qCurveIdentity(source: qCurveData)
    let physics = Dictionary(uniqueKeysWithValues: physicsFields.map { key, field in
      (key, optionalText(wire, field))
    })
    guard let tileManifest = wire[.tileManifest] else {
      throw TiciDeploymentSnapshotDecodeError.missingField(.tileManifest)
    }
    let tile = try tileIdentity(
      manifest: tileManifest,
      topology: try requiredText(wire, .tileTopology)
    )

    return TiciDeploymentSnapshot(
      bootID: try optionalBootID(wire),
      isOffroad: try requiredBool(wire, .isOffroad),
      isOnroad: try requiredBool(wire, .isOnroad),
      mapLookaheadEnabled: try requiredBool(wire, .mapLookaheadEnabled),
      branch: try requiredText(wire, .branch),
      head: try requiredText(wire, .head),
      dirty: try requiredBool(wire, .dirty),
      physicsParams: physics,
      qCurveSHA256: qCurve.sha256,
      mapdReleaseVersion: release,
      mapdVersion: version,
      activeMapdSHA256: optionalText(wire, .activeMapdSHA256) ?? "",
      cachedMapdPath: cached?.path ?? "",
      cachedMapdSHA256: cached?.sha256,
      activeTileSetID: tile.id,
      activeTileTopology: tile.topology
    )
  }

  static func qCurveIdentity(source: Data) throws -> TiciQCurveIdentity {
    // Python's `splitlines()` intentionally removes line terminators before
    // hashing the selected rows back with exactly one LF each.
    let lines = String(decoding: source, as: UTF8.self).split(
      omittingEmptySubsequences: false,
      whereSeparator: \.isNewline
    ).map(String.init)
    guard let enabledLine = lines.first(where: { $0.hasPrefix("Q_CURVE_ENABLED") }) else {
      throw TiciDeploymentSnapshotDecodeError.qCurveMarkerMissing("Q_CURVE_ENABLED")
    }
    guard let firstPointIndex = lines.firstIndex(where: { $0.hasPrefix("Q_CURVE_POINTS") }) else {
      throw TiciDeploymentSnapshotDecodeError.qCurveMarkerMissing("Q_CURVE_POINTS")
    }

    var selected = [enabledLine, lines[firstPointIndex]]
    if !lines[firstPointIndex].trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("]") {
      for line in lines.dropFirst(firstPointIndex + 1) {
        selected.append(line)
        if line.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("]") { break }
      }
    }
    let digest = TuneDeploymentIdentity.sha256Hex(Data((selected.joined(separator: "\n") + "\n").utf8))
    let enabled = enabledLine
      .split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
      .last
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) == "True" }
      ?? false
    let pointLines = selected.dropFirst(2).dropLast()
    let pointCount = pointLines.reduce(into: 0) { count, line in
      if line.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("(") { count += 1 }
    }
    return TiciQCurveIdentity(sha256: digest, enabled: enabled, pointCount: pointCount)
  }

  private static func requiredData(
    _ wire: TiciSnapshotWireSnapshot,
    _ field: TiciSnapshotWireField
  ) throws -> Data {
    guard let value = wire[field], !value.isEmpty else {
      throw TiciDeploymentSnapshotDecodeError.missingField(field)
    }
    return value
  }

  private static func requiredText(
    _ wire: TiciSnapshotWireSnapshot,
    _ field: TiciSnapshotWireField
  ) throws -> String {
    let value = String(decoding: try requiredData(wire, field), as: UTF8.self)
    guard !value.isEmpty else { throw TiciDeploymentSnapshotDecodeError.missingField(field) }
    return value
  }

  private static func optionalText(
    _ wire: TiciSnapshotWireSnapshot,
    _ field: TiciSnapshotWireField
  ) -> String? {
    guard let value = wire[field], !value.isEmpty else { return nil }
    return String(decoding: value, as: UTF8.self)
  }

  private static func firstNonempty(_ first: Data?, _ second: Data?) -> Data? {
    if let first, !first.isEmpty { return first }
    if let second, !second.isEmpty { return second }
    return nil
  }

  private static func requiredBool(
    _ wire: TiciSnapshotWireSnapshot,
    _ field: TiciSnapshotWireField
  ) throws -> Bool {
    let text = try requiredText(wire, field)
    switch text {
    case "0": return false
    case "1": return true
    default: throw TiciDeploymentSnapshotDecodeError.invalidBoolean(field, text)
    }
  }

  private static func requiredEpochMilliseconds(_ wire: TiciSnapshotWireSnapshot) throws -> Int64 {
    let text = try requiredText(wire, .remoteEpochMilliseconds)
    guard text.range(of: #"^[0-9]{13,16}$"#, options: .regularExpression) != nil,
          let value = Int64(text), value > 0
    else { throw TiciDeploymentSnapshotDecodeError.invalidEpochMilliseconds(text) }
    return value
  }

  private static func optionalBootID(_ wire: TiciSnapshotWireSnapshot) throws -> String? {
    guard let text = optionalText(wire, .bootID) else { return nil }
    guard TiciBootIdentity.isValid(text) else {
      throw TiciDeploymentSnapshotDecodeError.invalidBootID(text)
    }
    return text
  }

  private static func requiredLiveMapDataControllerStatus(
    _ wire: TiciSnapshotWireSnapshot
  ) throws -> TiciLiveMapDataControllerStatus {
    let text = try requiredText(wire, .liveMapDataControllerStatus)
    let fields = text.split(separator: "|", omittingEmptySubsequences: false)
    guard fields.count == 5,
          [fields[0], fields[1], fields[3]].allSatisfy({ $0 == "0" || $0 == "1" }),
          fields[2].range(of: #"^[0-9]{1,20}$"#, options: .regularExpression) != nil,
          fields[4].range(of: #"^[0-9]{1,20}$"#, options: .regularExpression) != nil,
          let logMonoTimeNs = UInt64(fields[2]),
          let sampleMonoTimeNs = UInt64(fields[4])
    else { throw TiciDeploymentSnapshotDecodeError.malformedLiveMapDataControllerStatus(text) }
    return TiciLiveMapDataControllerStatus(
      updated: fields[0] == "1",
      valid: fields[1] == "1",
      logMonoTimeNs: logMonoTimeNs,
      roadGeometryValid: fields[3] == "1",
      sampleMonoTimeNs: sampleMonoTimeNs
    )
  }

  private struct CacheEntry: Equatable, Sendable {
    var path: String
    var sha256: String
  }

  private static func decodeCacheEntries(_ listing: String) throws -> [CacheEntry] {
    guard !listing.isEmpty else { return [] }
    return try listing.split(whereSeparator: \.isNewline).map { rawLine in
      let fields = rawLine.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
      guard fields.count == 2,
            !fields[0].isEmpty,
            fields[1].range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil
      else { throw TiciDeploymentSnapshotDecodeError.malformedCacheListing(String(rawLine)) }
      return CacheEntry(path: String(fields[0]), sha256: String(fields[1]))
    }
  }

  private static func selectCachedMapd(releaseID: String, entries: [CacheEntry]) -> CacheEntry? {
    let releaseDigest = TuneDeploymentIdentity.sha256Hex(Data(releaseID.utf8))
    let prefix = "mapd-\(releaseDigest.prefix(16))-"
    return entries
      .filter { URL(fileURLWithPath: $0.path).lastPathComponent.hasPrefix(prefix) }
      .sorted { $0.path < $1.path }
      .last
  }

  private static func tileIdentity(
    manifest raw: Data,
    topology rawTopology: String
  ) throws -> (id: String?, topology: TiciActiveTileTopology) {
    if rawTopology == "direct-unidentified" {
      guard raw.isEmpty else {
        throw TiciDeploymentSnapshotDecodeError.malformedTileManifest(
          "direct-unidentified topology carried a manifest"
        )
      }
      return (nil, .directUnidentified)
    }
    guard !raw.isEmpty,
          let object = try? JSONSerialization.jsonObject(with: raw) as? [String: Any],
          let tileSetID = object["tile_set_id"] as? String,
          tileSetID.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"#, options: .regularExpression) != nil
    else {
      throw TiciDeploymentSnapshotDecodeError.malformedTileManifest(
        "\(rawTopology):\(String(decoding: raw, as: UTF8.self))"
      )
    }
    if rawTopology == "direct-identified" {
      return (tileSetID, .directIdentified)
    }
    let prefix = "canonical:"
    guard rawTopology.hasPrefix(prefix) else {
      throw TiciDeploymentSnapshotDecodeError.malformedTileManifest(rawTopology)
    }
    let linkID = String(rawTopology.dropFirst(prefix.count))
    guard linkID == tileSetID,
          linkID.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"#, options: .regularExpression) != nil else {
      throw TiciDeploymentSnapshotDecodeError.malformedTileManifest(
        "canonical link identity \(linkID) differs from manifest \(tileSetID)"
      )
    }
    return (tileSetID, .canonical)
  }
}
