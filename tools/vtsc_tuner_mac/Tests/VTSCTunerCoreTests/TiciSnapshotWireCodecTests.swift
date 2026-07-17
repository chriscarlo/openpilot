import Foundation
import Testing
@testable import VTSCTunerCore

@Test func ticiSnapshotWireRoundTripsTypedBase64Payloads() throws {
  let expected = TiciSnapshotWireSnapshot(rawValues: [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data("5a4251fcf".utf8),
    .dirty: Data("0".utf8),
    .physicsAmplitude: Data("4.11".utf8),
    .qCurveFile: Data("Q_CURVE_ENABLED = True\nQ_CURVE_POINTS = [(0.1, 1.0)]\n".utf8),
    .tileManifest: Data("{\"tile_set_id\":\"current\"}".utf8),
    .tileTopology: Data("generation:current".utf8),
    .tileTreeSHA256: Data(String(repeating: "e", count: 64).utf8),
    .mapdCacheListing: Data("/data/media/0/osm/binaries/mapd-a\tabc123\n".utf8),
  ])

  let wire = TiciSnapshotWireCodec.encode(expected)
  #expect(wire.contains("q_curve_file\tUV9DVVJWRV9FTkFCTEVE"))
  #expect(try TiciSnapshotWireCodec.decode(wire) == expected)
}

@Test func ticiSnapshotWireKeepsMissingFieldsDistinctFromEmptyPayloads() throws {
  let wire = "branch\tY2hhdWZmZXVyLWV4cDAx\nmapd_version\t\n"
  let decoded = try TiciSnapshotWireCodec.decode(wire)

  #expect(decoded.text(for: .branch) == "chauffeur-exp01")
  #expect(decoded[.isOffroad] == nil)
  #expect(decoded[.mapdVersion] == Data())
}

@Test func ticiSnapshotWireRejectsMalformedAndDuplicateFields() throws {
  #expect(throws: TiciSnapshotWireCodecError.malformedLine(line: 1)) {
    try TiciSnapshotWireCodec.decode("branch Y2hhdWZmZXVyLWV4cDAx")
  }
  #expect(throws: TiciSnapshotWireCodecError.invalidBase64(field: .branch, line: 1)) {
    try TiciSnapshotWireCodec.decode("branch\tnot-base64!")
  }
  #expect(throws: TiciSnapshotWireCodecError.duplicateField(.branch, line: 2)) {
    try TiciSnapshotWireCodec.decode("branch\tYQ==\nbranch\tYg==")
  }
}

@Test func ticiSnapshotWireCommandUsesBoundedPythonOnlyForFastTileDigestAndRuntimeProbe() throws {
  let command = TiciSnapshotWireCommandBuilder.inspectionCommand()
  #expect(command.contains("tile_digest_python='/usr/local/venv/bin/python3'"))
  #expect(command.contains("hashlib.sha256"))
  #expect(!command.contains("for candidate do"))
  #expect(command.contains("base64"))
  #expect(command.contains("git -C"))
  #expect(command.contains(#"git_dirty=$(git -C "$repo" status --porcelain 2>/dev/null) || git_status_ok=0"#))
  #expect(command.contains(#"[ "$git_status_ok" = 1 ] && [ -z "$git_dirty" ]"#))
  #expect(command.contains("sha256sum"))
  #expect(command.contains("/data/params/d"))
  #expect(command.contains("/proc/sys/kernel/random/boot_id"))
  #expect(command.contains("q_curve_file"))
  #expect(command.contains("tile_manifest"))
  #expect(command.contains(#"tile_embedded="$tile_generation_offline/.tileset-manifest.json""#))
  #expect(command.contains("/data/media/0/osm/offline.manifest.json"))
  #expect(command.contains(#"[ -L "$tile_offline" ]"#))
  #expect(command.contains(#"[ -d "$tile_offline" ] && [ ! -L "$tile_offline" ]"#))
  #expect(command.contains("mapd_cache_listing"))

  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-n"]
  let input = Pipe()
  process.standardInput = input
  try process.run()
  input.fileHandleForWriting.write(Data(command.utf8))
  input.fileHandleForWriting.closeFile()
  process.waitUntilExit()
  #expect(process.terminationStatus == 0)
}

@Test func tileManifestProbeValidatesCanonicalPointersAndDirectIdentityStates() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-tile-probe-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  let offline = root.appendingPathComponent("offline", isDirectory: true)
  let adjacent = root.appendingPathComponent("offline.manifest.json")
  let tileID = String(repeating: "a", count: 64)
  try FileManager.default.createDirectory(at: offline, withIntermediateDirectories: true)
  try Data("tile-data".utf8).write(to: offline.appendingPathComponent("32--122"))
  try "{\"tile_set_id\":\"\(tileID)\"}\n".write(to: adjacent, atomically: true, encoding: .utf8)

  let direct = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  var fields = runtimeSnapshotMinimumFields()
  fields[.tileManifest] = Data(direct.manifest.utf8)
  fields[.tileTopology] = Data(direct.topology.utf8)
  fields[.tileTreeSHA256] = Data(direct.treeSHA256.utf8)
  let directRead = try TiciDeploymentSnapshotDecoder.decodeRead(
    TiciSnapshotWireCodec.encode(.init(rawValues: fields))
  ).snapshot
  #expect(directRead.activeTileSetID == tileID)
  #expect(directRead.activeTileTopology == .directIdentified)

  try FileManager.default.removeItem(at: adjacent)
  let unidentified = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  fields[.tileManifest] = Data(unidentified.manifest.utf8)
  fields[.tileTopology] = Data(unidentified.topology.utf8)
  fields[.tileTreeSHA256] = Data(unidentified.treeSHA256.utf8)
  let unidentifiedRead = try TiciDeploymentSnapshotDecoder.decodeRead(
    TiciSnapshotWireCodec.encode(.init(rawValues: fields))
  ).snapshot
  #expect(unidentifiedRead.activeTileSetID == nil)
  #expect(unidentifiedRead.activeTileTopology == .directUnidentified)
  try "{\"tile_set_id\":\"\(tileID)\"}\n".write(to: adjacent, atomically: true, encoding: .utf8)

  func expectInvalidTopology(_ label: String) throws {
    let result = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
    #expect(result.topology.hasPrefix("invalid:"), "\(label) did not emit explicit invalid topology")
    var invalidFields = runtimeSnapshotMinimumFields()
    invalidFields[.tileManifest] = Data(result.manifest.utf8)
    invalidFields[.tileTopology] = Data(result.topology.utf8)
    invalidFields[.tileTreeSHA256] = Data(String(repeating: "e", count: 64).utf8)
    #expect(throws: TiciDeploymentSnapshotDecodeError.self) {
      try TiciDeploymentSnapshotDecoder.decodeRead(
        TiciSnapshotWireCodec.encode(.init(rawValues: invalidFields))
      )
    }
  }

  let embeddedDirect = offline.appendingPathComponent(".tileset-manifest.json")
  try FileManager.default.createSymbolicLink(
    at: embeddedDirect,
    withDestinationURL: offline.appendingPathComponent("missing-embedded-manifest")
  )
  try expectInvalidTopology("direct tree with dangling embedded manifest")
  try FileManager.default.removeItem(at: embeddedDirect)

  try FileManager.default.removeItem(at: adjacent)
  try FileManager.default.createSymbolicLink(
    at: adjacent,
    withDestinationURL: root.appendingPathComponent("missing-adjacent-manifest")
  )
  try expectInvalidTopology("direct tree with dangling adjacent manifest")
  try FileManager.default.removeItem(at: adjacent)
  try "{\"tile_set_id\":\"\(tileID)\"}\n".write(to: adjacent, atomically: true, encoding: .utf8)

  try FileManager.default.createDirectory(at: embeddedDirect, withIntermediateDirectories: false)
  try expectInvalidTopology("direct tree with non-regular embedded manifest")
  try FileManager.default.removeItem(at: embeddedDirect)

  try FileManager.default.removeItem(at: adjacent)
  let previous = root.appendingPathComponent("offline.previous")
  try FileManager.default.createSymbolicLink(at: previous, withDestinationURL: offline)
  try expectInvalidTopology("direct unidentified tree with previous pointer")
  try FileManager.default.removeItem(at: previous)

  let generations = root.appendingPathComponent("tile-generations/foreign", isDirectory: true)
  try FileManager.default.createDirectory(at: generations, withIntermediateDirectories: true)
  try expectInvalidTopology("direct unidentified tree with generation artifact")
  try FileManager.default.removeItem(at: root.appendingPathComponent("tile-generations"))

  let transaction = root.appendingPathComponent(".tileset-transaction.json")
  try "{}\n".write(to: transaction, atomically: true, encoding: .utf8)
  try expectInvalidTopology("direct unidentified tree with helper transaction")
  try FileManager.default.removeItem(at: transaction)

  let authority = root.appendingPathComponent(".tileset-activation-authority.json")
  try "{}\n".write(to: authority, atomically: true, encoding: .utf8)
  try expectInvalidTopology("direct unidentified tree with helper activation authority")
  try FileManager.default.removeItem(at: authority)
  try FileManager.default.createSymbolicLink(
    at: transaction,
    withDestinationURL: root.appendingPathComponent("missing-transaction")
  )
  try expectInvalidTopology("direct unidentified tree with broken helper transaction link")
  try FileManager.default.removeItem(at: transaction)

  let tombstone = root.appendingPathComponent(".vtsc-cleanup-target")
  try FileManager.default.createDirectory(at: tombstone, withIntermediateDirectories: true)
  try expectInvalidTopology("direct unidentified tree with cleanup tombstone")
  try FileManager.default.removeItem(at: tombstone)
  try "{\"tile_set_id\":\"\(tileID)\"}\n".write(to: adjacent, atomically: true, encoding: .utf8)

  try FileManager.default.removeItem(at: offline)
  try expectInvalidTopology("missing active path with stale adjacent manifest")

  try FileManager.default.createSymbolicLink(
    at: offline,
    withDestinationURL: root.appendingPathComponent("missing-generation/offline")
  )
  try expectInvalidTopology("broken generation symlink")

  try FileManager.default.removeItem(at: offline)
  let generationID = "current"
  let generation = root.appendingPathComponent("tile-generations/\(generationID)/offline", isDirectory: true)
  try FileManager.default.createDirectory(at: generation, withIntermediateDirectories: true)
  try Data("tile-data".utf8).write(to: generation.appendingPathComponent("32--122"))
  try FileManager.default.createSymbolicLink(
    atPath: offline.path,
    withDestinationPath: "tile-generations/\(generationID)/offline"
  )
  try expectInvalidTopology("generation symlink without embedded manifest")

  let embedded = generation.appendingPathComponent(".tileset-manifest.json")
  try "{\"tile_set_id\":\"\(generationID)\"}\n".write(to: embedded, atomically: true, encoding: .utf8)
  let canonical = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  #expect(canonical.topology == "generation:\(generationID)")
  var canonicalFields = runtimeSnapshotMinimumFields()
  canonicalFields[.tileManifest] = Data(canonical.manifest.utf8)
  canonicalFields[.tileTopology] = Data(canonical.topology.utf8)
  canonicalFields[.tileTreeSHA256] = Data(canonical.treeSHA256.utf8)
  let canonicalRead = try TiciDeploymentSnapshotDecoder.decodeRead(
    TiciSnapshotWireCodec.encode(.init(rawValues: canonicalFields))
  ).snapshot
  #expect(canonicalRead.activeTileSetID == generationID)
  #expect(canonicalRead.activeTileTopology == .canonical)

  try FileManager.default.removeItem(at: offline)
  let legacyContainer = "legacy-0123456789abcdef"
  let logicalLegacyID = String(repeating: "c", count: 64)
  let legacyTargetID = String(repeating: "d", count: 64)
  let legacyGeneration = root.appendingPathComponent(
    "tile-generations/\(legacyContainer)/offline", isDirectory: true
  )
  try FileManager.default.createDirectory(at: legacyGeneration, withIntermediateDirectories: true)
  try Data("preserved-legacy-tile".utf8).write(to: legacyGeneration.appendingPathComponent("32--122"))
  let legacyEmbedded = legacyGeneration.appendingPathComponent(".tileset-manifest.json")
  func writeLegacyManifest(treeSHA256: String, container: String = legacyContainer) throws {
    let payload: [String: Any] = [
      "tile_set_id": logicalLegacyID,
      "legacy": true,
      "legacy_container_id": container,
      "legacy_migration_target_tile_set_id": legacyTargetID,
      "legacy_tree_sha256": treeSHA256,
    ]
    try JSONSerialization.data(withJSONObject: payload).write(to: legacyEmbedded)
  }
  try writeLegacyManifest(treeSHA256: String(repeating: "0", count: 64))
  try FileManager.default.createSymbolicLink(
    atPath: offline.path,
    withDestinationPath: "tile-generations/\(legacyContainer)/offline"
  )
  let legacyDigestProbe = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  try writeLegacyManifest(treeSHA256: legacyDigestProbe.treeSHA256)
  let legacy = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  var legacyFields = runtimeSnapshotMinimumFields()
  legacyFields[.tileManifest] = Data(legacy.manifest.utf8)
  legacyFields[.tileTopology] = Data(legacy.topology.utf8)
  legacyFields[.tileTreeSHA256] = Data(legacy.treeSHA256.utf8)
  let legacyRead = try TiciDeploymentSnapshotDecoder.decodeRead(
    TiciSnapshotWireCodec.encode(.init(rawValues: legacyFields))
  ).snapshot
  #expect(legacyRead.activeTileSetID == logicalLegacyID)
  #expect(legacyRead.activeTileTopology == .legacyMigration)
  #expect(legacyRead.activeTileContainerID == legacyContainer)
  #expect(legacyRead.activeTileLegacyMigrationTargetID == legacyTargetID)
  #expect(legacyRead.activeTileLegacyTreeSHA256 == legacy.treeSHA256)

  try writeLegacyManifest(treeSHA256: legacy.treeSHA256, container: "legacy-ffffffffffffffff")
  let wrongContainer = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  legacyFields[.tileManifest] = Data(wrongContainer.manifest.utf8)
  #expect(throws: TiciDeploymentSnapshotDecodeError.self) {
    try TiciDeploymentSnapshotDecoder.decodeRead(
      TiciSnapshotWireCodec.encode(.init(rawValues: legacyFields))
    )
  }
  for (label, manifestMutation, wireDigest) in [
    ("migration target", ["legacy_migration_target_tile_set_id": "../unsafe"], legacy.treeSHA256),
    ("tree digest", ["legacy_tree_sha256": String(repeating: "f", count: 64)], legacy.treeSHA256),
  ] {
    var object = try #require(
      JSONSerialization.jsonObject(with: Data(legacy.manifest.utf8)) as? [String: Any]
    )
    for (key, value) in manifestMutation { object[key] = value }
    legacyFields[.tileManifest] = try JSONSerialization.data(withJSONObject: object)
    legacyFields[.tileTreeSHA256] = Data(wireDigest.utf8)
    #expect(throws: TiciDeploymentSnapshotDecodeError.self, "legacy \(label) mismatch was accepted") {
      try TiciDeploymentSnapshotDecoder.decodeRead(
        TiciSnapshotWireCodec.encode(.init(rawValues: legacyFields))
      )
    }
  }

  try FileManager.default.removeItem(at: offline)

  let external = root.appendingPathComponent("external/offline", isDirectory: true)
  try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
  try "{\"tile_set_id\":\"external\"}\n".write(
    to: external.appendingPathComponent(".tileset-manifest.json"), atomically: true, encoding: .utf8
  )
  try FileManager.default.createSymbolicLink(at: offline, withDestinationURL: external)
  try expectInvalidTopology("absolute external symlink")

  for (label, target) in [
    ("traversal symlink", "../tile-generations/\(generationID)/offline"),
    ("wrong-shape symlink", "tile-generations/\(generationID)/extra/offline"),
  ] {
    try FileManager.default.removeItem(at: offline)
    try FileManager.default.createSymbolicLink(atPath: offline.path, withDestinationPath: target)
    try expectInvalidTopology(label)
  }

  try FileManager.default.removeItem(at: offline)
  try FileManager.default.createSymbolicLink(
    atPath: offline.path,
    withDestinationPath: "tile-generations/\(generationID)/offline"
  )
  try "{\"tile_set_id\":\"different\"}\n".write(to: embedded, atomically: true, encoding: .utf8)
  let mismatch = try runTileManifestProbeShell(offline: offline, adjacent: adjacent)
  #expect(mismatch.topology == "generation:\(generationID)")
  var mismatchFields = runtimeSnapshotMinimumFields()
  mismatchFields[.tileManifest] = Data(mismatch.manifest.utf8)
  mismatchFields[.tileTopology] = Data(mismatch.topology.utf8)
  mismatchFields[.tileTreeSHA256] = Data(mismatch.treeSHA256.utf8)
  #expect(throws: TiciDeploymentSnapshotDecodeError.self) {
    try TiciDeploymentSnapshotDecoder.decodeRead(
      TiciSnapshotWireCodec.encode(.init(rawValues: mismatchFields))
    )
  }
}

@Test func deploymentSnapshotDecoderOwnsQCacheAndManifestInterpretation() throws {
  let releaseID = "release-v1"
  let releaseDigest = TuneDeploymentIdentity.sha256Hex(Data(releaseID.utf8))
  let cachePath = "/data/media/0/osm/binaries/mapd-\(releaseDigest.prefix(16))-abcdef0123456789"
  let qSource = """
  Q_CURVE_ENABLED = True
  Q_CURVE_POINTS: list[tuple[float, float]] = [
    (1.000000e-04, 1.0000),
  ]
  """
  let fields: [TiciSnapshotWireField: Data] = [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data(String(repeating: "a", count: 40).utf8),
    .dirty: Data("0".utf8),
    .isOffroad: Data("1".utf8),
    .isOnroad: Data("0".utf8),
    .mapLookaheadEnabled: Data("0".utf8),
    .physicsAmplitude: Data("-1.658965".utf8),
    .physicsSteepness: Data("-1395.055546".utf8),
    .physicsCenter: Data("0.005397".utf8),
    .physicsBaseline: Data("4.107103".utf8),
    .physicsMinLatAccel: Data("2.4481".utf8),
    .physicsMaxLatAccel: Data("4.1071".utf8),
    .mapdReleaseVersion: Data(releaseID.utf8),
    .mapdVersion: Data(releaseID.utf8),
    .activeMapdSHA256: Data(String(repeating: "b", count: 64).utf8),
    .qCurveFile: Data(qSource.utf8),
    .tileManifest: Data("{\"tile_set_id\":\"tiles-v1\"}".utf8),
    .tileTopology: Data("generation:tiles-v1".utf8),
    .tileTreeSHA256: Data(String(repeating: "e", count: 64).utf8),
    .mapdCacheListing: Data("\(cachePath)\t\(String(repeating: "c", count: 64))\n".utf8),
  ]
  let read = try TiciDeploymentSnapshotDecoder.decodeRead(
    TiciSnapshotWireCodec.encode(.init(rawValues: fields))
  )

  #expect(read.snapshot.cachedMapdPath == cachePath)
  #expect(read.snapshot.cachedMapdSHA256 == String(repeating: "c", count: 64))
  #expect(read.snapshot.activeTileSetID == "tiles-v1")
  #expect(read.qCurve.enabled)
  #expect(read.qCurve.pointCount == 1)
  // The device-side source can contain unrelated whitespace. The Swift parser
  // deliberately hashes only the selected Q-curve declaration, matching the
  // legacy Python parser's identity semantics.
  let canonicalSelectedCurve = [
    "Q_CURVE_ENABLED = True",
    "Q_CURVE_POINTS: list[tuple[float, float]] = [",
    "  (1.000000e-04, 1.0000),",
    "]",
  ].joined(separator: "\n") + "\n"
  #expect(read.qCurve.sha256 == TuneDeploymentIdentity.sha256Hex(Data(canonicalSelectedCurve.utf8)))
}

@Test func deploymentSnapshotRejectsMalformedBootIdentity() throws {
  var fields = runtimeSnapshotMinimumFields()
  fields[.bootID] = Data("not-a-linux-boot-id".utf8)
  #expect(throws: TiciDeploymentSnapshotDecodeError.invalidBootID("not-a-linux-boot-id")) {
    try TiciDeploymentSnapshotDecoder.decodeRead(
      TiciSnapshotWireCodec.encode(.init(rawValues: fields))
    )
  }
}

@Test func runtimeSnapshotIncludesRawPostflightInputsAndBoundedControllerProbe() throws {
  let qSource = TuneDeploymentIdentity.canonicalQCurveSource(parameters: .checkoutFallback, bands: [])
  var raw: [TiciSnapshotWireField: Data] = [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data(String(repeating: "a", count: 40).utf8),
    .dirty: Data("0".utf8),
    .isOffroad: Data("1".utf8),
    .isOnroad: Data("0".utf8),
    .mapLookaheadEnabled: Data("0".utf8),
    .qCurveFile: Data(qSource.utf8),
    .activeMapdSHA256: Data(String(repeating: "b", count: 64).utf8),
    .activeMapdBuildInfo: Data("{}".utf8),
    .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
    .managerRunning: Data("1".utf8),
    .mapdRunning: Data("1".utf8),
    .remoteEpochMilliseconds: Data("1800000000000".utf8),
    .liveMapDataControllerStatus: Data("1|1|123456789|1|123456999".utf8),
    .runtimeEndIsOffroad: Data("1".utf8),
    .runtimeEndIsOnroad: Data("0".utf8),
    .runtimeEndMapLookaheadEnabled: Data("0".utf8),
    .tileManifest: Data(),
    .tileTopology: Data("direct-unidentified".utf8),
    .tileTreeSHA256: Data(String(repeating: "e", count: 64).utf8),
    .memoryWholeCurveProfile: Data("memory-profile".utf8),
    .persistentWholeCurveProfile: Data("persistent-profile".utf8),
    .persistentLastGPSPosition: Data("gps".utf8),
  ]
  let physics: [(TiciSnapshotWireField, String)] = [
    (.physicsAmplitude, "-1"), (.physicsSteepness, "-100"), (.physicsCenter, "0.01"),
    (.physicsBaseline, "4"), (.physicsMinLatAccel, "2"), (.physicsMaxLatAccel, "4"),
  ]
  for (field, value) in physics { raw[field] = Data(value.utf8) }
  let read = try TiciDeploymentSnapshotDecoder.decodeRuntimePostflight(
    TiciSnapshotWireCodec.encode(.init(rawValues: raw))
  )
  #expect(read.mapdRunning)
  #expect(read.managerRunning)
  #expect(read.remoteEpochMilliseconds == 1_800_000_000_000)
  #expect(read.wholeCurveProfile == Data("memory-profile".utf8))
  #expect(read.lastGPSPosition == Data("gps".utf8))
  #expect(read.liveMapDataControllerStatus == TiciLiveMapDataControllerStatus(
    updated: true,
    valid: true,
    logMonoTimeNs: 123_456_789,
    roadGeometryValid: true,
    sampleMonoTimeNs: 123_456_999
  ))
  #expect(read.runtimeEndIsOffroad)
  #expect(!read.runtimeEndIsOnroad)
  #expect(!read.runtimeEndMapLookaheadEnabled)

  let command = TiciSnapshotWireCommandBuilder.inspectionCommand(includeRuntimePostflight: true)
  #expect(command.contains("memory_params_root=/dev/shm/params/d"))
  #expect(command.contains(#"emit_file memory_whole_curve_profile "$memory_params_root/MapWholeCurveProfile""#))
  #expect(command.contains(#"emit_file memory_last_gps_position "$memory_params_root/LastGPSPosition""#))
  #expect(!command.contains("memory_params_root=/dev/shm/params\n"))
  #expect(command.contains("active_mapd_build_info"))
  #expect(command.contains("memory_whole_curve_profile"))
  #expect(command.contains("mapd_running"))
  #expect(command.contains("manager_running"))
  #expect(command.contains(#"[m]anager\.py"#))
  #expect(command.contains(#"$manager_proc/$manager_pid/cwd"#))
  #expect(command.contains(#"$manager_repo/system/manager"#))
  let staticCommand = TiciSnapshotWireCommandBuilder.inspectionCommand(includeStaticPostflight: true)
  #expect(staticCommand.contains("manager_running"))
  #expect(staticCommand.contains("mapd_running"))
  #expect(staticCommand.contains("active_mapd_build_info"))
  #expect(staticCommand.contains("runtime_end_is_offroad"))
  #expect(!staticCommand.contains("live_map_data_controller_status"))
  #expect(!staticCommand.contains("memory_whole_curve_profile"))
  #expect(!staticCommand.contains("memory_last_gps_position"))
  #expect(command.contains("remote_epoch_milliseconds"))
  #expect(command.contains("date +%s%3N"))
  #expect(command.contains("emit_command live_map_data_controller_status timeout 5"))
  #expect(command.contains("/usr/local/venv/bin/python -c"))
  #expect(command.contains("sm.update(2000)"))
  #expect(command.contains("time.clock_gettime_ns(time.CLOCK_BOOTTIME)"))
  #expect(!command.contains("time.monotonic_ns()"))
  #expect(command.contains(#"end="""#))
  #expect(command.contains("roadGeometryValid"))
  #expect(command.contains(#"emit_file runtime_end_is_offroad "$params_root/IsOffroad""#))
  #expect(command.contains(#"emit_file runtime_end_is_onroad "$params_root/IsOnroad""#))
  #expect(command.contains(#"emit_file runtime_end_map_lookahead_enabled "$params_root/MTSCLookaheadEnabled""#))
  let buildInfoIndex = try #require(command.range(of: "emit_command active_mapd_build_info")?.lowerBound)
  let mapdRunningIndex = try #require(command.range(of: "emit_text mapd_running")?.lowerBound)
  let controllerProbeIndex = try #require(command.range(of: "emit_command live_map_data_controller_status")?.lowerBound)
  let profileReadIndex = try #require(command.range(of: "emit_file memory_whole_curve_profile")?.lowerBound)
  let gpsReadIndex = try #require(command.range(of: "emit_file memory_last_gps_position")?.lowerBound)
  let endStateIndex = try #require(command.range(of: "emit_file runtime_end_is_offroad")?.lowerBound)
  let remoteTimeIndex = try #require(command.range(of: "emit_text remote_epoch_milliseconds")?.lowerBound)
  #expect(controllerProbeIndex < profileReadIndex)
  #expect(controllerProbeIndex < gpsReadIndex)
  #expect(profileReadIndex < endStateIndex)
  #expect(gpsReadIndex < endStateIndex)
  #expect(buildInfoIndex < endStateIndex)
  #expect(mapdRunningIndex < endStateIndex)
  #expect(endStateIndex < remoteTimeIndex)
}

@Test func managerProbeAcceptsOnlyTheActualSupervisedManagerProcess() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-manager-probe-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let repository = root.appendingPathComponent("openpilot", isDirectory: true)
  let managerDirectory = repository.appendingPathComponent("system/manager", isDirectory: true)
  let unrelatedDirectory = repository.appendingPathComponent("sunnypilot/models", isDirectory: true)
  let procRoot = root.appendingPathComponent("proc", isDirectory: true)
  try FileManager.default.createDirectory(at: managerDirectory, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: unrelatedDirectory, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: procRoot, withIntermediateDirectories: true)
  let canonicalManagerDirectory = URL(fileURLWithPath: try runShell(
    "cd '\(managerDirectory.path)' && pwd -P"
  ).trimmingCharacters(in: .whitespacesAndNewlines))

  func runFixture(pid: Int, cwd: URL, arguments: [String]) throws -> String {
    let processDirectory = procRoot.appendingPathComponent(String(pid), isDirectory: true)
    try? FileManager.default.removeItem(at: processDirectory)
    try FileManager.default.createDirectory(at: processDirectory, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
      at: processDirectory.appendingPathComponent("cwd"),
      withDestinationURL: cwd
    )
    var commandLine = Data()
    for argument in arguments {
      commandLine.append(Data(argument.utf8))
      commandLine.append(0)
    }
    try commandLine.write(to: processDirectory.appendingPathComponent("cmdline"))
    let fakePgrep = root.appendingPathComponent("pgrep")
    try "#!/bin/sh\nprintf '%s\\n' '\(pid)'\n".write(to: fakePgrep, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakePgrep.path)
    return try runManagerProbeShell(
      repository: repository.resolvingSymlinksInPath(),
      procRoot: procRoot,
      pgrep: fakePgrep
    )
  }

  let relativeResult = try runFixture(
    pid: 101,
    cwd: managerDirectory,
    arguments: ["python3", "./manager.py"]
  )
  #expect(relativeResult == "1")
  let absoluteResult = try runFixture(
    pid: 102,
    cwd: unrelatedDirectory,
    arguments: [
      "/usr/bin/python3",
      canonicalManagerDirectory.appendingPathComponent("manager.py").path,
    ]
  )
  #expect(absoluteResult == "1")
  let unrelatedResult = try runFixture(
    pid: 103,
    cwd: unrelatedDirectory,
    arguments: ["python3", "./manager.py"]
  )
  #expect(unrelatedResult == "0")

  let selfPgrep = root.appendingPathComponent("pgrep-self")
  let pidFile = root.appendingPathComponent("probe-shell-pid")
  try "#!/bin/sh\ncat '\(pidFile.path)'\n".write(to: selfPgrep, atomically: true, encoding: .utf8)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: selfPgrep.path)
  let fragment = TiciSnapshotWireCommandBuilder.managerProbeShellFragment(
    repositoryPath: repository.resolvingSymlinksInPath().path,
    procRoot: procRoot.path,
    pgrepCommand: selfPgrep.path
  )
  let selfScript = """
  set -eu
  self_pid=$$
  printf '%s' "$self_pid" > '\(pidFile.path)'
  mkdir -p '\(procRoot.path)'/"$self_pid"
  ln -s '\(managerDirectory.path)' '\(procRoot.path)'/"$self_pid"/cwd
  printf 'python3\\000./manager.py\\000' > '\(procRoot.path)'/"$self_pid"/cmdline
  manager_probe_result=
  emit_text() { manager_probe_result=$2; }
  \(fragment)
  printf '%s' "$manager_probe_result"
  """
  #expect(try runShell(selfScript) == "0")
}

@Test func runtimeSnapshotRejectsMalformedControllerProbeOutput() throws {
  var raw = runtimeSnapshotMinimumFields()
  raw[.liveMapDataControllerStatus] = Data("1|1|123|2|124".utf8)
  #expect(throws: TiciDeploymentSnapshotDecodeError.malformedLiveMapDataControllerStatus("1|1|123|2|124")) {
    try TiciDeploymentSnapshotDecoder.decodeRuntimePostflight(
      TiciSnapshotWireCodec.encode(.init(rawValues: raw))
    )
  }
}

@Test func runtimeSnapshotRejectsUnexpectedControllerProbeLineEnding() throws {
  var raw = runtimeSnapshotMinimumFields()
  raw[.liveMapDataControllerStatus] = Data("1|1|123|1|124\n".utf8)
  #expect(throws: TiciDeploymentSnapshotDecodeError.malformedLiveMapDataControllerStatus("1|1|123|1|124\n")) {
    try TiciDeploymentSnapshotDecoder.decodeRuntimePostflight(
      TiciSnapshotWireCodec.encode(.init(rawValues: raw))
    )
  }
}

private func runtimeSnapshotMinimumFields() -> [TiciSnapshotWireField: Data] {
  var raw: [TiciSnapshotWireField: Data] = [
    .branch: Data("chauffeur-exp01".utf8),
    .head: Data(String(repeating: "a", count: 40).utf8),
    .dirty: Data("0".utf8),
    .isOffroad: Data("1".utf8),
    .isOnroad: Data("0".utf8),
    .mapLookaheadEnabled: Data("0".utf8),
    .qCurveFile: Data(TuneDeploymentIdentity.canonicalQCurveSource(
      parameters: .checkoutFallback,
      bands: []
    ).utf8),
    .activeMapdSHA256: Data(String(repeating: "b", count: 64).utf8),
    .activeMapdBuildInfo: Data("{}".utf8),
    .activeMapdELFHeader: Data([0x7f, 0x45, 0x4c, 0x46, 2, 1] + Array(repeating: 0, count: 12) + [183, 0]),
    .managerRunning: Data("1".utf8),
    .mapdRunning: Data("1".utf8),
    .remoteEpochMilliseconds: Data("1800000000000".utf8),
    .liveMapDataControllerStatus: Data("1|1|123|1|124".utf8),
    .runtimeEndIsOffroad: Data("1".utf8),
    .runtimeEndIsOnroad: Data("0".utf8),
    .runtimeEndMapLookaheadEnabled: Data("0".utf8),
    .tileManifest: Data(),
    .tileTopology: Data("direct-unidentified".utf8),
    .tileTreeSHA256: Data(String(repeating: "e", count: 64).utf8),
  ]
  for (field, value) in [
    (TiciSnapshotWireField.physicsAmplitude, "-1"),
    (.physicsSteepness, "-100"),
    (.physicsCenter, "0.01"),
    (.physicsBaseline, "4"),
    (.physicsMinLatAccel, "2"),
    (.physicsMaxLatAccel, "4"),
  ] {
    raw[field] = Data(value.utf8)
  }
  return raw
}

private func runManagerProbeShell(repository: URL, procRoot: URL, pgrep: URL) throws -> String {
  let fragment = TiciSnapshotWireCommandBuilder.managerProbeShellFragment(
    repositoryPath: repository.path,
    procRoot: procRoot.path,
    pgrepCommand: pgrep.path
  )
  return try runShell("""
  set -eu
  manager_probe_result=
  emit_text() { manager_probe_result=$2; }
  \(fragment)
  printf '%s' "$manager_probe_result"
  """)
}

private func runTileManifestProbeShell(
  offline: URL,
  adjacent: URL
) throws -> (topology: String, manifest: String, treeSHA256: String) {
  let fragment = TiciSnapshotWireCommandBuilder.tileManifestProbeShellFragment(
    offlinePath: offline.path,
    adjacentManifestPath: adjacent.path,
    pythonCommand: "/usr/bin/python3"
  )
  let output = try runShell("""
  set -eu
  tile_probe_topology=
  tile_probe_manifest=
  tile_probe_tree_sha256=
  compact_base64() {
    while IFS= read -r chunk || [ -n "$chunk" ]; do printf '%s' "$chunk"; done
  }
  file_sha256() {
    /usr/bin/shasum -a 256 "$1" | awk '{ print $1 }'
  }
  sha256sum() {
    /usr/bin/shasum -a 256 "$@"
  }
  emit_file() {
    if [ "$1" = tile_manifest ]; then tile_probe_manifest=$(cat "$2"); fi
  }
  emit_text() {
    if [ "$1" = tile_topology ]; then tile_probe_topology=$2; fi
    if [ "$1" = tile_manifest ]; then tile_probe_manifest=$2; fi
    if [ "$1" = tile_tree_sha256 ]; then tile_probe_tree_sha256=$2; fi
  }
  \(fragment)
  printf '%s\n%s\n%s' "$tile_probe_topology" "$tile_probe_manifest" "$tile_probe_tree_sha256"
  """)
  let lines = output.split(separator: "\n", omittingEmptySubsequences: false)
  guard lines.count >= 3 else {
    throw NSError(domain: "TileManifestProbe", code: 1)
  }
  return (String(lines[0]), String(lines[1]), String(lines[2]))
}

private func runShell(_ script: String) throws -> String {
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", script]
  let output = Pipe()
  let errors = Pipe()
  process.standardOutput = output
  process.standardError = errors
  try process.run()
  process.waitUntilExit()
  let standardOutput = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
  let standardError = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
  guard process.terminationStatus == 0 else {
    throw NSError(
      domain: "ManagerProbeShell",
      code: Int(process.terminationStatus),
      userInfo: [NSLocalizedDescriptionKey: standardError]
    )
  }
  return standardOutput
}
