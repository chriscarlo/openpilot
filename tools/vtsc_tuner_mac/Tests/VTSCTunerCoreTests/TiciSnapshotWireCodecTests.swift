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

@Test func ticiSnapshotWireCommandUsesOnlyShellToolsAndNoPython() throws {
  let command = TiciSnapshotWireCommandBuilder.inspectionCommand()
  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("base64"))
  #expect(command.contains("git -C"))
  #expect(command.contains("sha256sum"))
  #expect(command.contains("/data/params/d"))
  #expect(command.contains("q_curve_file"))
  #expect(command.contains("tile_manifest"))
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

@Test func runtimeSnapshotIncludesRawPostflightInputsWithoutPython() throws {
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
    .mapdRunning: Data("1".utf8),
    .remoteEpochMilliseconds: Data("1800000000000".utf8),
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
  #expect(read.remoteEpochMilliseconds == 1_800_000_000_000)
  #expect(read.wholeCurveProfile == Data("memory-profile".utf8))
  #expect(read.lastGPSPosition == Data("gps".utf8))

  let command = TiciSnapshotWireCommandBuilder.inspectionCommand(includeRuntimePostflight: true)
  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("active_mapd_build_info"))
  #expect(command.contains("memory_whole_curve_profile"))
  #expect(command.contains("mapd_running"))
  #expect(command.contains("remote_epoch_milliseconds"))
  #expect(command.contains("date +%s%3N"))
}
