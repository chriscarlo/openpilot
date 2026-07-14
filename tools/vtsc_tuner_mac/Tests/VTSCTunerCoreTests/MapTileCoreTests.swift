import Foundation
import Testing
@testable import VTSCTunerCore

@Test func mapBakeMathMatchesGoReferences() {
  let parameters = SigmoidParameters.checkoutFallback
  #expect(MapBakeMath.sigmoidHash(parameters: parameters) == "22bd4f8d9bae")
  #expect(abs(MapBakeMath.bakedSpeedMPS(curvature: 0.01, parameters: parameters) - 15.655_132_108_333_34) < 1.0e-10)
  #expect(MapBakeMath.bakedSpeedMPS(curvature: 0, parameters: parameters) == 70)

  let previous = MapTileNode(latitude: 38.729_606, longitude: -120.801_668)
  let current = MapTileNode(latitude: 38.729_109, longitude: -120.803_160)
  let next = MapTileNode(latitude: 38.729_230, longitude: -120.803_468)
  let curvature = MapBakeMath.curvature(previous: previous, current: current, next: next)
  #expect(abs(curvature - 0.009_454_930_222_862_406) < 1.0e-12)
}

@Test func mapBakeEndpointsAndInteriorSpeedsMatchNodeCount() {
  let nodes = [
    MapTileNode(latitude: 38.0, longitude: -121.0),
    MapTileNode(latitude: 38.000_5, longitude: -120.999_5),
    MapTileNode(latitude: 38.001_0, longitude: -121.0),
    MapTileNode(latitude: 38.001_5, longitude: -120.999_5),
  ]
  let speeds = MapBakeMath.bakedSpeeds(nodes: nodes, parameters: .checkoutFallback)
  #expect(speeds.count == nodes.count)
  #expect(speeds.first == 70)
  #expect(speeds.last == 70)
  #expect(speeds[1] < 70)
}

@Test func runtimeSmoothedCurvatureMatchesProductionUS50Reference() throws {
  // Actual US 50 source nodes around saved bank item b432...:4.  The center
  // raw vertex is a digitization spike; production mapd averages the three
  // represented triplets over these five nodes.
  let nodes = [
    MapTileNode(latitude: 38.731_895_9, longitude: -120.746_461_8),
    MapTileNode(latitude: 38.731_805_2, longitude: -120.746_579_5),
    MapTileNode(latitude: 38.731_738_6, longitude: -120.746_668_9),
    MapTileNode(latitude: 38.731_675_6, longitude: -120.746_771_3),
    MapTileNode(latitude: 38.731_610_4, longitude: -120.746_885_6),
  ]
  let raw = MapBakeMath.curvature(previous: nodes[1], current: nodes[2], next: nodes[3])
  let estimate = try #require(MapBakeMath.runtimeCurvatureEstimate(nodes: nodes, nodeIndex: 2))

  #expect(abs(raw - 0.008_578_403_275_769_536) < 1.0e-11)
  #expect(abs(estimate.curvature - 0.004_181_845_727_569) < 1.0e-11)
  #expect(abs(estimate.supportMeters - 48.683_769_3) < 1.0e-4)
  #expect(raw / estimate.curvature > 2.0)
}

@Test func runtimeSmoothedCurvatureRequiresFiveNodeContext() {
  let nodes = [
    MapTileNode(latitude: 38.0, longitude: -121.0),
    MapTileNode(latitude: 38.001, longitude: -120.999),
    MapTileNode(latitude: 38.002, longitude: -121.0),
    MapTileNode(latitude: 38.003, longitude: -120.999),
  ]
  #expect(MapBakeMath.runtimeSmoothedCurvature(nodes: nodes, nodeIndex: 1) == nil)
  #expect(MapBakeMath.runtimeSmoothedCurvature(nodes: nodes, nodeIndex: 2) == nil)
}

@Test func runtimeSmoothedCurvatureMatchesMapdLaneIncreaseCorrection() throws {
  // Actual US 50 bank item 6b532...:1.  The selected four-lane way begins
  // after a three-lane one-way segment, so mapd clamps the raw transition
  // measurement to 0.0015 before applying its weighted average.
  let nodes = [
    MapTileNode(latitude: 38.729_059_6, longitude: -120.809_917_1),
    MapTileNode(latitude: 38.728_602_6, longitude: -120.809_233_6),
    MapTileNode(latitude: 38.728_499_1, longitude: -120.809_074_8),
    MapTileNode(latitude: 38.728_414_3, longitude: -120.808_895_7),
    MapTileNode(latitude: 38.728_334_8, longitude: -120.808_659_6),
    MapTileNode(latitude: 38.728_267_8, longitude: -120.808_402_2),
  ]
  let uncorrected = try #require(
    MapBakeMath.runtimeCurvatureEstimate(nodes: nodes, nodeIndex: 3)
  )
  let corrected = try #require(MapBakeMath.runtimeCurvatureEstimate(
    nodes: nodes,
    nodeIndex: 3,
    mergeOrSplitNodeIndices: [2]
  ))

  #expect(abs(uncorrected.curvature - 0.006_101_982_700_964_803) < 1.0e-11)
  #expect(abs(corrected.curvature - 0.002_332_571_7) < 1.0e-9)
  #expect(uncorrected.curvature / corrected.curvature > 2.6)
}

@Test func tileFilenameIndexFindsOnlyIntersectingCells() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-index-\(UUID().uuidString)/offline", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
  let group = root.appendingPathComponent("34/-120", isDirectory: true)
  try FileManager.default.createDirectory(at: group, withIntermediateDirectories: true)
  for name in [
    "34.000000_-120.000000_34.250000_-119.750000",
    "34.250000_-120.000000_34.500000_-119.750000",
    "not_a_tile",
  ] {
    #expect(FileManager.default.createFile(atPath: group.appendingPathComponent(name).path, contents: Data()))
  }
  let index = try MapTileIndex(rootURL: root)
  #expect(index.entries.count == 2)
  let visible = MapTileBounds(
    minLatitude: 34.05,
    minLongitude: -119.95,
    maxLatitude: 34.10,
    maxLongitude: -119.90
  )
  #expect(index.entries(intersecting: visible).count == 1)
  #expect(index.bounds == MapTileBounds(
    minLatitude: 34.0,
    minLongitude: -120.0,
    maxLatitude: 34.5,
    maxLongitude: -119.75
  ))
}

@Test func helperDecoderMapsCompactJSONToPublicModels() async throws {
  let tileURL = URL(fileURLWithPath: "/tmp/34.000000_-120.000000_34.250000_-119.750000")
  let json = #"{"source_path":"/tmp/34.000000_-120.000000_34.250000_-119.750000","bounds":{"min_latitude":34,"min_longitude":-120,"max_latitude":34.25,"max_longitude":-119.75},"overlap":0.01,"schema_version":1,"sigmoid_hash":"f9d38ab3357c","ways":[{"stable_id":"abc","name":"Curve Road","reference":"CA 1","bounds":{"min_latitude":34.1,"min_longitude":-119.9,"max_latitude":34.2,"max_longitude":-119.8},"nodes":[{"latitude":34.1,"longitude":-119.9,"baked_speed_mps":12.5}],"max_speed_mps":20,"max_speed_forward_mps":21,"max_speed_backward_mps":19,"advisory_speed_mps":13,"lanes":2,"hazard":"","one_way":true,"winding_forward_level":3,"winding_backward_level":2,"winding_forward_score":140,"winding_backward_score":120,"winding_forward_confidence":200,"winding_backward_confidence":180}]}"#
  let helper = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-helper-\(UUID().uuidString)")
  #expect(FileManager.default.createFile(atPath: helper.path, contents: Data()))
  try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
  defer { try? FileManager.default.removeItem(at: helper) }
  let decoder = MapTileHelperDecoder(
    helperURL: helper,
    processRunner: FixedProcessRunner(result: ProcessResult(
      terminationStatus: 0,
      standardOutput: json + "\n",
      standardError: ""
    ))
  )
  let tiles = try await decoder.decode(fileURLs: [tileURL])
  #expect(tiles.count == 1)
  #expect(tiles[0].ways.first?.name == "Curve Road")
  #expect(tiles[0].ways.first?.reference == "CA 1")
  #expect(tiles[0].ways.first?.nodes.first?.bakedSpeedMPS == 12.5)
  #expect(tiles[0].ways.first?.oneWay == true)
}

@Test func helperLocatorSupportsSwiftRunDevelopmentBuild() throws {
  let package = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-helper-locator-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: package) }
  let helper = package
    .appendingPathComponent(".build-tools/bin/vtsc-tile-decoder", isDirectory: false)
  try FileManager.default.createDirectory(
    at: helper.deletingLastPathComponent(),
    withIntermediateDirectories: true
  )
  #expect(FileManager.default.createFile(atPath: helper.path, contents: Data()))
  try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)

  let resolved = try MapTileHelperLocator.bundledURL(
    environment: [:],
    currentDirectoryURL: package
  )
  #expect(resolved == helper.standardizedFileURL)
}

@Test func tileStoreCachesDecodedVisibleTiles() async throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-store-\(UUID().uuidString)/offline", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
  let group = root.appendingPathComponent("34/-120", isDirectory: true)
  try FileManager.default.createDirectory(at: group, withIntermediateDirectories: true)
  let tileURL = group.appendingPathComponent("34.000000_-120.000000_34.250000_-119.750000")
  #expect(FileManager.default.createFile(atPath: tileURL.path, contents: Data()))
  let decoder = CountingTileDecoder()
  let store = try MapTileStore(rootURL: root, decoder: decoder)
  let visible = MapTileBounds(
    minLatitude: 34.0,
    minLongitude: -120.0,
    maxLatitude: 34.2,
    maxLongitude: -119.8
  )
  _ = try await store.tiles(intersecting: visible)
  _ = try await store.tiles(intersecting: visible)
  #expect(await decoder.decodeCount == 1)
}

@Test func tileStoreKeepsGoodVisibleTileWhenNeighborDecodeFails() async throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-store-partial-\(UUID().uuidString)/offline", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  let bad = root.appendingPathComponent("34.000000_-120.000000_34.250000_-119.750000")
  let good = root.appendingPathComponent("34.000000_-119.750000_34.250000_-119.500000")
  #expect(FileManager.default.createFile(atPath: bad.path, contents: Data()))
  #expect(FileManager.default.createFile(atPath: good.path, contents: Data()))
  let store = try MapTileStore(rootURL: root, decoder: SelectiveTileDecoder(badURL: bad))
  let tiles = try await store.tiles(intersecting: MapTileBounds(
    minLatitude: 34.0,
    minLongitude: -120.0,
    maxLatitude: 34.2,
    maxLongitude: -119.55
  ))
  #expect(tiles.map(\.sourcePath) == [good.standardizedFileURL.path])
}

@Test func ticiSyncStagesAndSwapsWithoutMixingOldAndCurrentTiles() async throws {
  let destination = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-sync-\(UUID().uuidString)/offline", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: destination.deletingLastPathComponent()) }
  let group = destination.appendingPathComponent("34/-120", isDirectory: true)
  try FileManager.default.createDirectory(at: group, withIntermediateDirectories: true)
  let stale = group.appendingPathComponent("34.000000_-120.000000_34.250000_-119.750000")
  #expect(FileManager.default.createFile(
    atPath: stale.path,
    contents: Data()
  ))
  let runner = RecordingProcessRunner()
  let service = TiciMapTileSyncService(processRunner: runner)
  let result = try await service.sync(profile: "commaAdb", destinationRootURL: destination)
  #expect(result.fileCount == 1)
  #expect(!FileManager.default.fileExists(atPath: stale.path))
  #expect(FileManager.default.fileExists(
    atPath: destination
      .appendingPathComponent("36.000000_-122.000000_36.250000_-121.750000").path
  ))
  #expect(FileManager.default.fileExists(
    atPath: destination.deletingLastPathComponent()
      .appendingPathComponent("offline.previous")
      .appendingPathComponent("34/-120")
      .appendingPathComponent(stale.lastPathComponent).path
  ))
  let request = try #require(await runner.requests.first)
  #expect(request.arguments.contains("commaAdb:/data/media/0/osm/offline/"))
  #expect(request.arguments.last?.contains(".offline.sync-") == true)
}

private struct FixedProcessRunner: ProcessRunning {
  let result: ProcessResult

  func run(_ request: ProcessRequest) async throws -> ProcessResult { result }
}

private actor CountingTileDecoder: MapTileDecoding {
  private(set) var decodeCount = 0

  func decode(fileURLs: [URL]) async throws -> [MapTile] {
    decodeCount += 1
    return fileURLs.map { url in
      MapTile(
        sourcePath: url.path,
        bounds: MapTileIndex.parseBounds(filename: url.lastPathComponent)!,
        overlap: 0.01,
        schemaVersion: 1,
        sigmoidHash: "f9d38ab3357c",
        ways: []
      )
    }
  }
}

private struct SelectiveTileDecoder: MapTileDecoding {
  let badURL: URL

  func decode(fileURLs: [URL]) async throws -> [MapTile] {
    let url = fileURLs[0].standardizedFileURL
    if url == badURL.standardizedFileURL {
      throw MapTileDecoderError.invalidOutput("synthetic corrupt neighbor")
    }
    return [MapTile(
      sourcePath: url.path,
      bounds: MapTileIndex.parseBounds(filename: url.lastPathComponent)!,
      overlap: 0.01,
      schemaVersion: 1,
      sigmoidHash: "f9d38ab3357c",
      ways: []
    )]
  }
}

private actor RecordingProcessRunner: ProcessRunning {
  private(set) var requests: [ProcessRequest] = []

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if let destinationArgument = request.arguments.last {
      let destination = URL(fileURLWithPath: String(destinationArgument.dropLast()))
      #expect(FileManager.default.createFile(
        atPath: destination
          .appendingPathComponent("36.000000_-122.000000_36.250000_-121.750000").path,
        contents: Data()
      ))
    }
    return ProcessResult(terminationStatus: 0, standardOutput: "synced", standardError: "")
  }
}
