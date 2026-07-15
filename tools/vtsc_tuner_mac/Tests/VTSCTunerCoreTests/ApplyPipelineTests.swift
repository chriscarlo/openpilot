import Foundation
import Testing
@testable import VTSCTunerCore

@Test func applyActionsKeepRustSupersetOrder() {
  #expect(ApplyAction.allCases == [
    .local,
    .commit,
    .push,
    .pullOnTici,
    .rebuildTilesAndReboot,
  ])
  #expect(ApplyAction.pullOnTici.label.contains("keep current tiles"))
  #expect(ApplyAction.rebuildTilesAndReboot.label.contains("build canonical tiles"))
}

@Test func productionTilePlansSeparateRuntimePullFromCanonicalRebuild() {
  let tune = Tune(params: .checkoutFallback)
  let repository = URL(fileURLWithPath: "/repo")
  let explicit = URL(fileURLWithPath: "/tmp/exact-tile-set")
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .pullOnTici,
    tune: tune,
    repositoryRoot: repository
  )) == .unchanged)
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: repository
  )) == .generateCanonical)
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .pullOnTici,
    tune: tune,
    repositoryRoot: repository,
    tileSetArtifactURL: explicit
  )) == .validateExplicit(explicit))
  #expect(ApplyPipeline.productionTilePlan(for: ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: repository,
    tileSetArtifactURL: explicit
  )) == .validateExplicit(explicit))
}

@Test func mapdConfigDecodesRustCompatiblePaths() throws {
  let data = #"""
  {
    "pbf_path": "/tmp/ready.osm.pbf",
    "mapd_repo_path": "/tmp/openpilot-mapd",
    "mapd_binary_path": "/tmp/mapd-darwin",
    "regions_override": [{"min_lat": 32, "min_lon": -118}]
  }
  """#.data(using: .utf8)!
  let config = try JSONDecoder().decode(MapdConfig.self, from: data)
  #expect(config.pbfURL.path == "/tmp/ready.osm.pbf")
  #expect(config.effectiveBinaryURL.path == "/tmp/mapd-darwin")
  #expect(config.regionsOverride == [RegionBox(minLatitude: 32, minLongitude: -118)])
}

@Test func mapdConfigRejectsAnExecutableELFBinary() throws {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-mapd-test-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let binaryURL = directory.appendingPathComponent("mapd")
  try Data([0x7f, 0x45, 0x4c, 0x46, 0, 0, 0, 0]).write(to: binaryURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binaryURL.path)
  let config = MapdConfig(
    pbfURL: directory.appendingPathComponent("map.osm.pbf"),
    mapdRepositoryURL: directory,
    mapdBinaryURL: binaryURL
  )
  #expect(throws: MapdConfigError.notDarwinBinary(binaryURL)) {
    try config.validateDarwinBinary()
  }
}

@Test func mapdGenerationPassesEveryTunedPhysicsValue() {
  let parameters = SigmoidParameters.checkoutFallback
  let arguments = MapdCommandBuilder.generationArguments(
    parameters: parameters,
    region: RegionBox(minLatitude: 32, minLongitude: -118)
  )
  #expect(arguments.contains("--minlat=32"))
  #expect(arguments.contains("--maxlon=-116"))
  #expect(arguments.contains("--phys-a=-1.658965"))
  #expect(arguments.contains("--phys-b=-1395.055546"))
  #expect(arguments.contains("--phys-c=0.005397"))
  #expect(arguments.contains("--phys-d=4.107103"))
  #expect(arguments.contains("--phys-min-lat=2.4481"))
  #expect(arguments.contains("--phys-max-lat=4.1071"))
}

@Test func rebuildGenerationPersistsAndFullDecodesCanonicalArtifact() async throws {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-canonical-generation-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }

  let mapdRepository = directory.appendingPathComponent("openpilot-mapd", isDirectory: true)
  try FileManager.default.createDirectory(at: mapdRepository, withIntermediateDirectories: true)
  try "schema".write(
    to: mapdRepository.appendingPathComponent("offline.capnp"),
    atomically: true,
    encoding: .utf8
  )
  let pbfURL = directory.appendingPathComponent("prepared.osm.pbf")
  try Data(repeating: 0x51, count: 512).write(to: pbfURL)
  let generatorURL = directory.appendingPathComponent("mapd-darwin")
  try Data([0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0]).write(to: generatorURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: generatorURL.path)
  let decoderURL = directory.appendingPathComponent("vtsc-tile-decoder")
  try Data("#!/bin/sh\nexit 0\n".utf8).write(to: decoderURL)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: decoderURL.path)

  let region = RegionBox(minLatitude: 32, minLongitude: -118)
  let config = try MapdConfig(
    pbfURL: pbfURL,
    mapdRepositoryURL: mapdRepository,
    mapdBinaryURL: generatorURL,
    regionsOverride: [region]
  ).validated()
  try config.validateDarwinBinary()
  let tune = Tune(params: .checkoutFallback)
  let identity = TuneDeploymentIdentity(tune: tune)
  let release = ValidatedMapdReleaseArtifact(
    artifact: MapdReleaseArtifact(
      releaseID: "whole-curve-release-v1",
      buildID: "build-abc123",
      binaryURL: directory.appendingPathComponent("mapd-arm64"),
      sha256: String(repeating: "a", count: 64)
    ),
    byteCount: 1_000_000,
    persistentCacheFileName: "mapd-release"
  )
  let setsRoot = directory.appendingPathComponent("map_tiles/sets", isDirectory: true)
  let request = ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: tune,
    repositoryRoot: URL(fileURLWithPath: "/repo"),
    tileSetsRootURL: setsRoot,
    tileDecoderURL: decoderURL
  )
  let runner = CanonicalGenerationRunner(
    generatorURL: generatorURL,
    decoderURL: decoderURL,
    sigmoidHash: identity.tileSigmoidHash
  )
  let artifact = try await ApplyPipeline(processRunner: runner).generateCanonicalTileSet(
    request: request,
    config: config,
    regions: [region],
    release: release,
    tuneIdentity: identity,
    decoderURL: decoderURL
  )

  #expect(artifact.rootURL.deletingLastPathComponent() == setsRoot.standardizedFileURL)
  #expect(artifact.manifest.regions == [region])
  #expect(artifact.manifest.fileCount == 1)
  #expect(artifact.manifest.tuneIdentitySHA256 == identity.identitySHA256)
  #expect(artifact.manifest.tileSigmoidHash == identity.tileSigmoidHash)
  #expect(artifact.manifest.pbfSHA256 == (try FileSHA256.hex(pbfURL)))
  #expect(FileManager.default.fileExists(atPath: artifact.offlineURL.path))
  let requests = await runner.requests
  let generation = try #require(requests.first { $0.executableURL == generatorURL })
  #expect(generation.arguments == MapdCommandBuilder.generationArguments(
    parameters: .checkoutFallback,
    region: region
  ))
  #expect(requests.filter { $0.executableURL == decoderURL }.count == 2)
}

@Test func ticiPhysicsMigrationPassesEverySourceRoundedValue() {
  let command = TiciPhysicsCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)
  #expect(command.hasPrefix("cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 tools/vtsc/apply_physics_params.py"))
  #expect(command.contains("--amplitude -1.658965"))
  #expect(command.contains("--steepness -1395.055546"))
  #expect(command.contains("--center 0.005397"))
  #expect(command.contains("--baseline 4.107103"))
  #expect(command.contains("--min-lat 2.4481"))
  #expect(command.contains("--max-lat 4.1071"))
}

@Test func tuneSaveFailureStopsBeforeSourceMutation() async {
  let collector = ApplyEventCollector()
  let pipeline = ApplyPipeline()
  let request = ApplyRequest(
    action: .local,
    tune: Tune(created: "2026-07-12T00:00:00-07:00", params: .checkoutFallback),
    repositoryRoot: URL(fileURLWithPath: "/definitely/not/a/repository"),
    tuneURL: URL(fileURLWithPath: "/dev/null/current.tune.json")
  )
  let succeeded = await pipeline.apply(request) { event in
    await collector.append(event)
  }
  let events = await collector.events
  #expect(!succeeded)
  #expect(events.contains(.step(ApplyStepEvent(
    id: 1,
    status: .running,
    text: "Saving the tune file…"
  ))))
  #expect(events.contains { event in
    if case let .step(step) = event { step.id == 1 && step.status == .failed } else { false }
  })
  #expect(!events.contains { event in
    if case let .step(step) = event { step.id == 2 } else { false }
  })
  #expect(events.last == .finished(success: false))
}

@Test func rebuildPreflightFailsBeforeWritingTuneOrTouchingGit() async {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-rebuild-preflight-\(UUID().uuidString)", isDirectory: true)
  try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let tuneURL = directory.appendingPathComponent("should-not-exist.json")
  let collector = ApplyEventCollector()
  let pipeline = ApplyPipeline()
  let succeeded = await pipeline.apply(ApplyRequest(
    action: .rebuildTilesAndReboot,
    tune: Tune(created: "2026-07-12T00:00:00-07:00", params: .checkoutFallback),
    repositoryRoot: URL(fileURLWithPath: "/definitely/not/a/repository"),
    tuneURL: tuneURL,
    mapdConfigURL: directory.appendingPathComponent("missing-mapd.json")
  )) { event in
    await collector.append(event)
  }
  let events = await collector.events
  #expect(!succeeded)
  #expect(!FileManager.default.fileExists(atPath: tuneURL.path))
  #expect(events.contains { event in
    if case let .step(step) = event { step.id == 0 && step.status == .failed } else { false }
  })
  #expect(!events.contains { event in
    if case let .step(step) = event { step.id == 1 } else { false }
  })
}

private actor ApplyEventCollector {
  private(set) var events: [ApplyEvent] = []

  func append(_ event: ApplyEvent) {
    events.append(event)
  }
}

private actor CanonicalGenerationRunner: ProcessRunning {
  let generatorURL: URL
  let decoderURL: URL
  let sigmoidHash: String
  private(set) var requests: [ProcessRequest] = []

  init(generatorURL: URL, decoderURL: URL, sigmoidHash: String) {
    self.generatorURL = generatorURL
    self.decoderURL = decoderURL
    self.sigmoidHash = sigmoidHash
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == generatorURL {
      let latitude = try argumentValue("--minlat=", in: request.arguments)
      let longitude = try argumentValue("--minlon=", in: request.arguments)
      let output = try #require(request.currentDirectoryURL)
        .appendingPathComponent("offline", isDirectory: true)
        .appendingPathComponent(String(latitude), isDirectory: true)
        .appendingPathComponent(String(longitude), isDirectory: true)
      try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
      let name = String(
        format: "%d.000000_%d.000000_%d.000000_%d.000000",
        latitude, longitude, latitude + 2, longitude + 2
      )
      try Data(repeating: 0x77, count: 256).write(to: output.appendingPathComponent(name))
      return ProcessResult(terminationStatus: 0, standardOutput: "generated\n", standardError: "")
    }
    if request.executableURL == decoderURL {
      let encoder = JSONEncoder()
      let lines = try request.arguments.map { path -> String in
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let bounds = try #require(MapTileIndex.parseBounds(filename: url.lastPathComponent))
        let tile = MapTile(
          sourcePath: url.path,
          bounds: bounds,
          overlap: 0.01,
          schemaVersion: 1,
          sigmoidHash: sigmoidHash,
          ways: []
        )
        return String(decoding: try encoder.encode(tile), as: UTF8.self)
      }
      return ProcessResult(
        terminationStatus: 0,
        standardOutput: lines.joined(separator: "\n") + "\n",
        standardError: ""
      )
    }
    return ProcessResult(terminationStatus: 1, standardOutput: "", standardError: "unexpected executable")
  }

  private func argumentValue(_ prefix: String, in arguments: [String]) throws -> Int {
    let value = try #require(arguments.first { $0.hasPrefix(prefix) })
    return try #require(Int(value.dropFirst(prefix.count)))
  }
}
