import Foundation
import Testing
@testable import VTSCTunerCore

@Test func exactParkedGateFailsClosedForMissingEmptyGarbageAndContradictoryParams() throws {
  let cases: [(String?, String?, String?)] = [
    (nil, "0", "0"),
    ("1", nil, "0"),
    ("1", "0", nil),
    ("", "0", "0"),
    ("1", "", "0"),
    ("1", "0", ""),
    ("garbage", "0", "0"),
    ("1", "garbage", "0"),
    ("1", "0", "garbage"),
    ("1", "1", "0"),
    ("0", "0", "0"),
    ("1", "0", "1"),
  ]
  for (offroad, onroad, lookahead) in cases {
    let root = try makeGateFixture(offroad: offroad, onroad: onroad, lookahead: lookahead)
    defer { try? FileManager.default.removeItem(at: root) }
    let marker = root.appendingPathComponent("mutated")
    let command = TiciParkedMutationGate.guardedCommand(
      "touch \(shellQuoteForGateTest(marker.path))",
      paramsDirectory: root.appendingPathComponent("params").path,
      refusalMessage: "unsafe"
    )
    #expect(try runGateShell(command) != 0)
    #expect(!FileManager.default.fileExists(atPath: marker.path))
  }
}

@Test func exactParkedGateAllowsOnlyOneZeroZero() throws {
  let root = try makeGateFixture(offroad: "1", onroad: "0", lookahead: "0")
  defer { try? FileManager.default.removeItem(at: root) }
  let marker = root.appendingPathComponent("mutated")
  let command = TiciParkedMutationGate.guardedCommand(
    "touch \(shellQuoteForGateTest(marker.path))",
    paramsDirectory: root.appendingPathComponent("params").path,
    refusalMessage: "unsafe"
  )
  #expect(try runGateShell(command) == 0)
  #expect(FileManager.default.fileExists(atPath: marker.path))
}

@Test func allProductionMutationBuildersUseTheSharedExactPredicate() throws {
  let head = String(repeating: "a", count: 40)
  let git = try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(branch: "chauffeur-exp01", head: head)
  let physics = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)
  let reboot = TiciRebootCommandBuilder.command()
  let recovery = TiciMapdReleaseTransactionCommandBuilder.recoveryCommand()
  let rollback = try TiciProductionRollbackCommandBuilder.command(journal: rollbackJournalForGateTest())
  let tile = try TiciTileSetDeploymentService.atomicActivationCommand(
    helperPath: "/data/media/0/osm/binaries/vtsc-tile-transaction-\(String(repeating: "a", count: 16))",
    stagingRoot: TiciTileSetDeploymentService.stagingRoot(tileSetID: String(repeating: "b", count: 64)),
    tileSetID: String(repeating: "b", count: 64)
  )
  for command in [git, physics, reboot, recovery, rollback, tile] {
    #expect(command.contains("[ \"$offroad\" != '1' ] || [ \"$onroad\" != '0' ] || [ \"$lookahead\" != '0' ]"))
  }
  #expect(git.components(separatedBy: "[ \"$offroad\" != '1' ]").count - 1 == 2)
  #expect(physics.components(separatedBy: "[ \"$offroad\" != '1' ]").count - 1 == 2)
  #expect(rollback.components(separatedBy: "[ \"$offroad\" != '1' ]").count - 1 >= 4)
}

private func makeGateFixture(
  offroad: String?,
  onroad: String?,
  lookahead: String?
) throws -> URL {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-parked-gate-\(UUID().uuidString)", isDirectory: true)
  let params = root.appendingPathComponent("params", isDirectory: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  for (key, value) in [
    ("IsOffroad", offroad),
    ("IsOnroad", onroad),
    ("MTSCLookaheadEnabled", lookahead),
  ] {
    if let value {
      try value.write(to: params.appendingPathComponent(key), atomically: true, encoding: .utf8)
    }
  }
  return root
}

private func runGateShell(_ command: String) throws -> Int32 {
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", command]
  process.standardOutput = FileHandle.nullDevice
  process.standardError = FileHandle.nullDevice
  try process.run()
  process.waitUntilExit()
  return process.terminationStatus
}

private func shellQuoteForGateTest(_ value: String) -> String {
  "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private func rollbackJournalForGateTest() -> DeploymentRollbackJournal {
  DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "a", count: 40),
    targetHead: String(repeating: "b", count: 40),
    previousPhysicsParams: [:],
    previousQCurveSHA256: String(repeating: "c", count: 64),
    previousMapdReleaseVersion: "old",
    previousMapdVersion: "old",
    previousActiveMapdSHA256: String(repeating: "d", count: 64),
    previousCachedMapdPath: "/data/media/0/osm/binaries/mapd-old",
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-test"
  )
}
