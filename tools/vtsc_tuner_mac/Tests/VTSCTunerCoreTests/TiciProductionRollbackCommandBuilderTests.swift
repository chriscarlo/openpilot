import Foundation
import Testing
@testable import VTSCTunerCore

@Test func productionRollbackCommandIsShellOnlyAndRestoresTheCompleteParamSet() throws {
  let journal = rollbackJournalFixture()
  let command = try TiciProductionRollbackCommandBuilder.command(
    journal: journal,
    expectedCurrentHead: String(repeating: "a", count: 40)
  )

  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("git reset --hard \"$previous_head\""))
  #expect(command.contains("[ \"$(git rev-parse HEAD)\" = \"$expected_current_head\" ]"))
  #expect(command.components(separatedBy: "git status --porcelain").count - 1 == 2)
  #expect(command.range(of: "rollback source head changed immediately before Git mutation")!.lowerBound < command.range(of: "git reset --hard")!.lowerBound)
  #expect(command.range(of: "rollback source checkout became dirty immediately before Git mutation")!.lowerBound < command.range(of: "git reset --hard")!.lowerBound)
  #expect(command.contains("MTSCLookaheadEnabled"))
  #expect(command.contains("refusing rollback unless tici is exactly offroad"))
  #expect(command.contains("[ \"$offroad\" != '1' ] || [ \"$onroad\" != '0' ] || [ \"$lookahead\" != '0' ]"))
  #expect(command.range(of: "refusing rollback")!.lowerBound < command.range(of: "git reset --hard")!.lowerBound)
  #expect(command.contains("flock -x 9"))
  #expect(command.contains("rollback_params()"))
  #expect(command.contains("MapdReleaseVersion"))
  #expect(command.contains("MapdVersion"))
  #expect(command.contains("VisionTurnSpeedControlPhysicsAmplitude"))
  #expect(command.contains("current_active_sha"))
  #expect(command.contains("mapd is already restored"))
  #expect(command.contains("artifact is missing and active mapd does not match"))
  #expect(command.contains("base64 -d"))
  #expect(command.contains(TiciProductionRollbackCommandBuilder.resultMarker))

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

@Test func productionRollbackCommandRejectsUnsafeJournalAndDecodesOnlyExactResult() throws {
  var unsafe = rollbackJournalFixture()
  unsafe.previousHead = "not-a-head"
  #expect(throws: (any Error).self) {
    try TiciProductionRollbackCommandBuilder.command(
      journal: unsafe,
      expectedCurrentHead: String(repeating: "a", count: 40)
    )
  }

  let head = String(repeating: "a", count: 40)
  let output = "\(TiciProductionRollbackCommandBuilder.resultMarker)\t\(Data(head.utf8).base64EncodedString())\n"
  #expect(try TiciProductionRollbackCommandBuilder.restoredHead(from: output) == head)
  #expect(throws: (any Error).self) {
    try TiciProductionRollbackCommandBuilder.restoredHead(from: "bad output")
  }
}

@Test func dirtyCheckoutImmediatelyBeforeResetStopsAllRollbackMutation() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-dirty-rollback-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let repository = root.appendingPathComponent("repo", isDirectory: true)
  let params = root.appendingPathComponent("params", isDirectory: true)
  let bin = root.appendingPathComponent("bin", isDirectory: true)
  try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
  try "1".write(to: params.appendingPathComponent("IsOffroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("IsOnroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("MTSCLookaheadEnabled"), atomically: true, encoding: .utf8)

  let statusCount = root.appendingPathComponent("status-count")
  let resetMarker = root.appendingPathComponent("git-reset-ran")
  let previousHead = String(repeating: "a", count: 40)
  let currentHead = String(repeating: "b", count: 40)
  let fakeGit = bin.appendingPathComponent("git")
  try """
  #!/bin/sh
  case "$1 $2" in
    "branch --show-current") printf '%s\\n' chauffeur-exp01 ;;
    "rev-parse HEAD") printf '%s\\n' '\(currentHead)' ;;
    "status --porcelain")
      count="$(cat '\(statusCount.path)' 2>/dev/null || printf 0)"
      count=$((count + 1))
      printf '%s' "$count" > '\(statusCount.path)'
      [ "$count" -ge 2 ] && printf '%s\\n' user-owned-change || :
      ;;
    "reset --hard") /usr/bin/touch '\(resetMarker.path)' ;;
    *) exit 1 ;;
  esac
  """.write(to: fakeGit, atomically: true, encoding: .utf8)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeGit.path)
  for tool in ["flock", "sha256sum"] {
    let url = bin.appendingPathComponent(tool)
    try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
  }

  var journal = rollbackJournalFixture()
  journal.previousHead = previousHead
  let command = try TiciProductionRollbackCommandBuilder.command(
    journal: journal,
    expectedCurrentHead: currentHead,
    repositoryPath: repository.path,
    paramsDirectory: params.path,
    paramsLockPath: root.appendingPathComponent("params.lock").path
  )
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", command]
  process.environment = ["PATH": "\(bin.path):/usr/bin:/bin:/usr/sbin:/sbin"]
  try process.run()
  process.waitUntilExit()

  #expect(process.terminationStatus != 0)
  #expect(!FileManager.default.fileExists(atPath: resetMarker.path))
  #expect(FileManager.default.fileExists(atPath: statusCount.path))
  #expect(try String(contentsOf: statusCount, encoding: .utf8) == "2")
  #expect(try String(contentsOf: params.appendingPathComponent("IsOffroad"), encoding: .utf8) == "1")
}

@Test(arguments: [1, 2])
func unreadableGitStatusAtEitherRollbackGateStopsBeforeReset(failingRead: Int) throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-unreadable-status-\(failingRead)-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let repository = root.appendingPathComponent("repo", isDirectory: true)
  let params = root.appendingPathComponent("params", isDirectory: true)
  let bin = root.appendingPathComponent("bin", isDirectory: true)
  try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
  try "1".write(to: params.appendingPathComponent("IsOffroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("IsOnroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("MTSCLookaheadEnabled"), atomically: true, encoding: .utf8)

  let statusCount = root.appendingPathComponent("status-count")
  let resetMarker = root.appendingPathComponent("git-reset-ran")
  let currentHead = String(repeating: "b", count: 40)
  let fakeGit = bin.appendingPathComponent("git")
  try """
  #!/bin/sh
  case "$1 $2" in
    "branch --show-current") printf '%s\\n' chauffeur-exp01 ;;
    "rev-parse HEAD") printf '%s\\n' '\(currentHead)' ;;
    "status --porcelain")
      count="$(cat '\(statusCount.path)' 2>/dev/null || printf 0)"
      count=$((count + 1))
      printf '%s' "$count" > '\(statusCount.path)'
      if [ "$count" -eq '\(failingRead)' ]; then exit 1; fi
      exit 0
      ;;
    "reset --hard") /usr/bin/touch '\(resetMarker.path)' ;;
    *) exit 1 ;;
  esac
  """.write(to: fakeGit, atomically: true, encoding: .utf8)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeGit.path)
  for tool in ["flock", "sha256sum"] {
    let url = bin.appendingPathComponent(tool)
    try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
  }

  let command = try TiciProductionRollbackCommandBuilder.command(
    journal: rollbackJournalFixture(),
    expectedCurrentHead: currentHead,
    repositoryPath: repository.path,
    paramsDirectory: params.path,
    paramsLockPath: root.appendingPathComponent("params.lock").path
  )
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", command]
  process.environment = ["PATH": "\(bin.path):/usr/bin:/bin:/usr/sbin:/sbin"]
  try process.run()
  process.waitUntilExit()

  #expect(process.terminationStatus != 0)
  #expect(!FileManager.default.fileExists(atPath: resetMarker.path))
  let observedReads = try String(contentsOf: statusCount, encoding: .utf8)
  #expect(observedReads == String(failingRead))
}

@Test func mapdRollbackReplayNeedsNoArtifactWhenTheActiveBinaryIsAlreadyExact() {
  let previous = String(repeating: "a", count: 64)
  #expect(TiciProductionRollbackCommandBuilder.mapdRestoreDecision(
    currentActiveSHA256: previous,
    expectedPreviousSHA256: previous,
    rollbackArtifactPresent: false
  ) == .alreadyRestored)
  #expect(TiciProductionRollbackCommandBuilder.mapdRestoreDecision(
    currentActiveSHA256: String(repeating: "b", count: 64),
    expectedPreviousSHA256: previous,
    rollbackArtifactPresent: true
  ) == .restoreRecordedArtifact)
  #expect(TiciProductionRollbackCommandBuilder.mapdRestoreDecision(
    currentActiveSHA256: String(repeating: "b", count: 64),
    expectedPreviousSHA256: previous,
    rollbackArtifactPresent: false
  ) == .failMissingArtifact)
}

private func rollbackJournalFixture() -> DeploymentRollbackJournal {
  let physics: [String: String?] = [
    "VisionTurnSpeedControlPhysicsAmplitude": "-1.658965",
    "VisionTurnSpeedControlPhysicsSteepness": "-1395.055546",
    "VisionTurnSpeedControlPhysicsCenter": "0.005397",
    "VisionTurnSpeedControlPhysicsBaseline": "4.107103",
    "VisionTurnSpeedControlPhysicsMinLatAccel": "2.4481",
    "VisionTurnSpeedControlPhysicsMaxLatAccel": "4.1071",
  ]
  return DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "a", count: 40),
    previousPhysicsParams: physics,
    previousQCurveSHA256: String(repeating: "b", count: 64),
    previousMapdReleaseVersion: "release-v1",
    previousMapdVersion: "release-v1",
    previousActiveMapdSHA256: String(repeating: "c", count: 64),
    previousCachedMapdPath: "/data/media/0/osm/binaries/mapd-old",
    previousCachedMapdSHA256: String(repeating: "d", count: 64),
    mapdRollbackPath: "/data/media/0/osm/binaries/mapd-rollback-01234567-89ab-cdef-0123-456789abcdef"
  )
}
