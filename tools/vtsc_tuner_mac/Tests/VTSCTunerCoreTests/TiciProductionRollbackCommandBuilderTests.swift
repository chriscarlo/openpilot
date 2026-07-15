import Foundation
import Testing
@testable import VTSCTunerCore

@Test func productionRollbackCommandIsShellOnlyAndRestoresTheCompleteParamSet() throws {
  let journal = rollbackJournalFixture()
  let command = try TiciProductionRollbackCommandBuilder.command(journal: journal)

  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("git reset --hard \"$previous_head\""))
  #expect(command.contains("MTSCLookaheadEnabled"))
  #expect(command.contains("refusing rollback unless tici is offroad"))
  #expect(command.contains("flock -x 9"))
  #expect(command.contains("rollback_params()"))
  #expect(command.contains("MapdReleaseVersion"))
  #expect(command.contains("MapdVersion"))
  #expect(command.contains("VisionTurnSpeedControlPhysicsAmplitude"))
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
    try TiciProductionRollbackCommandBuilder.command(journal: unsafe)
  }

  let head = String(repeating: "a", count: 40)
  let output = "\(TiciProductionRollbackCommandBuilder.resultMarker)\t\(Data(head.utf8).base64EncodedString())\n"
  #expect(try TiciProductionRollbackCommandBuilder.restoredHead(from: output) == head)
  #expect(throws: (any Error).self) {
    try TiciProductionRollbackCommandBuilder.restoredHead(from: "bad output")
  }
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
