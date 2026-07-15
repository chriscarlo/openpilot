import Foundation
import Testing
@testable import VTSCTunerCore

@Test func ticiGitFastForwardIsShellOnlyAndPinsTheExactOriginHead() throws {
  let head = String(repeating: "a", count: 40)
  let command = try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
    branch: "chauffeur-exp01",
    head: head
  )
  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("git fetch --no-tags origin refs/heads/chauffeur-exp01"))
  #expect(command.contains("test \"$(git rev-parse FETCH_HEAD)\" = '\(head)'"))
  #expect(command.contains("git merge --ff-only '\(head)'"))

  #expect(throws: (any Error).self) {
    try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
      branch: "chauffeur-exp01; touch nope",
      head: head
    )
  }
  #expect(throws: (any Error).self) {
    try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
      branch: "chauffeur-exp01",
      head: "not-a-git-head"
    )
  }

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
