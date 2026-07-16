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
  #expect(command.contains("[ \"$offroad\" != '1' ] || [ \"$onroad\" != '0' ] || [ \"$lookahead\" != '0' ]"))
  #expect(command.range(of: "refusing Git deployment")!.lowerBound < command.range(of: "git fetch --no-tags")!.lowerBound)

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

@Test func unsafeRemoteParkedStateStopsBeforeAnyGitCommand() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-unsafe-git-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let params = root.appendingPathComponent("params", isDirectory: true)
  let repository = root.appendingPathComponent("repo", isDirectory: true)
  let bin = root.appendingPathComponent("bin", isDirectory: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
  try "0".write(to: params.appendingPathComponent("IsOffroad"), atomically: true, encoding: .utf8)
  try "1".write(to: params.appendingPathComponent("IsOnroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("MTSCLookaheadEnabled"), atomically: true, encoding: .utf8)
  let marker = root.appendingPathComponent("git-ran")
  let fakeGit = bin.appendingPathComponent("git")
  try "#!/bin/sh\n/usr/bin/touch '\(marker.path)'\nexit 1\n"
    .write(to: fakeGit, atomically: true, encoding: .utf8)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeGit.path)

  let unsafe = try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
    branch: "chauffeur-exp01",
    head: String(repeating: "a", count: 40),
    repositoryPath: repository.path,
    paramsDirectory: params.path
  )
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", unsafe]
  process.environment = ["PATH": "\(bin.path):/usr/bin:/bin"]
  try process.run()
  process.waitUntilExit()

  #expect(process.terminationStatus != 0)
  #expect(!FileManager.default.fileExists(atPath: marker.path))
}

@Test func stateFlipDuringFetchStopsBeforeMergeMutation() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-fetch-flip-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let params = root.appendingPathComponent("params", isDirectory: true)
  let repository = root.appendingPathComponent("repo", isDirectory: true)
  let bin = root.appendingPathComponent("bin", isDirectory: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
  try "1".write(to: params.appendingPathComponent("IsOffroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("IsOnroad"), atomically: true, encoding: .utf8)
  try "0".write(to: params.appendingPathComponent("MTSCLookaheadEnabled"), atomically: true, encoding: .utf8)
  let marker = root.appendingPathComponent("merge-ran")
  let head = String(repeating: "a", count: 40)
  let fakeGit = bin.appendingPathComponent("git")
  try """
  #!/bin/sh
  case "$1 $2" in
    "branch --show-current") printf '%s\\n' chauffeur-exp01 ;;
    "status --porcelain") ;;
    "fetch --no-tags") printf '%s' 1 > '\(params.appendingPathComponent("IsOnroad").path)' ;;
    "rev-parse FETCH_HEAD") printf '%s\\n' '\(head)' ;;
    "merge --ff-only") /usr/bin/touch '\(marker.path)' ;;
    *) exit 1 ;;
  esac
  """
    .write(to: fakeGit, atomically: true, encoding: .utf8)
  try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeGit.path)

  let command = try TiciGitDeploymentCommandBuilder.exactFastForwardCommand(
    branch: "chauffeur-exp01",
    head: head,
    repositoryPath: repository.path,
    paramsDirectory: params.path
  )
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", command]
  process.environment = ["PATH": "\(bin.path):/usr/bin:/bin"]
  try process.run()
  process.waitUntilExit()
  #expect(process.terminationStatus != 0)
  #expect(!FileManager.default.fileExists(atPath: marker.path))
}
