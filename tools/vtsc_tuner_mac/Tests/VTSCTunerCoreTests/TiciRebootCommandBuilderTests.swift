import Foundation
import Testing
@testable import VTSCTunerCore

@Test func rebootCommandRequiresExactRemoteParkedStateImmediatelyBeforeReboot() throws {
  let command = TiciRebootCommandBuilder.command()
  #expect(command.contains("/data/params/d"))
  #expect(command.contains("[ \"$offroad\" != '1' ] || [ \"$onroad\" != '0' ] || [ \"$lookahead\" != '0' ]"))
  #expect(command.range(of: "refusing reboot")!.lowerBound < command.range(of: "nohup sudo reboot")!.lowerBound)
}

@Test func unsafeRemoteStateCannotRunTheRebootPayload() throws {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-reboot-gate-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: root) }
  let params = root.appendingPathComponent("params", isDirectory: true)
  try FileManager.default.createDirectory(at: params, withIntermediateDirectories: true)
  let marker = root.appendingPathComponent("reboot-ran")

  func setState(offroad: String, onroad: String, lookahead: String) throws {
    try offroad.write(to: params.appendingPathComponent("IsOffroad"), atomically: true, encoding: .utf8)
    try onroad.write(to: params.appendingPathComponent("IsOnroad"), atomically: true, encoding: .utf8)
    try lookahead.write(to: params.appendingPathComponent("MTSCLookaheadEnabled"), atomically: true, encoding: .utf8)
  }

  try setState(offroad: "0", onroad: "1", lookahead: "0")
  let unsafe = TiciRebootCommandBuilder.command(
    paramsDirectory: params.path,
    rebootCommand: "/usr/bin/touch '\(marker.path)'"
  )
  #expect(try runShell(unsafe) != 0)
  #expect(!FileManager.default.fileExists(atPath: marker.path))

  try setState(offroad: "1", onroad: "0", lookahead: "0")
  let safe = TiciRebootCommandBuilder.command(
    paramsDirectory: params.path,
    rebootCommand: "/usr/bin/touch '\(marker.path)'"
  )
  #expect(try runShell(safe) == 0)
  #expect(FileManager.default.fileExists(atPath: marker.path))
}

private func runShell(_ command: String) throws -> Int32 {
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-c", command]
  try process.run()
  process.waitUntilExit()
  return process.terminationStatus
}
