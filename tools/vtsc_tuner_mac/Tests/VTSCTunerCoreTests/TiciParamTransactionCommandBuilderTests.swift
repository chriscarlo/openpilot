import Foundation
import Testing
@testable import VTSCTunerCore

@Test func transactionCommandContainsOnlyTheSixVTSCPhysicsKeys() throws {
  let command = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)
  let expected = [
    "VisionTurnSpeedControlPhysicsAmplitude",
    "VisionTurnSpeedControlPhysicsSteepness",
    "VisionTurnSpeedControlPhysicsCenter",
    "VisionTurnSpeedControlPhysicsBaseline",
    "VisionTurnSpeedControlPhysicsMinLatAccel",
    "VisionTurnSpeedControlPhysicsMaxLatAccel",
  ]

  let valueCaseKeys = command.split(separator: "\n").compactMap { line -> String? in
    let prefix = "'VisionTurnSpeedControlPhysics"
    let suffix = "') printf '%s'"
    guard line.contains(prefix), line.contains(suffix),
          let start = line.firstIndex(of: "'"),
          let end = line.range(of: suffix)?.lowerBound else { return nil }
    return String(line[line.index(after: start) ..< end])
  }
  #expect(valueCaseKeys.count == 6)
  #expect(Set(valueCaseKeys) == Set(expected))
  #expect(!command.lowercased().contains("python"))
  #expect(!command.contains("apply_physics_params.py"))
}

@Test func transactionCommandHasParkedGuardLockAtomicWritesRollbackAndReadback() throws {
  let command = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)

  #expect(command.contains("params_dir='/data/params/d'"))
  #expect(command.contains("lock_file='/data/params/.lock'"))
  #expect(command.contains("IsOffroad"))
  #expect(command.contains("IsOnroad"))
  #expect(command.contains("MTSCLookaheadEnabled"))
  #expect(command.contains("Map Lookahead is disabled"))
  #expect(command.contains("exec 9>\"$lock_file\""))
  #expect(command.contains("flock -x 9"))
  #expect(command.contains("mktemp -d"))
  #expect(command.contains("mktemp \"$stage_dir/${key}.XXXXXX\""))
  #expect(command.contains("sync -f \"$temporary\""))
  #expect(command.contains("mv -f \"$stage_dir/$key\" \"$params_dir/$key\""))
  #expect(command.contains("rollback()"))
  #expect(command.contains("trap cleanup 0"))
  #expect(command.contains("actual=\"$(cat \"$params_dir/$key\")\""))
}

@Test func transactionCommandExpandsEveryTemplateAndPassesPOSIXShellSyntaxCheck() throws {
  let command = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .checkoutFallback)
  #expect(!command.contains("(keys)"))
  #expect(!command.contains("(cases)"))

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

@Test func transactionBuilderRejectsNonFiniteOutOfRangeAndInvertedValues() {
  do {
    _ = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .init(
      a: .nan, b: -1_395, c: 0.005, d: 4.1, minLat: 2, maxLat: 4
    ))
    Issue.record("Expected non-finite amplitude to be rejected")
  } catch let error as TiciParamTransactionCommandBuilderError {
    #expect(error == .nonFiniteValue(key: "VisionTurnSpeedControlPhysicsAmplitude"))
  } catch {
    Issue.record("Unexpected error: \(error)")
  }

  do {
    _ = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .init(
      a: -1, b: -1_395, c: 0.005, d: 7, minLat: 2, maxLat: 4
    ))
    Issue.record("Expected unsafe baseline to be rejected")
  } catch let error as TiciParamTransactionCommandBuilderError {
    #expect(error == .outOfRange(key: "VisionTurnSpeedControlPhysicsBaseline", value: 7))
  } catch {
    Issue.record("Unexpected error: \(error)")
  }

  do {
    _ = try TiciParamTransactionCommandBuilder.synchronizeAndVerifyCommand(parameters: .init(
      a: -1, b: -1_395, c: 0.005, d: 4.1, minLat: 3, maxLat: 2
    ))
    Issue.record("Expected inverted lateral acceleration clamps to be rejected")
  } catch let error as TiciParamTransactionCommandBuilderError {
    #expect(error == .minimumExceedsMaximum(minimum: 3, maximum: 2))
  } catch {
    Issue.record("Unexpected error: \(error)")
  }
}
