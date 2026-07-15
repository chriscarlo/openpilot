import Foundation
import Testing
@testable import VTSCTunerCore

@Test func mapdReleaseTransactionIsShellOnlyAndUsesDurableAtomicPrimitives() throws {
  let fixture = try makeValidatedMapdReleaseFixture()
  let recovery = TiciMapdReleaseTransactionCommandBuilder.recoveryCommand()
  let probe = try TiciMapdReleaseTransactionCommandBuilder.probeCommand(
    release: fixture.release,
    stagedPath: fixture.stagedPath
  )
  let command = try TiciMapdReleaseTransactionCommandBuilder.installCommand(
    release: fixture.release,
    stagedPath: fixture.stagedPath,
    rollbackPath: fixture.rollbackPath
  )

  #expect(!probe.lowercased().contains("python"))
  #expect(probe.contains("mapd --build-info"))
  #expect(probe.contains(TiciMapdReleaseTransactionCommandBuilder.probeMarker))
  #expect(probe.contains("19"))
  #expect(probe.contains("20"))
  #expect(!recovery.lowercased().contains("python"))
  #expect(recovery.contains("transaction_dir='\(TiciMapdReleaseTransactionCommandBuilder.transactionDirectory)'"))
  #expect(recovery.contains("transaction_lock=\"$transaction_dir/.lock\""))
  #expect(recovery.contains("chmod 700 \"$transaction_dir\""))
  #expect(recovery.contains("flock -x 8"))
  #expect(recovery.contains("flock -x 9"))
  #expect(recovery.contains("restore_uncommitted_journal()"))
  #expect(recovery.contains("prepared)"))
  #expect(recovery.contains("committed)"))
  #expect(recovery.contains("'recovered'"))
  #expect(recovery.contains("'clean'"))
  #expect(recovery.contains(TiciMapdReleaseTransactionCommandBuilder.recoveryMarker))
  #expect(!command.lowercased().contains("python"))
  #expect(command.contains("od -An -v -t u1 -N 20"))
  #expect(command.contains("sha256sum"))
  #expect(command.contains("grep -F -q"))
  #expect(command.contains("MapdReleaseID:release-v1"))
  #expect(command.contains("MapdBuildID:build-v1"))
  #expect(command.contains("chmod 755 \"$staged\""))
  #expect(command.contains("\"$staged\" --build-info"))
  #expect(command.contains("mv -f \"$staged\" \"$cache\""))
  #expect(command.contains("mv -f \"$active_temporary\" \"$active\""))
  #expect(command.contains("sync -f \"$1\""))
  #expect(command.contains("durable_file \"$active\""))
  #expect(command.contains("params_dir='/data/params/d'"))
  #expect(command.contains("params_lock='/data/params/.lock'"))
  #expect(command.contains("MTSCLookaheadEnabled"))
  #expect(command.contains("refusing mapd release mutation unless tici is offroad"))
  #expect(command.contains("transaction_dir='\(TiciMapdReleaseTransactionCommandBuilder.transactionDirectory)'"))
  #expect(command.contains("transaction_lock=\"$transaction_dir/.lock\""))
  #expect(command.contains("chmod 700 \"$transaction_dir\""))
  #expect(command.contains("flock -x 8"))
  #expect(command.contains("flock -x 9"))
  #expect(command.contains("MapdReleaseVersion MapdVersion"))
  #expect(command.contains("prepare_transaction_journal()"))
  #expect(command.contains("recover_prior_journal()"))
  #expect(command.contains("restore_uncommitted_journal()"))
  #expect(command.contains("journal_temporary=\"$(mktemp -d \"$transaction_dir/.journal.new.XXXXXX\")\""))
  #expect(command.contains("journal_rollback=\"$journal/rollback\""))
  #expect(command.contains("$journal_temporary_rollback/mapd.sha256"))
  #expect(command.contains("printf '%s' 'prepared'"))
  #expect(command.contains("write_journal_state 'committed'"))
  #expect(command.contains("transaction_committed=1"))
  #expect(command.contains("A completed install is never rolled back"))
  #expect(command.contains("journal cleanup deferred"))
  #expect(!command.contains("param_work_dir"))
  #expect(command.contains(TiciMapdReleaseTransactionCommandBuilder.resultMarker))
  #expect(command.contains("base64 | tr -d"))

  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/bin/sh")
  process.arguments = ["-n"]
  let input = Pipe()
  process.standardInput = input
  try process.run()
  input.fileHandleForWriting.write(Data((recovery + "\n" + probe + "\n" + command).utf8))
  input.fileHandleForWriting.closeFile()
  process.waitUntilExit()
  #expect(process.terminationStatus == 0)
}

@Test func mapdRecoveryOutcomeIsStrictAndMachineReadable() throws {
  let marker = TiciMapdReleaseTransactionCommandBuilder.recoveryMarker
  #expect(try TiciMapdReleaseTransactionCommandBuilder.decodeRecoveryOutcome(
    "diagnostic\n\(marker)\tclean\n"
  ) == .clean)
  #expect(try TiciMapdReleaseTransactionCommandBuilder.decodeRecoveryOutcome(
    "\(marker)\trecovered\n"
  ) == .recovered)

  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidResult("\(marker)\tunknown")) {
    try TiciMapdReleaseTransactionCommandBuilder.decodeRecoveryOutcome("\(marker)\tunknown\n")
  }
  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidResult("no recovery marker\n")) {
    try TiciMapdReleaseTransactionCommandBuilder.decodeRecoveryOutcome("no recovery marker\n")
  }
}

@Test func mapdReleaseTransactionRejectsEscapingAndNonIdentityInputs() throws {
  let fixture = try makeValidatedMapdReleaseFixture()

  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidStagedPath("/tmp/mapd.partial")) {
    try TiciMapdReleaseTransactionCommandBuilder.installCommand(
      release: fixture.release,
      stagedPath: "/tmp/mapd.partial",
      rollbackPath: fixture.rollbackPath
    )
  }
  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidRollbackPath("/tmp/mapd.rollback")) {
    try TiciMapdReleaseTransactionCommandBuilder.installCommand(
      release: fixture.release,
      stagedPath: fixture.stagedPath,
      rollbackPath: "/tmp/mapd.rollback"
    )
  }

  var invalidCache = fixture.release
  invalidCache.persistentCacheFileName = "mapd-not-an-identity-key"
  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidPersistentCacheFileName("mapd-not-an-identity-key")) {
    try TiciMapdReleaseTransactionCommandBuilder.installCommand(
      release: invalidCache,
      stagedPath: fixture.stagedPath,
      rollbackPath: fixture.rollbackPath
    )
  }

  var injectedMarker = fixture.release
  injectedMarker.artifact.capability = "capability$(touch /tmp/should-not-run)"
  #expect(throws: TiciMapdReleaseTransactionCommandBuilderError.invalidReleaseIdentity("release-v1")) {
    try TiciMapdReleaseTransactionCommandBuilder.probeCommand(
      release: injectedMarker,
      stagedPath: fixture.stagedPath
    )
  }
}

@Test func mapdReleaseTransactionDecodesAndSwiftValidatesBuildInfo() throws {
  let fixture = try makeValidatedMapdReleaseFixture()
  let artifact = fixture.release.artifact
  let cachePath = "/data/media/0/osm/binaries/\(fixture.release.persistentCacheFileName)"
  let buildInfo = TiciMapdReleaseBuildInfo(
    releaseID: artifact.releaseID,
    buildID: artifact.buildID,
    estimatorVersion: artifact.estimatorVersion,
    capabilities: [artifact.capability],
    identityMarkers: ["MapdReleaseID:\(artifact.releaseID)", "MapdBuildID:\(artifact.buildID)"]
  )
  let buildInfoData = try JSONEncoder().encode(buildInfo)
  let wire = [
    TiciMapdReleaseTransactionCommandBuilder.resultMarker,
    Data(artifact.releaseID.utf8).base64EncodedString(),
    artifact.sha256,
    Data(cachePath.utf8).base64EncodedString(),
    Data(fixture.rollbackPath.utf8).base64EncodedString(),
    buildInfoData.base64EncodedString(),
  ].joined(separator: "\t")

  let decoded = try TiciMapdReleaseTransactionCommandBuilder.decodeResult("remote diagnostic\n\(wire)\n")
  #expect(decoded.releaseID == artifact.releaseID)
  #expect(decoded.sha256 == artifact.sha256)
  #expect(decoded.cachePath == cachePath)
  #expect(decoded.rollbackPath == fixture.rollbackPath)
  #expect(try TiciMapdReleaseTransactionCommandBuilder.validate(
    decoded,
    matches: fixture.release,
    rollbackPath: fixture.rollbackPath
  ) == buildInfo)

  var malformed = decoded
  malformed.buildInfoJSON = Data("not json".utf8)
  do {
    try TiciMapdReleaseTransactionCommandBuilder.validate(
      malformed,
      matches: fixture.release,
      rollbackPath: fixture.rollbackPath
    )
    Issue.record("Malformed mapd --build-info unexpectedly passed validation")
  } catch let error as TiciMapdReleaseTransactionCommandBuilderError {
    guard case .invalidBuildInfo = error else {
      Issue.record("Unexpected mapd build-info validation error: \(error)")
      return
    }
  } catch {
    Issue.record("Unexpected error type: \(error)")
  }

  let probeWire = [
    TiciMapdReleaseTransactionCommandBuilder.probeMarker,
    buildInfoData.base64EncodedString(),
  ].joined(separator: "\t")
  #expect(try TiciMapdReleaseTransactionCommandBuilder.validateBuildInfo(
    TiciMapdReleaseTransactionCommandBuilder.decodeProbe(probeWire + "\n"),
    matches: fixture.release
  ) == buildInfo)
}

private func makeValidatedMapdReleaseFixture() throws -> (
  release: ValidatedMapdReleaseArtifact,
  stagedPath: String,
  rollbackPath: String
) {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-mapd-release-transaction-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let binary = directory.appendingPathComponent("mapd")
  var bytes = Data(repeating: 0, count: 128)
  bytes.replaceSubrange(0 ..< 6, with: [0x7f, 0x45, 0x4c, 0x46, 2, 1])
  bytes[18] = 183
  bytes[19] = 0
  for marker in [
    "MapdReleaseID:release-v1",
    "MapdBuildID:build-v1",
    "MapWholeCurveProfile:whole-curve-v1",
  ] {
    bytes.append(Data(marker.utf8))
    bytes.append(0)
  }
  try bytes.write(to: binary)
  let artifact = MapdReleaseArtifact(
    releaseID: "release-v1",
    buildID: "build-v1",
    binaryURL: binary,
    sha256: MapdReleaseArtifact.sha256Hex(bytes)
  )
  return (
    try artifact.validated(),
    "/data/media/0/osm/binaries/.mapd-release-01234567-89ab-cdef-0123-456789abcdef.partial",
    "/data/media/0/osm/binaries/mapd-rollback-01234567-89ab-cdef-0123-456789abcdef"
  )
}
