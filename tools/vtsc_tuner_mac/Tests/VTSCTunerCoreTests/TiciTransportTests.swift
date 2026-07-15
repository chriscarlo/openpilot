import Foundation
import Testing
@testable import VTSCTunerCore

@Test func adbFallbackCreatesForwardAndRestoresItAfterLoss() async throws {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: """
  List of devices attached
  e521630c device usb:2-1 transport_id:16

  """)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)

  let selected = try await pipeline.pickTiciProfile(preferred: "commaAdb")
  #expect(selected == "commaAdb")
  var requests = await runner.requests
  #expect(requests.contains {
    $0.executableURL == adbURL && $0.arguments == ["devices", "-l"]
  })
  #expect(requests.contains {
    $0.executableURL == adbURL
      && $0.arguments == ["-s", "e521630c", "forward", "tcp:2222", "tcp:22"]
  })

  await runner.dropForward()
  #expect(await pipeline.probeProfile("commaAdb"))
  requests = await runner.requests
  #expect(requests.filter {
    $0.executableURL == adbURL && $0.arguments.contains("forward")
  }.count == 2)
}

@Test func adbFallbackReportsUnauthorizedDeviceWithoutForwarding() async {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: """
  List of devices attached
  e521630c unauthorized usb:2-1 transport_id:16

  """)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)

  do {
    _ = try await pipeline.pickTiciProfile(preferred: "commaAdb")
    Issue.record("unauthorized ADB device unexpectedly resolved a tici transport")
  } catch let error as ApplyPipelineError {
    guard case let .noReachableTici(detail) = error else {
      Issue.record("unexpected transport error: \(error)")
      return
    }
    #expect(detail?.contains("e521630c=unauthorized") == true)
  } catch {
    Issue.record("unexpected transport error: \(error)")
  }

  let requests = await runner.requests
  #expect(!requests.contains { $0.executableURL == adbURL && $0.arguments.contains("forward") })
}

@Test func adbFallbackRefusesToChooseBetweenMultipleUSBDevices() async {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: """
  List of devices attached
  first device usb:2-1 transport_id:16
  second device usb:3-1 transport_id:17

  """)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)

  do {
    _ = try await pipeline.pickTiciProfile(preferred: "commaAdb")
    Issue.record("ambiguous ADB devices unexpectedly resolved a tici transport")
  } catch let error as ApplyPipelineError {
    guard case let .noReachableTici(detail) = error else {
      Issue.record("unexpected transport error: \(error)")
      return
    }
    #expect(detail?.contains("Multiple ADB devices") == true)
    #expect(detail?.contains("first=device, second=device") == true)
  } catch {
    Issue.record("unexpected transport error: \(error)")
  }

  let requests = await runner.requests
  #expect(!requests.contains { $0.executableURL == adbURL && $0.arguments.contains("forward") })
}

@Test func adbFallbackRefusesMixedReadyAndUnavailableDevices() async {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: """
  List of devices attached
  ready device usb:2-1 transport_id:16
  intended unauthorized usb:3-1 transport_id:17

  """)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)

  do {
    _ = try await pipeline.pickTiciProfile(preferred: "commaAdb")
    Issue.record("mixed ADB device states unexpectedly selected the wrong endpoint")
  } catch let error as ApplyPipelineError {
    guard case let .noReachableTici(detail) = error else {
      Issue.record("unexpected transport error: \(error)")
      return
    }
    #expect(detail?.contains("intended=unauthorized") == true)
    #expect(detail?.contains("ready=device") == true)
  } catch {
    Issue.record("unexpected transport error: \(error)")
  }

  let requests = await runner.requests
  #expect(!requests.contains { $0.executableURL == adbURL && $0.arguments.contains("forward") })
}

@Test func workingSSHProfilesNeverInvokeADB() async throws {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: "", forwardReady: true)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)

  #expect(try await pipeline.pickTiciProfile(preferred: "commaAdb") == "commaAdb")
  #expect(try await pipeline.pickTiciProfile(preferred: "commaHome") == "commaHome")
  let requests = await runner.requests
  #expect(!requests.contains { $0.executableURL == adbURL })
}

@Test func postRebootTimeoutKeepsTheLastADBDiagnosis() async {
  let adbURL = URL(fileURLWithPath: "/test/bin/adb")
  let runner = ADBBridgeRunner(deviceList: """
  List of devices attached
  e521630c unauthorized usb:2-1 transport_id:16

  """)
  let pipeline = ApplyPipeline(processRunner: runner, adbURL: adbURL)
  #expect(!(await pipeline.probeProfile("commaAdb")))

  do {
    try await pipeline.waitForTici(
      profile: "commaAdb",
      timeout: 0,
      initialDelayNanoseconds: 0,
      pollDelayNanoseconds: 0
    )
    Issue.record("unreachable tici unexpectedly returned from its reboot wait")
  } catch let error as ApplyPipelineError {
    guard case let .commandFailed(context, _, output) = error else {
      Issue.record("unexpected wait error: \(error)")
      return
    }
    #expect(context == "wait for tici")
    #expect(output.contains("USB / ADB"))
    #expect(output.contains("e521630c=unauthorized"))
  } catch {
    Issue.record("unexpected wait error: \(error)")
  }
}

@Test func adbLocatorIncludesFinderSafeUserInstallPaths() {
  let urls = ADBExecutableLocator.candidateURLs(environment: [
    "HOME": "/Users/example",
    "PATH": "/usr/bin:/bin",
  ])
  #expect(urls.map(\.path).contains("/Users/example/.local/bin/adb"))
  #expect(urls.map(\.path).contains("/Users/example/Library/Android/sdk/platform-tools/adb"))
}

private actor ADBBridgeRunner: ProcessRunning {
  let deviceList: String
  private var forwardReady = false
  private(set) var requests: [ProcessRequest] = []

  init(deviceList: String, forwardReady: Bool = false) {
    self.deviceList = deviceList
    self.forwardReady = forwardReady
  }

  func dropForward() {
    forwardReady = false
  }

  func run(_ request: ProcessRequest) async throws -> ProcessResult {
    requests.append(request)
    if request.executableURL == ApplyPipeline.networkSetupURL {
      return failure("Wi-Fi state unavailable")
    }
    if request.executableURL.path == "/test/bin/adb" {
      if request.arguments == ["devices", "-l"] {
        return success(deviceList)
      }
      if request.arguments.count == 5,
         request.arguments[0] == "-s",
         request.arguments[2...] == ["forward", "tcp:2222", "tcp:22"]
      {
        forwardReady = true
        return success("2222\n")
      }
      return failure("unexpected adb request: \(request.arguments)")
    }
    if request.executableURL == ApplyPipeline.sshURL {
      return forwardReady ? success("") : failure("connection refused")
    }
    return failure("unexpected process: \(request.executableURL.path)")
  }

  private func success(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 0, standardOutput: output, standardError: "")
  }

  private func failure(_ output: String) -> ProcessResult {
    ProcessResult(terminationStatus: 1, standardOutput: "", standardError: output)
  }
}
