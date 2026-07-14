import Foundation

public struct ProcessRequest: Equatable, Sendable {
  public var executableURL: URL
  public var arguments: [String]
  public var currentDirectoryURL: URL?
  public var environment: [String: String]
  public var timeout: TimeInterval?

  public init(
    executableURL: URL,
    arguments: [String] = [],
    currentDirectoryURL: URL? = nil,
    environment: [String: String] = [:],
    timeout: TimeInterval? = nil
  ) {
    self.executableURL = executableURL
    self.arguments = arguments
    self.currentDirectoryURL = currentDirectoryURL
    self.environment = environment
    self.timeout = timeout
  }
}

public struct ProcessResult: Equatable, Sendable {
  public var terminationStatus: Int32
  public var standardOutput: String
  public var standardError: String

  public init(terminationStatus: Int32, standardOutput: String, standardError: String) {
    self.terminationStatus = terminationStatus
    self.standardOutput = standardOutput
    self.standardError = standardError
  }

  public var succeeded: Bool { terminationStatus == 0 }

  public var combinedOutput: String {
    [standardOutput, standardError]
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
      .joined(separator: "\n")
  }
}

public enum ProcessRunnerError: LocalizedError, Sendable {
  case launchFailed(URL, String)
  case timedOut(URL, TimeInterval)
  case cancelled(URL)

  public var errorDescription: String? {
    switch self {
    case let .launchFailed(url, reason):
      "Could not launch \(url.path): \(reason)"
    case let .timedOut(url, seconds):
      "\(url.lastPathComponent) did not finish within \(Int(seconds)) seconds."
    case let .cancelled(url):
      "\(url.lastPathComponent) was cancelled."
    }
  }
}

public protocol ProcessRunning: Sendable {
  func run(_ request: ProcessRequest) async throws -> ProcessResult
}

/// `Foundation.Process` wrapped for Swift concurrency. Output is redirected to
/// temporary files rather than pipes so a verbose git, mapd, or rsync process
/// cannot block after filling a pipe buffer.
public struct SystemProcessRunner: ProcessRunning {
  public init() {}

  public func run(_ request: ProcessRequest) async throws -> ProcessResult {
    let state = RunningProcessState(executableURL: request.executableURL)
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .utility).async {
          do {
            continuation.resume(returning: try Self.runSynchronously(request, state: state))
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
    } onCancel: {
      state.cancel()
    }
  }

  private static func runSynchronously(
    _ request: ProcessRequest,
    state: RunningProcessState
  ) throws -> ProcessResult {
    let fileManager = FileManager.default
    let outputDirectory = fileManager.temporaryDirectory
      .appendingPathComponent("vtsc-tuner-process-\(UUID().uuidString)", isDirectory: true)
    try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: outputDirectory) }

    let stdoutURL = outputDirectory.appendingPathComponent("stdout")
    let stderrURL = outputDirectory.appendingPathComponent("stderr")
    guard fileManager.createFile(atPath: stdoutURL.path, contents: nil),
          fileManager.createFile(atPath: stderrURL.path, contents: nil)
    else {
      throw ProcessRunnerError.launchFailed(request.executableURL, "could not create output files")
    }

    let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
    let stderrHandle = try FileHandle(forWritingTo: stderrURL)
    defer {
      try? stdoutHandle.close()
      try? stderrHandle.close()
    }

    let process = Process()
    process.executableURL = request.executableURL
    process.arguments = request.arguments
    process.currentDirectoryURL = request.currentDirectoryURL
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = stdoutHandle
    process.standardError = stderrHandle
    if !request.environment.isEmpty {
      process.environment = ProcessInfo.processInfo.environment.merging(request.environment) { _, supplied in
        supplied
      }
    }

    do {
      try process.run()
    } catch {
      throw ProcessRunnerError.launchFailed(request.executableURL, error.localizedDescription)
    }
    state.install(process)

    let started = Date()
    while process.isRunning {
      if state.isCancelled {
        process.terminate()
        process.waitUntilExit()
        throw ProcessRunnerError.cancelled(request.executableURL)
      }
      if let timeout = request.timeout, Date().timeIntervalSince(started) >= timeout {
        process.terminate()
        process.waitUntilExit()
        throw ProcessRunnerError.timedOut(request.executableURL, timeout)
      }
      Thread.sleep(forTimeInterval: 0.05)
    }
    process.waitUntilExit()
    try stdoutHandle.synchronize()
    try stderrHandle.synchronize()
    try stdoutHandle.close()
    try stderrHandle.close()

    let stdout = String(decoding: try Data(contentsOf: stdoutURL), as: UTF8.self)
    let stderr = String(decoding: try Data(contentsOf: stderrURL), as: UTF8.self)
    return ProcessResult(
      terminationStatus: process.terminationStatus,
      standardOutput: stdout,
      standardError: stderr
    )
  }
}

private final class RunningProcessState: @unchecked Sendable {
  private let lock = NSLock()
  private let executableURL: URL
  private var process: Process?
  private var cancelled = false

  init(executableURL: URL) {
    self.executableURL = executableURL
  }

  var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  func install(_ process: Process) {
    lock.lock()
    self.process = process
    let shouldCancel = cancelled
    lock.unlock()
    if shouldCancel, process.isRunning { process.terminate() }
  }

  func cancel() {
    lock.lock()
    cancelled = true
    let process = process
    lock.unlock()
    if let process, process.isRunning { process.terminate() }
  }
}
