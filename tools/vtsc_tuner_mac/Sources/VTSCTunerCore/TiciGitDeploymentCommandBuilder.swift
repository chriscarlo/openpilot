import Foundation

/// The narrow Git primitive that must still execute on the tici. Swift owns
/// the branch/head policy and only emits a fixed, injection-safe fast-forward
/// command after the local and remote preflight identities agree.
enum TiciGitDeploymentCommandBuilder {
  static func exactFastForwardCommand(branch: String, head: String) throws -> String {
    guard branch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
          !branch.contains(".."), !branch.hasPrefix("/"),
          head.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
    else {
      throw ApplyPipelineError.invalidDeploymentOutput("unsafe tici Git deployment identity")
    }
    return """
    cd /data/openpilot && \
    test "$(git branch --show-current)" = \(shellQuote(branch)) && \
    test -z "$(git status --porcelain)" && \
    git fetch --no-tags origin refs/heads/\(branch) && \
    test "$(git rev-parse FETCH_HEAD)" = \(shellQuote(head)) && \
    git merge --ff-only \(shellQuote(head)) && \
    test "$(git rev-parse HEAD)" = \(shellQuote(head))
    """
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
