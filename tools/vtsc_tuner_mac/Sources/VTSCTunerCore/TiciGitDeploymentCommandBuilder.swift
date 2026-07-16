import Foundation

/// The narrow Git primitive that must still execute on the tici. Swift owns
/// the branch/head policy and only emits a fixed, injection-safe fast-forward
/// command after the local and remote preflight identities agree.
enum TiciGitDeploymentCommandBuilder {
  static func exactFastForwardCommand(
    branch: String,
    head: String,
    repositoryPath: String = "/data/openpilot",
    paramsDirectory: String = "/data/params/d"
  ) throws -> String {
    guard branch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
          !branch.contains(".."), !branch.hasPrefix("/"),
          head.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
    else {
      throw ApplyPipelineError.invalidDeploymentOutput("unsafe tici Git deployment identity")
    }
    return """
    set -eu
    params_dir=\(shellQuote(paramsDirectory))
    offroad="$(cat "$params_dir/IsOffroad" 2>/dev/null || true)"
    onroad="$(cat "$params_dir/IsOnroad" 2>/dev/null || true)"
    lookahead="$(cat "$params_dir/MTSCLookaheadEnabled" 2>/dev/null || true)"
    if [ "$offroad" != '1' ] || [ "$onroad" != '0' ] || [ "$lookahead" != '0' ]; then
      printf '%s\n' 'refusing Git deployment unless tici is exactly offroad and Map Lookahead is disabled' >&2
      exit 1
    fi
    cd \(shellQuote(repositoryPath)) && \
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
