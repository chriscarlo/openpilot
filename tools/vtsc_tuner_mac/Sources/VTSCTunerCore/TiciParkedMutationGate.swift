import Foundation

/// One fail-closed remote safety predicate for every tici mutation primitive.
/// Callers define `params_dir` immediately before inserting this fragment.
/// Missing, empty, malformed, or contradictory values all fail because only
/// the exact parked/kill-switch tuple 1 / 0 / 0 is accepted.
enum TiciParkedMutationGate {
  static let defaultParamsDirectory = "/data/params/d"

  static func shellFragment(refusalMessage: String) -> String {
    let message = shellQuote(refusalMessage)
    return """
    [ -d "$params_dir" ] || { printf '%s\\n' 'Params directory is missing' >&2; exit 1; }
    offroad="$(cat "$params_dir/IsOffroad" 2>/dev/null || true)"
    onroad="$(cat "$params_dir/IsOnroad" 2>/dev/null || true)"
    lookahead="$(cat "$params_dir/MTSCLookaheadEnabled" 2>/dev/null || true)"
    if [ "$offroad" != '1' ] || [ "$onroad" != '0' ] || [ "$lookahead" != '0' ]; then
      printf '%s\\n' \(message) >&2
      exit 1
    fi
    """
  }

  /// `rsync` normally starts its remote receiver directly. Wrap that receiver
  /// in the same exact gate so a state change cannot mutate a staging tree
  /// merely because an earlier SSH preflight was safe.
  static func gatedRsyncPath(refusalMessage: String) -> String {
    let body = """
    params_dir=\(shellQuote(defaultParamsDirectory))
    \(shellFragment(refusalMessage: refusalMessage))
    exec rsync "$@"
    """
    return "sh -c \(shellQuote(body)) sh"
  }

  static func guardedCommand(
    _ mutation: String,
    paramsDirectory: String = defaultParamsDirectory,
    refusalMessage: String
  ) -> String {
    """
    set -eu
    params_dir=\(shellQuote(paramsDirectory))
    \(shellFragment(refusalMessage: refusalMessage))
    \(mutation)
    """
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
