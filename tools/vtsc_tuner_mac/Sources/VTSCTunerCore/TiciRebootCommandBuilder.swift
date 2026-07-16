import Foundation

/// A reboot is itself a car-facing mutation. The remote shell re-reads the
/// exact parked and kill-switch Params immediately before backgrounding it;
/// host snapshots and journal writes cannot substitute for this final gate.
enum TiciRebootCommandBuilder {
  static let productionRebootCommand = "nohup sudo reboot >/dev/null 2>&1 </dev/null &"

  static func command(
    paramsDirectory: String = "/data/params/d",
    rebootCommand: String = productionRebootCommand
  ) -> String {
    """
    set -eu
    params_dir=\(shellQuote(paramsDirectory))
    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing reboot unless tici is exactly offroad and Map Lookahead is disabled"
    ))
    \(rebootCommand)
    """
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
