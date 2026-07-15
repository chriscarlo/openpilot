import Foundation

/// Errors raised before a device command is ever constructed. The command only
/// contains the fixed VTSC physics Param allowlist and decimal values rendered
/// from a finite, in-range `SigmoidParameters` value.
public enum TiciParamTransactionCommandBuilderError: Error, Equatable, Sendable {
  case invalidParameterSet
  case nonFiniteValue(key: String)
  case unsafeNumericValue(key: String, value: String)
  case outOfRange(key: String, value: Double)
  case minimumExceedsMaximum(minimum: Double, maximum: Double)
}

/// Builds the tici-side, shell-only physics Param transaction.
///
/// `Params.put` is deliberately not used: importing the openpilot Python
/// package made this otherwise tiny deployment step depend on the device's
/// Python environment. Each Param replacement is atomic (`mktemp` + `mv`),
/// serialized with `/data/params/.lock`, read back exactly, and rolled back as
/// a complete six-key set if any replacement or verification fails.
public enum TiciParamTransactionCommandBuilder {
  private struct Entry: Sendable {
    let key: String
    let renderedValue: String
    let deployedValue: Double
    let allowedRange: ClosedRange<Double>
  }

  private static let expectedKeys = [
    "VisionTurnSpeedControlPhysicsAmplitude",
    "VisionTurnSpeedControlPhysicsSteepness",
    "VisionTurnSpeedControlPhysicsCenter",
    "VisionTurnSpeedControlPhysicsBaseline",
    "VisionTurnSpeedControlPhysicsMinLatAccel",
    "VisionTurnSpeedControlPhysicsMaxLatAccel",
  ]

  /// Returns one POSIX `sh` command suitable as the final SSH argument.
  /// It never invokes Python and it only targets the six known physics keys.
  public static func synchronizeAndVerifyCommand(parameters: SigmoidParameters) throws -> String {
    let entries = try validatedEntries(for: parameters)
    let keys = shellKeyList(entries)
    let cases = shellValueCases(entries)

    return """
    set -eu
    params_dir='/data/params/d'
    lock_file='/data/params/.lock'

    if [ ! -d "$params_dir" ]; then
      printf '%s\\n' 'VTSC Param directory is missing' >&2
      exit 1
    fi
    offroad="$(cat "$params_dir/IsOffroad" 2>/dev/null || true)"
    onroad="$(cat "$params_dir/IsOnroad" 2>/dev/null || true)"
    lookahead="$(cat "$params_dir/MTSCLookaheadEnabled" 2>/dev/null || true)"
    if [ "$offroad" != '1' ] || [ "$onroad" = '1' ] || [ "$lookahead" = '1' ]; then
      printf '%s\\n' 'refusing to change VTSC physics Params unless tici is offroad and Map Lookahead is disabled' >&2
      exit 1
    fi

    umask 077
    exec 9>"$lock_file"
    flock -x 9
    work_dir="$(mktemp -d "$params_dir/.vtsc-physics.XXXXXX")"
    rollback_dir="$work_dir/rollback"
    stage_dir="$work_dir/stage"
    mkdir "$rollback_dir" "$stage_dir"
    committed=0
    rollback_ready=0
    mutation_started=0

    value_for_key() {
      case "$1" in
    \(cases)
        *) return 1 ;;
      esac
    }

    rollback() {
      for key in \\
    \(keys)
      do
        if [ -f "$rollback_dir/$key.present" ]; then
          temporary="$(mktemp "$params_dir/.${key}.rollback.XXXXXX")"
          cat "$rollback_dir/$key.value" > "$temporary"
          sync -f "$temporary"
          mv -f "$temporary" "$params_dir/$key"
          sync -f "$params_dir/$key"
          sync -f "$params_dir"
        else
          rm -f "$params_dir/$key"
          sync -f "$params_dir"
        fi
      done
    }

    cleanup() {
      status=$?
      trap - 0 1 2 15
      if [ "$mutation_started" -eq 1 ] && [ "$rollback_ready" -eq 1 ] && [ "$committed" -ne 1 ]; then
        rollback || printf '%s\\n' 'VTSC physics Param rollback failed' >&2
      fi
      rm -rf "$work_dir"
      exit "$status"
    }
    trap cleanup 0
    trap 'exit 1' 1 2 15

    # Snapshot every key before the first atomic replacement.
    for key in \\
    \(keys)
    do
      if [ -e "$params_dir/$key" ]; then
        : > "$rollback_dir/$key.present"
        cat "$params_dir/$key" > "$rollback_dir/$key.value"
        sync -f "$rollback_dir/$key.value"
      else
        : > "$rollback_dir/$key.absent"
      fi
    done
    rollback_ready=1

    # Prepare and fsync all six new values before moving any into Params.
    for key in \\
    \(keys)
    do
      value="$(value_for_key "$key")"
      temporary="$(mktemp "$stage_dir/${key}.XXXXXX")"
      printf '%s' "$value" > "$temporary"
      sync -f "$temporary"
      mv -f "$temporary" "$stage_dir/$key"
      sync -f "$stage_dir/$key"
    done

    # Each replacement is atomic; the EXIT trap restores the complete snapshot
    # if a later replacement or read-back check fails.
    for key in \\
    \(keys)
    do
      mutation_started=1
      mv -f "$stage_dir/$key" "$params_dir/$key"
      sync -f "$params_dir/$key"
      sync -f "$params_dir"
    done

    for key in \\
    \(keys)
    do
      value="$(value_for_key "$key")"
      actual="$(cat "$params_dir/$key")"
      if [ "$actual" != "$value" ]; then
        printf '%s\\n' "VTSC physics Param read-back mismatch: $key" >&2
        exit 1
      fi
    done

    committed=1
    printf '{'
    separator=''
    for key in \\
    \(keys)
    do
      value="$(value_for_key "$key")"
      printf '%s"%s":%s' "$separator" "$key" "$value"
      separator=','
    done
    printf '}\\n'
    """
  }

  private static func validatedEntries(for parameters: SigmoidParameters) throws -> [Entry] {
    let locale = Locale(identifier: "en_US_POSIX")
    let candidates: [(String, Double, String, ClosedRange<Double>)] = [
      ("VisionTurnSpeedControlPhysicsAmplitude", parameters.a, String(format: "%.6f", locale: locale, parameters.a), -5.0 ... -0.2),
      ("VisionTurnSpeedControlPhysicsSteepness", parameters.b, String(format: "%.6f", locale: locale, parameters.b), -100_000.0 ... -100.0),
      ("VisionTurnSpeedControlPhysicsCenter", parameters.c, String(format: "%.6f", locale: locale, parameters.c), 0.000_01 ... 0.1),
      ("VisionTurnSpeedControlPhysicsBaseline", parameters.d, String(format: "%.6f", locale: locale, parameters.d), 2.0 ... 6.5),
      ("VisionTurnSpeedControlPhysicsMinLatAccel", parameters.minLat, String(format: "%.4f", locale: locale, parameters.minLat), 1.0 ... 3.0),
      ("VisionTurnSpeedControlPhysicsMaxLatAccel", parameters.maxLat, String(format: "%.4f", locale: locale, parameters.maxLat), 2.0 ... 5.5),
    ]
    guard candidates.map(\.0) == expectedKeys, Set(candidates.map(\.0)) == Set(expectedKeys) else {
      throw TiciParamTransactionCommandBuilderError.invalidParameterSet
    }

    var entries: [Entry] = []
    for (key, rawValue, renderedValue, allowedRange) in candidates {
      guard rawValue.isFinite else {
        throw TiciParamTransactionCommandBuilderError.nonFiniteValue(key: key)
      }
      guard isSafeDecimal(renderedValue), let deployedValue = Double(renderedValue), deployedValue.isFinite else {
        throw TiciParamTransactionCommandBuilderError.unsafeNumericValue(key: key, value: renderedValue)
      }
      guard allowedRange.contains(deployedValue) else {
        throw TiciParamTransactionCommandBuilderError.outOfRange(key: key, value: deployedValue)
      }
      entries.append(.init(key: key, renderedValue: renderedValue, deployedValue: deployedValue, allowedRange: allowedRange))
    }

    let values = Dictionary(uniqueKeysWithValues: entries.map { ($0.key, $0.deployedValue) })
    guard let minimum = values["VisionTurnSpeedControlPhysicsMinLatAccel"],
          let maximum = values["VisionTurnSpeedControlPhysicsMaxLatAccel"] else {
      throw TiciParamTransactionCommandBuilderError.invalidParameterSet
    }
    guard minimum <= maximum else {
      throw TiciParamTransactionCommandBuilderError.minimumExceedsMaximum(minimum: minimum, maximum: maximum)
    }
    return entries
  }

  private static func isSafeDecimal(_ value: String) -> Bool {
    let bytes = Array(value.utf8)
    guard !bytes.isEmpty else { return false }
    var index = 0
    if bytes[index] == 45 { index += 1 } // '-'
    guard index < bytes.count else { return false }

    var sawDecimal = false
    var digitsBeforeDecimal = 0
    var digitsAfterDecimal = 0
    while index < bytes.count {
      switch bytes[index] {
      case 48 ... 57:
        if sawDecimal { digitsAfterDecimal += 1 } else { digitsBeforeDecimal += 1 }
      case 46 where !sawDecimal: // '.'
        sawDecimal = true
      default:
        return false
      }
      index += 1
    }
    return digitsBeforeDecimal > 0 && (!sawDecimal || digitsAfterDecimal > 0)
  }

  private static func shellKeyList(_ entries: [Entry]) -> String {
    entries.enumerated().map { offset, entry in
      let suffix = offset == entries.count - 1 ? "" : " \\"
      return "  \(shellQuote(entry.key))\(suffix)"
    }.joined(separator: "\n")
  }

  private static func shellValueCases(_ entries: [Entry]) -> String {
    entries.map { entry in
      "    \(shellQuote(entry.key))) printf '%s' \(shellQuote(entry.renderedValue)) ;;"
    }.joined(separator: "\n")
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
