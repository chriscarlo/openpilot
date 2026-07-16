import Foundation

/// Reverts the pieces the Swift-owned production deployer mutates on a tici.
/// It intentionally uses only Git and POSIX file primitives: the device does
/// not import the openpilot Python package during recovery.
enum TiciProductionRollbackCommandBuilder {
  enum MapdRestoreDecision: Equatable, Sendable {
    case alreadyRestored
    case restoreRecordedArtifact
    case failMissingArtifact
  }

  static let resultMarker = "VTSC_ROLLBACK_V1"

  static func mapdRestoreDecision(
    currentActiveSHA256: String,
    expectedPreviousSHA256: String,
    rollbackArtifactPresent: Bool
  ) -> MapdRestoreDecision {
    if !expectedPreviousSHA256.isEmpty,
       currentActiveSHA256 == expectedPreviousSHA256 {
      return .alreadyRestored
    }
    if rollbackArtifactPresent { return .restoreRecordedArtifact }
    return expectedPreviousSHA256.isEmpty ? .alreadyRestored : .failMissingArtifact
  }

  static func command(journal: DeploymentRollbackJournal) throws -> String {
    try validate(journal)
    let values = physicsAndMapdParamValues(journal)
    let keys = values.map(\.key)
    let keyList = shellKeyList(keys)
    let stateCases = shellStateCases(values)
    let valueCases = shellValueCases(values)
    let expectedActiveSHA = shellQuote(journal.previousActiveMapdSHA256)

    return """
    set -eu
    repo='/data/openpilot'
    params_dir='/data/params/d'
    params_lock='/data/params/.lock'
    previous_head=\(shellQuote(journal.previousHead))
    expected_branch=\(shellQuote(journal.branch))
    rollback_mapd=\(shellQuote(journal.mapdRollbackPath))
    expected_active_sha=\(expectedActiveSHA)

    fail() {
      printf '%s\\n' "$1" >&2
      exit 1
    }

    [ -d "$params_dir" ] || fail 'Params directory is missing'
    offroad="$(cat "$params_dir/IsOffroad" 2>/dev/null || true)"
    onroad="$(cat "$params_dir/IsOnroad" 2>/dev/null || true)"
    lookahead="$(cat "$params_dir/MTSCLookaheadEnabled" 2>/dev/null || true)"
    if [ "$offroad" != '1' ] || [ "$onroad" != '0' ] || [ "$lookahead" != '0' ]; then
      fail 'refusing rollback unless tici is exactly offroad and Map Lookahead is disabled'
    fi
    for tool in git cat flock mktemp sync mv rm mkdir chmod dirname sha256sum awk base64 tr
    do
      command -v "$tool" >/dev/null 2>&1 || fail "required rollback tool is unavailable: $tool"
    done

    cd "$repo"
    [ "$(git branch --show-current)" = "$expected_branch" ] || fail 'rollback branch changed unexpectedly'
    git reset --hard "$previous_head"
    [ "$(git rev-parse HEAD)" = "$previous_head" ] || fail 'git rollback head mismatch'

    active="$repo/third_party/mapd/mapd"
    current_active_sha=''
    if [ -f "$active" ]; then
      current_active_sha="$(sha256sum "$active" | awk '{print $1}')"
    fi
    if [ -n "$expected_active_sha" ] && [ "$current_active_sha" = "$expected_active_sha" ]; then
      : # mapd is already restored; rollback replay is an exact no-op
    elif [ -f "$rollback_mapd" ]; then
      active_dir="$(dirname "$active")"
      [ -n "$expected_active_sha" ] && [ "$(sha256sum "$rollback_mapd" | awk '{print $1}')" = "$expected_active_sha" ] || [ -z "$expected_active_sha" ] || fail 'rollback mapd digest mismatch'
      temporary="$(mktemp "$active_dir/.mapd-rollback.XXXXXX")"
      cat "$rollback_mapd" > "$temporary"
      chmod 755 "$temporary"
      sync -f "$temporary"
      mv -f "$temporary" "$active"
      sync -f "$active"
      sync -f "$active_dir" || sync
      [ -z "$expected_active_sha" ] || [ "$(sha256sum "$active" | awk '{print $1}')" = "$expected_active_sha" ] || fail 'restored active mapd digest mismatch'
    elif [ -n "$expected_active_sha" ]; then
      fail 'rollback mapd artifact is missing and active mapd does not match the recorded previous digest'
    fi

    desired_present() {
      case "$1" in
    \(stateCases)
        *) return 1 ;;
      esac
    }

    desired_value() {
      case "$1" in
    \(valueCases)
        *) return 1 ;;
      esac
    }

    umask 077
    exec 9>"$params_lock"
    flock -x 9
    work_dir="$(mktemp -d "$params_dir/.vtsc-rollback.XXXXXX")"
    rollback_dir="$work_dir/rollback"
    stage_dir="$work_dir/stage"
    mkdir "$rollback_dir" "$stage_dir"
    committed=0
    rollback_ready=0
    mutation_started=0

    rollback_params() {
      for key in \\
    \(keyList)
      do
        if [ -f "$rollback_dir/$key.present" ]; then
          temporary="$(mktemp "$params_dir/.${key}.rollback.XXXXXX")"
          cat "$rollback_dir/$key.value" > "$temporary"
          sync -f "$temporary"
          mv -f "$temporary" "$params_dir/$key"
          sync -f "$params_dir/$key"
          sync -f "$params_dir" || sync
        else
          rm -f "$params_dir/$key"
          sync -f "$params_dir" || sync
        fi
      done
    }

    cleanup() {
      status=$?
      trap - 0 1 2 15
      if [ "$mutation_started" -eq 1 ] && [ "$rollback_ready" -eq 1 ] && [ "$committed" -ne 1 ]; then
        rollback_params || printf '%s\\n' 'tici rollback Param restoration failed' >&2
      fi
      rm -rf "$work_dir"
      exit "$status"
    }
    trap cleanup 0
    trap 'exit 1' 1 2 15

    for key in \\
    \(keyList)
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

    for key in \\
    \(keyList)
    do
      if [ "$(desired_present "$key")" = 1 ]; then
        temporary="$(mktemp "$stage_dir/${key}.XXXXXX")"
        desired_value "$key" > "$temporary"
        sync -f "$temporary"
        mv -f "$temporary" "$stage_dir/$key"
        sync -f "$stage_dir/$key"
      fi
    done

    for key in \\
    \(keyList)
    do
      mutation_started=1
      if [ "$(desired_present "$key")" = 1 ]; then
        mv -f "$stage_dir/$key" "$params_dir/$key"
        sync -f "$params_dir/$key"
      else
        rm -f "$params_dir/$key"
      fi
      sync -f "$params_dir" || sync
    done

    for key in \\
    \(keyList)
    do
      if [ "$(desired_present "$key")" = 1 ]; then
        expected="$(desired_value "$key")"
        actual="$(cat "$params_dir/$key")"
        [ "$actual" = "$expected" ] || fail "rollback Param read-back mismatch: $key"
      else
        [ ! -e "$params_dir/$key" ] || fail "rollback Param should be absent: $key"
      fi
    done
    committed=1
    printf '%s\\t%s\\n' '\(resultMarker)' "$(printf '%s' "$previous_head" | base64 | tr -d '\\n')"
    """
  }

  static func restoredHead(from output: String) throws -> String {
    let records = output.split(whereSeparator: \.isNewline).filter { $0.hasPrefix("\(resultMarker)\t") }
    guard records.count == 1, let record = records.first else {
      throw ApplyPipelineError.invalidDeploymentOutput(output)
    }
    let fields = record.split(separator: "\t", omittingEmptySubsequences: false)
    guard fields.count == 2,
          let data = Data(base64Encoded: String(fields[1])),
          let head = String(data: data, encoding: .utf8),
          head.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil
    else { throw ApplyPipelineError.invalidDeploymentOutput(String(record)) }
    return head
  }

  private struct Value: Sendable {
    var key: String
    var value: String?
  }

  private static func physicsAndMapdParamValues(_ journal: DeploymentRollbackJournal) -> [Value] {
    let physicsKeys = [
      "VisionTurnSpeedControlPhysicsAmplitude",
      "VisionTurnSpeedControlPhysicsSteepness",
      "VisionTurnSpeedControlPhysicsCenter",
      "VisionTurnSpeedControlPhysicsBaseline",
      "VisionTurnSpeedControlPhysicsMinLatAccel",
      "VisionTurnSpeedControlPhysicsMaxLatAccel",
    ]
    return physicsKeys.map { Value(key: $0, value: journal.previousPhysicsParams[$0] ?? nil) } + [
      Value(key: "MapdReleaseVersion", value: journal.previousMapdReleaseVersion),
      Value(key: "MapdVersion", value: journal.previousMapdVersion),
    ]
  }

  private static func validate(_ journal: DeploymentRollbackJournal) throws {
    guard journal.previousHead.range(of: #"^[0-9a-f]{40}$"#, options: .regularExpression) != nil,
          journal.branch.range(of: #"^[A-Za-z0-9._/-]+$"#, options: .regularExpression) != nil,
          !journal.branch.contains(".."), !journal.branch.hasPrefix("/"),
          journal.mapdRollbackPath.range(of: #"^/data/media/0/osm/binaries/mapd-rollback-[A-Za-z0-9._-]{1,128}$"#, options: .regularExpression) != nil,
          journal.previousActiveMapdSHA256.isEmpty || journal.previousActiveMapdSHA256.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil
    else { throw ApplyPipelineError.invalidDeploymentOutput("invalid production rollback journal") }
    for value in physicsAndMapdParamValues(journal).compactMap(\.value) {
      guard value.utf8.count <= 4_096, !value.utf8.contains(0) else {
        throw ApplyPipelineError.invalidDeploymentOutput("rollback journal contains an unsafe Param value")
      }
    }
  }

  private static func shellKeyList(_ keys: [String]) -> String {
    keys.enumerated().map { index, key in
      let suffix = index == keys.count - 1 ? "" : " \\"
      return "  \(shellQuote(key))\(suffix)"
    }.joined(separator: "\n")
  }

  private static func shellStateCases(_ values: [Value]) -> String {
    values.map { value in
      "    \(shellQuote(value.key))) printf '%s' \(value.value == nil ? "'0'" : "'1'") ;;"
    }.joined(separator: "\n")
  }

  private static func shellValueCases(_ values: [Value]) -> String {
    values.compactMap { value in
      guard let raw = value.value else { return nil }
      return "    \(shellQuote(value.key))) printf '%s' \(shellQuote(raw.data(using: .utf8)!.base64EncodedString())) | base64 -d ;;"
    }.joined(separator: "\n")
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
