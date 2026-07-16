import Foundation

/// Fields emitted by the tici's dependency-free deployment snapshot command.
///
/// Values are kept as raw bytes because the Q-curve source, tile manifest, and
/// persistent Params files must be inspected before any lossy text conversion.
public enum TiciSnapshotWireField: String, CaseIterable, Codable, Sendable {
  case bootID = "boot_id"
  case branch
  case head
  case dirty
  case isOffroad = "is_offroad"
  case isOnroad = "is_onroad"
  case mapLookaheadEnabled = "map_lookahead_enabled"
  case physicsAmplitude = "physics_amplitude"
  case physicsSteepness = "physics_steepness"
  case physicsCenter = "physics_center"
  case physicsBaseline = "physics_baseline"
  case physicsMinLatAccel = "physics_min_lat_accel"
  case physicsMaxLatAccel = "physics_max_lat_accel"
  case mapdReleaseVersion = "mapd_release_version"
  case mapdVersion = "mapd_version"
  case activeMapdSHA256 = "active_mapd_sha256"
  case qCurveFile = "q_curve_file"
  case tileManifest = "tile_manifest"
  case mapdCacheListing = "mapd_cache_listing"
  case memoryWholeCurveProfile = "memory_whole_curve_profile"
  case persistentWholeCurveProfile = "persistent_whole_curve_profile"
  case memoryLastGPSPosition = "memory_last_gps_position"
  case persistentLastGPSPosition = "persistent_last_gps_position"
  case runtimeEndIsOffroad = "runtime_end_is_offroad"
  case runtimeEndIsOnroad = "runtime_end_is_onroad"
  case runtimeEndMapLookaheadEnabled = "runtime_end_map_lookahead_enabled"
  case liveMapDataControllerStatus = "live_map_data_controller_status"
  case activeMapdBuildInfo = "active_mapd_build_info"
  case activeMapdELFHeader = "active_mapd_elf_header"
  case managerRunning = "manager_running"
  case mapdRunning = "mapd_running"
  case remoteEpochMilliseconds = "remote_epoch_milliseconds"
}

/// A lossless, typed representation of one remote tici snapshot.
///
/// A missing field is intentionally different from a present field with an
/// empty payload: callers can require the former while allowing an empty file
/// or unset persistent Param to be reported faithfully.
public struct TiciSnapshotWireSnapshot: Equatable, Sendable {
  public typealias RawValues = [TiciSnapshotWireField: Data]

  public let rawValues: RawValues

  public init(rawValues: RawValues = [:]) {
    self.rawValues = rawValues
  }

  public subscript(_ field: TiciSnapshotWireField) -> Data? {
    rawValues[field]
  }

  /// Returns a UTF-8 view when the field is present and textual.
  public func text(for field: TiciSnapshotWireField) -> String? {
    guard let value = rawValues[field] else { return nil }
    return String(data: value, encoding: .utf8)
  }
}

public enum TiciSnapshotWireCodecError: Error, Equatable, Sendable {
  case malformedLine(line: Int)
  case unknownField(name: String, line: Int)
  case duplicateField(TiciSnapshotWireField, line: Int)
  case invalidBase64(field: TiciSnapshotWireField, line: Int)
}

/// Strict codec for one-record-per-line `field<TAB>base64payload` snapshots.
///
/// It deliberately does not infer defaults. The deployment layer decides which
/// values are required for a given operation after it has a complete raw view.
public enum TiciSnapshotWireCodec {
  public static func encode(_ snapshot: TiciSnapshotWireSnapshot) -> String {
    TiciSnapshotWireField.allCases.compactMap { field in
      guard let value = snapshot.rawValues[field] else { return nil }
      return "\(field.rawValue)\t\(value.base64EncodedString())"
    }.joined(separator: "\n")
  }

  public static func decode(_ wire: String) throws -> TiciSnapshotWireSnapshot {
    guard !wire.isEmpty else { return TiciSnapshotWireSnapshot() }

    var values: TiciSnapshotWireSnapshot.RawValues = [:]
    let lines = wire.split(separator: "\n", omittingEmptySubsequences: false)

    for (offset, originalLine) in lines.enumerated() {
      let lineNumber = offset + 1
      var line = Substring(originalLine)
      if line.last == "\r" { line.removeLast() }

      // A single trailing newline is the normal command output terminator.
      if line.isEmpty, offset == lines.count - 1 { continue }

      let tabs = line.indices.filter { line[$0] == "\t" }
      guard tabs.count == 1, let tab = tabs.first else {
        throw TiciSnapshotWireCodecError.malformedLine(line: lineNumber)
      }

      let rawName = String(line[..<tab])
      guard let field = TiciSnapshotWireField(rawValue: rawName) else {
        throw TiciSnapshotWireCodecError.unknownField(name: rawName, line: lineNumber)
      }
      guard values[field] == nil else {
        throw TiciSnapshotWireCodecError.duplicateField(field, line: lineNumber)
      }

      let payload = String(line[line.index(after: tab)...])
      guard let decoded = Data(base64Encoded: payload) else {
        throw TiciSnapshotWireCodecError.invalidBase64(field: field, line: lineNumber)
      }
      values[field] = decoded
    }

    return TiciSnapshotWireSnapshot(rawValues: values)
  }
}

/// Builds the tici-side snapshot probe. File and process identity stays in
/// dependency-free shell; the bounded liveMapDataSP record uses the tici's
/// checked-out cereal runtime because Cap'n Proto interpretation belongs with
/// the exact deployed schema. Swift still owns every acceptance decision.
///
/// `compact_base64` removes any implementation-specific line wrapping using
/// POSIX shell built-ins, so a large Q-curve or manifest remains one record.
public enum TiciSnapshotWireCommandBuilder {
  static func tileManifestProbeShellFragment(
    offlinePath: String = "/data/media/0/osm/offline",
    adjacentManifestPath: String = "/data/media/0/osm/offline.manifest.json"
  ) -> String {
    let offline = shellQuote(offlinePath)
    let embedded = shellQuote(offlinePath + "/.tileset-manifest.json")
    let adjacent = shellQuote(adjacentManifestPath)
    return """
    tile_offline=\(offline)
    tile_embedded=\(embedded)
    tile_adjacent=\(adjacent)
    if [ -L "$tile_offline" ]; then
      if [ -f "$tile_embedded" ]; then
        emit_file tile_manifest "$tile_embedded"
      else
        emit_text tile_manifest 'invalid:canonical-pointer-missing-embedded-manifest'
      fi
    elif [ -d "$tile_offline" ] && [ ! -L "$tile_offline" ]; then
      if [ ! -f "$tile_embedded" ] && [ -f "$tile_adjacent" ]; then
        emit_file tile_manifest "$tile_adjacent"
      else
        emit_text tile_manifest 'invalid:direct-tree-manifest-topology'
      fi
    else
      emit_text tile_manifest 'invalid:active-offline-path'
    fi
    """
  }

  static func managerProbeShellFragment(
    repositoryPath: String = "/data/openpilot",
    procRoot: String = "/proc",
    pgrepCommand: String = "pgrep"
  ) -> String {
    let repository = shellQuote(repositoryPath)
    let proc = shellQuote(procRoot)
    let pgrep = shellQuote(pgrepCommand)
    return """
    manager_running=0
    manager_repo_input=\(repository)
    manager_repo=$(cd "$manager_repo_input" 2>/dev/null && pwd -P || :)
    manager_proc=\(proc)
    for manager_pid in $(\(pgrep) -f '(^|[ /])(\\./)?[m]anager\\.py([[:space:]]|$)' 2>/dev/null || :); do
      [ "$manager_pid" != "$$" ] || continue
      [ -d "$manager_proc/$manager_pid" ] || continue
      [ -r "$manager_proc/$manager_pid/cmdline" ] || continue
      manager_cwd=$(cd "$manager_proc/$manager_pid/cwd" 2>/dev/null && pwd -P || :)
      manager_cmdline=" $(tr '\\000' ' ' < "$manager_proc/$manager_pid/cmdline" 2>/dev/null || :) "
      manager_expected_dir="$manager_repo/system/manager"
      manager_expected_absolute="$manager_expected_dir/manager.py"
      manager_matches=0
      case "$manager_cmdline" in
        *" $manager_expected_absolute "*) manager_matches=1 ;;
      esac
      if [ "$manager_cwd" = "$manager_expected_dir" ]; then
        case "$manager_cmdline" in
          *" ./manager.py "*) manager_matches=1 ;;
        esac
      fi
      if [ "$manager_matches" = 1 ]; then
        manager_running=1
        break
      fi
    done
    emit_text manager_running "$manager_running"
    """
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }

  public static func inspectionCommand(
    includeRuntimePostflight: Bool = false,
    includeStaticPostflight: Bool = false,
    includeMapdBuildIdentity: Bool = true
  ) -> String {
    let buildIdentityRecords: String
    if includeRuntimePostflight || (includeStaticPostflight && includeMapdBuildIdentity) {
      buildIdentityRecords = """
      emit_first_bytes active_mapd_elf_header "$active_mapd" 20
      if [ -x "$active_mapd" ]; then
        emit_command active_mapd_build_info "$active_mapd" --build-info
      fi
      """
    } else {
      buildIdentityRecords = ""
    }
    let staticRecords: String
    if includeRuntimePostflight || includeStaticPostflight {
      staticRecords = """
      \(managerProbeShellFragment())
      if pgrep -x mapd >/dev/null 2>&1; then
        emit_text mapd_running 1
      else
        emit_text mapd_running 0
      fi
      """
    } else {
      staticRecords = ""
    }
    let endStateRecords: String
    if includeRuntimePostflight || includeStaticPostflight {
      endStateRecords = """
      emit_file runtime_end_is_offroad "$params_root/IsOffroad"
      emit_file runtime_end_is_onroad "$params_root/IsOnroad"
      emit_file runtime_end_map_lookahead_enabled "$params_root/MTSCLookaheadEnabled"
      """
    } else {
      endStateRecords = ""
    }
    let runtimeRecords: String
    if includeRuntimePostflight {
      runtimeRecords = """
      emit_command live_map_data_controller_status timeout 5 env PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c 'import time; from cereal import messaging; sm=messaging.SubMaster(["liveMapDataSP"], ignore_avg_freq=["liveMapDataSP"]); sm.update(2000); now=time.clock_gettime_ns(time.CLOCK_BOOTTIME); print("{}|{}|{}|{}|{}".format(int(sm.updated["liveMapDataSP"]), int(sm.valid["liveMapDataSP"]), int(sm.logMonoTime["liveMapDataSP"]), int(sm["liveMapDataSP"].roadGeometryValid), now), end="")'
      emit_file memory_whole_curve_profile "$memory_params_root/MapWholeCurveProfile"
      emit_file persistent_whole_curve_profile "$params_root/MapWholeCurveProfile"
      emit_file memory_last_gps_position "$memory_params_root/LastGPSPosition"
      emit_file persistent_last_gps_position "$params_root/LastGPSPosition"
      """
    } else {
      runtimeRecords = ""
    }
    return """
    /bin/sh <<'VTSC_SNAPSHOT_SH'
    set -eu

    repo=/data/openpilot
    params_root=/data/params/d
    memory_params_root=/dev/shm/params/d
    cache_root=/data/media/0/osm/binaries
    active_mapd=/data/openpilot/third_party/mapd/mapd

    compact_base64() {
      while IFS= read -r chunk || [ -n "$chunk" ]; do
        printf '%s' "$chunk"
      done
    }

    emit_text() {
      field=$1
      value=$2
      printf '%s\\t' "$field"
      printf '%s' "$value" | base64 | compact_base64
      printf '\\n'
    }

    emit_file() {
      field=$1
      path=$2
      [ -f "$path" ] || return 0
      printf '%s\\t' "$field"
      base64 < "$path" | compact_base64
      printf '\\n'
    }

    emit_first_bytes() {
      field=$1
      path=$2
      count=$3
      [ -f "$path" ] || return 0
      printf '%s\\t' "$field"
      dd if="$path" bs=1 count="$count" 2>/dev/null | base64 | compact_base64
      printf '\\n'
    }

    emit_command() {
      field=$1
      shift
      printf '%s\\t' "$field"
      "$@" 2>/dev/null | base64 | compact_base64 || :
      printf '\\n'
    }

    file_sha256() {
      path=$1
      if [ -f "$path" ]; then
        set -- $(sha256sum "$path")
        printf '%s' "$1"
      fi
    }

    cache_listing() {
      for candidate in "$cache_root"/*; do
        [ -f "$candidate" ] || continue
        printf '%s\\t%s\\n' "$candidate" "$(file_sha256 "$candidate")"
      done
    }

    git_dirty=$(git -C "$repo" status --porcelain 2>/dev/null || :)
    if [ -n "$git_dirty" ]; then dirty=1; else dirty=0; fi

    emit_text boot_id "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || :)"
    emit_text branch "$(git -C "$repo" branch --show-current 2>/dev/null || :)"
    emit_text head "$(git -C "$repo" rev-parse HEAD 2>/dev/null || :)"
    emit_text dirty "$dirty"
    emit_file is_offroad "$params_root/IsOffroad"
    emit_file is_onroad "$params_root/IsOnroad"
    emit_file map_lookahead_enabled "$params_root/MTSCLookaheadEnabled"
    emit_file physics_amplitude "$params_root/VisionTurnSpeedControlPhysicsAmplitude"
    emit_file physics_steepness "$params_root/VisionTurnSpeedControlPhysicsSteepness"
    emit_file physics_center "$params_root/VisionTurnSpeedControlPhysicsCenter"
    emit_file physics_baseline "$params_root/VisionTurnSpeedControlPhysicsBaseline"
    emit_file physics_min_lat_accel "$params_root/VisionTurnSpeedControlPhysicsMinLatAccel"
    emit_file physics_max_lat_accel "$params_root/VisionTurnSpeedControlPhysicsMaxLatAccel"
    emit_file mapd_release_version "$params_root/MapdReleaseVersion"
    emit_file mapd_version "$params_root/MapdVersion"
    emit_text active_mapd_sha256 "$(file_sha256 "$active_mapd")"
    emit_file q_curve_file "$repo/sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py"
    \(tileManifestProbeShellFragment())
    emit_text mapd_cache_listing "$(cache_listing)"
    \(buildIdentityRecords)
    \(staticRecords)
    \(runtimeRecords)
    \(endStateRecords)
    \(includeRuntimePostflight ? "emit_text remote_epoch_milliseconds \"$(date +%s%3N)\"" : "")
    VTSC_SNAPSHOT_SH
    """
  }
}
