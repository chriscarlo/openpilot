import Foundation

/// Fields emitted by the tici's dependency-free deployment snapshot command.
///
/// Values are kept as raw bytes because the Q-curve source, tile manifest, and
/// persistent Params files must be inspected before any lossy text conversion.
public enum TiciSnapshotWireField: String, CaseIterable, Codable, Sendable {
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
    case activeMapdBuildInfo = "active_mapd_build_info"
    case activeMapdELFHeader = "active_mapd_elf_header"
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

/// Builds the tici-side snapshot probe without importing the checked-out
/// openpilot Python environment. It emits only the wire records decoded above.
///
/// `compact_base64` removes any implementation-specific line wrapping using
/// POSIX shell built-ins, so a large Q-curve or manifest remains one record.
public enum TiciSnapshotWireCommandBuilder {
  public static func inspectionCommand(includeRuntimePostflight: Bool = false) -> String {
    let runtimeRecords: String
    if includeRuntimePostflight {
      runtimeRecords = """
      emit_file memory_whole_curve_profile "$memory_params_root/MapWholeCurveProfile"
      emit_file persistent_whole_curve_profile "$params_root/MapWholeCurveProfile"
      emit_file memory_last_gps_position "$memory_params_root/LastGPSPosition"
      emit_file persistent_last_gps_position "$params_root/LastGPSPosition"
      emit_first_bytes active_mapd_elf_header "$active_mapd" 20
      if [ -x "$active_mapd" ]; then
        emit_command active_mapd_build_info "$active_mapd" --build-info
      fi
      if pgrep -x mapd >/dev/null 2>&1; then
        emit_text mapd_running 1
      else
        emit_text mapd_running 0
      fi
      emit_text remote_epoch_milliseconds "$(date +%s%3N)"
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
    emit_file tile_manifest /data/media/0/osm/offline/.tileset-manifest.json
    emit_text mapd_cache_listing "$(cache_listing)"
    \(runtimeRecords)
    VTSC_SNAPSHOT_SH
    """
  }
}
