import Foundation

/// Errors raised before a mapd-release transaction is handed to the tici.
///
/// The command intentionally accepts only the immutable release layout used by
/// the deployment journal. That keeps a malformed journal or a future caller
/// from turning the SSH helper into an arbitrary-file writer.
public enum TiciMapdReleaseTransactionCommandBuilderError: Error, Equatable, Sendable {
  case invalidReleaseIdentity(String)
  case invalidPersistentCacheFileName(String)
  case invalidStagedPath(String)
  case invalidRollbackPath(String)
  case invalidResult(String)
  case invalidBuildInfo(String)
  case identityMismatch(field: String, expected: String, actual: String)
}

/// The success record emitted by `TiciMapdReleaseTransactionCommandBuilder`.
///
/// `buildInfoJSON` is deliberately left raw until it reaches the Mac. The
/// tici command proves the binary can run, but Swift performs the semantic
/// `--build-info` identity validation so the device never needs a JSON parser.
public struct TiciMapdReleaseTransactionResult: Equatable, Sendable {
  public var releaseID: String
  public var sha256: String
  public var cachePath: String
  public var rollbackPath: String
  public var buildInfoJSON: Data

  public init(
    releaseID: String,
    sha256: String,
    cachePath: String,
    rollbackPath: String,
    buildInfoJSON: Data
  ) {
    self.releaseID = releaseID
    self.sha256 = sha256
    self.cachePath = cachePath
    self.rollbackPath = rollbackPath
    self.buildInfoJSON = buildInfoJSON
  }
}

/// The device-side journal state observed by the preflight recovery command.
/// `clean` means no uncommitted deployment required restoration; `recovered`
/// means the command restored a durable pre-mutation snapshot before returning.
public enum TiciMapdReleaseRecoveryOutcome: String, Equatable, Sendable {
  case clean
  case recovered
}

/// The subset of mapd's `--build-info` document that establishes the immutable
/// release identity used by the VTSC deployment pipeline.
public struct TiciMapdReleaseBuildInfo: Codable, Equatable, Sendable {
  public var releaseID: String
  public var buildID: String
  public var estimatorVersion: String
  public var capabilities: [String]
  public var identityMarkers: [String]

  enum CodingKeys: String, CodingKey {
    case releaseID
    case buildID
    case estimatorVersion
    case capabilities
    case identityMarkers
  }

  public init(
    releaseID: String,
    buildID: String,
    estimatorVersion: String,
    capabilities: [String],
    identityMarkers: [String]
  ) {
    self.releaseID = releaseID
    self.buildID = buildID
    self.estimatorVersion = estimatorVersion
    self.capabilities = capabilities
    self.identityMarkers = identityMarkers
  }
}

/// Builds the mapd-release half of a tici deployment from shell primitives.
///
/// The tici is only asked to validate bytes, move durable files, and atomically
/// update Params. Swift owns the release-manifest and build-info semantics.
/// The result line has a fixed tab-separated, base64-encoded wire format so a
/// noisy remote shell cannot corrupt binary-safe `--build-info` output.
public enum TiciMapdReleaseTransactionCommandBuilder {
  public static let probeMarker = "VTSC_MAPD_PROBE_V1"
  public static let resultMarker = "VTSC_MAPD_RELEASE_V1"
  public static let recoveryMarker = "VTSC_MAPD_RECOVERY_V1"
  /// The persistent, device-side journal for a single mapd transaction. The
  /// directory is deliberately fixed and retained after cleanup so its lock
  /// file remains a stable serialization point across SSH invocations.
  public static let transactionDirectory = "/data/media/0/osm/binaries/.vtsc-mapd-release-transaction-v1"

  /// Runs every executable-identity check while the release is still a
  /// disposable staging file. Swift validates the returned build-info before
  /// the later command can replace the active runtime binary.
  public static func probeCommand(
    release: ValidatedMapdReleaseArtifact,
    stagedPath: String
  ) throws -> String {
    try validateReleaseIdentity(release)
    try validateStagedPath(stagedPath)

    let artifact = release.artifact
    let markers = [
      "MapdReleaseID:\(artifact.releaseID)",
      "MapdBuildID:\(artifact.buildID)",
      artifact.capability,
    ]
    let markerChecks = markers.map { marker in
      """
      if ! LC_ALL=C grep -F -q \(shellQuote(marker)) "$staged"; then
        fail 'staged mapd is missing an expected literal identity marker'
      fi
      """
    }.joined(separator: "\n")

    return """
    set -eu
    staged=\(shellQuote(stagedPath))
    expected_sha=\(shellQuote(artifact.sha256))
    params_dir='/data/params/d'

    fail() {
      printf '%s\\n' "$1" >&2
      exit 1
    }

    [ -f "$staged" ] || fail 'staged mapd release is missing'
    for tool in od sha256sum grep base64 tr chmod
    do
      command -v "$tool" >/dev/null 2>&1 || fail "required mapd probe tool is unavailable: $tool"
    done

    set -- $(LC_ALL=C od -An -v -t u1 -N 20 "$staged")
    if [ "$#" -lt 20 ] || [ "$1" != 127 ] || [ "$2" != 69 ] || [ "$3" != 76 ] || [ "$4" != 70 ] || [ "$5" != 2 ] || [ "$6" != 1 ] || [ "${19}" != 183 ] || [ "${20}" != 0 ]; then
      fail 'staged mapd is not ELF64 little-endian AArch64'
    fi
    set -- $(sha256sum "$staged")
    [ "$#" -ge 1 ] && [ "$1" = "$expected_sha" ] || fail 'staged mapd digest mismatch'

    \(markerChecks)

    \(TiciParkedMutationGate.shellFragment(
      refusalMessage: "refusing mapd probe chmod/execute unless tici is exactly offroad and Map Lookahead is disabled"
    ))
    chmod 755 "$staged"
    build_info="$("$staged" --build-info)"
    [ -n "$build_info" ] || fail 'mapd --build-info produced no output'
    printf '%s\\t%s\\n' '\(probeMarker)' "$(printf '%s' "$build_info" | base64 | tr -d '\\n')"
    """
  }

  /// Recovers an interrupted mapd transaction without requiring a new release
  /// artifact. It is safe for `ApplyPipeline` to run before preflight: a clean
  /// device emits `VTSC_MAPD_RECOVERY_V1\tclean`; a restored prepared journal
  /// emits `VTSC_MAPD_RECOVERY_V1\trecovered`. A committed journal is only
  /// cleaned, never rolled back.
  public static func recoveryCommand() -> String {
    """
    set -eu
    active='/data/openpilot/third_party/mapd/mapd'
    params_dir='/data/params/d'
    params_lock='/data/params/.lock'
    transaction_dir=\(shellQuote(transactionDirectory))
    transaction_lock="$transaction_dir/.lock"
    journal="$transaction_dir/journal"
    journal_state="$journal/state"
    journal_rollback="$journal/rollback"
    recovery_marker=\(shellQuote(recoveryMarker))

    fail() {
      printf '%s\\n' "$1" >&2
      exit 1
    }

    durable_file() {
      sync -f "$1"
    }

    durable_directory() {
      sync -f "$1" || sync
    }

    digest() {
      set -- $(sha256sum "$1")
      [ "$#" -ge 1 ] || fail 'sha256sum did not produce a digest'
      printf '%s' "$1"
    }

    require_safe_state() {
      \(TiciParkedMutationGate.shellFragment(
        refusalMessage: "refusing mapd transaction recovery unless tici is exactly offroad and Map Lookahead is disabled"
      ))
    }

    restore_mapd_params_from_journal() {
      for key in MapdReleaseVersion MapdVersion
      do
        if [ -f "$journal_rollback/$key.present" ] && [ ! -e "$journal_rollback/$key.absent" ]; then
          [ -f "$journal_rollback/$key.value" ] || return 1
          temporary="$(mktemp "$params_dir/.${key}.rollback.XXXXXX")"
          cat "$journal_rollback/$key.value" > "$temporary"
          durable_file "$temporary"
          mv -f "$temporary" "$params_dir/$key"
          durable_file "$params_dir/$key"
          durable_directory "$params_dir"
          expected="$(cat "$journal_rollback/$key.value")"
          actual="$(cat "$params_dir/$key")"
          [ "$actual" = "$expected" ] || return 1
        elif [ -f "$journal_rollback/$key.absent" ] && [ ! -e "$journal_rollback/$key.present" ]; then
          rm -f "$params_dir/$key"
          durable_directory "$params_dir"
          [ ! -e "$params_dir/$key" ] || return 1
        else
          return 1
        fi
      done
    }

    restore_active_mapd_from_journal() {
      [ -f "$journal_rollback/mapd" ] || return 1
      [ -f "$journal_rollback/mapd.sha256" ] || return 1
      expected="$(cat "$journal_rollback/mapd.sha256")"
      printf '%s\\n' "$expected" | LC_ALL=C grep -E -q '^[0-9a-f]{64}$' || return 1
      [ "$(digest "$journal_rollback/mapd")" = "$expected" ] || return 1
      active_dir="$(dirname "$active")"
      temporary="$(mktemp "$active_dir/.mapd-recovery.XXXXXX")"
      cat "$journal_rollback/mapd" > "$temporary"
      chmod 755 "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$active"
      durable_file "$active"
      durable_directory "$active_dir"
      [ "$(digest "$active")" = "$expected" ] || return 1
    }

    restore_uncommitted_journal() {
      restore_active_mapd_from_journal && restore_mapd_params_from_journal
    }

    cleanup_journal() {
      [ -e "$journal" ] || return 0
      rm -rf "$journal"
      durable_directory "$transaction_dir"
    }

    for tool in sha256sum grep flock mktemp sync mv mkdir rm cat chmod dirname
    do
      command -v "$tool" >/dev/null 2>&1 || fail "required mapd recovery tool is unavailable: $tool"
    done

    # Merely creating/locking the persistent transaction directory cannot
    # affect the active binary. If no journal exists, report clean without an
    # offroad gate so preflight can observe a parked or live device safely.
    umask 077
    mkdir -p "$transaction_dir"
    chmod 700 "$transaction_dir"
    durable_directory "$transaction_dir"
    exec 8>"$transaction_lock"
    flock -x 8
    if [ ! -e "$journal" ]; then
      printf '%s\\t%s\\n' "$recovery_marker" 'clean'
      exit 0
    fi

    [ -d "$journal" ] || fail 'refusing mapd transaction recovery: journal is not a directory'
    [ -f "$journal_state" ] || fail 'refusing mapd transaction recovery: journal state is missing'
    require_safe_state
    exec 9>"$params_lock"
    flock -x 9
    require_safe_state

    state="$(cat "$journal_state")"
    case "$state" in
      prepared)
        restore_uncommitted_journal || fail 'refusing mapd transaction recovery: could not safely restore journal'
        cleanup_journal || fail 'refusing mapd transaction recovery: could not clear restored journal'
        printf '%s\\t%s\\n' "$recovery_marker" 'recovered'
        ;;
      committed)
        # Do not revert a successful install just because its journal survived
        # an interruption during cleanup.
        cleanup_journal || fail 'refusing mapd transaction recovery: could not clear committed journal'
        printf '%s\\t%s\\n' "$recovery_marker" 'clean'
        ;;
      *)
        fail 'refusing mapd transaction recovery: journal has an unknown state'
        ;;
    esac
    """
  }

  public static func installCommand(
    release: ValidatedMapdReleaseArtifact,
    stagedPath: String,
    rollbackPath: String
  ) throws -> String {
    try validateInputs(release: release, stagedPath: stagedPath, rollbackPath: rollbackPath)

    let artifact = release.artifact
    let cachePath = "/data/media/0/osm/binaries/\(release.persistentCacheFileName)"
    let markers = [
      "MapdReleaseID:\(artifact.releaseID)",
      "MapdBuildID:\(artifact.buildID)",
      artifact.capability,
    ]
    let markerChecks = markers.map { marker in
      """
      if ! LC_ALL=C grep -F -q \(shellQuote(marker)) "$staged"; then
        fail 'staged mapd is missing an expected literal identity marker'
      fi
      """
    }.joined(separator: "\n")

    return """
    set -eu
    staged=\(shellQuote(stagedPath))
    cache=\(shellQuote(cachePath))
    rollback=\(shellQuote(rollbackPath))
    active='/data/openpilot/third_party/mapd/mapd'
    params_dir='/data/params/d'
    params_lock='/data/params/.lock'
    transaction_dir=\(shellQuote(transactionDirectory))
    transaction_lock="$transaction_dir/.lock"
    journal="$transaction_dir/journal"
    journal_state="$journal/state"
    journal_rollback="$journal/rollback"
    release_id=\(shellQuote(artifact.releaseID))
    expected_sha=\(shellQuote(artifact.sha256))
    result_marker=\(shellQuote(resultMarker))
    journal_temporary=''
    transaction_journal_ready=0
    transaction_committed=0

    fail() {
      printf '%s\\n' "$1" >&2
      exit 1
    }

    durable_file() {
      sync -f "$1"
    }

    durable_directory() {
      sync -f "$1" || sync
    }

    digest() {
      set -- $(sha256sum "$1")
      [ "$#" -ge 1 ] || fail 'sha256sum did not produce a digest'
      printf '%s' "$1"
    }

    require_safe_state() {
      \(TiciParkedMutationGate.shellFragment(
        refusalMessage: "refusing mapd release mutation unless tici is exactly offroad and Map Lookahead is disabled"
      ))
    }

    journal_is_committed() {
      [ -f "$journal_state" ] || return 1
      [ "$(cat "$journal_state")" = 'committed' ]
    }

    write_journal_state() {
      state="$1"
      temporary="$(mktemp "$journal/.state.XXXXXX")"
      printf '%s' "$state" > "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$journal_state"
      durable_file "$journal_state"
      durable_directory "$journal"
    }

    restore_mapd_params_from_journal() {
      for key in MapdReleaseVersion MapdVersion
      do
        if [ -f "$journal_rollback/$key.present" ] && [ ! -e "$journal_rollback/$key.absent" ]; then
          [ -f "$journal_rollback/$key.value" ] || return 1
          temporary="$(mktemp "$params_dir/.${key}.rollback.XXXXXX")"
          cat "$journal_rollback/$key.value" > "$temporary"
          durable_file "$temporary"
          mv -f "$temporary" "$params_dir/$key"
          durable_file "$params_dir/$key"
          durable_directory "$params_dir"
          expected="$(cat "$journal_rollback/$key.value")"
          actual="$(cat "$params_dir/$key")"
          [ "$actual" = "$expected" ] || return 1
        elif [ -f "$journal_rollback/$key.absent" ] && [ ! -e "$journal_rollback/$key.present" ]; then
          rm -f "$params_dir/$key"
          durable_directory "$params_dir"
          [ ! -e "$params_dir/$key" ] || return 1
        else
          return 1
        fi
      done
    }

    restore_active_mapd_from_journal() {
      [ -f "$journal_rollback/mapd" ] || return 1
      [ -f "$journal_rollback/mapd.sha256" ] || return 1
      expected="$(cat "$journal_rollback/mapd.sha256")"
      printf '%s\\n' "$expected" | LC_ALL=C grep -E -q '^[0-9a-f]{64}$' || return 1
      [ "$(digest "$journal_rollback/mapd")" = "$expected" ] || return 1
      active_dir="$(dirname "$active")"
      temporary="$(mktemp "$active_dir/.mapd-recovery.XXXXXX")"
      cat "$journal_rollback/mapd" > "$temporary"
      chmod 755 "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$active"
      durable_file "$active"
      durable_directory "$active_dir"
      [ "$(digest "$active")" = "$expected" ] || return 1
    }

    restore_uncommitted_journal() {
      restore_active_mapd_from_journal && restore_mapd_params_from_journal
    }

    cleanup_journal() {
      [ -e "$journal" ] || return 0
      rm -rf "$journal"
      durable_directory "$transaction_dir"
    }

    recover_prior_journal() {
      [ -e "$journal" ] || return 0
      [ -d "$journal" ] || fail 'refusing mapd release install: transaction journal is not a directory'
      [ -f "$journal_state" ] || fail 'refusing mapd release install: transaction journal state is missing'
      state="$(cat "$journal_state")"
      case "$state" in
        prepared)
          restore_uncommitted_journal || fail 'refusing mapd release install: could not safely restore uncommitted transaction'
          cleanup_journal || fail 'refusing mapd release install: could not clear recovered transaction journal'
          ;;
        committed)
          # A completed install is never rolled back merely because a prior
          # process was interrupted while removing its journal.
          cleanup_journal || fail 'refusing mapd release install: could not clear committed transaction journal'
          ;;
        *)
          fail 'refusing mapd release install: transaction journal has an unknown state'
          ;;
      esac
    }

    prepare_transaction_journal() {
      [ ! -e "$journal" ] || fail 'transaction journal still exists after recovery'
      [ -f "$active" ] || fail 'active mapd is missing; refusing to replace without rollback material'
      journal_temporary="$(mktemp -d "$transaction_dir/.journal.new.XXXXXX")"
      journal_temporary_rollback="$journal_temporary/rollback"
      mkdir "$journal_temporary_rollback"

      temporary="$(mktemp "$journal_temporary_rollback/.mapd.XXXXXX")"
      cat "$active" > "$temporary"
      chmod 755 "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$journal_temporary_rollback/mapd"
      durable_file "$journal_temporary_rollback/mapd"
      snapshot_sha="$(digest "$journal_temporary_rollback/mapd")"
      printf '%s' "$snapshot_sha" > "$journal_temporary_rollback/mapd.sha256"
      durable_file "$journal_temporary_rollback/mapd.sha256"

      for key in MapdReleaseVersion MapdVersion
      do
        if [ -e "$params_dir/$key" ]; then
          printf '%s' 'present' > "$journal_temporary_rollback/$key.present"
          durable_file "$journal_temporary_rollback/$key.present"
          temporary="$(mktemp "$journal_temporary_rollback/.${key}.XXXXXX")"
          cat "$params_dir/$key" > "$temporary"
          durable_file "$temporary"
          mv -f "$temporary" "$journal_temporary_rollback/$key.value"
          durable_file "$journal_temporary_rollback/$key.value"
        else
          printf '%s' 'absent' > "$journal_temporary_rollback/$key.absent"
          durable_file "$journal_temporary_rollback/$key.absent"
        fi
      done
      durable_directory "$journal_temporary_rollback"

      temporary="$(mktemp "$journal_temporary/.state.XXXXXX")"
      printf '%s' 'prepared' > "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$journal_temporary/state"
      durable_file "$journal_temporary/state"
      durable_directory "$journal_temporary"

      # Publish the fully durable rollback snapshot atomically. No active
      # mapd or Mapd* Param changes occur until this name is visible.
      mv -f "$journal_temporary" "$journal"
      journal_temporary=''
      durable_directory "$transaction_dir"
      transaction_journal_ready=1
    }

    persist_external_rollback() {
      [ -f "$journal_rollback/mapd" ] || return 1
      [ -f "$journal_rollback/mapd.sha256" ] || return 1
      expected="$(cat "$journal_rollback/mapd.sha256")"
      [ "$(digest "$journal_rollback/mapd")" = "$expected" ] || return 1
      rollback_dir="$(dirname "$rollback")"
      mkdir -p "$rollback_dir"
      temporary="$(mktemp "$rollback_dir/.mapd-rollback.XXXXXX")"
      cat "$journal_rollback/mapd" > "$temporary"
      chmod 755 "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$rollback"
      durable_file "$rollback"
      durable_directory "$rollback_dir"
      [ "$(digest "$rollback")" = "$expected" ] || return 1
    }

    cleanup_transaction() {
      status=$?
      trap - 0 1 2 15
      # The committed state is authoritative. This keeps normal cleanup (and
      # a later installer) from reverting a release that fully committed.
      if [ "$transaction_journal_ready" -eq 1 ] && [ "$transaction_committed" -ne 1 ] && ! journal_is_committed; then
        if ! restore_uncommitted_journal; then
          printf '%s\\n' 'mapd release transaction rollback failed; journal retained for recovery' >&2
        else
          cleanup_journal || printf '%s\\n' 'mapd release recovered journal cleanup failed' >&2
        fi
      fi
      if [ -n "$journal_temporary" ] && [ -d "$journal_temporary" ]; then
        rm -rf "$journal_temporary" || true
      fi
      exit "$status"
    }

    [ -f "$staged" ] || fail 'staged mapd release is missing'
    for tool in od sha256sum grep base64 tr flock mktemp sync mv mkdir rm cat chmod dirname
    do
      command -v "$tool" >/dev/null 2>&1 || fail "required deployment tool is unavailable: $tool"
    done

    # The transaction lock covers stale-journal recovery, the active binary,
    # and the mapd-version Params. Retaining its parent directory means an
    # interrupted cleanup cannot accidentally create a second lock inode.
    umask 077
    mkdir -p "$transaction_dir"
    chmod 700 "$transaction_dir"
    durable_directory "$transaction_dir"
    exec 8>"$transaction_lock"
    flock -x 8
    require_safe_state
    exec 9>"$params_lock"
    flock -x 9
    require_safe_state
    trap cleanup_transaction 0
    trap 'exit 1' 1 2 15

    # A prepared journal is a durable pre-mutation snapshot. Recover it before
    # considering the new staged release; an unknown journal is preserved and
    # blocks deployment rather than guessing at the prior active state.
    recover_prior_journal

    set -- $(LC_ALL=C od -An -v -t u1 -N 20 "$staged")
    if [ "$#" -lt 20 ] || [ "$1" != 127 ] || [ "$2" != 69 ] || [ "$3" != 76 ] || [ "$4" != 70 ] || [ "$5" != 2 ] || [ "$6" != 1 ] || [ "${19}" != 183 ] || [ "${20}" != 0 ]; then
      fail 'staged mapd is not ELF64 little-endian AArch64'
    fi

    actual_sha="$(digest "$staged")"
    [ "$actual_sha" = "$expected_sha" ] || fail 'staged mapd digest mismatch'

    \(markerChecks)

    chmod 755 "$staged"
    build_info="$("$staged" --build-info)"
    [ -n "$build_info" ] || fail 'mapd --build-info produced no output'

    # A staged binary can take time to self-report. Recheck after that work and
    # immediately before we create the rollback point / mutate the active path.
    require_safe_state

    cache_dir="$(dirname "$cache")"
    mkdir -p "$cache_dir"
    durable_file "$staged"
    mv -f "$staged" "$cache"
    durable_file "$cache"
    durable_directory "$cache_dir"
    [ "$(digest "$cache")" = "$expected_sha" ] || fail 'persistent mapd cache digest mismatch'

    prepare_transaction_journal
    persist_external_rollback || fail 'could not persist mapd rollback artifact from transaction journal'

    active_dir="$(dirname "$active")"
    active_temporary="$(mktemp "$active_dir/.mapd-deploy.XXXXXX")"
    cat "$cache" > "$active_temporary"
    chmod 755 "$active_temporary"
    durable_file "$active_temporary"
    mv -f "$active_temporary" "$active"
    durable_file "$active"
    durable_directory "$active_dir"
    [ "$(digest "$active")" = "$expected_sha" ] || fail 'active mapd digest mismatch'

    param_stage="$journal/stage"
    mkdir "$param_stage"
    durable_directory "$journal"

    for key in MapdReleaseVersion MapdVersion
    do
      temporary="$(mktemp "$param_stage/${key}.XXXXXX")"
      printf '%s' "$release_id" > "$temporary"
      durable_file "$temporary"
      mv -f "$temporary" "$param_stage/$key"
      durable_file "$param_stage/$key"
    done

    for key in MapdReleaseVersion MapdVersion
    do
      mv -f "$param_stage/$key" "$params_dir/$key"
      durable_file "$params_dir/$key"
      durable_directory "$params_dir"
    done
    for key in MapdReleaseVersion MapdVersion
    do
      actual="$(cat "$params_dir/$key")"
      [ "$actual" = "$release_id" ] || fail "mapd release Param read-back mismatch: $key"
    done
    write_journal_state 'committed'
    transaction_committed=1
    # Cleanup is deliberately best-effort only after the durable committed
    # marker. Reporting this install as failed would make the host's broader
    # rollback journal undo a completed release merely because deletion was
    # interrupted; the next recovery command will remove it safely instead.
    cleanup_journal || printf '%s\\n' 'mapd release committed; journal cleanup deferred' >&2

    release_b64="$(printf '%s' "$release_id" | base64 | tr -d '\\n')"
    cache_b64="$(printf '%s' "$cache" | base64 | tr -d '\\n')"
    rollback_b64="$(printf '%s' "$rollback" | base64 | tr -d '\\n')"
    build_info_b64="$(printf '%s' "$build_info" | base64 | tr -d '\\n')"
    printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$result_marker" "$release_b64" "$expected_sha" "$cache_b64" "$rollback_b64" "$build_info_b64"
    """
  }

  /// Decodes the last success record in an SSH command's stdout. Diagnostics
  /// before the marker are intentionally ignored, but malformed or duplicate
  /// fields are never silently accepted.
  public static func decodeResult(_ output: String) throws -> TiciMapdReleaseTransactionResult {
    let matchingLines = output.split(whereSeparator: \.isNewline).filter {
      $0.hasPrefix("\(resultMarker)\t")
    }
    guard matchingLines.count == 1, let line = matchingLines.first else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(output)
    }
    let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
    guard fields.count == 6, fields[0] == Substring(resultMarker) else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(String(line))
    }
    guard let releaseData = Data(base64Encoded: String(fields[1])),
          let releaseID = String(data: releaseData, encoding: .utf8),
          let cacheData = Data(base64Encoded: String(fields[3])),
          let cachePath = String(data: cacheData, encoding: .utf8),
          let rollbackData = Data(base64Encoded: String(fields[4])),
          let rollbackPath = String(data: rollbackData, encoding: .utf8),
          let buildInfo = Data(base64Encoded: String(fields[5]))
    else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(String(line))
    }
    return TiciMapdReleaseTransactionResult(
      releaseID: releaseID,
      sha256: String(fields[2]),
      cachePath: cachePath,
      rollbackPath: rollbackPath,
      buildInfoJSON: buildInfo
    )
  }

  public static func decodeProbe(_ output: String) throws -> Data {
    let matchingLines = output.split(whereSeparator: \.isNewline).filter {
      $0.hasPrefix("\(probeMarker)\t")
    }
    guard matchingLines.count == 1, let line = matchingLines.first else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(output)
    }
    let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
    guard fields.count == 2, fields[0] == Substring(probeMarker),
          let data = Data(base64Encoded: String(fields[1])), !data.isEmpty
    else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(String(line))
    }
    return data
  }

  /// Decodes the one-line outcome of `recoveryCommand()`. The command is
  /// intentionally machine-readable so callers can surface recovery before a
  /// new release preflight without scraping human diagnostics.
  public static func decodeRecoveryOutcome(_ output: String) throws -> TiciMapdReleaseRecoveryOutcome {
    let matchingLines = output.split(whereSeparator: \.isNewline).filter {
      $0.hasPrefix("\(recoveryMarker)\t")
    }
    guard matchingLines.count == 1, let line = matchingLines.first else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(output)
    }
    let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
    guard fields.count == 2,
          fields[0] == Substring(recoveryMarker),
          let outcome = TiciMapdReleaseRecoveryOutcome(rawValue: String(fields[1]))
    else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidResult(String(line))
    }
    return outcome
  }

  /// Validates every release identity fact that the remote shell intentionally
  /// leaves opaque, including the raw JSON returned by `mapd --build-info`.
  @discardableResult
  public static func validate(
    _ result: TiciMapdReleaseTransactionResult,
    matches release: ValidatedMapdReleaseArtifact,
    rollbackPath: String
  ) throws -> TiciMapdReleaseBuildInfo {
    try validateReleaseIdentity(release)
    try validateRollbackPath(rollbackPath)
    let artifact = release.artifact
    let expectedCache = "/data/media/0/osm/binaries/\(release.persistentCacheFileName)"
    try requireEqual(result.releaseID, artifact.releaseID, field: "release_id")
    try requireEqual(result.sha256, artifact.sha256, field: "sha256")
    try requireEqual(result.cachePath, expectedCache, field: "cache_path")
    try requireEqual(result.rollbackPath, rollbackPath, field: "rollback_path")

    return try validateBuildInfo(result.buildInfoJSON, matches: release)
  }

  @discardableResult
  public static func validateBuildInfo(
    _ data: Data,
    matches release: ValidatedMapdReleaseArtifact
  ) throws -> TiciMapdReleaseBuildInfo {
    let artifact = release.artifact
    let buildInfo: TiciMapdReleaseBuildInfo
    do {
      buildInfo = try JSONDecoder().decode(TiciMapdReleaseBuildInfo.self, from: data)
    } catch {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidBuildInfo(error.localizedDescription)
    }
    try requireEqual(buildInfo.releaseID, artifact.releaseID, field: "build_info.releaseID")
    try requireEqual(buildInfo.buildID, artifact.buildID, field: "build_info.buildID")
    try requireEqual(buildInfo.estimatorVersion, artifact.estimatorVersion, field: "build_info.estimatorVersion")
    guard buildInfo.capabilities.contains(artifact.capability) else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidBuildInfo("missing capability \(artifact.capability)")
    }
    for marker in ["MapdReleaseID:\(artifact.releaseID)", "MapdBuildID:\(artifact.buildID)"] {
      guard buildInfo.identityMarkers.contains(marker) else {
        throw TiciMapdReleaseTransactionCommandBuilderError.invalidBuildInfo("missing identity marker \(marker)")
      }
    }
    return buildInfo
  }

  private static func validateInputs(
    release: ValidatedMapdReleaseArtifact,
    stagedPath: String,
    rollbackPath: String
  ) throws {
    try validateReleaseIdentity(release)
    try validateRollbackPath(rollbackPath)
    try validateStagedPath(stagedPath)
  }

  private static func validateStagedPath(_ stagedPath: String) throws {
    guard stagedPath.range(of: #"^/data/media/0/osm/binaries/\.mapd-release-[A-Za-z0-9._-]{1,128}\.partial$"#, options: .regularExpression) != nil else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidStagedPath(stagedPath)
    }
  }

  private static func validateReleaseIdentity(_ release: ValidatedMapdReleaseArtifact) throws {
    let artifact = release.artifact
    guard isSafeIdentifier(artifact.releaseID),
          isSafeIdentifier(artifact.buildID),
          isSafeMarker(artifact.estimatorVersion),
          isSafeMarker(artifact.capability),
          artifact.sha256.range(of: #"^[0-9a-f]{64}$"#, options: .regularExpression) != nil,
          release.byteCount > 0
    else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidReleaseIdentity(artifact.releaseID)
    }

    let expectedCacheName = [
      "mapd",
      String(MapdReleaseArtifact.sha256Hex(Data(artifact.releaseID.utf8)).prefix(16)),
      String(artifact.sha256.prefix(16)),
    ].joined(separator: "-")
    guard release.persistentCacheFileName == expectedCacheName else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidPersistentCacheFileName(release.persistentCacheFileName)
    }
  }

  private static func validateRollbackPath(_ rollbackPath: String) throws {
    guard rollbackPath.range(of: #"^/data/media/0/osm/binaries/mapd-rollback-[A-Za-z0-9._-]{1,128}$"#, options: .regularExpression) != nil else {
      throw TiciMapdReleaseTransactionCommandBuilderError.invalidRollbackPath(rollbackPath)
    }
  }

  private static func requireEqual(_ actual: String, _ expected: String, field: String) throws {
    guard actual == expected else {
      throw TiciMapdReleaseTransactionCommandBuilderError.identityMismatch(
        field: field,
        expected: expected,
        actual: actual
      )
    }
  }

  private static func isSafeIdentifier(_ value: String) -> Bool {
    !value.isEmpty && value.range(of: #"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$"#, options: .regularExpression) != nil
  }

  private static func isSafeMarker(_ value: String) -> Bool {
    !value.isEmpty && value.range(
      of: #"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$"#,
      options: .regularExpression
    ) != nil
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
