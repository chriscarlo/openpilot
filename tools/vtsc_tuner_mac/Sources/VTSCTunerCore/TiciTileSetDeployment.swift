import Foundation

public struct TiciStagedTileSet: Equatable, Sendable {
  public var profile: String
  public var tileSetID: String
  public var remoteStagingRoot: String

  public init(profile: String, tileSetID: String, remoteStagingRoot: String) {
    self.profile = profile
    self.tileSetID = tileSetID
    self.remoteStagingRoot = remoteStagingRoot
  }
}

public struct TiciTileActivationResult: Equatable, Sendable {
  public var tileSetID: String
  public var commandOutput: String

  public init(tileSetID: String, commandOutput: String) {
    self.tileSetID = tileSetID
    self.commandOutput = commandOutput
  }
}

enum TileTransactionRecoveryDecision: Equatable, Sendable {
  case restartBeforeSwitch
  case finishAfterSwitch
  case alreadyRolledBack
  case performRollback
  case manualRecoveryRequired
}

enum TileTransactionRecovery {
  static func activation(
    pendingID: String,
    expectedID: String,
    activeTarget: String,
    newTarget: String,
    previousTarget: String
  ) -> TileTransactionRecoveryDecision {
    guard pendingID == expectedID else { return .manualRecoveryRequired }
    if activeTarget == newTarget { return .finishAfterSwitch }
    if activeTarget.isEmpty || activeTarget == previousTarget { return .restartBeforeSwitch }
    return .manualRecoveryRequired
  }

  static func rollback(
    activeTarget: String,
    previousTarget: String,
    recordedActiveTarget: String,
    recordedPreviousTarget: String
  ) -> TileTransactionRecoveryDecision {
    if activeTarget == recordedPreviousTarget, previousTarget == recordedActiveTarget {
      return .alreadyRolledBack
    }
    if activeTarget == recordedActiveTarget, previousTarget == recordedPreviousTarget {
      return .performRollback
    }
    return .manualRecoveryRequired
  }
}

/// Production deployment for an already-built canonical tile set. Transfers
/// are confined to a named staging tree. The live `offline` directory is only
/// changed by one rename-exchange activation command.
public struct TiciTileSetDeploymentService: Sendable {
  public static let sshURL = URL(fileURLWithPath: "/usr/bin/ssh")
  public static let rsyncURL = URL(fileURLWithPath: "/usr/bin/rsync")
  public static let remoteRoot = "/data/media/0/osm"
  public static let activeOfflinePath = "/data/media/0/osm/offline"
  public static let previousOfflinePath = "/data/media/0/osm/offline.previous"
  public static let generationRoot = "/data/media/0/osm/tile-generations"
  public static let embeddedManifestName = ".tileset-manifest.json"
  public static let transactionPath = "/data/media/0/osm/.tileset-transaction.json"

  private let processRunner: any ProcessRunning

  public init(processRunner: any ProcessRunning = SystemProcessRunner()) {
    self.processRunner = processRunner
  }

  public func stageAndVerify(
    artifact: CanonicalTileSetArtifact,
    profile: String
  ) async throws -> TiciStagedTileSet {
    try Self.validateProfile(profile)
    let manifest = try artifact.manifest.validatedMetadata()
    let stagingRoot = Self.stagingRoot(tileSetID: manifest.tileSetID)
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          "rm -rf \(stagingRoot) && mkdir -p \(stagingRoot)/offline",
        ],
        timeout: 30
      ),
      context: "prepare remote tile staging"
    )

    let tileRsync = ProcessRequest(
      executableURL: Self.rsyncURL,
      arguments: [
        "-a", "--delete", "--partial-dir=.rsync-partial",
        "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
        artifact.offlineURL.path + "/",
        "\(profile):\(stagingRoot)/offline/",
      ],
      timeout: 3_600
    )
    guard !tileRsync.arguments.contains(where: { $0 == "\(profile):\(Self.activeOfflinePath)/" }) else {
      throw TiciTileSetDeploymentError.activePathTransferForbidden
    }
    _ = try await checked(tileRsync, context: "stage canonical tile files")
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.rsyncURL,
        arguments: [
          "-a",
          "-e", "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new",
          artifact.manifestURL.path,
          "\(profile):\(stagingRoot)/manifest.json",
        ],
        timeout: 120
      ),
      context: "stage tile manifest"
    )
    let verified = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          Self.remoteManifestVerificationCommand(stagingRoot: stagingRoot, tileSetID: manifest.tileSetID),
        ],
        timeout: max(300, TimeInterval(manifest.fileCount) * 2)
      ),
      context: "verify staged tile manifest"
    )
    guard verified.standardOutput.contains(manifest.tileSetID) else {
      throw TiciTileSetDeploymentError.verificationIdentityMissing(manifest.tileSetID)
    }
    return TiciStagedTileSet(profile: profile, tileSetID: manifest.tileSetID, remoteStagingRoot: stagingRoot)
  }

  public func activate(_ staged: TiciStagedTileSet) async throws -> TiciTileActivationResult {
    try Self.validateProfile(staged.profile)
    guard staged.remoteStagingRoot == Self.stagingRoot(tileSetID: staged.tileSetID) else {
      throw TiciTileSetDeploymentError.invalidStagingRoot(staged.remoteStagingRoot)
    }
    let result = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          staged.profile,
          Self.atomicActivationCommand(stagingRoot: staged.remoteStagingRoot, tileSetID: staged.tileSetID),
        ],
        timeout: 120
      ),
      context: "atomically activate tile set"
    )
    guard result.standardOutput.contains(staged.tileSetID) else {
      throw TiciTileSetDeploymentError.activationIdentityMissing(staged.tileSetID)
    }
    return TiciTileActivationResult(tileSetID: staged.tileSetID, commandOutput: result.combinedOutput)
  }

  public func rollback(profile: String, expectedActivatedTileSetID: String? = nil) async throws {
    try Self.validateProfile(profile)
    _ = try await checked(
      ProcessRequest(
        executableURL: Self.sshURL,
        arguments: Self.sshOptions(connectTimeout: 10) + [
          profile,
          Self.atomicRollbackCommand(expectedActivatedTileSetID: expectedActivatedTileSetID),
        ],
        timeout: 120
      ),
      context: "roll back active tile set"
    )
  }

  public static func stagingRoot(tileSetID: String) -> String {
    "\(remoteRoot)/.tileset-\(tileSetID).partial"
  }

  static func remoteManifestVerificationCommand(stagingRoot: String, tileSetID: String) -> String {
    """
    \(TiciDeploymentCommandBuilder.ticiPython) - <<'PY'
    import hashlib, json, pathlib
    root = pathlib.Path(\(pythonLiteral(stagingRoot)))
    expected_id = \(pythonLiteral(tileSetID))
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("tile_set_id") != expected_id:
      raise SystemExit("tile-set identity mismatch")
    entries = manifest.get("files", [])
    expected_paths = [entry["path"] for entry in entries]
    if expected_paths != sorted(set(expected_paths)):
      raise SystemExit("manifest paths are not canonical")
    actual_paths = sorted(
      str(path.relative_to(root)) for path in (root / "offline").rglob("*")
      if path.is_file() and path.name != ".tileset-manifest.json"
    )
    if actual_paths != expected_paths:
      raise SystemExit("remote tile file set differs from manifest")
    total = 0
    for entry in entries:
      rel = pathlib.PurePosixPath(entry["path"])
      if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "offline":
        raise SystemExit("unsafe manifest path")
      path = root.joinpath(*rel.parts)
      size = path.stat().st_size
      if size != int(entry["byte_count"]):
        raise SystemExit(f"size mismatch: {rel}")
      digest = hashlib.sha256()
      with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
          digest.update(block)
      if digest.hexdigest() != entry["sha256"]:
        raise SystemExit(f"digest mismatch: {rel}")
      total += size
    if len(entries) != int(manifest.get("file_count", -1)) or total != int(manifest.get("total_bytes", -1)):
      raise SystemExit("manifest aggregate mismatch")
    print(json.dumps({"tile_set_id": expected_id, "file_count": len(entries), "total_bytes": total}, sort_keys=True))
    PY
    """
  }

  static func atomicActivationCommand(
    stagingRoot: String,
    tileSetID: String,
    injectedFailurePoint: String? = nil
  ) -> String {
    """
    \(TiciDeploymentCommandBuilder.ticiPython) - <<'PY'
    import ctypes, hashlib, json, os, pathlib, shutil, stat, time
    root = pathlib.Path(\(pythonLiteral(remoteRoot)))
    stage = pathlib.Path(\(pythonLiteral(stagingRoot)))
    active = pathlib.Path(\(pythonLiteral(activeOfflinePath)))
    previous = pathlib.Path(\(pythonLiteral(previousOfflinePath)))
    generations = pathlib.Path(\(pythonLiteral(generationRoot)))
    transaction = pathlib.Path(\(pythonLiteral(transactionPath)))
    expected_id = \(pythonLiteral(tileSetID))
    injected_failure = \(pythonLiteral(injectedFailurePoint ?? ""))
    generation = generations / expected_id
    new_target = os.path.relpath(generation / "offline", root)
    switch = root / (".offline-switch-" + expected_id)

    def fsync_dir(path):
      descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
      try: os.fsync(descriptor)
      finally: os.close(descriptor)

    def fail(point):
      if injected_failure == point: raise RuntimeError("injected tile activation failure: " + point)

    def write_transaction(payload):
      temporary = transaction.with_suffix(".tmp")
      with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary, transaction)
      fsync_dir(root)

    def clear_transaction():
      if transaction.exists(): transaction.unlink()
      fsync_dir(root)

    def remove_or_retain(path, label):
      if not path.exists() and not path.is_symlink(): return
      if path.is_symlink() or path.is_file():
        path.unlink()
      else:
        retained = root / (".retained-" + label + "-" + str(time.time_ns()))
        os.replace(path, retained)

    def install_previous_link(target):
      temporary = root / (".offline-previous-link-" + expected_id)
      remove_or_retain(temporary, "previous-temp")
      os.symlink(target, temporary)
      fsync_dir(root)
      remove_or_retain(previous, "older-previous")
      os.replace(temporary, previous)
      fsync_dir(root)

    def cleanup_exchange(path):
      if path.is_symlink() or path.is_file():
        path.unlink()
      elif path.exists():
        retained = root / (".retained-pre-generation-" + expected_id)
        if retained.exists(): retained = root / (retained.name + "-" + str(time.time_ns()))
        os.replace(path, retained)
      fsync_dir(root)

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int

    # Recover an interrupted activation before starting or retrying. The
    # single active symlink always resolves to an offline tree containing its
    # own immutable manifest, so every observable state is coherent.
    if transaction.exists():
      pending = json.loads(transaction.read_text())
      if pending.get("kind") != "activation" or pending.get("newID") != expected_id:
        raise SystemExit("another tile transaction requires recovery")
      active_target = os.readlink(active) if active.is_symlink() else ""
      if active_target == pending["newTarget"]:
        install_previous_link(pending["previousTarget"])
        cleanup_exchange(pathlib.Path(pending["switchPath"]))
        clear_transaction()
        print(json.dumps({"activated_tile_set_id": expected_id, "recovered": True}, sort_keys=True))
        raise SystemExit(0)
      if active_target and active_target != pending["previousTarget"]:
        raise SystemExit("active tile pointer diverged during recovery")
      cleanup_exchange(pathlib.Path(pending["switchPath"]))
      clear_transaction()

    generations.mkdir(parents=True, exist_ok=True)
    if not generation.exists():
      staged_manifest = stage / "manifest.json"
      staged_offline = stage / "offline"
      if not staged_offline.is_dir() or json.loads(staged_manifest.read_text()).get("tile_set_id") != expected_id:
        raise SystemExit("staged generation identity changed")
      partial = staged_offline / ".rsync-partial"
      if partial.exists(): shutil.rmtree(partial)
      embedded_manifest = staged_offline / \(pythonLiteral(embeddedManifestName))
      shutil.copy2(staged_manifest, embedded_manifest)
      with embedded_manifest.open("rb") as stream: os.fsync(stream.fileno())
      for path in sorted(staged_offline.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
      os.chmod(staged_offline, 0o555)
      os.chmod(staged_manifest, 0o444)
      os.replace(stage, generation)
      fsync_dir(generations)
    embedded = generation / "offline" / \(pythonLiteral(embeddedManifestName))
    if json.loads(embedded.read_text()).get("tile_set_id") != expected_id:
      raise SystemExit("immutable generation manifest mismatch")
    fail("after_generation")

    if active.is_symlink():
      previous_target = os.readlink(active)
    elif active.is_dir():
      # One-time migration: first copy the legacy tree into a complete
      # immutable generation. The original is retained until after the atomic
      # pointer exchange, so a crash cannot destroy rollback material.
      legacy_seed = str(active.stat().st_mtime_ns) + ":" + str(active.stat().st_size)
      legacy_id = "legacy-" + hashlib.sha256(legacy_seed.encode()).hexdigest()[:16]
      legacy = generations / legacy_id
      if not legacy.exists():
        building = generations / ("." + legacy_id + ".building")
        if building.exists(): shutil.rmtree(building)
        shutil.copytree(active, building / "offline", symlinks=False)
        legacy_manifest = {"tile_set_id": legacy_id, "legacy": True}
        adjacent = root / "offline.manifest.json"
        if adjacent.is_file():
          try: legacy_manifest = json.loads(adjacent.read_text())
          except Exception: pass
        legacy_manifest["tile_set_id"] = legacy_manifest.get("tile_set_id") or legacy_id
        manifest_bytes = (json.dumps(legacy_manifest, sort_keys=True) + "\\n").encode()
        (building / "manifest.json").write_bytes(manifest_bytes)
        (building / "offline" / \(pythonLiteral(embeddedManifestName))).write_bytes(manifest_bytes)
        os.replace(building, legacy)
        fsync_dir(generations)
      previous_target = os.path.relpath(legacy / "offline", root)
    else:
      raise SystemExit("no complete active tile generation is available")

    remove_or_retain(switch, "switch-temp")
    os.symlink(new_target, switch)
    fsync_dir(root)
    payload = {
      "kind": "activation", "newID": expected_id,
      "newTarget": new_target, "previousTarget": previous_target,
      "switchPath": str(switch),
    }
    write_transaction(payload)
    fail("after_journal")
    if renameat2(-100, os.fsencode(switch), -100, os.fsencode(active), 2) != 0:
      error_number = ctypes.get_errno()
      raise OSError(error_number, os.strerror(error_number))
    fsync_dir(root)
    fail("after_switch")
    install_previous_link(previous_target)
    fail("after_previous")
    cleanup_exchange(switch)
    clear_transaction()
    print(json.dumps({"activated_tile_set_id": expected_id}, sort_keys=True))
    PY
    """
  }

  static func atomicRollbackCommand(
    expectedActivatedTileSetID: String? = nil,
    injectedFailurePoint: String? = nil
  ) -> String {
    """
    \(TiciDeploymentCommandBuilder.ticiPython) - <<'PY'
    import ctypes, json, os, pathlib, time
    root = pathlib.Path(\(pythonLiteral(remoteRoot)))
    active = pathlib.Path(\(pythonLiteral(activeOfflinePath)))
    previous = pathlib.Path(\(pythonLiteral(previousOfflinePath)))
    transaction = pathlib.Path(\(pythonLiteral(transactionPath)))
    expected_activated_id = \(expectedActivatedTileSetID.map(pythonLiteral) ?? "None")
    injected_failure = \(pythonLiteral(injectedFailurePoint ?? ""))
    def fsync_dir(path):
      descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
      try: os.fsync(descriptor)
      finally: os.close(descriptor)
    def fail(point):
      if injected_failure == point: raise RuntimeError("injected tile rollback failure: " + point)
    def write_transaction(payload):
      temporary = transaction.with_suffix(".tmp")
      with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush(); os.fsync(stream.fileno())
      os.replace(temporary, transaction); fsync_dir(root)
    def clear_transaction():
      if transaction.exists(): transaction.unlink()
      fsync_dir(root)
    def target(path): return os.readlink(path) if path.is_symlink() else ""
    def manifest_id(path):
      manifest = path / \(pythonLiteral(embeddedManifestName))
      try: return json.loads(manifest.read_text()).get("tile_set_id") if manifest.is_file() else None
      except Exception: return None
    def remove_or_retain(path, label):
      if not path.exists() and not path.is_symlink(): return
      if path.is_symlink() or path.is_file():
        path.unlink()
      else:
        retained = root / (".retained-" + label + "-" + str(time.time_ns()))
        os.replace(path, retained)
      fsync_dir(root)
    def install_previous_link(previous_target):
      temporary = root / ".offline-previous-rollback-recovery"
      remove_or_retain(temporary, "rollback-previous-temp")
      os.symlink(previous_target, temporary)
      fsync_dir(root)
      remove_or_retain(previous, "rollback-older-previous")
      os.replace(temporary, previous)
      fsync_dir(root)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if transaction.exists():
      pending = json.loads(transaction.read_text())
      if pending.get("kind") == "activation":
        active_target = target(active)
        switch_path = pathlib.Path(pending["switchPath"])
        if active_target == pending["newTarget"]:
          install_previous_link(pending["previousTarget"])
          remove_or_retain(switch_path, "activation-exchange")
          clear_transaction()
        elif not active.is_symlink() or active_target == pending["previousTarget"]:
          remove_or_retain(switch_path, "activation-before-switch")
          clear_transaction()
          print(json.dumps({"tile_activation_not_switched": True, "recovered": True}, sort_keys=True))
          raise SystemExit(0)
        else:
          raise SystemExit("activation pointers diverged during rollback recovery")
      elif pending.get("kind") == "rollback":
        if target(active) == pending["previousTarget"] and target(previous) == pending["activeTarget"]:
          clear_transaction()
          print(json.dumps({"rolled_back_tile_set_id": manifest_id(active), "recovered": True}, sort_keys=True))
          raise SystemExit(0)
        if target(active) != pending["activeTarget"] or target(previous) != pending["previousTarget"]:
          raise SystemExit("tile rollback pointers diverged during recovery")
      else:
        raise SystemExit("unknown tile transaction requires manual recovery")
    else:
      if expected_activated_id is not None and manifest_id(active) != expected_activated_id:
        print(json.dumps({"tile_activation_not_observed": True, "active_tile_set_id": manifest_id(active)}, sort_keys=True))
        raise SystemExit(0)
    if not active.is_symlink() or not previous.is_symlink() or not manifest_id(active) or not manifest_id(previous):
      raise SystemExit("rollback requires two complete immutable tile generations")
    if not transaction.exists():
      pending = {"kind": "rollback", "activeTarget": target(active), "previousTarget": target(previous)}
      write_transaction(pending)
    fail("after_journal")
    if renameat2(-100, os.fsencode(active), -100, os.fsencode(previous), 2) != 0:
      error_number = ctypes.get_errno()
      raise OSError(error_number, os.strerror(error_number))
    fsync_dir(root)
    fail("after_switch")
    clear_transaction()
    print(json.dumps({"rolled_back_tile_set_id": manifest_id(active)}, sort_keys=True))
    PY
    """
  }

  private func checked(_ request: ProcessRequest, context: String) async throws -> ProcessResult {
    try Task.checkCancellation()
    let result = try await processRunner.run(request)
    guard result.succeeded else {
      throw TiciTileSetDeploymentError.commandFailed(context, result.terminationStatus, result.combinedOutput)
    }
    return result
  }

  private static func validateProfile(_ profile: String) throws {
    guard profile.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
      throw TiciTileSetDeploymentError.invalidProfile(profile)
    }
  }

  private static func sshOptions(connectTimeout: Int) -> [String] {
    [
      "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=\(connectTimeout)",
      "-o", "StrictHostKeyChecking=accept-new",
    ]
  }

  private static func pythonLiteral(_ value: String) -> String {
    let escaped = value
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "'", with: "\\'")
    return "'\(escaped)'"
  }
}

public enum TiciTileSetDeploymentError: LocalizedError, Equatable, Sendable {
  case invalidProfile(String)
  case invalidStagingRoot(String)
  case activePathTransferForbidden
  case commandFailed(String, Int32, String)
  case verificationIdentityMissing(String)
  case activationIdentityMissing(String)

  public var errorDescription: String? {
    switch self {
    case let .invalidProfile(profile): "Invalid SSH profile: \(profile)"
    case let .invalidStagingRoot(path): "Invalid remote tile staging root: \(path)"
    case .activePathTransferForbidden: "Refusing to transfer files directly into the active tile directory."
    case let .commandFailed(context, status, output): "\(context) exited \(status): \(output)"
    case let .verificationIdentityMissing(identity): "Remote verification did not confirm tile set \(identity)."
    case let .activationIdentityMissing(identity): "Atomic activation did not confirm tile set \(identity)."
    }
  }
}
