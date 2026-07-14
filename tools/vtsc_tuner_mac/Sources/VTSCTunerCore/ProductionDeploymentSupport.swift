import Foundation

struct GitDeploymentPreflight: Equatable, Sendable {
  var branch: String
  var localHead: String
  var originHead: String
  var upstream: String
}

struct TiciDeploymentSnapshot: Codable, Equatable, Sendable {
  var isOffroad: Bool
  var isOnroad: Bool
  var mapLookaheadEnabled: Bool
  var branch: String
  var head: String
  var dirty: Bool
  var physicsParams: [String: String?]
  var qCurveSHA256: String
  var mapdReleaseVersion: String?
  var mapdVersion: String?
  var activeMapdSHA256: String
  var cachedMapdPath: String
  var cachedMapdSHA256: String?
  var activeTileSetID: String?

  enum CodingKeys: String, CodingKey {
    case isOffroad = "is_offroad"
    case isOnroad = "is_onroad"
    case mapLookaheadEnabled = "map_lookahead_enabled"
    case branch, head, dirty
    case physicsParams = "physics_params"
    case qCurveSHA256 = "q_curve_sha256"
    case mapdReleaseVersion = "mapd_release_version"
    case mapdVersion = "mapd_version"
    case activeMapdSHA256 = "active_mapd_sha256"
    case cachedMapdPath = "cached_mapd_path"
    case cachedMapdSHA256 = "cached_mapd_sha256"
    case activeTileSetID = "active_tile_set_id"
  }
}

struct TiciDeploymentPostflight: Codable, Equatable, Sendable {
  var isOffroad: Bool
  var isOnroad: Bool
  var mapLookaheadEnabled: Bool
  var head: String
  var physicsMatches: Bool
  var qCurveSHA256: String
  var qCurveEnabled: Bool
  var qCurvePointCount: Int
  var mapdReleaseVersion: String
  var mapdVersion: String
  var activeMapdSHA256: String
  var cachedMapdSHA256: String
  var mapdRunning: Bool
  var capabilityPresent: Bool
  var activeELFARM64: Bool
  var buildInfoMatches: Bool
  var profileEstimatorVersion: String
  var profileRouteFingerprint: String
  var profileFresh: Bool
  var profileValuesFinite: Bool
  var gpsStatus: String
  var profileValidationStatus: String
  var profilePointCount: Int
  var profileEventCount: Int
  var activeTileSetID: String?

  enum CodingKeys: String, CodingKey {
    case isOffroad = "is_offroad"
    case isOnroad = "is_onroad"
    case mapLookaheadEnabled = "map_lookahead_enabled"
    case head
    case physicsMatches = "physics_matches"
    case qCurveSHA256 = "q_curve_sha256"
    case qCurveEnabled = "q_curve_enabled"
    case qCurvePointCount = "q_curve_point_count"
    case mapdReleaseVersion = "mapd_release_version"
    case mapdVersion = "mapd_version"
    case activeMapdSHA256 = "active_mapd_sha256"
    case cachedMapdSHA256 = "cached_mapd_sha256"
    case mapdRunning = "mapd_running"
    case capabilityPresent = "capability_present"
    case activeELFARM64 = "active_elf_arm64"
    case buildInfoMatches = "build_info_matches"
    case profileEstimatorVersion = "profile_estimator_version"
    case profileRouteFingerprint = "profile_route_fingerprint"
    case profileFresh = "profile_fresh"
    case profileValuesFinite = "profile_values_finite"
    case gpsStatus = "gps_status"
    case profileValidationStatus = "profile_validation_status"
    case profilePointCount = "profile_point_count"
    case profileEventCount = "profile_event_count"
    case activeTileSetID = "active_tile_set_id"
  }
}

struct RuntimeDeploymentPreflight: Sendable {
  var git: GitDeploymentPreflight
  var profile: String
  var release: ValidatedMapdReleaseArtifact
  var tileSet: CanonicalTileSetArtifact?
  var journal: DeploymentRollbackJournal
  var journalURL: URL
}

enum ProductionVerificationSuite {
  static func requests(repositoryRoot: URL) -> [ProcessRequest] {
    let app = repositoryRoot.appendingPathComponent("tools/vtsc_tuner_mac", isDirectory: true)
    let mapd = repositoryRoot.appendingPathComponent("mapd_repo/openpilot-mapd", isDirectory: true)
    let python = repositoryRoot.appendingPathComponent(".venv/bin/python", isDirectory: false)
    let pinnedGo = app.appendingPathComponent(".build-tools/go-1.26.5-darwin-arm64/bin/go", isDirectory: false)
    let goEnvironment = [
      "GOTOOLCHAIN": "local",
      "CGO_ENABLED": "0",
      "GOMODCACHE": app.appendingPathComponent(".build-tools/go-mod-cache", isDirectory: true).path,
      "GOCACHE": app.appendingPathComponent(".build-tools/go-build-cache", isDirectory: true).path,
    ]
    return [
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/bin/bash"),
        arguments: ["scripts/build_tile_decoder.sh", "--test"],
        currentDirectoryURL: app,
        timeout: 600
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/usr/bin/swift"),
        arguments: ["test"],
        currentDirectoryURL: app,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/bin/bash"),
        arguments: ["scripts/build_app.sh"],
        currentDirectoryURL: app,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: URL(fileURLWithPath: "/usr/bin/codesign"),
        arguments: ["--verify", "--deep", "--strict", "--verbose=2", "dist/VTSC Tuner.app"],
        currentDirectoryURL: app,
        timeout: 120
      ),
      ProcessRequest(
        executableURL: pinnedGo,
        arguments: ["test", "./..."],
        currentDirectoryURL: mapd,
        environment: goEnvironment,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: python,
        arguments: ["-m", "pytest", "-q", "sunnypilot/mapd/tests"],
        currentDirectoryURL: repositoryRoot,
        timeout: 1_200
      ),
      ProcessRequest(
        executableURL: python,
        arguments: [
          "-m", "pytest", "--noconftest", "-o", "addopts=",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_scenarios.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_whole_curve_profile.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_response_model.py",
          "sunnypilot/selfdrive/controls/lib/tests/vtsc/test_longitudinal_planner_vtsc_map_timing.py",
        ],
        currentDirectoryURL: repositoryRoot,
        timeout: 1_800
      ),
      ProcessRequest(
        executableURL: python,
        arguments: ["-m", "pytest", "-q", "tools/vtsc/tests/test_apply_physics_params.py"],
        currentDirectoryURL: repositoryRoot,
        timeout: 600
      ),
      ProcessRequest(
        executableURL: ApplyPipeline.gitURL,
        arguments: ["diff", "--check"],
        currentDirectoryURL: repositoryRoot,
        timeout: 60
      ),
    ]
  }
}

enum TiciDeploymentCommandBuilder {
  static let physicsKeys = [
    "VisionTurnSpeedControlPhysicsAmplitude",
    "VisionTurnSpeedControlPhysicsSteepness",
    "VisionTurnSpeedControlPhysicsCenter",
    "VisionTurnSpeedControlPhysicsBaseline",
    "VisionTurnSpeedControlPhysicsMinLatAccel",
    "VisionTurnSpeedControlPhysicsMaxLatAccel",
  ]

  static func preflightInspectionCommand() -> String {
    let keys = pythonStringList(physicsKeys)
    return """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import hashlib, json, pathlib, subprocess
    from openpilot.common.params import Params
    params = Params()
    keys = \(keys)
    def text(key):
      value = params.get(key)
      return value.decode("utf-8", "replace") if isinstance(value, bytes) else (str(value) if value is not None else "")
    def digest(path):
      path = pathlib.Path(path)
      if not path.is_file(): return ""
      h = hashlib.sha256()
      with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
      return h.hexdigest()
    def q_curve_digest(path):
      lines = path.read_text().splitlines()
      enabled = next(line for line in lines if line.startswith("Q_CURVE_ENABLED"))
      start = next(i for i, line in enumerate(lines) if line.startswith("Q_CURVE_POINTS"))
      selected = [enabled, lines[start]]
      if not lines[start].rstrip().endswith("]"):
        for line in lines[start + 1:]:
          selected.append(line)
          if line.rstrip().endswith("]"): break
      return hashlib.sha256(("\\n".join(selected) + "\\n").encode()).hexdigest()
    release = text("MapdReleaseVersion") or text("MapdVersion")
    cache = ""
    if release:
      prefix = "mapd-" + hashlib.sha256(release.encode()).hexdigest()[:16] + "-"
      matches = sorted(pathlib.Path("/data/media/0/osm/binaries").glob(prefix + "*"))
      cache = str(matches[-1]) if matches else ""
    tile_id = None
    manifest_path = pathlib.Path("/data/media/0/osm/offline/.tileset-manifest.json")
    if manifest_path.is_file():
      try: tile_id = json.loads(manifest_path.read_text()).get("tile_set_id")
      except Exception: tile_id = None
    state = {
      "is_offroad": params.get_bool("IsOffroad"),
      "is_onroad": params.get_bool("IsOnroad"),
      "map_lookahead_enabled": params.get_bool("MTSCLookaheadEnabled"),
      "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
      "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
      "physics_params": {key: (text(key) or None) for key in keys},
      "q_curve_sha256": q_curve_digest(pathlib.Path("sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py")),
      "mapd_release_version": text("MapdReleaseVersion") or None,
      "mapd_version": text("MapdVersion") or None,
      "active_mapd_sha256": digest("/data/openpilot/third_party/mapd/mapd"),
      "cached_mapd_path": cache,
      "cached_mapd_sha256": digest(cache) if cache else None,
      "active_tile_set_id": tile_id,
    }
    print(json.dumps(state, sort_keys=True))
    PY
    """
  }

  static func exactFastForwardCommand(branch: String, head: String) -> String {
    """
    cd /data/openpilot && \
    test "$(git branch --show-current)" = \(shellQuote(branch)) && \
    test -z "$(git status --porcelain)" && \
    git fetch --no-tags origin refs/heads/\(branch) && \
    test "$(git rev-parse FETCH_HEAD)" = \(shellQuote(head)) && \
    git merge --ff-only \(shellQuote(head)) && \
    test "$(git rev-parse HEAD)" = \(shellQuote(head))
    """
  }

  static func installReleaseCommand(
    release: ValidatedMapdReleaseArtifact,
    stagedPath: String,
    rollbackPath: String
  ) -> String {
    let artifact = release.artifact
    let cachePath = "/data/media/0/osm/binaries/\(release.persistentCacheFileName)"
    let markers = pythonStringList([
      "MapdReleaseID:\(artifact.releaseID)", "MapdBuildID:\(artifact.buildID)", artifact.capability,
    ])
    return """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import hashlib, json, os, pathlib, shutil, struct, subprocess
    from openpilot.common.params import Params
    staged = pathlib.Path(\(pythonLiteral(stagedPath)))
    cache = pathlib.Path(\(pythonLiteral(cachePath)))
    active = pathlib.Path("/data/openpilot/third_party/mapd/mapd")
    rollback = pathlib.Path(\(pythonLiteral(rollbackPath)))
    expected = \(pythonLiteral(artifact.sha256))
    markers = \(markers)
    def digest(path):
      h = hashlib.sha256()
      with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
      return h.hexdigest()
    data = staged.read_bytes()
    if len(data) < 20 or data[:4] != b"\\x7fELF" or data[4] != 2 or data[5] != 1 or struct.unpack_from("<H", data, 18)[0] != 183:
      raise SystemExit("staged mapd is not ELF64 little-endian AArch64")
    if hashlib.sha256(data).hexdigest() != expected:
      raise SystemExit("staged mapd digest mismatch")
    for marker in markers:
      if marker.encode() not in data: raise SystemExit(f"missing mapd marker: {marker}")
    os.chmod(staged, 0o755)
    build_info = json.loads(subprocess.check_output([str(staged), "--build-info"], text=True, timeout=15))
    if build_info.get("releaseID") != \(pythonLiteral(artifact.releaseID)):
      raise SystemExit("mapd --build-info releaseID mismatch")
    if build_info.get("buildID") != \(pythonLiteral(artifact.buildID)):
      raise SystemExit("mapd --build-info buildID mismatch")
    if build_info.get("estimatorVersion") != \(pythonLiteral(artifact.estimatorVersion)):
      raise SystemExit("mapd --build-info estimatorVersion mismatch")
    if \(pythonLiteral(artifact.capability)) not in build_info.get("capabilities", []):
      raise SystemExit("mapd --build-info capability mismatch")
    if not set(markers[:2]).issubset(set(build_info.get("identityMarkers", []))):
      raise SystemExit("mapd --build-info identity markers mismatch")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(staged, cache)
    if digest(cache) != expected: raise SystemExit("persistent cache digest mismatch")
    rollback.parent.mkdir(parents=True, exist_ok=True)
    if active.is_file():
      shutil.copy2(active, rollback)
      os.chmod(rollback, 0o755)
    temporary = active.with_name(".mapd-deploy.tmp")
    shutil.copy2(cache, temporary)
    os.chmod(temporary, 0o755)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(temporary, active)
    if digest(active) != expected: raise SystemExit("active mapd digest mismatch")
    params = Params()
    params.put("MapdReleaseVersion", \(pythonLiteral(artifact.releaseID)))
    params.put("MapdVersion", \(pythonLiteral(artifact.releaseID)))
    print(json.dumps({"release_id": \(pythonLiteral(artifact.releaseID)), "sha256": expected, "cache": str(cache)}, sort_keys=True))
    PY
    """
  }

  static func qCurveVerificationCommand(identity: TuneDeploymentIdentity) -> String {
    """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import hashlib, json, pathlib
    path = pathlib.Path("sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py")
    lines = path.read_text().splitlines()
    enabled_line = next(line for line in lines if line.startswith("Q_CURVE_ENABLED"))
    start = next(i for i, line in enumerate(lines) if line.startswith("Q_CURVE_POINTS"))
    selected = [enabled_line, lines[start]]
    if not lines[start].rstrip().endswith("]"):
      for line in lines[start + 1:]:
        selected.append(line)
        if line.rstrip().endswith("]"): break
    digest = hashlib.sha256(("\\n".join(selected) + "\\n").encode()).hexdigest()
    enabled = enabled_line.rsplit("=", 1)[1].strip() == "True"
    count = sum(1 for line in selected[2:-1] if line.lstrip().startswith("("))
    if digest != \(pythonLiteral(identity.qCurveSHA256)) or enabled != \(identity.qCurveEnabled ? "True" : "False") or count != \(identity.qCurvePointCount):
      raise SystemExit("Q-curve identity/readback mismatch")
    print(json.dumps({"q_curve_sha256": digest, "enabled": enabled, "point_count": count}, sort_keys=True))
    PY
    """
  }

  static func postflightInspectionCommand(
    head: String,
    release: ValidatedMapdReleaseArtifact,
    identity: TuneDeploymentIdentity,
    expectedTileSetID: String?
  ) -> String {
    let artifact = release.artifact
    let cachePath = "/data/media/0/osm/binaries/\(release.persistentCacheFileName)"
    let physics = Dictionary(uniqueKeysWithValues: identity.physics.map { ($0.paramKey, $0.value) })
    let physicsJSON = String(data: try! JSONEncoder().encode(physics), encoding: .utf8)!
    return """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import hashlib, json, math, pathlib, struct, subprocess
    from openpilot.common.params import Params
    from openpilot.sunnypilot.selfdrive.controls.lib.vision_turn_controller import VisionTurnController
    params = Params()
    expected_physics = json.loads(\(pythonLiteral(physicsJSON)))
    def text(key, source=params):
      value = source.get(key)
      return value.decode("utf-8", "replace") if isinstance(value, bytes) else (str(value) if value is not None else "")
    def digest(path):
      path = pathlib.Path(path)
      if not path.is_file(): return ""
      h = hashlib.sha256()
      with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
      return h.hexdigest()
    lines = pathlib.Path("sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py").read_text().splitlines()
    enabled_line = next(line for line in lines if line.startswith("Q_CURVE_ENABLED"))
    start = next(i for i, line in enumerate(lines) if line.startswith("Q_CURVE_POINTS"))
    selected = [enabled_line, lines[start]]
    if not lines[start].rstrip().endswith("]"):
      for line in lines[start + 1:]:
        selected.append(line)
        if line.rstrip().endswith("]"): break
    q_digest = hashlib.sha256(("\\n".join(selected) + "\\n").encode()).hexdigest()
    q_enabled = enabled_line.rsplit("=", 1)[1].strip() == "True"
    q_count = sum(1 for line in selected[2:-1] if line.lstrip().startswith("("))
    memory = Params("/dev/shm/params")
    raw_profile = memory.get("MapWholeCurveProfile") or params.get("MapWholeCurveProfile")
    raw_gps = memory.get("LastGPSPosition") or params.get("LastGPSPosition")
    profile = {}
    profile_valid = False
    profile_validation_status = "profile_pending"
    gps_status = "pending"
    gps_pose = None
    def haversine_m(lat1, lon1, lat2, lon2):
      radius_m = 6371000.0
      phi1, phi2 = math.radians(lat1), math.radians(lat2)
      dphi = math.radians(lat2 - lat1)
      dlambda = math.radians(lon2 - lon1)
      a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
      return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))
    try:
      if not raw_gps:
        raise LookupError("gps_missing")
      gps_text = raw_gps.decode("utf-8") if isinstance(raw_gps, bytes) else str(raw_gps or "")
      gps = json.loads(gps_text)
      latitude = float(gps["latitude"])
      longitude = float(gps["longitude"])
      bearing = float(gps.get("bearing", gps.get("bearingDeg")))
      if not (math.isfinite(latitude) and math.isfinite(longitude) and math.isfinite(bearing)):
        raise ValueError("gps_non_finite")
      if not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and 0 <= bearing < 360):
        raise ValueError("gps_out_of_range")
      if latitude == 0.0 and longitude == 0.0:
        raise LookupError("gps_pending")
      gps_pose = (latitude, longitude, bearing)
      gps_status = "valid"
    except LookupError:
      gps_status = "pending"
    except Exception as exc:
      gps_status = "invalid:" + str(exc)
    try:
      if not raw_profile:
        raise LookupError("profile_missing")
      profile_text = raw_profile.decode("utf-8") if isinstance(raw_profile, bytes) else str(raw_profile or "")
      profile = json.loads(profile_text)
      if gps_pose is None:
        profile_validation_status = "pending_real_gps"
      else:
        VisionTurnController._parse_map_whole_curve_profile(profile_text, gps_pose)
        points = profile.get("points", [])
        distances = [
          haversine_m(gps_pose[0], gps_pose[1], float(point["latitude"]), float(point["longitude"]))
          for point in points
        ]
        nearest_index = min(range(len(distances)), key=distances.__getitem__)
        nearest_distance = float(distances[nearest_index])
        if not math.isfinite(nearest_distance) or nearest_distance > 75.0:
          raise ValueError("route_not_near_real_gps")
        if nearest_index > len(points) - 3:
          raise ValueError("insufficient_forward_context")
        profile_valid = True
        profile_validation_status = "valid"
    except LookupError:
      profile_validation_status = "profile_pending"
    except Exception as exc:
      profile_validation_status = "rejected:" + str(exc)
    active_manifest = pathlib.Path("/data/media/0/osm/offline/.tileset-manifest.json")
    try: tile_id = json.loads(active_manifest.read_text()).get("tile_set_id") if active_manifest.is_file() else None
    except Exception: tile_id = None
    binary = pathlib.Path("/data/openpilot/third_party/mapd/mapd")
    binary_data = binary.read_bytes() if binary.is_file() else b""
    try:
      build_info = json.loads(subprocess.check_output([str(binary), "--build-info"], text=True, timeout=15))
    except Exception:
      build_info = {}
    elf_arm64 = (len(binary_data) >= 20 and binary_data[:4] == b"\\x7fELF" and
                 binary_data[4] == 2 and binary_data[5] == 1 and struct.unpack_from("<H", binary_data, 18)[0] == 183)
    build_info_matches = (
      build_info.get("releaseID") == \(pythonLiteral(artifact.releaseID)) and
      build_info.get("buildID") == \(pythonLiteral(artifact.buildID)) and
      build_info.get("estimatorVersion") == \(pythonLiteral(artifact.estimatorVersion)) and
      \(pythonLiteral(artifact.capability)) in build_info.get("capabilities", [])
    )
    result = {
      "is_offroad": params.get_bool("IsOffroad"),
      "is_onroad": params.get_bool("IsOnroad"),
      "map_lookahead_enabled": params.get_bool("MTSCLookaheadEnabled"),
      "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "physics_matches": all(text(key) == value for key, value in expected_physics.items()),
      "q_curve_sha256": q_digest,
      "q_curve_enabled": q_enabled,
      "q_curve_point_count": q_count,
      "mapd_release_version": text("MapdReleaseVersion"),
      "mapd_version": text("MapdVersion"),
      "active_mapd_sha256": digest(binary),
      "cached_mapd_sha256": digest(\(pythonLiteral(cachePath))),
      "mapd_running": subprocess.run(["pgrep", "-x", "mapd"], stdout=subprocess.DEVNULL).returncode == 0,
      "capability_present": \(pythonLiteral(artifact.capability)).encode() in binary_data,
      "active_elf_arm64": elf_arm64,
      "build_info_matches": build_info_matches,
      "profile_estimator_version": str(profile.get("estimatorVersion", "")),
      "profile_route_fingerprint": str(profile.get("routeFingerprint", "")),
      "profile_fresh": profile_valid,
      "profile_values_finite": profile_valid,
      "gps_status": gps_status,
      "profile_validation_status": profile_validation_status,
      "profile_point_count": len(profile.get("points", [])),
      "profile_event_count": len(profile.get("events", [])),
      "active_tile_set_id": tile_id,
    }
    print(json.dumps(result, sort_keys=True))
    PY
    """
  }

  static func rollbackCommand(journal: DeploymentRollbackJournal, rollbackMapdPath: String) -> String {
    let paramsJSON = String(data: try! JSONEncoder().encode(journal.previousPhysicsParams), encoding: .utf8)!
    return """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import json, os, pathlib, shutil, subprocess
    from openpilot.common.params import Params
    previous_head = \(pythonLiteral(journal.previousHead))
    subprocess.run(["git", "reset", "--hard", previous_head], check=True)
    params = Params()
    previous = json.loads(\(pythonLiteral(paramsJSON)))
    for key, value in previous.items():
      params.remove(key) if value is None else params.put(key, value)
    previous_release = \(journal.previousMapdReleaseVersion.map(pythonLiteral) ?? "None")
    previous_version = \(journal.previousMapdVersion.map(pythonLiteral) ?? "None")
    params.remove("MapdReleaseVersion") if previous_release is None else params.put("MapdReleaseVersion", previous_release)
    params.remove("MapdVersion") if previous_version is None else params.put("MapdVersion", previous_version)
    rollback = pathlib.Path(\(pythonLiteral(rollbackMapdPath)))
    active = pathlib.Path("/data/openpilot/third_party/mapd/mapd")
    if rollback.is_file():
      temporary = active.with_name(".mapd-rollback.tmp")
      shutil.copy2(rollback, temporary)
      os.chmod(temporary, 0o755)
      os.replace(temporary, active)
    print(json.dumps({"rolled_back_head": previous_head}, sort_keys=True))
    PY
    """
  }

  static func rollbackVerificationCommand(
    journal: DeploymentRollbackJournal,
    tilesWereTouched: Bool
  ) -> String {
    let paramsJSON = String(data: try! JSONEncoder().encode(journal.previousPhysicsParams), encoding: .utf8)!
    let expectedTileID = journal.previousTileSetID.map(pythonLiteral) ?? "None"
    let targetTileID = journal.targetTileSetID.map(pythonLiteral) ?? "None"
    return """
    cd /data/openpilot && PYTHONPATH=/data/openpilot python3 - <<'PY'
    import hashlib, json, pathlib, subprocess
    from openpilot.common.params import Params
    params = Params()
    expected_physics = json.loads(\(pythonLiteral(paramsJSON)))
    expected_head = \(pythonLiteral(journal.previousHead))
    expected_q = \(pythonLiteral(journal.previousQCurveSHA256))
    expected_release = \(journal.previousMapdReleaseVersion.map(pythonLiteral) ?? "None")
    expected_version = \(journal.previousMapdVersion.map(pythonLiteral) ?? "None")
    expected_active_sha = \(pythonLiteral(journal.previousActiveMapdSHA256))
    expected_cache_path = \(pythonLiteral(journal.previousCachedMapdPath))
    expected_cache_sha = \(pythonLiteral(journal.previousCachedMapdSHA256 ?? ""))
    expected_tile_id = \(expectedTileID)
    target_tile_id = \(targetTileID)
    tiles_were_touched = \(tilesWereTouched ? "True" : "False")
    def text(key):
      value = params.get(key)
      return value.decode("utf-8", "replace") if isinstance(value, bytes) else (str(value) if value is not None else "")
    def digest(path):
      path = pathlib.Path(path)
      if not path.is_file(): return ""
      value = hashlib.sha256()
      with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
      return value.hexdigest()
    def q_curve_digest(path):
      lines = path.read_text().splitlines()
      enabled = next(line for line in lines if line.startswith("Q_CURVE_ENABLED"))
      start = next(i for i, line in enumerate(lines) if line.startswith("Q_CURVE_POINTS"))
      selected = [enabled, lines[start]]
      if not lines[start].rstrip().endswith("]"):
        for line in lines[start + 1:]:
          selected.append(line)
          if line.rstrip().endswith("]"): break
      return hashlib.sha256(("\\n".join(selected) + "\\n").encode()).hexdigest()
    manifest = pathlib.Path("/data/media/0/osm/offline/.tileset-manifest.json")
    try: tile_id = json.loads(manifest.read_text()).get("tile_set_id") if manifest.is_file() else None
    except Exception: tile_id = None
    actual = {
      "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
      "physics": {key: (text(key) or None) for key in expected_physics},
      "q": q_curve_digest(pathlib.Path("sunnypilot/selfdrive/controls/lib/vtsc_curve_tuning.py")),
      "release": text("MapdReleaseVersion") or None,
      "version": text("MapdVersion") or None,
      "active_sha": digest("/data/openpilot/third_party/mapd/mapd"),
      "cache_sha": digest(expected_cache_path) if expected_cache_path else "",
      "tile_id": tile_id,
      "mapd_running": subprocess.run(
        ["pgrep", "-x", "mapd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
      ).returncode == 0,
      "is_offroad": params.get_bool("IsOffroad"),
      "is_onroad": params.get_bool("IsOnroad"),
      "lookahead": params.get_bool("MTSCLookaheadEnabled"),
    }
    errors = []
    if actual["head"] != expected_head: errors.append("head")
    if actual["dirty"]: errors.append("dirty")
    if actual["physics"] != expected_physics: errors.append("physics")
    if actual["q"] != expected_q: errors.append("q_curve")
    if actual["release"] != expected_release or actual["version"] != expected_version: errors.append("mapd_version")
    if actual["active_sha"] != expected_active_sha: errors.append("active_mapd")
    if expected_cache_sha and actual["cache_sha"] != expected_cache_sha: errors.append("cached_mapd")
    if not actual["mapd_running"]: errors.append("mapd_process")
    if not actual["is_offroad"] or actual["is_onroad"] or actual["lookahead"]: errors.append("parked_state")
    if expected_tile_id is not None and actual["tile_id"] != expected_tile_id: errors.append("tile_id")
    if tiles_were_touched and target_tile_id is not None and actual["tile_id"] == target_tile_id:
      errors.append("target_tile_still_active")
    if tiles_were_touched and not pathlib.Path("/data/media/0/osm/offline").is_dir(): errors.append("tile_tree")
    if errors: raise SystemExit(json.dumps({"rollback_verification_failed": errors, "actual": actual}, sort_keys=True))
    print(json.dumps({"rollback_verified": True, "head": actual["head"], "tile_id": actual["tile_id"]}, sort_keys=True))
    PY
    """
  }

  static func decodeLastJSONLine<T: Decodable>(_ type: T.Type, output: String) throws -> T {
    let lines = output.split(whereSeparator: \.isNewline)
    guard let line = lines.last else { throw ApplyPipelineError.invalidDeploymentOutput(output) }
    do { return try JSONDecoder().decode(T.self, from: Data(line.utf8)) }
    catch { throw ApplyPipelineError.invalidDeploymentOutput("\(output)\n\(error.localizedDescription)") }
  }

  private static func pythonStringList(_ values: [String]) -> String {
    "[" + values.map(pythonLiteral).joined(separator: ", ") + "]"
  }

  private static func pythonLiteral(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'") + "'"
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}
