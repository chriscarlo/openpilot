#!/usr/bin/env python3

import hashlib
import json
import stat
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.mapd import mapd_installer


class DummyParams:
  def __init__(self, values: dict[str, str | bytes | None]):
    self._values = values

  def get(self, key, *args, **kwargs):
    return self._values.get(key)

  def put(self, key, value):
    self._values[key] = value


class DummyReporter:
  def __init__(self):
    self.messages = []

  def update(self, msg):
    self.messages.append(msg)

  def close(self):
    pass


class FakeSubMaster:
  def __init__(self, services):
    self.device_state = SimpleNamespace(networkMetered=False)

  def update(self, *args, **kwargs):
    pass

  def __getitem__(self, key):
    assert key == "deviceState"
    return self.device_state


class MeteredSubMaster(FakeSubMaster):
  def __init__(self, services):
    super().__init__(services)
    self.device_state.networkMetered = True


class FakeResponse:
  def __init__(self, content: bytes):
    self.content = content

  def raise_for_status(self):
    pass


def make_release_and_binary(*, machine=mapd_installer.EM_AARCH64,
                            release_marker=True, build_marker=True,
                            capability_marker=True):
  release = mapd_installer.MapdReleaseSpec(
    version=mapd_installer.DEFAULT_VERSION,
    release_id="chauffeur-whole-curve-v3",
    build_id="test-build-20260713",
    sha256="f" * 64,
    capability=mapd_installer.MAP_WHOLE_CURVE_CAPABILITY,
    binary_url="https://example.test/chauffeur-whole-curve-v3/mapd",
  )
  header = bytearray(256)
  header[:4] = b"\x7fELF"
  header[4] = mapd_installer.ELFCLASS64
  header[5] = mapd_installer.ELFDATA2LSB
  header[6] = 1
  struct.pack_into("<H", header, 18, machine)
  markers = []
  if release_marker:
    markers.append(f"MapdReleaseID:{release.release_id}".encode())
  if build_marker:
    markers.append(f"MapdBuildID:{release.build_id}".encode())
  if capability_marker:
    markers.append(release.capability.encode())
  content = bytes(header) + b"\0" + b"\0".join(markers)
  content += b"\0" * max(0, 1_000_128 - len(content))
  release = replace(release, sha256=hashlib.sha256(content).hexdigest())
  return release, content


def write_binary(path: Path, content: bytes):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(content)
  path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture(autouse=True)
def trusted_release(monkeypatch):
  monkeypatch.delenv("SP_MAPD_RELEASE_VERSION", raising=False)
  monkeypatch.delenv("SP_MAPD_BINARY_URL", raising=False)
  release, content = make_release_and_binary()
  monkeypatch.setattr(mapd_installer, "DEFAULT_RELEASE", release)
  return release, content


def configure_install_paths(monkeypatch, tmp_path):
  binary_dir = tmp_path / "checkout" / "third_party" / "mapd"
  mapd_root = tmp_path / "persistent-osm"
  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary_dir / "mapd"))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(binary_dir))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  return binary_dir / "mapd", mapd_root


def test_target_version_defaults():
  assert mapd_installer.get_target_version(DummyParams({})) == mapd_installer.DEFAULT_VERSION


def test_default_release_is_bound_to_the_v2_artifact(monkeypatch):
  monkeypatch.undo()
  assert mapd_installer.DEFAULT_RELEASE == mapd_installer.MapdReleaseSpec(
    version="chauffeur-whole-curve-v3",
    release_id="chauffeur-whole-curve-v3",
    build_id="tree-fc148f05d6574ff4c114",
    sha256="6c6911a90722a0defe28263798272b515c9a027ccd23775095d895e2969532f8",
    capability="MapWholeCurveProfile:whole-curve-v3",
    binary_url="https://github.com/chriscarlo/mapd/releases/download/chauffeur-whole-curve-v3/mapd",
  )


def test_unregistered_target_version_is_rejected():
  params = DummyParams({"MapdReleaseVersion": b"v9.9.9"})
  with pytest.raises(ValueError, match="untrusted mapd release"):
    mapd_installer.get_target_release(params)


def test_target_binary_url_override_preserves_immutable_identity(trusted_release):
  release, _ = trusted_release
  params = DummyParams({"MapdBinaryUrl": "https://maps.example.test/releases/{version}/mapd"})

  target = mapd_installer.get_target_release(params)

  assert target.binary_url == f"https://maps.example.test/releases/{release.version}/mapd"
  assert replace(target, binary_url=release.binary_url) == release


def test_unbound_production_release_is_rejected(monkeypatch):
  placeholder = replace(
    mapd_installer.DEFAULT_RELEASE,
    build_id="REPLACE_WITH_FINAL_MAPD_BUILD_ID",
    sha256="0" * 64,
  )
  monkeypatch.setattr(mapd_installer, "DEFAULT_RELEASE", placeholder)

  with pytest.raises(ValueError, match="production artifact"):
    mapd_installer.get_target_release(DummyParams({}))


def test_verify_installed_binary_accepts_exact_linux_arm64_release(tmp_path, trusted_release):
  release, content = trusted_release
  binary = tmp_path / "mapd"
  write_binary(binary, content)

  mapd_installer.MapdInstallManager._verify_installed_binary(str(binary), release)


def test_verify_installed_binary_rejects_wrong_digest(tmp_path, trusted_release):
  release, content = trusted_release
  binary = tmp_path / "mapd"
  write_binary(binary, content)

  with pytest.raises(OSError, match="SHA-256"):
    mapd_installer.MapdInstallManager._verify_installed_binary(
      str(binary), replace(release, sha256="1" * 64)
    )


def test_verify_installed_binary_rejects_wrong_architecture(tmp_path):
  release, content = make_release_and_binary(machine=62)
  binary = tmp_path / "mapd"
  write_binary(binary, content)

  with pytest.raises(OSError, match="expected Linux ARM64"):
    mapd_installer.MapdInstallManager._verify_installed_binary(str(binary), release)


@pytest.mark.parametrize(
  ("missing_marker", "expected"),
  (("release", "MapdReleaseID"), ("build", "MapdBuildID"), ("capability", "MapWholeCurveProfile")),
)
def test_verify_installed_binary_rejects_missing_identity_marker(tmp_path, missing_marker, expected):
  release, content = make_release_and_binary(
    release_marker=missing_marker != "release",
    build_marker=missing_marker != "build",
    capability_marker=missing_marker != "capability",
  )
  binary = tmp_path / "mapd"
  write_binary(binary, content)

  with pytest.raises(OSError, match=expected):
    mapd_installer.MapdInstallManager._verify_installed_binary(str(binary), release)


def test_verify_runtime_build_info_accepts_exact_identity(monkeypatch, trusted_release):
  release, _ = trusted_release
  info = {
    "releaseID": release.release_id,
    "buildID": release.build_id,
    "capabilities": ["legacy", release.capability],
  }
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr=""),
  )

  assert mapd_installer.MapdInstallManager._verify_runtime_build_info("/mapd", release) == info


@pytest.mark.parametrize(
  ("info_update", "expected"),
  (
    ({"releaseID": "wrong"}, "release ID"),
    ({"buildID": "wrong"}, "build ID"),
    ({"capabilities": []}, "required capability"),
  ),
)
def test_verify_runtime_build_info_rejects_wrong_identity(monkeypatch, trusted_release, info_update, expected):
  release, _ = trusted_release
  info = {
    "releaseID": release.release_id,
    "buildID": release.build_id,
    "capabilities": [release.capability],
  }
  info.update(info_update)
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr=""),
  )

  with pytest.raises(OSError, match=expected):
    mapd_installer.MapdInstallManager._verify_runtime_build_info("/mapd", release)


def test_invalid_download_never_replaces_active_binary(monkeypatch, tmp_path, trusted_release):
  release, _ = trusted_release
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  old_content = b"known-active-binary"
  write_binary(binary, old_content)
  monkeypatch.setattr(mapd_installer.requests, "get", lambda *args, **kwargs: FakeResponse(b"not-a-mapd-release"))
  monkeypatch.setattr(mapd_installer.time, "sleep", lambda *_: None)
  manager = mapd_installer.MapdInstallManager(DummyReporter(), DummyParams({}))

  with pytest.raises(RuntimeError, match="failed after 1 retries"):
    manager._download_file(release.binary_url, release, num_retries=1)

  assert binary.read_bytes() == old_content
  assert not Path(f"{binary}.tmp").exists()


def test_terminal_download_failure_does_not_stamp_version(monkeypatch, tmp_path):
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  old_content = b"known-active-binary"
  write_binary(binary, old_content)
  params = DummyParams({"MapdVersion": "known-old-release"})
  monkeypatch.setattr(
    mapd_installer.requests,
    "get",
    lambda *args, **kwargs: (_ for _ in ()).throw(mapd_installer.requests.exceptions.ConnectionError("offline")),
  )
  monkeypatch.setattr(mapd_installer.time, "sleep", lambda *_: None)

  with pytest.raises(RuntimeError, match="failed after 5 retries"):
    mapd_installer.MapdInstallManager(DummyReporter(), params).download()

  assert params.get("MapdVersion") == "known-old-release"
  assert binary.read_bytes() == old_content


def test_successful_download_atomically_installs_caches_and_stamps(monkeypatch, tmp_path, trusted_release):
  release, content = trusted_release
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  write_binary(binary, b"old-active")
  params = DummyParams({"MapdVersion": "old-release"})
  monkeypatch.setattr(mapd_installer.requests, "get", lambda *args, **kwargs: FakeResponse(content))

  mapd_installer.MapdInstallManager(DummyReporter(), params).download()

  assert params.get("MapdVersion") == release.version
  mapd_installer.MapdInstallManager._verify_installed_binary(str(binary), release)
  cache = mapd_installer.get_persistent_binary_cache_path(release)
  mapd_installer.MapdInstallManager._verify_installed_binary(cache, release)


def test_ensure_exact_binary_repairs_version_without_download(monkeypatch, tmp_path, trusted_release):
  release, content = trusted_release
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  write_binary(binary, content)
  params = DummyParams({"MapdVersion": "stale-param"})
  calls = []
  monkeypatch.setattr(mapd_installer.MapdInstallManager, "download", lambda self: calls.append("download"))

  assert mapd_installer.ensure_mapd_installed(params, DummyReporter())
  assert calls == []
  assert params.get("MapdVersion") == release.version
  mapd_installer.MapdInstallManager._verify_installed_binary(
    mapd_installer.get_persistent_binary_cache_path(release), release
  )


def test_ensure_restores_exact_persistent_cache_while_offline(monkeypatch, tmp_path, trusted_release):
  release, content = trusted_release
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  params = DummyParams({"MapdVersion": release.version})
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", MeteredSubMaster)
  cache = Path(mapd_installer.get_persistent_binary_cache_path(release))
  write_binary(cache, content)

  assert mapd_installer.ensure_mapd_installed(params, DummyReporter())
  mapd_installer.MapdInstallManager._verify_installed_binary(str(binary), release)


def test_ensure_rejects_wrong_digest_cache_while_offline(monkeypatch, tmp_path, trusted_release):
  release, content = trusted_release
  binary, _ = configure_install_paths(monkeypatch, tmp_path)
  params = DummyParams({"MapdVersion": release.version})
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", MeteredSubMaster)
  cache = Path(mapd_installer.get_persistent_binary_cache_path(release))
  write_binary(cache, content[:-1] + b"x")

  assert not mapd_installer.ensure_mapd_installed(params, DummyReporter())
  assert not binary.exists()


def test_mapd_ready_rejects_version_mismatch_before_launch(monkeypatch, tmp_path, trusted_release):
  from openpilot.system.manager import process_config

  release, content = trusted_release
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  mapd_root.mkdir()
  write_binary(binary, content)
  runtime_calls = []
  monkeypatch.setattr(process_config, "MAPD_PATH", str(binary))
  monkeypatch.setattr(process_config.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(process_config, "_MAPD_READY_IDENTITY_CACHE", None)
  monkeypatch.setattr(
    process_config.MapdInstallManager,
    "_verify_runtime_build_info",
    lambda *args: runtime_calls.append(args),
  )

  assert process_config.mapd_ready(False, DummyParams({"MapdVersion": "wrong"}), SimpleNamespace()) is False
  assert runtime_calls == []
  assert release.version != "wrong"


def test_mapd_ready_rejects_failed_runtime_identity(monkeypatch, tmp_path, trusted_release):
  from openpilot.system.manager import process_config

  release, content = trusted_release
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  mapd_root.mkdir()
  write_binary(binary, content)
  monkeypatch.setattr(process_config, "MAPD_PATH", str(binary))
  monkeypatch.setattr(process_config.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(process_config, "_MAPD_READY_IDENTITY_CACHE", None)
  monkeypatch.setattr(
    process_config.MapdInstallManager,
    "_verify_runtime_build_info",
    lambda *args: (_ for _ in ()).throw(OSError("wrong build ID")),
  )

  assert process_config.mapd_ready(
    False, DummyParams({"MapdVersion": release.version}), SimpleNamespace()
  ) is False


def test_mapd_ready_accepts_exact_release_and_caches_unchanged_identity(monkeypatch, tmp_path, trusted_release):
  from openpilot.system.manager import process_config

  release, content = trusted_release
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  mapd_root.mkdir()
  write_binary(binary, content)
  runtime_calls = []
  monkeypatch.setattr(process_config, "MAPD_PATH", str(binary))
  monkeypatch.setattr(process_config.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(process_config, "_MAPD_READY_IDENTITY_CACHE", None)
  monkeypatch.setattr(
    process_config.MapdInstallManager,
    "_verify_runtime_build_info",
    lambda *args: runtime_calls.append(args) or {},
  )
  params = DummyParams({"MapdVersion": release.version})

  assert process_config.mapd_ready(False, params, SimpleNamespace()) is True
  assert process_config.mapd_ready(False, params, SimpleNamespace()) is True
  assert len(runtime_calls) == 1
