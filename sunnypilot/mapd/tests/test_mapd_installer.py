#!/usr/bin/env python3

import stat
from pathlib import Path
from types import SimpleNamespace

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


def write_fake_mapd(path, *, chauffeur_markers: bool):
  markers = b" ".join(mapd_installer._REQUIRED_MAPD_BINARY_MARKERS) if chauffeur_markers else b"legacy-pfeifer-mapd"
  content = b"\x7fELF" + b"\0" * 128 + markers
  content += b"\0" * max(0, 1_000_000 - len(content))
  path.write_bytes(content)
  path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_target_version_defaults(monkeypatch):
  monkeypatch.delenv("SP_MAPD_RELEASE_VERSION", raising=False)
  assert mapd_installer.get_target_version(DummyParams({})) == mapd_installer.DEFAULT_VERSION


def test_target_version_param_override(monkeypatch):
  monkeypatch.delenv("SP_MAPD_RELEASE_VERSION", raising=False)
  params = DummyParams({"MapdReleaseVersion": b"v9.9.9"})
  assert mapd_installer.get_target_version(params) == "v9.9.9"


def test_target_version_env_override_wins(monkeypatch):
  monkeypatch.setenv("SP_MAPD_RELEASE_VERSION", "v8.8.8")
  params = DummyParams({"MapdReleaseVersion": b"v9.9.9"})
  assert mapd_installer.get_target_version(params) == "v8.8.8"


def test_target_binary_url_defaults(monkeypatch):
  monkeypatch.delenv("SP_MAPD_BINARY_URL", raising=False)
  assert mapd_installer.get_target_binary_url("v1.2.3", DummyParams({})) == \
    "https://github.com/chriscarlo/mapd/releases/download/v1.2.3/mapd"


def test_target_binary_url_param_override_supports_version_template(monkeypatch):
  monkeypatch.delenv("SP_MAPD_BINARY_URL", raising=False)
  params = DummyParams({"MapdBinaryUrl": "https://maps.example.com/releases/{version}/mapd"})
  assert mapd_installer.get_target_binary_url("v1.2.3", params) == \
    "https://maps.example.com/releases/v1.2.3/mapd"


def test_target_binary_url_env_override_wins(monkeypatch):
  monkeypatch.setenv("SP_MAPD_BINARY_URL", "https://env.example.com/mapd-{version}")
  params = DummyParams({"MapdBinaryUrl": "https://param.example.com/mapd-{version}"})
  assert mapd_installer.get_target_binary_url("v1.2.3", params) == "https://env.example.com/mapd-v1.2.3"


def test_verify_installed_binary_accepts_chauffeur_build_markers(tmp_path):
  binary = tmp_path / "mapd"
  write_fake_mapd(binary, chauffeur_markers=True)

  mapd_installer.MapdInstallManager._verify_installed_binary(str(binary))


def test_verify_installed_binary_rejects_stale_pfeifer_build(tmp_path):
  binary = tmp_path / "mapd"
  write_fake_mapd(binary, chauffeur_markers=False)

  try:
    mapd_installer.MapdInstallManager._verify_installed_binary(str(binary))
  except OSError as e:
    assert "not the chauffeur-bake mapd build" in str(e)
    assert "MapPreCurveSpeeds" in str(e)
  else:
    raise AssertionError("stale mapd binary passed chauffeur-bake verification")


def test_download_needed_rejects_current_version_with_stale_binary(monkeypatch, tmp_path):
  binary = tmp_path / "mapd"
  write_fake_mapd(binary, chauffeur_markers=False)
  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))

  manager = mapd_installer.MapdInstallManager(DummyReporter())
  manager._params = DummyParams({"MapdVersion": mapd_installer.DEFAULT_VERSION})

  assert manager.download_needed()


def test_ensure_mapd_installed_downloads_invalid_binary_even_when_prebuilt(monkeypatch, tmp_path):
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  write_fake_mapd(binary, chauffeur_markers=False)
  calls = []

  def fake_download(self):
    calls.append("download")
    write_fake_mapd(binary, chauffeur_markers=True)

  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(tmp_path))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(mapd_installer, "is_prebuilt", lambda: True)
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", FakeSubMaster)
  monkeypatch.setattr(mapd_installer.MapdInstallManager, "download", fake_download)

  assert mapd_installer.ensure_mapd_installed(DummyParams({}), DummyReporter())
  assert calls == ["download"]


def test_ensure_mapd_installed_downloads_when_target_version_changes(monkeypatch, tmp_path):
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  write_fake_mapd(binary, chauffeur_markers=True)
  params = DummyParams({"MapdVersion": "old-mapd-release"})
  calls = []

  def fake_download(self):
    calls.append(self._params)
    write_fake_mapd(binary, chauffeur_markers=True)
    mapd_installer.update_installed_version(mapd_installer.get_target_version(self._params), self._params)

  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(tmp_path))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", FakeSubMaster)
  monkeypatch.setattr(mapd_installer.MapdInstallManager, "download", fake_download)

  assert mapd_installer.ensure_mapd_installed(params, DummyReporter())
  assert calls == [params]
  assert params.get("MapdVersion") == mapd_installer.DEFAULT_VERSION


def test_ensure_mapd_installed_keeps_matching_valid_binary(monkeypatch, tmp_path):
  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  write_fake_mapd(binary, chauffeur_markers=True)
  params = DummyParams({"MapdVersion": mapd_installer.DEFAULT_VERSION})
  calls = []

  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(tmp_path))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(mapd_installer.MapdInstallManager, "download", lambda self: calls.append("download"))

  assert mapd_installer.ensure_mapd_installed(params, DummyReporter())
  assert calls == []
  cache = mapd_installer.get_persistent_binary_cache_path(mapd_installer.DEFAULT_VERSION)
  mapd_installer.MapdInstallManager._verify_installed_binary(cache)


def test_ensure_mapd_installed_restores_persistent_cache_while_offline(monkeypatch, tmp_path):
  binary_dir = tmp_path / "checkout" / "third_party" / "mapd"
  binary_dir.mkdir(parents=True)
  binary = binary_dir / "mapd"
  mapd_root = tmp_path / "persistent-osm"
  cache_dir = mapd_root / mapd_installer._PERSISTENT_BINARY_CACHE_DIR
  cache_dir.mkdir(parents=True)
  params = DummyParams({"MapdVersion": mapd_installer.DEFAULT_VERSION})

  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(binary_dir))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", MeteredSubMaster)
  cache = mapd_installer.get_persistent_binary_cache_path(mapd_installer.DEFAULT_VERSION)
  write_fake_mapd(Path(cache), chauffeur_markers=True)

  assert mapd_installer.ensure_mapd_installed(params, DummyReporter())
  mapd_installer.MapdInstallManager._verify_installed_binary(str(binary))


def test_ensure_mapd_installed_rejects_invalid_cache_while_offline(monkeypatch, tmp_path):
  binary_dir = tmp_path / "checkout" / "third_party" / "mapd"
  binary_dir.mkdir(parents=True)
  binary = binary_dir / "mapd"
  mapd_root = tmp_path / "persistent-osm"
  cache_dir = mapd_root / mapd_installer._PERSISTENT_BINARY_CACHE_DIR
  cache_dir.mkdir(parents=True)
  params = DummyParams({"MapdVersion": mapd_installer.DEFAULT_VERSION})

  monkeypatch.setattr(mapd_installer, "MAPD_PATH", str(binary))
  monkeypatch.setattr(mapd_installer, "MAPD_BIN_DIR", str(binary_dir))
  monkeypatch.setattr(mapd_installer.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))
  monkeypatch.setattr(mapd_installer.messaging, "SubMaster", MeteredSubMaster)
  cache = mapd_installer.get_persistent_binary_cache_path(mapd_installer.DEFAULT_VERSION)
  write_fake_mapd(Path(cache), chauffeur_markers=False)

  assert not mapd_installer.ensure_mapd_installed(params, DummyReporter())
  assert not binary.exists()


def test_mapd_ready_rejects_stale_binary_before_native_launch(monkeypatch, tmp_path):
  from openpilot.system.manager import process_config

  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  mapd_root.mkdir()
  write_fake_mapd(binary, chauffeur_markers=False)

  monkeypatch.setattr(process_config, "MAPD_PATH", str(binary))
  monkeypatch.setattr(process_config.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))

  assert process_config.mapd_ready(False, DummyParams({}), SimpleNamespace()) is False


def test_mapd_ready_accepts_chauffeur_binary(monkeypatch, tmp_path):
  from openpilot.system.manager import process_config

  binary = tmp_path / "mapd"
  mapd_root = tmp_path / "osm"
  mapd_root.mkdir()
  write_fake_mapd(binary, chauffeur_markers=True)

  monkeypatch.setattr(process_config, "MAPD_PATH", str(binary))
  monkeypatch.setattr(process_config.Paths, "mapd_root", staticmethod(lambda: str(mapd_root)))

  assert process_config.mapd_ready(False, DummyParams({}), SimpleNamespace()) is True
