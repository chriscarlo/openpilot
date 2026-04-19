#!/usr/bin/env python3

from openpilot.sunnypilot.mapd import mapd_installer


class DummyParams:
  def __init__(self, values: dict[str, str | bytes | None]):
    self._values = values

  def get(self, key, *args, **kwargs):
    return self._values.get(key)


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
