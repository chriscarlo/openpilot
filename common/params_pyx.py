import builtins
import datetime
import json
import os
import re
import sys
from enum import IntEnum, IntFlag
from pathlib import Path


if sys.platform != "win32":
  raise ImportError("openpilot.common.params_pyx must be built with scons outside Windows")


class ParamKeyFlag(IntFlag):
  PERSISTENT = 0x02
  CLEAR_ON_MANAGER_START = 0x04
  CLEAR_ON_ONROAD_TRANSITION = 0x08
  CLEAR_ON_OFFROAD_TRANSITION = 0x10
  DONT_LOG = 0x20
  DEVELOPMENT_ONLY = 0x40
  CLEAR_ON_IGNITION_ON = 0x80
  BACKUP = 0x100
  ALL = 0xFFFFFFFF


class ParamKeyType(IntEnum):
  STRING = 0
  BOOL = 1
  INT = 2
  FLOAT = 3
  TIME = 4
  JSON = 5
  BYTES = 6


class UnknownKeyName(Exception):
  pass


PYTHON_2_CPP = {
  (str, ParamKeyType.STRING): lambda v: v.encode("utf-8"),
  (builtins.bool, ParamKeyType.BOOL): lambda v: b"1" if v else b"0",
  (int, ParamKeyType.INT): lambda v: str(v).encode("utf-8"),
  (float, ParamKeyType.FLOAT): lambda v: str(v).encode("utf-8"),
  (datetime.datetime, ParamKeyType.TIME): lambda v: v.isoformat().encode("utf-8"),
  (dict, ParamKeyType.JSON): lambda v: json.dumps(v).encode("utf-8"),
  (list, ParamKeyType.JSON): lambda v: json.dumps(v).encode("utf-8"),
  (bytes, ParamKeyType.BYTES): lambda v: v,
}

CPP_2_PYTHON = {
  ParamKeyType.STRING: lambda v: v.decode("utf-8"),
  ParamKeyType.BOOL: lambda v: v == b"1",
  ParamKeyType.INT: int,
  ParamKeyType.FLOAT: float,
  ParamKeyType.TIME: lambda v: datetime.datetime.fromisoformat(v.decode("utf-8")),
  ParamKeyType.JSON: json.loads,
  ParamKeyType.BYTES: lambda v: v,
}


_PARAM_LINE_RE = re.compile(
  r'\{"(?P<key>[^"]+)",\s*\{(?P<flags>[^,]+),\s*(?P<type>[A-Z]+)(?:,\s*(?P<default>.*?))?\}\}'
)


def ensure_bytes(v):
  return v.encode("utf-8") if isinstance(v, str) else v


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _parse_flag_expr(expr: str) -> ParamKeyFlag:
  value = 0
  for name in expr.split("|"):
    name = name.strip()
    if not name:
      continue
    value |= int(getattr(ParamKeyFlag, name))
  return ParamKeyFlag(value)


def _parse_default(raw: str | None) -> bytes | None:
  if raw is None:
    return None
  raw = raw.strip().rstrip(",")
  quoted = re.match(r'"(.*)"', raw)
  if quoted is not None:
    return quoted.group(1).encode("utf-8")
  if "LongitudinalPersonality::STANDARD" in raw:
    return b"1"
  return None


def _load_key_attributes() -> dict[str, tuple[ParamKeyFlag, ParamKeyType, bytes | None]]:
  attrs: dict[str, tuple[ParamKeyFlag, ParamKeyType, bytes | None]] = {}
  params_keys = _repo_root() / "common" / "params_keys.h"
  for line in params_keys.read_text(encoding="utf-8").splitlines():
    match = _PARAM_LINE_RE.search(line)
    if match is None:
      continue
    key = match.group("key")
    flags = _parse_flag_expr(match.group("flags"))
    key_type = getattr(ParamKeyType, match.group("type"))
    default = _parse_default(match.group("default"))
    attrs[key] = (flags, key_type, default)
  return attrs


_KEY_ATTRIBUTES = _load_key_attributes()


class Params:
  def __init__(self, d=""):
    self.d = d
    params_root = Path(d) if d else Path(os.environ.get("PARAMS_ROOT", Path.home() / f".comma{os.environ.get('OPENPILOT_PREFIX', '')}" / "params"))
    self.params_prefix = os.environ.get("OPENPILOT_PREFIX", "d")
    self.params_path = params_root
    self.key_path = self.params_path / self.params_prefix
    self.key_path.mkdir(parents=True, exist_ok=True)

  def __reduce__(self):
    return (type(self), (self.d,))

  def clear_all(self, tx_flag=ParamKeyFlag.ALL):
    for key in self.all_keys(tx_flag):
      self.remove(key)
    for child in self.key_path.iterdir() if self.key_path.exists() else []:
      if child.is_file() and child.name not in _KEY_ATTRIBUTES:
        child.unlink(missing_ok=True)

  def check_key(self, key):
    if isinstance(key, bytes):
      key = key.decode("utf-8")
    if key not in _KEY_ATTRIBUTES:
      raise UnknownKeyName(key)
    return key

  def python2cpp(self, proposed_type, expected_type, value, key):
    cast = PYTHON_2_CPP.get((proposed_type, expected_type))
    if cast is not None:
      return cast(value)
    raise TypeError(f"Type mismatch while writing param {key}: proposed_type={proposed_type} expected_type={expected_type} value={value}")

  def _cpp2python(self, t, value, default, key):
    if value is None:
      return None
    try:
      return CPP_2_PYTHON[t](value)
    except (KeyError, TypeError, ValueError):
      return self._cpp2python(t, default, None, key)

  def get(self, key, block=False, return_default=False):
    key = self.check_key(key)
    key_type = self.get_type(key)
    default = self._default_bytes(key) if return_default else None
    path = self.key_path / key
    if not path.exists():
      return self._cpp2python(key_type, default, None, key)
    return self._cpp2python(key_type, path.read_bytes(), default, key)

  def get_bool(self, key, block=False):
    key = self.check_key(key)
    path = self.key_path / key
    return path.exists() and path.read_bytes() == b"1"

  def _put_cast(self, key, dat):
    key = self.check_key(key)
    return ensure_bytes(self.python2cpp(type(dat), self.get_type(key), dat, key))

  def put(self, key, dat):
    key = self.check_key(key)
    dat_bytes = self._put_cast(key, dat)
    self.key_path.mkdir(parents=True, exist_ok=True)
    tmp_path = self.key_path / f".tmp_{key}_{os.getpid()}"
    tmp_path.write_bytes(dat_bytes)
    os.replace(tmp_path, self.key_path / key)

  def put_bool(self, key, val):
    key = self.check_key(key)
    self.key_path.mkdir(parents=True, exist_ok=True)
    (self.key_path / key).write_bytes(b"1" if val else b"0")

  def put_nonblocking(self, key, dat):
    self.put(key, dat)

  def put_bool_nonblocking(self, key, val):
    self.put_bool(key, val)

  def remove(self, key):
    key = self.check_key(key)
    (self.key_path / key).unlink(missing_ok=True)

  def get_param_path(self, key=""):
    return str(self.key_path / key) if key else str(self.key_path)

  def get_type(self, key):
    key = self.check_key(key)
    return _KEY_ATTRIBUTES[key][1]

  def _default_bytes(self, key):
    return _KEY_ATTRIBUTES[self.check_key(key)][2]

  def all_keys(self, flag=ParamKeyFlag.ALL):
    flag = ParamKeyFlag(flag)
    return [key for key, (key_flags, _, _) in _KEY_ATTRIBUTES.items() if flag == ParamKeyFlag.ALL or bool(key_flags & flag)]

  def get_default_value(self, key):
    key = self.check_key(key)
    return self._cpp2python(self.get_type(key), self._default_bytes(key), None, key)

  def cpp2python(self, key, value):
    key = self.check_key(key)
    return self._cpp2python(self.get_type(key), value, None, key)
