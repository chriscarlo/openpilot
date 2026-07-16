from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PLANNER_REPLAY_INPUTS_VERSION = 1
RADARD_REPLAY_INPUTS_VERSION = 2
RADARD_REPLAY_INPUTS_SUPPORTED_VERSIONS = frozenset((1, RADARD_REPLAY_INPUTS_VERSION))


def is_supported_radard_replay_version(value: Any) -> bool:
  return (
    isinstance(value, int)
    and not isinstance(value, bool)
    and value in RADARD_REPLAY_INPUTS_SUPPORTED_VERSIONS
  )


def is_exact_supported_radard_replay_contract(contract: Any) -> bool:
  if not isinstance(contract, Mapping) or contract.get("status") != "exact":
    return False
  return is_supported_radard_replay_version(contract.get("version"))
