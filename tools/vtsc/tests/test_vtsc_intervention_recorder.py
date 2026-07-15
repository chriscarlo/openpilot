"""Regression coverage for recorder-side arbitration provenance reads."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtsc.vtsc_intervention_recorder import _read_map_vision_arbitration


def test_arbitration_reader_marks_old_wire_defaults_unavailable():
  # A new PyCapnp reader exposes these names for a pre-schema producer, but
  # their defaults are not real map/vision evidence.
  old_wire = SimpleNamespace(
    mapStrategyState='',
    mapFloorActive=False,
    mapStrategicCap=0.0,
  )

  fields = _read_map_vision_arbitration(old_wire)

  assert all(value is None for value in fields.values())


def test_arbitration_reader_preserves_false_and_zero_from_new_producer():
  new_wire = SimpleNamespace(
    mapStrategyState='idle',
    mapStrategyMode='strategic',
    mapFloorActive=False,
    mapFloorReason='',
    visionRelaxAllowed=False,
    visionRelaxReason='',
    mapAdvisoryCap=0.0,
    mapStrategicCap=0.0,
    visionLocalCap=0.0,
    selectedCap=0.0,
    mapAnchorDistanceM=0.0,
    mapAnchorCurvature=0.0,
    mapAnchorIndex=-1,
    mapTakeoverDwellS=0.0,
    mapCounterevidenceDwellS=0.0,
  )

  fields = _read_map_vision_arbitration(new_wire)

  assert fields['map_strategy_state'] == 'idle'
  assert fields['map_strategy_mode'] == 'strategic'
  assert fields['map_floor_active'] is False
  assert fields['vision_relax_allowed'] is False
  assert fields['map_strategic_cap'] == 0.0
  assert fields['map_anchor_index'] == -1
