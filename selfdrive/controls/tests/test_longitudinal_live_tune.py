import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import (
  LEAD_RESPONSE_TUNE_SPECS,
  LEAD_RESPONSE_TUNE_SPECS_BY_ATTR,
  read_lead_response_tuning_config,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.test.longitudinal_harness.config import DEFAULT_PARAM_VALUES, REPLAY_PARAM_MANIFEST_KEYS
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_TUNE_SCRIPT = REPO_ROOT / ".codex/skills/openpilot-longitudinal-tuner/scripts/live_lead_tune.py"


def _make_lead(*, status=True, d_rel=58.0, v_lead=34.3, a_lead=0.2):
  return SimpleNamespace(
    status=status,
    dRel=d_rel,
    vLead=v_lead,
    aLeadK=a_lead,
  )


def _configure_vibe_follow(headway=1.3):
  params = Params()
  params.put_bool('VibePersonalityEnabled', True)
  params.put_bool('VibeFollowPersonalityEnabled', True)
  params.put_bool('VibeAccelPersonalityEnabled', False)
  params.put('LongitudinalPersonality', int(log.LongitudinalPersonality.standard))
  for idx in range(4):
    params.put(f'VibeTune.Follow.Standard.Headway{idx}', float(headway))


def _run_live_tune_script(*args: str) -> subprocess.CompletedProcess[str]:
  env = dict(os.environ)
  pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
  return subprocess.run(
    [sys.executable, str(LIVE_TUNE_SCRIPT), *args],
    cwd=REPO_ROOT,
    capture_output=True,
    check=True,
    text=True,
    env=env,
  )


@pytest.fixture(autouse=True)
def _restore_live_tune_params():
  params = Params()
  saved_values = {
    spec.key: params.get(spec.key)
    for spec in LEAD_RESPONSE_TUNE_SPECS
  }
  yield
  for spec in LEAD_RESPONSE_TUNE_SPECS:
    raw_value = saved_values[spec.key]
    if raw_value is None:
      params.remove(spec.key)
    else:
      params.put(spec.key, raw_value)


@pytest.fixture(autouse=True)
def _planner_test_setup(monkeypatch):
  _configure_vibe_follow()
  monkeypatch.setattr(
    LongitudinalPlannerSP,
    "update_v_cruise",
    lambda self, sm, v_ego, a_ego, v_cruise: v_cruise,
  )


class TestLeadResponseTuneConfig:
  def test_param_key_defaults_match_shared_spec_defaults(self):
    params = Params()

    for spec in LEAD_RESPONSE_TUNE_SPECS:
      assert params.get_default_value(spec.key) == pytest.approx(spec.default)

  def test_read_config_clamps_invalid_values(self):
    params = Params()
    params.put("Longitudinal.LiveTune.LeadPreviewStrength", 4.0)
    params.put("Longitudinal.LiveTune.GapReclaimMaxAccel", -1.0)

    config = read_lead_response_tuning_config(params)

    assert config.lead_preview_strength == pytest.approx(
      LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["lead_preview_strength"].maximum,
    )
    assert config.gap_reclaim_max_accel == pytest.approx(0.0)

  def test_raw_cruise_cap_qualification_cannot_be_disabled_by_live_values(self):
    params = Params()
    requests_that_reduce_cap_authority = {
      "CruiseCapRawLeadAcquireDwellS": 99.0,
      "CruiseCapRawLeadReleaseHoldS": 0.0,
      "CruiseCapRawLeadMaxSampleGapS": 0.0,
      "CruiseCapRawLeadMaxDRelStepM": 0.0,
      "CruiseCapRawLeadMaxDPathStepM": 0.0,
      "CruiseCapRawLeadMaxYRelStepM": 0.0,
      "CruiseCapRawLeadSuspectYRelM": 0.0,
      "CruiseCapRawLeadSuspectVLatMps": 0.0,
    }
    for suffix, value in requests_that_reduce_cap_authority.items():
      params.put(f"Longitudinal.LiveTune.{suffix}", value)

    config = read_lead_response_tuning_config(params)

    assert config.cruise_cap_raw_lead_acquire_dwell_s == pytest.approx(0.60)
    assert config.cruise_cap_raw_lead_release_hold_s == pytest.approx(0.50)
    assert config.cruise_cap_raw_lead_max_sample_gap_s == pytest.approx(0.30)
    assert config.cruise_cap_raw_lead_max_drel_step_m == pytest.approx(6.0)
    assert config.cruise_cap_raw_lead_max_dpath_step_m == pytest.approx(0.75)
    assert config.cruise_cap_raw_lead_max_yrel_step_m == pytest.approx(1.50)
    assert config.cruise_cap_raw_lead_suspect_yrel_m == pytest.approx(4.0)
    assert config.cruise_cap_raw_lead_suspect_vlat_mps == pytest.approx(4.0)

  def test_raw_cruise_cap_params_are_in_synthetic_defaults_and_exact_manifest(self):
    raw_cap_specs = [
      spec for spec in LEAD_RESPONSE_TUNE_SPECS
      if spec.attr.startswith("cruise_cap_raw_lead_")
    ]

    assert len(raw_cap_specs) == 8
    for spec in raw_cap_specs:
      assert float(DEFAULT_PARAM_VALUES[spec.key]) == pytest.approx(spec.default)
      assert spec.key in REPLAY_PARAM_MANIFEST_KEYS

  def test_stopping_release_jerk_only_allows_smoother_tuning(self):
    params = Params()
    spec = LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["stopping_release_jerk_mps3"]

    params.put(spec.key, 99.0)
    assert read_lead_response_tuning_config(params).stopping_release_jerk_mps3 == pytest.approx(6.0)

    params.put(spec.key, 0.0)
    assert read_lead_response_tuning_config(params).stopping_release_jerk_mps3 == pytest.approx(1.0)
    assert float(DEFAULT_PARAM_VALUES[spec.key]) == pytest.approx(spec.default)
    assert spec.key in REPLAY_PARAM_MANIFEST_KEYS


class TestLiveLeadTuneScript:
  def test_script_show_set_and_reset_cycle(self):
    _run_live_tune_script("reset")
    _run_live_tune_script(
      "set",
      "--gap-reclaim-strength", "1.25",
      "--lead-preview-gap-min-m", "2.0",
      "--cruise-cap-raw-lead-acquire-dwell-s", "0.30",
    )

    payload = json.loads(_run_live_tune_script("show", "--json").stdout)
    assert payload["gap_reclaim_strength"]["stored"] == pytest.approx(1.25)
    assert payload["gap_reclaim_strength"]["effective"] == pytest.approx(1.25)
    assert payload["lead_preview_gap_min_m"]["stored"] == pytest.approx(2.0)
    assert payload["cruise_cap_raw_lead_acquire_dwell_s"]["stored"] == pytest.approx(0.30)

    _run_live_tune_script("reset")
    payload = json.loads(_run_live_tune_script("show", "--json").stdout)
    assert payload["gap_reclaim_strength"]["stored"] is None
    assert payload["gap_reclaim_strength"]["effective"] == pytest.approx(
      LEAD_RESPONSE_TUNE_SPECS_BY_ATTR["gap_reclaim_strength"].default,
    )
    assert payload["cruise_cap_raw_lead_acquire_dwell_s"]["stored"] is None
    assert payload["cruise_cap_raw_lead_acquire_dwell_s"]["effective"] == pytest.approx(0.60)


class TestLongitudinalMpcLiveRefresh:
  def test_same_mpc_instance_picks_up_param_change_after_refresh_interval(self):
    params = Params()
    params.put("Longitudinal.LiveTune.GapReclaimStrength", 1.0)

    mpc = LongitudinalMpc()
    lead = _make_lead()
    mpc.mode = 'acc'
    mpc.set_cur_state(33.5, 0.0)
    mpc.control_leads = (lead, None)
    mpc.current_t_follow = 1.3
    mpc.last_v_cruise_clipped = np.array([33.5, 34.1])

    mpc._refresh_live_tune(100.0, force=True)
    assert mpc.get_gap_reclaim_floor() > 0.05

    params.put("Longitudinal.LiveTune.GapReclaimStrength", 0.0)
    mpc._refresh_live_tune(100.2)
    assert mpc.get_gap_reclaim_floor() > 0.05

    mpc._refresh_live_tune(100.6)
    assert mpc.get_gap_reclaim_floor() == pytest.approx(0.0)
