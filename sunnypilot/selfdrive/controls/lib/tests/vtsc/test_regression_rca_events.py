#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_response_model import build_cruise_response_model

from .harness import Step, mk_vtsc_with_params, _mk_sm


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "rca_events"


def _default_replay_response_model():
  # Match the shared cruise-response helper used onroad as closely as this
  # controller-only replay can, without standing up the full planner/MPC path.
  return build_cruise_response_model(actuation_delay_s=float(DT_MDL))


def _load_fixture(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _fill_forward(values: list[float | None], *, fallback: float) -> list[float]:
  out: list[float] = []
  cur = float(fallback)
  for v in values:
    if v is not None:
      cur = float(v)
    out.append(cur)
  return out


def _simulate_controller_from_fixture(fixture: dict) -> tuple[list[float], list[float], list[float]]:
  steps_raw = list(fixture.get("steps") or [])
  assert steps_raw, "fixture has no steps"

  t_rel = [float(r.get("t", 0.0)) for r in steps_raw]
  v_ego = _fill_forward([r.get("v_ego") for r in steps_raw], fallback=0.0)
  a_ego = _fill_forward([r.get("a_ego") for r in steps_raw], fallback=0.0)
  v_cruise = _fill_forward([r.get("v_cruise") for r in steps_raw], fallback=0.0)

  ctrl = mk_vtsc_with_params()
  ctrl.set_longitudinal_response_model(_default_replay_response_model())
  v_turn_out: list[float] = []

  for i, r in enumerate(steps_raw):
    st = Step(
      curvature=float(r.get("curvature") or 0.0),
      curvature_ahead=r.get("curvature_ahead"),
      confidence=float(r.get("confidence") or 1.0),
      lead_d_rel_m=r.get("lead_d_rel_m"),
      steering_angle_deg=float(r.get("steering_angle_deg") or 0.0),
    )
    # Use relative time (shifted positive) so controller dwell timers work deterministically.
    t_sim = float(t_rel[i] + float(fixture.get("pre_s") or 10.0))
    sm = _mk_sm(st.curvature, st.curvature_ahead, v_ego[i], st.confidence, st.lead_d_rel_m, st.steering_angle_deg)
    with patch("sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.time", lambda: t_sim), \
         patch("sunnypilot.selfdrive.controls.lib.vision_turn_controller.time.monotonic", lambda: t_sim):
      ctrl.update(sm, True, v_ego[i], a_ego[i], v_cruise[i])
    v_turn_out.append(float(ctrl.v_turn))

  return t_rel, v_cruise, v_turn_out


def _one_frame_limiting_pulses(*, t_rel: list[float], v_cruise: list[float], v_turn: list[float],
                              t0_window: tuple[float, float]) -> int:
  assert len(t_rel) == len(v_cruise) == len(v_turn)
  t_min, t_max = t0_window
  idx = [i for i, t in enumerate(t_rel) if float(t_min) <= float(t) <= float(t_max)]
  if len(idx) < 3:
    return 0

  pulses = 0
  for j in range(1, len(idx) - 1):
    i0 = idx[j - 1]
    i1 = idx[j]
    i2 = idx[j + 1]
    # A "limiting" frame is one where VTSC requests a meaningful reduction vs cruise.
    lim1 = v_turn[i1] <= (v_cruise[i1] - 1.0)
    # A 1-frame pulse is a single limiting frame with non-limiting neighbors.
    lim0 = v_turn[i0] <= (v_cruise[i0] - 1.0)
    lim2 = v_turn[i2] <= (v_cruise[i2] - 1.0)
    if lim1 and (not lim0) and (not lim2):
      pulses += 1
  return pulses


@pytest.mark.parametrize("fixture_name", [
  "t3044_brake.json",
  "t3049_brake.json",
  "t3073_brake.json",
  "t3088_brake.json",
  "t5112_brake.json",
  "t5146_brake.json",
])
def test_rca_fixtures_have_no_one_frame_cap_pulses_pre_intervention(fixture_name: str):
  # Regression: real interventions showed VTSC cap "flapping" (single-frame or very short pulses).
  # The longitudinal planner often can't react to a <0.5s cap, which presents as late braking.
  #
  # This test uses anonymized fixtures extracted from rlogs and asserts we do not produce any
  # single-frame *limiting* caps in the pre-intervention window.
  fix_path = _FIXTURES_DIR / fixture_name
  fixture = _load_fixture(fix_path)
  t_rel, v_cruise, v_turn = _simulate_controller_from_fixture(fixture)

  pulses = _one_frame_limiting_pulses(
    t_rel=t_rel,
    v_cruise=v_cruise,
    v_turn=v_turn,
    t0_window=(-2.0, 0.0),
  )
  assert pulses == 0
