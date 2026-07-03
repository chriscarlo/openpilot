"""Repro CD4 (SEV-2): radard ModelLeadTracker association gate keys on RAW yRel
instead of the path-relative dPath, so curve-induced yRel drift churns the
published track id and fabricates physically-impossible single-frame dRel steps.

Road evidence (00000200--8cbf2c9481--3, t=51-59 s, a single physical lead held
at prob ~0.98 through a curve): the raw model yRel drifts -1.77 -> -8.50 m
(6.72 m excursion) while the path-relative dPath the tracker already computes
stays within +/-0.84 m -- the lead never leaves the lane. Because the
association / same-frame-duplicate gates in
selfdrive/controls/radard.py:ModelLeadTracker compare RAW yRel against the
MODEL_LEAD_ASSOC_Y_GATE_M (3.0 m) / MODEL_LEAD_DUPLICATE_PATH_GATE_M (0.8 m)
tolerances, the two duplicate model slots stop merging and start flapping
between competing track ids as the raw dRel noise (50-83 m) shifts which track
wins slot 0. Each flap re-publishes the raw dRel of the freshly-landed track
with no filter continuity, so the published leadOne dRel jumps up to 16 m in a
single 50 ms frame while the same physical lead persists.

This is a UNIT-level replay: the recorded raw modelV2.leadsV3[0]/[1] series
(passed through the real get_RadarState_from_vision + add_path_relative_lead_metrics
runtime helpers at extraction time) is driven straight through the REAL
selfdrive/controls/radard.py ModelLeadTracker, exactly as RadarD.update feeds
it in production (both slots into one tracker per frame, leadOne = slot 0).

Device radarState leadOne over the same window shows the same signature: 15
track-id changes and a 14.3 m single-frame published dRel step at t=53.3 s.

ORACLE:
  scenario-validity (always runs): the raw yRel genuinely blows the 3.0 m gate
    while dPath stays lane-relevant, AND the real tracker actually churns the
    published track id -- proving the gate rejects continuity on this drive.
  behavioral (strict-xfail; flips XPASS when CD4 is fixed): (1) the published
    leadOne radarTrackId stays CONSTANT across the yRel excursion; (2) no
    single-frame published dRel step exceeds max(3 m, 0.1*dRel) while the same
    physical lead persists.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.controls.radard import (
  MODEL_LEAD_ASSOC_Y_GATE_M,
  ModelLeadTracker,
)

_DATA = (Path(__file__).parents[1] / "testdata" / "cd4_track_id_churn"
         / "frames_200_3.json")

# dPath tolerance the forensics reports the lead never exceeds on this curve
# (+/-1.8 m); the gate should treat the lead as continuous the whole time.
DPATH_LANE_RELEVANT_M = 1.8
# Published single-frame dRel step guard: while one physical lead persists, a
# filtered publish must not move more than the noise-floor-derived bound.
def _step_gate_m(published_drel: float) -> float:
  return max(3.0, 0.1 * published_drel)


@lru_cache(maxsize=1)
def _frames() -> list[dict]:
  return json.loads(_DATA.read_text())["frames"]


@lru_cache(maxsize=1)
def _replay() -> dict:
  """Drive the recorded two-slot raw-model series through the REAL tracker,
  exactly as RadarD.update does (both slots -> one tracker/frame, leadOne=slot0).
  """
  frames = _frames()
  tracker = ModelLeadTracker(Params())
  lead_one_ids: list[int] = []
  lead_one_drels: list[float] = []
  raw_yrel_slot0: list[float] = []
  raw_dpath_slot0: list[float] = []

  for frame in frames:
    tracker.begin_frame(frame["t_s"])
    lead_one: dict | None = None
    for slot in (0, 1):
      s = frame["slots"][slot]
      lead_dict = dict(s)
      lead_dict["status"] = True
      lead_dict["vLeadK"] = s["vLead"]
      out = tracker.update_from_vision(
        lead_dict, now=frame["t_s"], v_ego=frame["v_ego"], lead_slot=slot
      )
      if slot == 0:
        lead_one = out
    tracker.end_frame()
    assert lead_one is not None
    lead_one_ids.append(int(lead_one["radarTrackId"]))
    lead_one_drels.append(float(lead_one["dRel"]))
    raw_yrel_slot0.append(float(frame["slots"][0]["yRel"]))
    raw_dpath_slot0.append(float(frame["slots"][0]["dPath"]))

  # Largest single-frame published dRel step while the physical lead persists.
  max_step_m = 0.0
  max_step_t = None
  steps_over_gate = 0
  for i in range(1, len(lead_one_drels)):
    step = abs(lead_one_drels[i] - lead_one_drels[i - 1])
    if step > _step_gate_m(lead_one_drels[i]):
      steps_over_gate += 1
    if step > max_step_m:
      max_step_m = step
      max_step_t = frames[i]["t_s"]

  churn_events = sum(
    1 for i in range(1, len(lead_one_ids)) if lead_one_ids[i] != lead_one_ids[i - 1]
  )

  return {
    "n_frames": len(frames),
    "unique_ids": sorted(set(lead_one_ids)),
    "churn_events": churn_events,
    "max_step_m": max_step_m,
    "max_step_t_s": max_step_t,
    "steps_over_gate": steps_over_gate,
    "raw_yrel_min": min(raw_yrel_slot0),
    "raw_yrel_max": max(raw_yrel_slot0),
    "raw_yrel_excursion": max(raw_yrel_slot0) - min(raw_yrel_slot0),
    "dpath_abs_max": max(abs(y) for y in raw_dpath_slot0),
  }


def test_scenario_validity_gate_rejects_continuity_on_yrel_but_not_dpath() -> None:
  """Prove this is a genuine CD4 repro: the raw yRel excursion exceeds the
  association gate while the path-relative dPath stays lane-relevant, and the
  REAL tracker actually churns the published track id in response."""
  r = _replay()

  # A single physical lead on a curve: raw yRel drifts past the 3.0 m gate...
  assert r["raw_yrel_excursion"] > MODEL_LEAD_ASSOC_Y_GATE_M, (
    f"raw yRel excursion {r['raw_yrel_excursion']:.2f} m must exceed the "
    f"{MODEL_LEAD_ASSOC_Y_GATE_M} m gate for this to exercise the defect"
  )
  # ...while the path-relative dPath the tracker already computes never leaves
  # the lane, so the lead is continuously lane-relevant the whole window.
  assert r["dpath_abs_max"] < DPATH_LANE_RELEVANT_M, (
    f"dPath abs-max {r['dpath_abs_max']:.2f} m must stay within "
    f"{DPATH_LANE_RELEVANT_M} m; if not, the gap is not the yRel-vs-dPath defect"
  )
  # The gate rejects continuity today: the published track id churns.
  assert r["churn_events"] > 0, (
    "scenario does not reproduce CD4 track-id churn; the association gate must "
    "reject continuity on the raw-yRel excursion for the behavioral oracle to bite"
  )


@pytest.mark.xfail(
  strict=True,
  reason="CD4: ModelLeadTracker association / duplicate gates key on RAW yRel "
         "(MODEL_LEAD_ASSOC_Y_GATE_M / MODEL_LEAD_DUPLICATE_PATH_GATE_M) instead "
         "of the path-relative dPath, so curve-induced yRel drift churns the "
         "published leadOne track id and fabricates single-frame dRel steps "
         "(road 00000200--8cbf2c9481--3 t=51-59 s; forensics CD4).",
)
def test_track_id_constant_and_no_fabricated_drel_step() -> None:
  r = _replay()

  # (1) One physical lead => one published track id across the whole window.
  track_id_constant = len(r["unique_ids"]) == 1
  # (2) No physically-impossible published dRel step while the lead persists.
  no_fabricated_step = r["steps_over_gate"] == 0

  physics = (
    f"CD4 unit replay of real ModelLeadTracker over 00000200--8cbf2c9481--3 "
    f"t=51-59 s ({r['n_frames']} frames, one physical lead prob ~0.98):\n"
    f"  raw model yRel drifts {r['raw_yrel_max']:.2f} -> {r['raw_yrel_min']:.2f} m "
    f"(excursion {r['raw_yrel_excursion']:.2f} m, gate {MODEL_LEAD_ASSOC_Y_GATE_M} m) "
    f"while path-relative dPath abs-max stays {r['dpath_abs_max']:.2f} m "
    f"(< {DPATH_LANE_RELEVANT_M} m, still in-lane)\n"
    f"  published leadOne track ids: {r['unique_ids']} "
    f"({r['churn_events']} churn events; expect 1 stable id)\n"
    f"  max single-frame published dRel step: {r['max_step_m']:.2f} m at "
    f"t={r['max_step_t_s']} s; {r['steps_over_gate']} step(s) exceed "
    f"max(3 m, 0.1*dRel) while the same physical lead persists (expect 0)"
  )
  assert track_id_constant and no_fabricated_step, physics
