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
    while dPath stays lane-relevant, AND the pre-fix association gate (a faithful
    legacy reconstruction, keying lateral continuity on RAW yRel at the shared
    3.0 m gate with y_err in the same-frame-duplicate test) actually churns the
    published track id -- proving the LEGACY gate rejects continuity on this
    drive. The legacy reconstruction is fix-independent: it reuses the REAL
    ModelLeadTrack filter/publish and only restores the two pre-CD4 gating
    decisions, so it keeps witnessing the defect after the fix lands.
  behavioral (was strict-xfail; flips XPASS once CD4 is fixed): the REAL,
    current ModelLeadTracker over the same fixture publishes (1) a CONSTANT
    leadOne radarTrackId across the yRel excursion; (2) no single-frame published
    dRel step exceeding max(3 m, 0.1*dRel) while the same physical lead persists.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from openpilot.common.params import Params
from openpilot.selfdrive.controls.radard import (
  MODEL_LEAD_ASSOC_VREL_GATE_MPS,
  MODEL_LEAD_ASSOC_Y_GATE_M,
  MODEL_LEAD_DUPLICATE_CLOSER_KEEP_SEPARATE_M,
  MODEL_LEAD_DUPLICATE_CLOSING_KEEP_SEPARATE_MPS,
  MODEL_LEAD_DUPLICATE_DREL_GATE_M,
  MODEL_LEAD_DUPLICATE_PATH_GATE_M,
  MODEL_LEAD_DUPLICATE_VREL_GATE_MPS,
  MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M,
  MODEL_LEAD_TRACK_ID_START,
  MODEL_LEAD_TRACK_MAX_COUNT,
  MODEL_LEAD_TRACK_MAX_MISSES,
  ModelLeadTrack,
  ModelLeadTracker,
)
from openpilot.selfdrive.controls.lib.longitudinal_live_tune import LeadResponseTuningConfig

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


def _lead_dicts(frame: dict) -> list[dict]:
  out = []
  for slot in (0, 1):
    s = frame["slots"][slot]
    lead_dict = dict(s)
    lead_dict["status"] = True
    lead_dict["vLeadK"] = s["vLead"]
    out.append(lead_dict)
  return out


class _LegacyModelLeadTracker:
  """Faithful reconstruction of the PRE-CD4 ModelLeadTracker association and
  same-frame-duplicate gates (lateral continuity keyed on RAW yRel at the shared
  MODEL_LEAD_ASSOC_Y_GATE_M, y_err inside the same-frame-duplicate test). It
  reuses the REAL ModelLeadTrack filter/publish path verbatim -- only the two
  gating DECISIONS are restored to their pre-fix form -- so it is an independent,
  fix-proof witness that the legacy gate churns the published id on this fixture.
  """

  def __init__(self) -> None:
    self._cfg = LeadResponseTuningConfig.defaults()
    self._tracks: dict[int, ModelLeadTrack] = {}
    self._updated: set[int] = set()
    self._next_id = MODEL_LEAD_TRACK_ID_START

  def _new_id(self) -> int:
    i = self._next_id
    self._next_id -= 1
    return i

  def _assoc(self, track: ModelLeadTrack, ld: dict, now: float, slot: int) -> float | None:
    raw_drel = float(ld["dRel"])
    raw_dpath = float(ld.get("dPath", ld["yRel"]))
    raw_yrel = float(ld["yRel"])
    raw_vrel = float(ld["vRel"])
    pred_drel = track.predict_drel(now)
    same_slot_gate = 35.0 if track.last_slot == slot else float(self._cfg.model_lead_filter_assoc_drel_m)
    drel_gate = max(same_slot_gate, 0.18 * max(pred_drel, raw_drel, 1.0))
    drel_err = abs(pred_drel - raw_drel)
    path_err = abs(track.dPath - raw_dpath)
    y_err = abs(track.yRel - raw_yrel)
    vrel_err = abs(track.vRel - raw_vrel)
    same_frame_duplicate = (
      track.identifier in self._updated and
      track.last_slot != slot and
      path_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M and
      y_err <= MODEL_LEAD_DUPLICATE_PATH_GATE_M and
      vrel_err <= MODEL_LEAD_DUPLICATE_VREL_GATE_MPS
    )
    closer_safety_candidate = (
      raw_drel < (pred_drel - MODEL_LEAD_DUPLICATE_CLOSER_KEEP_SEPARATE_M) and
      raw_vrel < -MODEL_LEAD_DUPLICATE_CLOSING_KEEP_SEPARATE_MPS
    )
    if same_frame_duplicate and not closer_safety_candidate:
      drel_gate = max(drel_gate, MODEL_LEAD_DUPLICATE_DREL_GATE_M)
    if drel_err > drel_gate or path_err > MODEL_LEAD_ASSOC_Y_GATE_M or y_err > MODEL_LEAD_ASSOC_Y_GATE_M:
      return None
    if vrel_err > MODEL_LEAD_ASSOC_VREL_GATE_MPS:
      return None
    updated_penalty = 0.25 if track.identifier in self._updated else 0.0
    return float(
      drel_err / max(drel_gate, 1e-3) +
      path_err / MODEL_LEAD_ASSOC_Y_GATE_M +
      y_err / MODEL_LEAD_ASSOC_Y_GATE_M +
      vrel_err / MODEL_LEAD_ASSOC_VREL_GATE_MPS +
      updated_penalty
    )

  def _match(self, ld: dict, now: float, slot: int) -> ModelLeadTrack | None:
    best: tuple[float, ModelLeadTrack] | None = None
    for track in self._tracks.values():
      score = self._assoc(track, ld, now, slot)
      if score is None:
        continue
      if best is None or score < best[0]:
        best = (score, track)
    if best is not None:
      return best[1]
    raw_drel = float(ld["dRel"])
    raw_dpath = float(ld.get("dPath", ld["yRel"]))
    raw_yrel = float(ld["yRel"])
    raw_vrel = float(ld["vRel"])
    same_slot_best: tuple[float, ModelLeadTrack] | None = None
    for track in self._tracks.values():
      if track.last_slot != slot:
        continue
      pred_drel = track.predict_drel(now)
      drel_err = abs(pred_drel - raw_drel)
      path_err = abs(track.dPath - raw_dpath)
      y_err = abs(track.yRel - raw_yrel)
      vrel_err = abs(track.vRel - raw_vrel)
      if (
        drel_err > MODEL_LEAD_SAME_SLOT_RECOVER_DREL_GATE_M or
        path_err > MODEL_LEAD_ASSOC_Y_GATE_M or
        y_err > MODEL_LEAD_ASSOC_Y_GATE_M or
        vrel_err > MODEL_LEAD_ASSOC_VREL_GATE_MPS
      ):
        continue
      score = drel_err + 4.0 * path_err + 4.0 * y_err + vrel_err
      if same_slot_best is None or score < same_slot_best[0]:
        same_slot_best = (score, track)
    return None if same_slot_best is None else same_slot_best[1]

  def begin_frame(self) -> None:
    self._updated = set()

  def end_frame(self) -> None:
    for identifier, track in list(self._tracks.items()):
      if identifier not in self._updated:
        track.missed += 1
      if track.missed > MODEL_LEAD_TRACK_MAX_MISSES:
        self._tracks.pop(identifier, None)

  def update_from_vision(self, ld: dict, now: float, v_ego: float, slot: int) -> dict:
    track = self._match(ld, now, slot)
    if track is None:
      track = ModelLeadTrack.from_lead_dict(self._new_id(), ld, now, slot)
      self._tracks[track.identifier] = track
      while len(self._tracks) > MODEL_LEAD_TRACK_MAX_COUNT:
        stale = max(self._tracks.values(), key=lambda t: (t.missed, -t.age)).identifier
        self._tracks.pop(stale, None)
    if track.identifier in self._updated:
      return track.get_RadarState(self._cfg)
    self._updated.add(track.identifier)
    return track.update(ld, now, v_ego, self._cfg, slot)


def _drive(tracker, *, legacy: bool) -> dict:
  """Drive the recorded two-slot raw-model series through a tracker exactly as
  RadarD.update does (both slots -> one tracker/frame, leadOne = slot 0)."""
  frames = _frames()
  lead_one_ids: list[int] = []
  lead_one_drels: list[float] = []
  raw_yrel_slot0: list[float] = []
  raw_dpath_slot0: list[float] = []

  for frame in frames:
    if legacy:
      tracker.begin_frame()
    else:
      tracker.begin_frame(frame["t_s"])
    lead_one: dict | None = None
    for slot, lead_dict in enumerate(_lead_dicts(frame)):
      if legacy:
        out = tracker.update_from_vision(lead_dict, frame["t_s"], frame["v_ego"], slot)
      else:
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


@lru_cache(maxsize=1)
def _replay() -> dict:
  """The REAL, current ModelLeadTracker over the fixture (the behavioral oracle)."""
  return _drive(ModelLeadTracker(Params()), legacy=False)


@lru_cache(maxsize=1)
def _replay_legacy() -> dict:
  """The pre-CD4 gate (faithful reconstruction) over the same fixture -- the
  fix-independent scenario-validity witness that the legacy gate churns."""
  return _drive(_LegacyModelLeadTracker(), legacy=True)


def test_scenario_validity_gate_rejects_continuity_on_yrel_but_not_dpath() -> None:
  """Prove this is a genuine CD4 repro: the raw yRel excursion exceeds the
  association gate while the path-relative dPath stays lane-relevant, and the
  pre-fix (legacy) gate actually churns the published track id in response."""
  r = _replay_legacy()

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
  # The LEGACY gate rejects continuity: the published track id churns.
  assert r["churn_events"] > 0, (
    "scenario does not reproduce CD4 track-id churn; the legacy association gate "
    "must reject continuity on the raw-yRel excursion for the behavioral oracle to bite"
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
