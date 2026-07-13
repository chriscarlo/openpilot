from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class LeadDirective:
  status: bool = False
  v_lead_mps: float = 0.0
  model_prob_target: float = 0.0
  d_rel_override_m: float | None = None
  measured_d_rel_m: float | None = None
  # Additive raw-measurement distance bias (m): models the vision model's
  # far-range x optimism against ground truth (road-measured on 200-13 EDGE1:
  # raw leadsV3 x ran +5.8..+7.3 m above the true gap through the deep close).
  measured_d_rel_bias_m: float = 0.0
  # Additive raw-measurement velocity bias (m/s): models the vision model's
  # far-range stopped-traffic OPTIMISM against ground truth (road-measured on
  # 200-13 EDGE2: while a far, newly-acquired stopped/slow lead is still
  # stopping, the published vLead ran ~+4 m/s high vs position-derived truth
  # during the 43-48 s window, decaying to truth as the gap closes). Positive =
  # published vLead runs HIGH (lead looks faster/less-urgent than it is). Flows
  # into the raw measured vRel -> leadsV3.v (radard_stage.py _fill_lead_v3
  # entry.v = model_v_ego + vRel) -> the REAL radard ModelLeadTracker vRel/vLead
  # EMA the MPC extrapolates with, exactly as measured_d_rel_bias_m does for x.
  v_lead_bias_mps: float = 0.0
  measured_v_rel_mps: float | None = None
  a_lead_k_mps2: float | None = None
  v_lead_k_mps: float | None = None
  y_rel_m: float = 0.0
  d_path_m: float | None = None
  v_lat_mps: float = 0.0
  fcw: bool = False
  radar: bool = False
  radar_track_id: int = -1
  acquisition_reset: bool = False
  # Per-step override of the vehicle config's aLeadTau (EV6 runtime publishes 0.3).
  a_lead_tau_s: float | None = None

  @classmethod
  def from_json(cls, payload: dict[str, Any] | None) -> LeadDirective:
    if not payload:
      return cls()
    return cls(**payload)

  def to_json(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class StepInput:
  t_s: float
  cruise_speed_mps: float
  lead_one: LeadDirective = field(default_factory=LeadDirective)
  lead_two: LeadDirective = field(default_factory=LeadDirective)
  event: str | None = None
  note: str = ""
  pitch_rad: float = 0.0
  force_decel: bool = False
  experimental_mode: bool = False

  @classmethod
  def from_json(cls, payload: dict[str, Any]) -> StepInput:
    return cls(
      t_s=float(payload["t"]),
      cruise_speed_mps=float(payload["cruiseSpeedMps"]),
      lead_one=LeadDirective.from_json(payload.get("leadOne")),
      lead_two=LeadDirective.from_json(payload.get("leadTwo")),
      event=payload.get("event"),
      note=str(payload.get("note", "")),
      pitch_rad=float(payload.get("pitchRad", 0.0)),
      force_decel=bool(payload.get("forceDecel", False)),
      experimental_mode=bool(payload.get("experimentalMode", False)),
    )

  def to_json(self) -> dict[str, Any]:
    return {
      "t": self.t_s,
      "cruiseSpeedMps": self.cruise_speed_mps,
      "leadOne": self.lead_one.to_json(),
      "leadTwo": self.lead_two.to_json(),
      "event": self.event,
      "note": self.note,
      "pitchRad": self.pitch_rad,
      "forceDecel": self.force_decel,
      "experimentalMode": self.experimental_mode,
    }


@dataclass
class SnapshotBundle:
  path: Path
  vehicle: dict[str, Any]
  params: dict[str, Any]
  timeline: list[StepInput]
  initial_speed_mps: float
  initial_accel_mps2: float = 0.0
  name: str = ""


BASE_SCENARIO_NAMES = (
  "approach",
  "slower_lead_acquisition",
  "decelerating_lead",
  "pullaway",
  "pullaway_close",
  "cutin",
  "dangerous_cutin",
  "multi_cutin",
  "handoff",
  "handoff_previewable",
  "handoff_previewable_early_deficit",
  "handoff_previewable_cruise_release",
  "duplicate_pair",
  "dropout",
  "cruise_lead_handoff",
  "far_cruise_slow_lead",
  "oscillating",
  "random_speed_wander",
  "goldilocks_speed_wander",
  "accordion_close",
)

CANONICAL_LEAD_PROFILE_NAMES = (
  "profile_steady_goldilocks",
  "profile_slow_lead_acquisition",
  "profile_gentle_pullaway",
  "profile_confirmed_pullaway",
  "profile_decelerating_lead",
  "profile_high_ttc_slowdown",
  "profile_emergency_ttc",
  "profile_stoplight_launch",
  "profile_varying_speed",
  "profile_random_speed_wander",
  "profile_goldilocks_speed_wander",
  "profile_benign_cutin",
  "profile_dangerous_cutin",
  "profile_duplicate_dropout",
)

SCENARIO_NAMES = BASE_SCENARIO_NAMES + CANONICAL_LEAD_PROFILE_NAMES


def _scenario_step_count(duration_s: float, dt_s: float) -> int:
  return max(1, int(round(duration_s / dt_s)))


_MPH_TO_MPS = 0.44704
_RANDOM_SPEED_WANDER_ANCHORS_MPH = (
  65.0,
  70.5,
  64.8,
  58.7,
  55.2,
  62.1,
  68.4,
  74.8,
  69.5,
  72.2,
  65.3,
  57.6,
  60.8,
)
_GOLDILOCKS_GAP_WANDER_ANCHORS_M = (
  44.0,
  43.2,
  45.1,
  44.4,
  42.8,
  43.7,
  45.3,
  44.1,
  43.4,
  44.8,
)


def _mph_to_mps(speed_mph: float) -> float:
  return speed_mph * _MPH_TO_MPS


def _smoothstep(progress: float) -> float:
  clamped = min(max(progress, 0.0), 1.0)
  return clamped * clamped * (3.0 - (2.0 * clamped))


def _smooth_profile_value(t_s: float, anchors: tuple[float, ...], interval_s: float) -> float:
  if len(anchors) == 1:
    return anchors[0]
  segment_idx = min(int(t_s // interval_s), len(anchors) - 2)
  segment_start_t = segment_idx * interval_s
  progress = _smoothstep((t_s - segment_start_t) / interval_s)
  return anchors[segment_idx] + ((anchors[segment_idx + 1] - anchors[segment_idx]) * progress)


def build_synthetic_scenario(name: str, *, duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  name = name.lower()
  if name == "approach":
    return _build_approach(duration_s, dt_s)
  if name == "slower_lead_acquisition":
    return _build_slower_lead_acquisition(duration_s, dt_s)
  if name == "decelerating_lead":
    return _build_decelerating_lead(duration_s, dt_s)
  if name == "pullaway":
    return _build_pullaway(duration_s, dt_s)
  if name == "pullaway_close":
    return _build_pullaway_close(duration_s, dt_s)
  if name == "cutin":
    return _build_cutin(duration_s, dt_s, dangerous=False)
  if name == "dangerous_cutin":
    return _build_cutin(duration_s, dt_s, dangerous=True)
  if name == "multi_cutin":
    return _build_multi_cutin(duration_s, dt_s)
  if name == "handoff":
    return _build_handoff(duration_s, dt_s)
  if name == "handoff_previewable":
    return _build_handoff_previewable(duration_s, dt_s)
  if name == "handoff_previewable_early_deficit":
    return _build_handoff_previewable_early_deficit(duration_s, dt_s)
  if name == "handoff_previewable_cruise_release":
    return _build_handoff_previewable_cruise_release(duration_s, dt_s)
  if name == "duplicate_pair":
    return _build_duplicate_pair(duration_s, dt_s)
  if name == "dropout":
    return _build_dropout(duration_s, dt_s)
  if name == "cruise_lead_handoff":
    return _build_cruise_lead_handoff(duration_s, dt_s)
  if name == "far_cruise_slow_lead":
    return _build_far_cruise_slow_lead(duration_s, dt_s)
  if name == "oscillating":
    return _build_oscillating(duration_s, dt_s)
  if name == "random_speed_wander":
    return _build_random_speed_wander(duration_s, dt_s)
  if name == "goldilocks_speed_wander":
    return _build_goldilocks_speed_wander(duration_s, dt_s)
  if name == "accordion_close":
    return _build_accordion_close(duration_s, dt_s)
  if name == "profile_steady_goldilocks":
    return _build_profile_steady_goldilocks(duration_s, dt_s)
  if name == "profile_slow_lead_acquisition":
    return _build_slower_lead_acquisition(duration_s, dt_s, canonical=True)
  if name == "profile_gentle_pullaway":
    return _build_profile_pullaway(duration_s, dt_s, confirmed=False)
  if name == "profile_confirmed_pullaway":
    return _build_profile_pullaway(duration_s, dt_s, confirmed=True)
  if name == "profile_decelerating_lead":
    return _build_decelerating_lead(duration_s, dt_s, canonical=True)
  if name == "profile_high_ttc_slowdown":
    return _build_profile_ttc_slowdown(duration_s, dt_s, emergency=False)
  if name == "profile_emergency_ttc":
    return _build_profile_ttc_slowdown(duration_s, dt_s, emergency=True)
  if name == "profile_stoplight_launch":
    return _build_profile_stoplight_launch(duration_s, dt_s)
  if name == "profile_varying_speed":
    return _build_profile_varying_speed(duration_s, dt_s)
  if name == "profile_random_speed_wander":
    return _build_random_speed_wander(duration_s, dt_s, canonical=True)
  if name == "profile_goldilocks_speed_wander":
    return _build_goldilocks_speed_wander(duration_s, dt_s, canonical=True)
  if name == "profile_benign_cutin":
    return _build_cutin(duration_s, dt_s, dangerous=False)
  if name == "profile_dangerous_cutin":
    return _build_cutin(duration_s, dt_s, dangerous=True)
  if name == "profile_duplicate_dropout":
    return _build_profile_duplicate_dropout(duration_s, dt_s)
  raise ValueError(f"unknown synthetic scenario '{name}'")


def load_snapshot_bundle(bundle_dir: str | Path) -> SnapshotBundle:
  root = Path(bundle_dir)
  vehicle = json.loads(root.joinpath("vehicle.json").read_text())
  params = json.loads(root.joinpath("params.json").read_text())
  timeline = [
    StepInput.from_json(json.loads(line))
    for line in root.joinpath("timeline.jsonl").read_text().splitlines()
    if line.strip()
  ]

  return SnapshotBundle(
    path=root,
    vehicle=vehicle,
    params=params,
    timeline=timeline,
    initial_speed_mps=float(vehicle.get("initialSpeedMps", 0.0)),
    initial_accel_mps2=float(vehicle.get("initialAccelMps2", 0.0)),
    name=str(vehicle.get("name", root.name)),
  )


def write_snapshot_bundle(bundle: SnapshotBundle, out_dir: str | Path) -> None:
  root = Path(out_dir)
  root.mkdir(parents=True, exist_ok=True)
  vehicle_payload = dict(bundle.vehicle)
  vehicle_payload.setdefault("initialSpeedMps", bundle.initial_speed_mps)
  vehicle_payload.setdefault("initialAccelMps2", bundle.initial_accel_mps2)
  root.joinpath("vehicle.json").write_text(json.dumps(vehicle_payload, indent=2, sort_keys=True))
  root.joinpath("params.json").write_text(json.dumps(bundle.params, indent=2, sort_keys=True))
  timeline_lines = [json.dumps(step.to_json(), sort_keys=True) for step in bundle.timeline]
  root.joinpath("timeline.jsonl").write_text("\n".join(timeline_lines) + ("\n" if timeline_lines else ""))


def _build_approach(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 35.0
  lead_speed = 20.0
  gap_m = 120.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_one = LeadDirective(status=True, v_lead_mps=lead_speed, model_prob_target=1.0,
                             d_rel_override_m=gap_m if idx == 0 else None,
                             acquisition_reset=idx == 0)
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, note="steady slower lead"))
  return initial_speed, 0.0, timeline


def _build_slower_lead_acquisition(duration_s: float, dt_s: float, *, canonical: bool = False) -> tuple[float, float, list[StepInput]]:
  initial_speed = _mph_to_mps(67.0)
  lead_speed = _mph_to_mps(47.0)
  reveal_t_s = 1.5
  reveal_gap_m = 60.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    reveal_now = abs(t_s - reveal_t_s) < (dt_s * 0.5)
    if t_s >= reveal_t_s:
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=lead_speed,
        model_prob_target=1.0,
        d_rel_override_m=reveal_gap_m if reveal_now else None,
        acquisition_reset=reveal_now,
      )
      event = "lead_reveal" if reveal_now else None
      note = "canonical slower lead acquisition" if canonical else "new lead appears 20 mph slower than ego"
    else:
      lead_one = LeadDirective()
      event = None
      note = "cruise before slower lead acquisition"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=_mph_to_mps(78.0), lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_decelerating_lead(duration_s: float, dt_s: float, *, canonical: bool = False) -> tuple[float, float, list[StepInput]]:
  initial_speed = _mph_to_mps(67.0)
  lead_base_speed = initial_speed
  lead_speed_drop = _mph_to_mps(16.0)
  decel_start_t = 1.5
  decel_duration_s = 7.0
  gap_m = 48.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < decel_start_t:
      lead_speed = lead_base_speed
      event = None
    elif t_s < decel_start_t + decel_duration_s:
      progress = _smoothstep((t_s - decel_start_t) / decel_duration_s)
      lead_speed = lead_base_speed - (lead_speed_drop * progress)
      event = "lead_decel_start" if abs(t_s - decel_start_t) < (dt_s * 0.5) else None
    else:
      lead_speed = lead_base_speed - lead_speed_drop
      event = None

    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=0.98 if canonical else 1.0,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    note = "canonical smooth lead deceleration" if canonical else "lead smoothly decelerates from freeway speed"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=_mph_to_mps(78.0), lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_pullaway(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 25.0
  base_lead_speed = 25.0
  gap_m = 40.0
  pull_start_t = 2.0
  pull_dur_s = 4.0
  pull_dv = 8.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < pull_start_t:
      lead_speed = base_lead_speed
      event = None
    elif t_s < pull_start_t + pull_dur_s:
      lead_speed = base_lead_speed + pull_dv * ((t_s - pull_start_t) / pull_dur_s)
      event = "pullaway_start" if abs(t_s - pull_start_t) < (dt_s * 0.5) else None
    else:
      lead_speed = base_lead_speed + pull_dv
      event = None

    lead_one = LeadDirective(status=True, v_lead_mps=lead_speed, model_prob_target=1.0,
                             d_rel_override_m=gap_m if idx == 0 else None,
                             acquisition_reset=idx == 0)
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=36.0, lead_one=lead_one, event=event, note="lead pullaway"))
  return initial_speed, 0.0, timeline


def _build_pullaway_close(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 8.0
  base_lead_speed = 8.0
  gap_m = 12.0
  pull_start_t = 2.0
  pull_dur_s = 3.0
  pull_dv = 5.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < pull_start_t:
      lead_speed = base_lead_speed
      event = None
    elif t_s < pull_start_t + pull_dur_s:
      lead_speed = base_lead_speed + pull_dv * ((t_s - pull_start_t) / pull_dur_s)
      event = "pullaway_start" if abs(t_s - pull_start_t) < (dt_s * 0.5) else None
    else:
      lead_speed = base_lead_speed + pull_dv
      event = None

    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=1.0,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=16.0,
      lead_one=lead_one,
      event=event,
      note="close-follow pullaway",
    ))
  return initial_speed, 0.0, timeline


def _build_cutin(duration_s: float, dt_s: float, *, dangerous: bool) -> tuple[float, float, list[StepInput]]:
  initial_speed = 30.0
  reveal_t = 3.0
  gap_m = 18.0 if dangerous else 30.0
  lead_speed = 20.0 if dangerous else 22.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < reveal_t:
      lead_one = LeadDirective()
      event = None
      note = "no lead"
    else:
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=lead_speed,
        model_prob_target=1.0,
        d_rel_override_m=gap_m if abs(t_s - reveal_t) < (dt_s * 0.5) else None,
        acquisition_reset=abs(t_s - reveal_t) < (dt_s * 0.5),
      )
      event = "lead_reveal" if abs(t_s - reveal_t) < (dt_s * 0.5) else None
      note = "dangerous cut-in" if dangerous else "benign cut-in"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=35.0, lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_multi_cutin(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 27.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    event = None
    note = "lead A follow"
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=27.0,
      model_prob_target=0.98,
      d_rel_override_m=30.0 if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    lead_two = LeadDirective()

    if 2.0 <= t_s < 3.8:
      note = "cut-in B"
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=27.0,
        model_prob_target=0.90,
        y_rel_m=1.3,
        d_path_m=1.3,
        v_lat_mps=0.8,
      )
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=24.0,
        model_prob_target=1.0,
        d_rel_override_m=23.0 if abs(t_s - 2.0) < (dt_s * 0.5) else None,
        y_rel_m=0.2,
        d_path_m=0.2,
        v_lat_mps=-0.6,
        acquisition_reset=abs(t_s - 2.0) < (dt_s * 0.5),
      )
      event = "lead_reveal" if abs(t_s - 2.0) < (dt_s * 0.5) else None
    elif 4.2 <= t_s < 6.2:
      note = "cut-in C"
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=24.0,
        model_prob_target=0.88,
        y_rel_m=-1.2,
        d_path_m=-1.2,
        v_lat_mps=-0.6,
      )
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=22.0,
        model_prob_target=1.0,
        d_rel_override_m=18.5 if abs(t_s - 4.2) < (dt_s * 0.5) else None,
        y_rel_m=0.15,
        d_path_m=0.15,
        v_lat_mps=0.7,
        acquisition_reset=abs(t_s - 4.2) < (dt_s * 0.5),
      )
      event = "lead_reveal" if abs(t_s - 4.2) < (dt_s * 0.5) else None

    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=33.0,
      lead_one=lead_one,
      lead_two=lead_two,
      event=event,
      note=note,
    ))
  return initial_speed, 0.0, timeline


def _build_handoff(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 35.0
  handoff_t = 3.0
  overlap_s = 0.5
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < handoff_t:
      lead_one = LeadDirective(status=True, v_lead_mps=32.0, model_prob_target=1.0,
                               d_rel_override_m=80.0 if idx == 0 else None,
                               acquisition_reset=idx == 0)
      lead_two = LeadDirective()
      event = None
      note = "lead A"
    elif t_s < handoff_t + overlap_s:
      lead_one = LeadDirective(status=True, v_lead_mps=32.0, model_prob_target=1.0, y_rel_m=1.6, d_path_m=1.6, v_lat_mps=0.7)
      lead_two = LeadDirective(status=True, v_lead_mps=22.0, model_prob_target=1.0,
                               d_rel_override_m=50.0 if abs(t_s - handoff_t) < (dt_s * 0.5) else None,
                               acquisition_reset=abs(t_s - handoff_t) < (dt_s * 0.5))
      event = "handoff_reveal" if abs(t_s - handoff_t) < (dt_s * 0.5) else None
      note = "lead A exits, lead B reveals"
    else:
      lead_one = LeadDirective()
      lead_two = LeadDirective(status=True, v_lead_mps=22.0, model_prob_target=1.0)
      event = None
      note = "following lead B"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, lead_two=lead_two, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_handoff_previewable(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 35.0
  lead_one_track_id = -1101
  lead_two_track_id = -1102
  # Keep lead A genuinely MPC-owning through the adjacent preview after the
  # real radard filter. At the old 80 m start the least-cost source was cruise,
  # so the fixture no longer exercised a lead0 -> lead1 handoff at all.
  lead_one_start_gap_m = 60.0
  preview_start_t = 1.7
  handoff_t = 3.0
  overlap_s = 0.5
  preview_duration_s = max(handoff_t - preview_start_t, dt_s)
  lead_two_start_gap_m = 76.0
  lead_two_start_path_m = 2.45
  lead_two_end_path_m = 1.8
  lead_two_v_lat_mps = -((lead_two_start_path_m - lead_two_end_path_m) / preview_duration_s)
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < preview_start_t:
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        d_rel_override_m=lead_one_start_gap_m if idx == 0 else None,
        acquisition_reset=idx == 0,
      )
      lead_two = LeadDirective()
      event = None
      note = "lead A only"
    elif t_s < handoff_t:
      progress = min(max((t_s - preview_start_t) / preview_duration_s, 0.0), 1.0)
      lead_two_path_m = lead_two_start_path_m + (lead_two_end_path_m - lead_two_start_path_m) * progress
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        y_rel_m=0.20 * progress,
        d_path_m=0.20 * progress,
        v_lat_mps=0.10,
      )
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=22.0,
        model_prob_target=0.80,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=lead_two_start_gap_m if abs(t_s - preview_start_t) < (dt_s * 0.5) else None,
        y_rel_m=lead_two_path_m,
        d_path_m=lead_two_path_m,
        v_lat_mps=lead_two_v_lat_mps,
        acquisition_reset=abs(t_s - preview_start_t) < (dt_s * 0.5),
      )
      event = None
      note = "lead B adjacent awareness preview"
    elif t_s < handoff_t + overlap_s:
      lead_one = LeadDirective(status=True, v_lead_mps=32.0, model_prob_target=0.98,
                               y_rel_m=1.6, d_path_m=1.6, v_lat_mps=0.7,
                               radar_track_id=lead_one_track_id)
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=22.0,
        model_prob_target=1.0,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=50.0 if abs(t_s - handoff_t) < (dt_s * 0.5) else None,
        y_rel_m=0.15,
        d_path_m=0.15,
        v_lat_mps=-0.2,
      )
      event = "handoff_reveal" if abs(t_s - handoff_t) < (dt_s * 0.5) else None
      note = "lead A exits, lead B handoff"
    else:
      lead_one = LeadDirective()
      lead_two = LeadDirective(status=True, v_lead_mps=22.0, model_prob_target=1.0,
                               radar_track_id=lead_two_track_id)
      event = None
      note = "following lead B"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, lead_two=lead_two, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_handoff_previewable_early_deficit(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 35.0
  lead_one_track_id = -1201
  lead_two_track_id = -1202
  lead_one_start_gap_m = 60.0
  preview_start_t = 1.0
  handoff_t = 3.0
  overlap_s = 0.5
  preview_duration_s = max(handoff_t - preview_start_t, dt_s)
  lead_two_start_gap_m = 70.0
  lead_two_start_path_m = 2.25
  lead_two_end_path_m = 1.55
  lead_two_v_lat_mps = -((lead_two_start_path_m - lead_two_end_path_m) / preview_duration_s)
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < preview_start_t:
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        d_rel_override_m=lead_one_start_gap_m if idx == 0 else None,
        acquisition_reset=idx == 0,
      )
      lead_two = LeadDirective()
      event = None
      note = "lead A only"
    elif t_s < handoff_t:
      progress = min(max((t_s - preview_start_t) / preview_duration_s, 0.0), 1.0)
      lead_two_path_m = lead_two_start_path_m + (lead_two_end_path_m - lead_two_start_path_m) * progress
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        y_rel_m=0.25 * progress,
        d_path_m=0.25 * progress,
        v_lat_mps=0.12,
      )
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=20.0,
        model_prob_target=0.84,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=lead_two_start_gap_m if abs(t_s - preview_start_t) < (dt_s * 0.5) else None,
        y_rel_m=lead_two_path_m,
        d_path_m=lead_two_path_m,
        v_lat_mps=lead_two_v_lat_mps,
        acquisition_reset=abs(t_s - preview_start_t) < (dt_s * 0.5),
      )
      event = None
      note = "lead B earlier adjacent deficit preview"
    elif t_s < handoff_t + overlap_s:
      lead_one = LeadDirective(status=True, v_lead_mps=32.0, model_prob_target=0.98,
                               y_rel_m=1.6, d_path_m=1.6, v_lat_mps=0.7,
                               radar_track_id=lead_one_track_id)
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=20.0,
        model_prob_target=1.0,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=44.0 if abs(t_s - handoff_t) < (dt_s * 0.5) else None,
        y_rel_m=0.12,
        d_path_m=0.12,
        v_lat_mps=-0.25,
      )
      event = "handoff_reveal" if abs(t_s - handoff_t) < (dt_s * 0.5) else None
      note = "lead A exits, lead B earlier deficit handoff"
    else:
      lead_one = LeadDirective()
      lead_two = LeadDirective(status=True, v_lead_mps=20.0, model_prob_target=1.0,
                               radar_track_id=lead_two_track_id)
      event = None
      note = "following earlier deficit lead B"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, lead_two=lead_two, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_handoff_previewable_cruise_release(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 35.0
  lead_one_track_id = -1301
  lead_two_track_id = -1302
  lead_one_start_gap_m = 60.0
  preview_start_t = 1.0
  late_release_t = 2.0
  handoff_t = 3.0
  overlap_s = 0.5
  preview_duration_s = max(handoff_t - preview_start_t, dt_s)
  lead_two_start_gap_m = 70.0
  lead_two_start_path_m = 2.25
  lead_two_end_path_m = 1.55
  lead_two_v_lat_mps = -((lead_two_start_path_m - lead_two_end_path_m) / preview_duration_s)
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < preview_start_t:
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        d_rel_override_m=lead_one_start_gap_m if idx == 0 else None,
        acquisition_reset=idx == 0,
      )
      lead_two = LeadDirective()
      event = None
      note = "lead A only"
    elif t_s < handoff_t:
      progress = min(max((t_s - preview_start_t) / preview_duration_s, 0.0), 1.0)
      lead_two_path_m = lead_two_start_path_m + (lead_two_end_path_m - lead_two_start_path_m) * progress
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=33.5 if t_s >= late_release_t else 32.0,
        model_prob_target=1.0,
        radar_track_id=lead_one_track_id,
        d_rel_override_m=lead_one_start_gap_m if abs(t_s - late_release_t) < (dt_s * 0.5) else None,
        y_rel_m=0.25 * progress,
        d_path_m=0.25 * progress,
        v_lat_mps=0.12,
        acquisition_reset=abs(t_s - late_release_t) < (dt_s * 0.5),
      )
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=20.0,
        model_prob_target=0.84,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=lead_two_start_gap_m if abs(t_s - preview_start_t) < (dt_s * 0.5) else None,
        y_rel_m=lead_two_path_m,
        d_path_m=lead_two_path_m,
        v_lat_mps=lead_two_v_lat_mps,
        acquisition_reset=abs(t_s - preview_start_t) < (dt_s * 0.5),
      )
      event = None
      note = "lead B preview, lead A late pullaway release"
    elif t_s < handoff_t + overlap_s:
      lead_one = LeadDirective(status=True, v_lead_mps=33.5, model_prob_target=0.98,
                               y_rel_m=1.6, d_path_m=1.6, v_lat_mps=0.7,
                               radar_track_id=lead_one_track_id)
      lead_two = LeadDirective(
        status=True,
        v_lead_mps=20.0,
        model_prob_target=1.0,
        radar_track_id=lead_two_track_id,
        d_rel_override_m=44.0 if abs(t_s - handoff_t) < (dt_s * 0.5) else None,
        y_rel_m=0.12,
        d_path_m=0.12,
        v_lat_mps=-0.25,
      )
      event = "handoff_reveal" if abs(t_s - handoff_t) < (dt_s * 0.5) else None
      note = "lead A exits after late cruise release, lead B handoff"
    else:
      lead_one = LeadDirective()
      lead_two = LeadDirective(status=True, v_lead_mps=20.0, model_prob_target=1.0,
                               radar_track_id=lead_two_track_id)
      event = None
      note = "following release-triggered lead B"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, lead_two=lead_two, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_duplicate_pair(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 29.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    jitter = (-0.12 if idx % 2 else 0.08)
    lead_one = LeadDirective(status=True, v_lead_mps=29.0, model_prob_target=0.96,
                             d_rel_override_m=44.8 + jitter if idx == 0 else None,
                             y_rel_m=0.04, d_path_m=0.04, v_lat_mps=0.4,
                             acquisition_reset=idx == 0)
    lead_two = LeadDirective(status=True, v_lead_mps=29.0, model_prob_target=0.92,
                             d_rel_override_m=44.9 - jitter if idx == 0 else None,
                             y_rel_m=0.07, d_path_m=0.07, v_lat_mps=4.0,
                             acquisition_reset=idx == 0)
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=31.0, lead_one=lead_one, lead_two=lead_two, note="duplicate pair jitter"))
  return initial_speed, 0.0, timeline


def _build_dropout(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 29.0
  dropout_start = 3.0
  dropout_end = 3.3
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if dropout_start <= t_s < dropout_end:
      lead_one = LeadDirective()
      event = "dropout_start" if abs(t_s - dropout_start) < (dt_s * 0.5) else None
      note = "brief dropout"
    else:
      lead_one = LeadDirective(status=True, v_lead_mps=29.0, model_prob_target=0.96,
                               d_rel_override_m=38.5 if idx == 0 else (38.5 if abs(t_s - dropout_end) < (dt_s * 0.5) else None),
                               acquisition_reset=idx == 0 or abs(t_s - dropout_end) < (dt_s * 0.5))
      event = "reacquire" if abs(t_s - dropout_end) < (dt_s * 0.5) else None
      note = "stable follow"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=31.0, lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_cruise_lead_handoff(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 33.5
  reveal_t = 1.5
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < reveal_t:
      lead_one = LeadDirective()
      event = None
      note = "cruise"
    else:
      lead_one = LeadDirective(status=True, v_lead_mps=27.0, model_prob_target=1.0,
                               d_rel_override_m=90.0 if abs(t_s - reveal_t) < (dt_s * 0.5) else None,
                               acquisition_reset=abs(t_s - reveal_t) < (dt_s * 0.5))
      event = "lead_reveal" if abs(t_s - reveal_t) < (dt_s * 0.5) else None
      note = "slower lead appears"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=40.0, lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_far_cruise_slow_lead(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 27.5
  reveal_t = 0.5
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < reveal_t:
      lead_one = LeadDirective()
      event = None
      note = "cruise before model lead"
    else:
      is_reveal = abs(t_s - reveal_t) < (dt_s * 0.5)
      lead_one = LeadDirective(
        status=True,
        v_lead_mps=26.3,
        model_prob_target=0.94,
        d_rel_override_m=86.0 if is_reveal else None,
        acquisition_reset=is_reveal,
        radar=False,
        radar_track_id=-1006,
      )
      event = "lead_reveal" if is_reveal else None
      note = "far stable slower model lead"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=32.0, lead_one=lead_one, event=event, note=note))
  return initial_speed, 0.0, timeline


def _build_oscillating(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  import math

  initial_speed = 28.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = 28.0 + 4.0 * math.sin((2.0 * math.pi * t_s) / 6.0)
    lead_one = LeadDirective(status=True, v_lead_mps=lead_speed, model_prob_target=1.0,
                             d_rel_override_m=40.0 if idx == 0 else None,
                             acquisition_reset=idx == 0)
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=31.0, lead_one=lead_one, note="oscillating lead"))
  return initial_speed, 0.0, timeline


def _build_random_speed_wander(duration_s: float, dt_s: float, *, canonical: bool = False) -> tuple[float, float, list[StepInput]]:
  anchors_mps = tuple(_mph_to_mps(speed_mph) for speed_mph in _RANDOM_SPEED_WANDER_ANCHORS_MPH)
  initial_speed = _mph_to_mps(65.0)
  gap_m = 52.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = _smooth_profile_value(t_s, anchors_mps, interval_s=7.0)
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=0.98 if canonical else 1.0,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    note = "canonical random-like freeway speed wander" if canonical else "random-like lead speed wander between 55 and 75 mph"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=_mph_to_mps(78.0), lead_one=lead_one, note=note))
  return initial_speed, 0.0, timeline


def _build_goldilocks_speed_wander(duration_s: float, dt_s: float, *, canonical: bool = False) -> tuple[float, float, list[StepInput]]:
  anchors_mps = tuple(_mph_to_mps(speed_mph) for speed_mph in _RANDOM_SPEED_WANDER_ANCHORS_MPH)
  initial_speed = _mph_to_mps(65.0)
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = _smooth_profile_value(t_s, anchors_mps, interval_s=7.0)
    gap_m = _smooth_profile_value(t_s, _GOLDILOCKS_GAP_WANDER_ANCHORS_M, interval_s=5.0)
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=0.98 if canonical else 1.0,
      d_rel_override_m=gap_m,
      acquisition_reset=idx == 0,
    )
    note = "canonical lead-owned goldilocks speed wander" if canonical else "lead-owned goldilocks speed wander"
    timeline.append(StepInput(t_s=t_s, cruise_speed_mps=_mph_to_mps(78.0), lead_one=lead_one, note=note))
  return initial_speed, 0.0, timeline


def _build_accordion_close(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  import math

  initial_speed = 20.0
  base_lead_speed = 20.0
  gap_m = 14.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = base_lead_speed + 3.5 * math.sin((2.0 * math.pi * t_s) / 4.5)
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=1.0,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=28.0,
      lead_one=lead_one,
      note="close-follow accordion lead",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_steady_goldilocks(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 29.0
  gap_m = 47.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=29.0,
      model_prob_target=0.96,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=idx * dt_s,
      cruise_speed_mps=31.0,
      lead_one=lead_one,
      note="canonical steady goldilocks follow",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_pullaway(duration_s: float, dt_s: float, *, confirmed: bool) -> tuple[float, float, list[StepInput]]:
  initial_speed = 29.0
  base_lead_speed = 29.0
  gap_m = 54.0 if confirmed else 46.0
  pull_start_t = 1.5 if confirmed else 2.0
  pull_dur_s = 3.5 if confirmed else 7.0
  pull_dv = 7.0 if confirmed else 2.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if t_s < pull_start_t:
      lead_speed = base_lead_speed
      event = None
    elif t_s < pull_start_t + pull_dur_s:
      progress = (t_s - pull_start_t) / pull_dur_s
      lead_speed = base_lead_speed + pull_dv * progress
      event = "pullaway_start" if abs(t_s - pull_start_t) < (dt_s * 0.5) else None
    else:
      lead_speed = base_lead_speed + pull_dv
      event = None

    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=0.98,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=40.0 if confirmed else 33.0,
      lead_one=lead_one,
      event=event,
      note="canonical confirmed pullaway" if confirmed else "canonical gentle pullaway",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_ttc_slowdown(duration_s: float, dt_s: float, *, emergency: bool) -> tuple[float, float, list[StepInput]]:
  initial_speed = 31.0
  lead_speed = 16.0 if emergency else 26.0
  gap_m = 24.0 if emergency else 70.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=1.0,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=idx * dt_s,
      cruise_speed_mps=40.0,
      lead_one=lead_one,
      event="emergency_ttc_start" if emergency and idx == 0 else ("high_ttc_start" if idx == 0 else None),
      note="canonical emergency TTC closing lead" if emergency else "canonical high TTC slower lead",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_stoplight_launch(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 0.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = min(5.0, max(0.0, (t_s - 1.0) * 2.5)) if t_s >= 1.0 else 0.0
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=1.0,
      d_rel_override_m=6.0 if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=15.0,
      lead_one=lead_one,
      event="lead_launch" if abs(t_s - 1.0) < (dt_s * 0.5) else None,
      note="canonical stopped lead launch",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_varying_speed(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  import math

  initial_speed = 28.0
  base_lead_speed = 28.0
  gap_m = 44.0
  timeline = []
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    lead_speed = base_lead_speed + 2.5 * math.sin((2.0 * math.pi * t_s) / 7.0)
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=lead_speed,
      model_prob_target=0.98,
      d_rel_override_m=gap_m if idx == 0 else None,
      acquisition_reset=idx == 0,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=34.0,
      lead_one=lead_one,
      note="canonical varying-speed lead",
    ))
  return initial_speed, 0.0, timeline


def _build_profile_duplicate_dropout(duration_s: float, dt_s: float) -> tuple[float, float, list[StepInput]]:
  initial_speed = 29.0
  timeline = []
  dropout_start = 3.0
  dropout_end = 3.35
  for idx in range(_scenario_step_count(duration_s, dt_s)):
    t_s = idx * dt_s
    if dropout_start <= t_s < dropout_end:
      timeline.append(StepInput(
        t_s=t_s,
        cruise_speed_mps=31.0,
        event="dropout_start" if abs(t_s - dropout_start) < (dt_s * 0.5) else None,
        note="canonical duplicate pair dropout",
      ))
      continue

    jitter = -0.12 if idx % 2 else 0.08
    reacquire = abs(t_s - dropout_end) < (dt_s * 0.5)
    lead_one = LeadDirective(
      status=True,
      v_lead_mps=29.0,
      model_prob_target=0.96,
      d_rel_override_m=44.8 + jitter if idx == 0 or reacquire else None,
      y_rel_m=0.04,
      d_path_m=0.04,
      v_lat_mps=0.4,
      acquisition_reset=idx == 0 or reacquire,
    )
    lead_two = LeadDirective(
      status=True,
      v_lead_mps=29.0,
      model_prob_target=0.92,
      d_rel_override_m=44.9 - jitter if idx == 0 or reacquire else None,
      y_rel_m=0.07,
      d_path_m=0.07,
      v_lat_mps=4.0,
      acquisition_reset=idx == 0 or reacquire,
    )
    timeline.append(StepInput(
      t_s=t_s,
      cruise_speed_mps=31.0,
      lead_one=lead_one,
      lead_two=lead_two,
      event="reacquire" if reacquire else None,
      note="canonical duplicate pair with brief dropout",
    ))
  return initial_speed, 0.0, timeline
