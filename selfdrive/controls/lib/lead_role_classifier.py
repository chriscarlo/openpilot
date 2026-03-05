#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

try:
  from openpilot.common.params import Params
except Exception:
  # Local test/dev fallback when openpilot runtime deps are unavailable.
  class Params:  # type: ignore[no-redef]
    def get(self, _key: str):
      return None

    def get_bool(self, _key: str) -> bool:
      raise KeyError(_key)


@dataclass
class ControlLead:
  status: bool = False
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  aRel: float = 0.0
  vLead: float = 0.0
  dPath: float = 0.0
  vLat: float = 0.0
  vLeadK: float = 0.0
  aLeadK: float = 0.0
  fcw: bool = False
  aLeadTau: float = 1.5
  modelProb: float = 0.0
  radar: bool = False
  radarTrackId: int = -1

  @classmethod
  def from_lead(cls, lead: Any) -> ControlLead:
    return cls(
      status=bool(getattr(lead, "status", False)),
      dRel=float(getattr(lead, "dRel", 0.0) or 0.0),
      yRel=float(getattr(lead, "yRel", 0.0) or 0.0),
      vRel=float(getattr(lead, "vRel", 0.0) or 0.0),
      aRel=float(getattr(lead, "aRel", 0.0) or 0.0),
      vLead=float(getattr(lead, "vLead", 0.0) or 0.0),
      dPath=float(getattr(lead, "dPath", 0.0) or 0.0),
      vLat=float(getattr(lead, "vLat", 0.0) or 0.0),
      vLeadK=float(getattr(lead, "vLeadK", 0.0) or 0.0),
      aLeadK=float(getattr(lead, "aLeadK", 0.0) or 0.0),
      fcw=bool(getattr(lead, "fcw", False)),
      aLeadTau=float(getattr(lead, "aLeadTau", 1.5) or 1.5),
      modelProb=float(getattr(lead, "modelProb", 0.0) or 0.0),
      radar=bool(getattr(lead, "radar", False)),
      radarTrackId=int(getattr(lead, "radarTrackId", -1) or -1),
    )


class LeadRoleClassifier:
  REFRESH_DT_S = 0.50
  CENTER_CONTROL = "center_control"
  ADJ_LEFT = "adjacent_awareness_left"
  ADJ_RIGHT = "adjacent_awareness_right"
  INVALID = "invalid"

  def __init__(self):
    self._params = Params()
    self._last_refresh_t = 0.0
    self._cfg = {
      "enabled": True,
      "center_y_abs_max_m": 2.2,
      "center_y_abs_min_m": 1.2,
      "center_hyst_m": 0.35,
      "cutin_drel_max_m": 55.0,
      "cutin_yrate_min_mps": 0.8,
      "low_speed_bypass_v_mps": 8.0,
      "dedupe_drel_eps_m": 2.0,
      "dedupe_yrel_eps_m": 0.6,
      "dedupe_vrel_eps_mps": 1.5,
      "debug_log_enabled": True,
    }
    # slot state keyed by lead slot index (0/1)
    self._slot_state: dict[int, dict[str, Any]] = {
      0: {"role": self.INVALID, "y_abs": None, "t": None},
      1: {"role": self.INVALID, "y_abs": None, "t": None},
    }

  @staticmethod
  def _read_float(params: Params, key: str, default: float) -> float:
    try:
      raw = params.get(key)
      if raw is None:
        return float(default)
      val = float(raw)
      return val if math.isfinite(val) else float(default)
    except Exception:
      return float(default)

  @staticmethod
  def _read_bool(params: Params, key: str, default: bool) -> bool:
    try:
      return bool(params.get_bool(key))
    except Exception:
      return bool(default)

  @staticmethod
  def _lead_valid(lead: Any) -> bool:
    try:
      if not bool(getattr(lead, "status", False)):
        return False
      d_rel = float(getattr(lead, "dRel", 0.0) or 0.0)
      y_rel = float(getattr(lead, "yRel", 0.0) or 0.0)
      v_rel = float(getattr(lead, "vRel", 0.0) or 0.0)
      return math.isfinite(d_rel) and math.isfinite(y_rel) and math.isfinite(v_rel)
    except Exception:
      return False

  def _refresh_params(self, now: float) -> None:
    if (now - self._last_refresh_t) < self.REFRESH_DT_S:
      return
    self._last_refresh_t = now

    self._cfg["enabled"] = self._read_bool(self._params, "VTSC.Expert.AdjLeadControlEnabled", self._cfg["enabled"])
    self._cfg["center_y_abs_max_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadCenterYAbsMaxM", self._cfg["center_y_abs_max_m"])
    self._cfg["center_y_abs_min_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadCenterYAbsMinM", self._cfg["center_y_abs_min_m"])
    self._cfg["center_hyst_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadCenterHystM", self._cfg["center_hyst_m"])
    self._cfg["cutin_drel_max_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadCutInDRelMaxM", self._cfg["cutin_drel_max_m"])
    self._cfg["cutin_yrate_min_mps"] = self._read_float(self._params, "VTSC.Expert.AdjLeadCutInYRateMinMps", self._cfg["cutin_yrate_min_mps"])
    self._cfg["low_speed_bypass_v_mps"] = self._read_float(self._params, "VTSC.Expert.AdjLeadLowSpeedBypassVMps", self._cfg["low_speed_bypass_v_mps"])
    self._cfg["dedupe_drel_eps_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadDedupeDRelEpsM", self._cfg["dedupe_drel_eps_m"])
    self._cfg["dedupe_yrel_eps_m"] = self._read_float(self._params, "VTSC.Expert.AdjLeadDedupeYRelEpsM", self._cfg["dedupe_yrel_eps_m"])
    self._cfg["dedupe_vrel_eps_mps"] = self._read_float(self._params, "VTSC.Expert.AdjLeadDedupeVRelEpsMps", self._cfg["dedupe_vrel_eps_mps"])
    self._cfg["debug_log_enabled"] = self._read_bool(
      self._params, "VTSC.Expert.AdjLeadDebugLogEnabled", self._cfg["debug_log_enabled"],
    )

  def _classify_slot(self, slot: int, lead: Any, v_ego: float, now: float, gate_active: bool) -> tuple[str, dict[str, Any]]:
    info: dict[str, Any] = {"reason": "-", "toward_center_mps": 0.0, "cutin_promoted": False}
    if not self._lead_valid(lead):
      self._slot_state[slot] = {"role": self.INVALID, "y_abs": None, "t": None}
      info["reason"] = "invalid_or_missing"
      return self.INVALID, info

    y_rel = float(getattr(lead, "yRel", 0.0) or 0.0)
    y_abs = abs(y_rel)
    d_rel = float(getattr(lead, "dRel", 0.0) or 0.0)

    prev = self._slot_state.get(slot, {})
    prev_role = str(prev.get("role", self.INVALID))
    prev_y_abs = prev.get("y_abs")
    prev_t = prev.get("t")

    toward_center_mps = 0.0
    if prev_y_abs is not None and prev_t is not None:
      dt = max(now - float(prev_t), 1e-3)
      toward_center_mps = max(0.0, (float(prev_y_abs) - y_abs) / dt)
    info["toward_center_mps"] = toward_center_mps

    if not gate_active:
      role = self.CENTER_CONTROL
      info["reason"] = "gate_bypassed"
    else:
      center_enter_m = self._cfg["center_y_abs_min_m"]
      center_exit_m = self._cfg["center_y_abs_max_m"] + self._cfg["center_hyst_m"]
      in_center = y_abs <= (center_exit_m if prev_role == self.CENTER_CONTROL else center_enter_m)

      cutin_ok = (
        (d_rel <= self._cfg["cutin_drel_max_m"])
        and (toward_center_mps >= self._cfg["cutin_yrate_min_mps"])
        and (y_abs <= center_exit_m)
      )
      if (not in_center) and cutin_ok:
        in_center = True
        info["cutin_promoted"] = True

      if in_center:
        role = self.CENTER_CONTROL
        info["reason"] = "center_lane"
      else:
        role = self.ADJ_LEFT if y_rel > 0.0 else self.ADJ_RIGHT
        info["reason"] = "adjacent_lane"

    self._slot_state[slot] = {"role": role, "y_abs": y_abs, "t": now}
    return role, info

  def _is_duplicate_pair(self, lead0: Any, lead1: Any) -> bool:
    if not (self._lead_valid(lead0) and self._lead_valid(lead1)):
      return False
    try:
      d_close = abs(float(lead0.dRel) - float(lead1.dRel)) <= self._cfg["dedupe_drel_eps_m"]
      y_close = abs(float(lead0.yRel) - float(lead1.yRel)) <= self._cfg["dedupe_yrel_eps_m"]
      v_close = abs(float(lead0.vRel) - float(lead1.vRel)) <= self._cfg["dedupe_vrel_eps_mps"]
      return d_close and y_close and v_close
    except Exception:
      return False

  def classify(self, v_ego: float, lead0: Any, lead1: Any, now: float | None = None) -> tuple[ControlLead, ControlLead, dict[str, Any]]:
    now = time.monotonic() if now is None else float(now)
    self._refresh_params(now)

    low_speed_bypass = float(v_ego) <= self._cfg["low_speed_bypass_v_mps"]
    gate_active = bool(self._cfg["enabled"]) and (not low_speed_bypass)

    role0, info0 = self._classify_slot(0, lead0, v_ego, now, gate_active)
    role1, info1 = self._classify_slot(1, lead1, v_ego, now, gate_active)

    duplicate_pair = self._is_duplicate_pair(lead0, lead1)
    dropped_slot = None
    if duplicate_pair:
      # Keep the nearer lead, demote the farther duplicate to awareness-only.
      d0 = float(getattr(lead0, "dRel", 1e9) or 1e9) if self._lead_valid(lead0) else 1e9
      d1 = float(getattr(lead1, "dRel", 1e9) or 1e9) if self._lead_valid(lead1) else 1e9
      dropped_slot = 1 if d0 <= d1 else 0

    control0 = ControlLead()
    control1 = ControlLead()
    if role0 == self.CENTER_CONTROL and dropped_slot != 0 and self._lead_valid(lead0):
      control0 = ControlLead.from_lead(lead0)
    if role1 == self.CENTER_CONTROL and dropped_slot != 1 and self._lead_valid(lead1):
      control1 = ControlLead.from_lead(lead1)

    awareness: list[dict[str, Any]] = []
    for idx, (lead, role) in enumerate(((lead0, role0), (lead1, role1))):
      if not self._lead_valid(lead):
        continue
      if role != self.CENTER_CONTROL or dropped_slot == idx:
        awareness.append({
          "slot": idx,
          "role": role,
          "dRel": float(getattr(lead, "dRel", 0.0) or 0.0),
          "yRel": float(getattr(lead, "yRel", 0.0) or 0.0),
          "vRel": float(getattr(lead, "vRel", 0.0) or 0.0),
          "dropped_duplicate": dropped_slot == idx,
        })

    debug = {
      "enabled": bool(self._cfg["enabled"]),
      "gate_active": bool(gate_active),
      "low_speed_bypass": bool(low_speed_bypass),
      "debug_log_enabled": bool(self._cfg["debug_log_enabled"]),
      "roles": {"lead0": role0, "lead1": role1},
      "reasons": {"lead0": info0["reason"], "lead1": info1["reason"]},
      "toward_center_mps": {
        "lead0": float(info0["toward_center_mps"]),
        "lead1": float(info1["toward_center_mps"]),
      },
      "cutin_promoted": {
        "lead0": bool(info0["cutin_promoted"]),
        "lead1": bool(info1["cutin_promoted"]),
      },
      "duplicate_pair": bool(duplicate_pair),
      "dropped_slot": dropped_slot,
      "control_status": {
        "lead0": bool(control0.status),
        "lead1": bool(control1.status),
      },
      "raw": {
        "lead0": {
          "status": bool(getattr(lead0, "status", False)),
          "dRel": float(getattr(lead0, "dRel", 0.0) or 0.0),
          "yRel": float(getattr(lead0, "yRel", 0.0) or 0.0),
          "vRel": float(getattr(lead0, "vRel", 0.0) or 0.0),
        },
        "lead1": {
          "status": bool(getattr(lead1, "status", False)),
          "dRel": float(getattr(lead1, "dRel", 0.0) or 0.0),
          "yRel": float(getattr(lead1, "yRel", 0.0) or 0.0),
          "vRel": float(getattr(lead1, "vRel", 0.0) or 0.0),
        },
      },
      "awareness": awareness,
      "cfg": {
        "center_y_abs_max_m": float(self._cfg["center_y_abs_max_m"]),
        "center_y_abs_min_m": float(self._cfg["center_y_abs_min_m"]),
        "center_hyst_m": float(self._cfg["center_hyst_m"]),
        "cutin_drel_max_m": float(self._cfg["cutin_drel_max_m"]),
        "cutin_yrate_min_mps": float(self._cfg["cutin_yrate_min_mps"]),
      },
    }
    return control0, control1, debug
