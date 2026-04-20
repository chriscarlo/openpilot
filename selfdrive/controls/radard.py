#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from cereal import messaging, log, car, custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead
    self.measured = measured   # measured or estimate

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]):
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    return prob_d * prob_y * prob_v

  track = max(tracks.values(), key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def _get_model_path_xy(model_msg: capnp._DynamicStructReader) -> tuple[np.ndarray, np.ndarray] | None:
  try:
    path_x = np.asarray(model_msg.position.x, dtype=float)
    path_y = np.asarray(model_msg.position.y, dtype=float)
  except Exception:
    return None

  if path_x.size < 2 or path_x.size != path_y.size:
    return None

  finite = np.isfinite(path_x) & np.isfinite(path_y)
  path_x = path_x[finite]
  path_y = path_y[finite]
  if path_x.size < 2:
    return None

  # Model path points are published in increasing longitudinal order; guard against
  # malformed inputs to keep interpolation stable.
  if np.any(np.diff(path_x) < 0.0):
    order = np.argsort(path_x)
    path_x = path_x[order]
    path_y = path_y[order]

  return path_x, path_y


def get_path_y_rel(model_msg: capnp._DynamicStructReader, d_rel: float) -> float:
  path_xy = _get_model_path_xy(model_msg)
  if path_xy is None or not math.isfinite(d_rel):
    return 0.0

  path_x, path_y = path_xy
  x_device = float(np.clip(d_rel + RADAR_TO_CAMERA, path_x[0], path_x[-1]))
  return float(-np.interp(x_device, path_x, path_y))


def get_path_relative_lead_metrics(lead_dict: dict[str, Any], model_msg: capnp._DynamicStructReader,
                                   lead_msg: capnp._DynamicStructReader | None = None) -> tuple[float, float]:
  y_rel = float(lead_dict.get("yRel", 0.0) or 0.0)
  d_rel = float(lead_dict.get("dRel", 0.0) or 0.0)
  d_path = y_rel - get_path_y_rel(model_msg, d_rel)
  v_lat = 0.0

  if lead_msg is not None:
    try:
      times = np.asarray(lead_msg.t, dtype=float)
      xs = np.asarray(lead_msg.x, dtype=float)
      ys = np.asarray(lead_msg.y, dtype=float)
      valid = np.isfinite(times) & np.isfinite(xs) & np.isfinite(ys)
      if np.any(valid):
        times = times[valid]
        xs = xs[valid]
        ys = ys[valid]
        future_idxs = np.where(times > (times[0] + 1e-3))[0]
        if future_idxs.size > 0:
          i = int(future_idxs[0])
          future_d_rel = float(xs[i] - RADAR_TO_CAMERA)
          future_y_rel = float(-ys[i])
          future_d_path = future_y_rel - get_path_y_rel(model_msg, future_d_rel)
          dt = float(times[i] - times[0])
          if dt > 1e-3:
            v_lat = float((future_d_path - d_path) / dt)
    except Exception:
      v_lat = 0.0

  return d_path, v_lat


def add_path_relative_lead_metrics(lead_dict: dict[str, Any], model_msg: capnp._DynamicStructReader,
                                   lead_msg: capnp._DynamicStructReader | None = None) -> dict[str, Any]:
  d_path, v_lat = get_path_relative_lead_metrics(lead_dict, model_msg, lead_msg)
  lead_dict["dPath"] = float(d_path)
  lead_dict["vLat"] = float(v_lat)
  return lead_dict


def _is_lead_prob_accepted(lead_prob: float, prev_latched: bool,
                           prob_enter: float, prob_exit: float) -> bool:
  """Asymmetric Schmitt trigger on vision lead prob. Previous latch state decides
  whether the current prob clears the enter or the exit threshold. Defaults to
  the classic `prob > 0.5` behavior when both thresholds collapse to 0.5."""
  if not math.isfinite(lead_prob):
    return False
  lower = min(prob_enter, prob_exit)
  upper = max(prob_enter, prob_exit)
  return lead_prob > (lower if prev_latched else upper)


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP, model_msg: capnp._DynamicStructReader,
             low_speed_override: bool = True, prev_latched: bool = False,
             prob_enter: float = 0.5, prob_exit: float = 0.5) -> tuple[dict[str, Any], bool]:
  # Determine leads, this is where the essential logic happens
  prob_accepted = _is_lead_prob_accepted(float(lead_msg.prob), prev_latched, prob_enter, prob_exit)
  if len(tracks) > 0 and ready and prob_accepted:
    track = match_vision_to_track(v_ego, lead_msg, tracks)
  else:
    track = None

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_msg.prob)
    lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)
  elif (track is None) and ready and prob_accepted:
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  if lead_dict.get("status", False):
    lead_dict = add_path_relative_lead_metrics(lead_dict, model_msg, lead_msg if ready else None)

  new_latched = bool(lead_dict.get("status", False))
  return lead_dict, new_latched


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead_dict: dict[str, Any],
                    lead_msg: capnp._DynamicStructReader) -> dict[str, Any]:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead_dict['yRel'] = float(-lead_msg.y[0])

  return lead_dict


class RadarD:
  LEAD_PROB_REFRESH_S = 1.0

  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

    self._lead_latched = [False, False]
    self._params = Params()
    self._lead_prob_enter = 0.5
    self._lead_prob_exit = 0.5
    self._last_prob_refresh_t = 0.0

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks(service_list=['modelV2', 'liveTracks'])
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    self._maybe_refresh_lead_prob_thresholds()

    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      lead_one, latched_one = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego,
                                       self.CP, self.CP_SP, sm['modelV2'], low_speed_override=True,
                                       prev_latched=self._lead_latched[0],
                                       prob_enter=self._lead_prob_enter, prob_exit=self._lead_prob_exit)
      lead_two, latched_two = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego,
                                       self.CP, self.CP_SP, sm['modelV2'], low_speed_override=False,
                                       prev_latched=self._lead_latched[1],
                                       prob_enter=self._lead_prob_enter, prob_exit=self._lead_prob_exit)
      self.radar_state.leadOne = lead_one
      self.radar_state.leadTwo = lead_two
      self._lead_latched = [latched_one, latched_two]

  def _maybe_refresh_lead_prob_thresholds(self) -> None:
    if (self.current_time - self._last_prob_refresh_t) < self.LEAD_PROB_REFRESH_S:
      return
    self._last_prob_refresh_t = self.current_time
    def _read(key: str, default: float) -> float:
      try:
        raw = self._params.get(key)
        if raw is None:
          return float(default)
        val = float(raw)
        return val if math.isfinite(val) else float(default)
      except Exception:
        return float(default)
    self._lead_prob_enter = max(0.0, min(1.0, _read("Longitudinal.LiveTune.LeadProbEnter", 0.6)))
    self._lead_prob_exit = max(0.0, min(1.0, _read("Longitudinal.LiveTune.LeadProbExit", 0.35)))

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(6, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(Params().get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_SP, CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
