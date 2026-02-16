#pragma once

#include <QPointF>
#include <vector>

#include "cereal/gen/cpp/custom.capnp.h"

struct StripMapPolyline {
  std::vector<QPointF> points_m;
  bool is_current_road = false;
  bool is_stub = false;
};

struct StripMapScene {
  std::vector<StripMapPolyline> polylines;
  bool valid = false;
  float heading_deg = 0.0f;
  float forward_m = 0.0f;
  float behind_m = 0.0f;
  float radius_m = 0.0f;
};

StripMapScene build_strip_map_scene(
  const cereal::LiveMapDataSP::Reader &live_map,
  double ego_lat,
  double ego_lon,
  float heading_deg,
  float forward_m,
  float behind_m);
