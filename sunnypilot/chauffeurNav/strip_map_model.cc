#include "sunnypilot/chauffeurNav/strip_map_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kEarthRadiusM = 6371000.0;
constexpr double kPi = 3.14159265358979323846;
constexpr float kStubGateM = 60.0f;
constexpr float kStubLengthM = 80.0f;
constexpr int kMaxStubs = 8;

struct CenterlineData {
  std::vector<QPointF> pts;
  std::vector<float> s;
};

static inline bool is_finite(double v) {
  return std::isfinite(v);
}

static QPointF xy_from_latlon(double lat, double lon, double lat0, double lon0) {
  const double lat_rad = lat * kPi / 180.0;
  const double lon_rad = lon * kPi / 180.0;
  const double lat0_rad = lat0 * kPi / 180.0;
  const double lon0_rad = lon0 * kPi / 180.0;
  const double dlat = lat_rad - lat0_rad;
  const double dlon = lon_rad - lon0_rad;
  const double x = dlon * std::cos(lat0_rad) * kEarthRadiusM;
  const double y = dlat * kEarthRadiusM;
  return QPointF(x, y);
}

static double dist2(const QPointF &a, const QPointF &b) {
  const double dx = a.x() - b.x();
  const double dy = a.y() - b.y();
  return dx * dx + dy * dy;
}

static CenterlineData build_centerline(const cereal::LiveMapDataSP::RoadSegment::Reader &seg,
                                       double ego_lat, double ego_lon) {
  CenterlineData out;
  const auto centerline = seg.getCenterline();
  const int n = centerline.size();
  if (n < 2) return out;

  out.pts.reserve(n);
  out.s.reserve(n);

  bool use_distances = true;
  float prev_s = -1.0f;
  for (int i = 0; i < n; ++i) {
    const auto coord = centerline[i];
    const double lat = coord.getLatitude();
    const double lon = coord.getLongitude();
    if (!is_finite(lat) || !is_finite(lon)) {
      use_distances = false;
      continue;
    }
    out.pts.push_back(xy_from_latlon(lat, lon, ego_lat, ego_lon));
    const float s_val = coord.getDistanceFromStart();
    if (!std::isfinite(s_val) || s_val < prev_s) {
      use_distances = false;
    }
    prev_s = s_val;
    out.s.push_back(s_val);
  }

  if (out.pts.size() < 2) {
    out.pts.clear();
    out.s.clear();
    return out;
  }

  if (!use_distances || prev_s <= 1.0f) {
    out.s.assign(out.pts.size(), 0.0f);
    for (size_t i = 1; i < out.pts.size(); ++i) {
      const double dx = out.pts[i].x() - out.pts[i - 1].x();
      const double dy = out.pts[i].y() - out.pts[i - 1].y();
      out.s[i] = out.s[i - 1] + static_cast<float>(std::hypot(dx, dy));
    }
  }

  return out;
}

struct Projection {
  float s = 0.0f;
  float dist = 0.0f;
};

static Projection project_onto_centerline(const CenterlineData &cl) {
  Projection proj;
  if (cl.pts.empty()) return proj;

  double best_d2 = std::numeric_limits<double>::infinity();
  float best_s = cl.s.front();

  for (size_t i = 0; i + 1 < cl.pts.size(); ++i) {
    const QPointF a = cl.pts[i];
    const QPointF b = cl.pts[i + 1];
    const double abx = b.x() - a.x();
    const double aby = b.y() - a.y();
    const double ab2 = abx * abx + aby * aby;
    if (ab2 < 1e-6) continue;

    double t = -(a.x() * abx + a.y() * aby) / ab2;  // ego at (0,0)
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    const QPointF p(a.x() + abx * t, a.y() + aby * t);
    const double d2 = p.x() * p.x() + p.y() * p.y();
    if (d2 < best_d2) {
      best_d2 = d2;
      const float s0 = cl.s[i];
      const float s1 = cl.s[i + 1];
      best_s = s0 + static_cast<float>(t) * (s1 - s0);
    }
  }

  proj.s = best_s;
  proj.dist = static_cast<float>(std::sqrt(std::max(0.0, best_d2)));
  return proj;
}

static std::vector<QPointF> clip_centerline_by_s(const CenterlineData &cl, float s_min, float s_max) {
  std::vector<QPointF> out;
  if (cl.pts.size() < 2) return out;

  struct SPoint {
    float s;
    QPointF p;
  };
  std::vector<SPoint> points;
  points.reserve(cl.pts.size() + 4);

  for (size_t i = 0; i < cl.pts.size(); ++i) {
    points.push_back({cl.s[i], cl.pts[i]});
  }

  for (size_t i = 0; i + 1 < cl.pts.size(); ++i) {
    const float s0 = cl.s[i];
    const float s1 = cl.s[i + 1];
    const QPointF p0 = cl.pts[i];
    const QPointF p1 = cl.pts[i + 1];
    const float ds = s1 - s0;
    if (std::abs(ds) < 1e-3f) continue;

    auto add_intersection = [&](float s_target) {
      if ((s_target < s0 && s_target < s1) || (s_target > s0 && s_target > s1)) return;
      const float t = (s_target - s0) / ds;
      if (t < 0.0f || t > 1.0f) return;
      const QPointF p(p0.x() + (p1.x() - p0.x()) * t,
                      p0.y() + (p1.y() - p0.y()) * t);
      points.push_back({s_target, p});
    };

    add_intersection(s_min);
    add_intersection(s_max);
  }

  std::sort(points.begin(), points.end(), [](const SPoint &a, const SPoint &b) {
    return a.s < b.s;
  });

  for (const auto &sp : points) {
    if (sp.s < s_min - 1e-3f || sp.s > s_max + 1e-3f) continue;
    if (!out.empty() && dist2(out.back(), sp.p) < 0.01) continue;
    out.push_back(sp.p);
  }

  return out;
}

static std::vector<QPointF> smooth_polyline(const std::vector<QPointF> &pts) {
  if (pts.size() < 3) return pts;
  std::vector<QPointF> out = pts;
  for (size_t i = 1; i + 1 < pts.size(); ++i) {
    const QPointF &a = pts[i - 1];
    const QPointF &b = pts[i];
    const QPointF &c = pts[i + 1];
    const double sx = a.x() * 0.25 + b.x() * 0.5 + c.x() * 0.25;
    const double sy = a.y() * 0.25 + b.y() * 0.5 + c.y() * 0.25;
    out[i] = QPointF(b.x() * 0.4 + sx * 0.6, b.y() * 0.4 + sy * 0.6);
  }
  return out;
}

}  // namespace

StripMapScene build_strip_map_scene(const cereal::LiveMapDataSP::Reader &live_map,
                                    double ego_lat, double ego_lon,
                                    float heading_deg,
                                    float forward_m,
                                    float behind_m) {
  StripMapScene scene;
  scene.heading_deg = heading_deg;
  scene.forward_m = forward_m;
  scene.behind_m = behind_m;
  scene.radius_m = 0.5f * (forward_m + behind_m);

  if (!live_map.getRoadGeometryValid()) return scene;

  const auto current_segment = live_map.getCurrentRoadSegment();
  CenterlineData current = build_centerline(current_segment, ego_lat, ego_lon);
  if (current.pts.size() < 2) return scene;

  const Projection current_proj = project_onto_centerline(current);
  const float s_min = current_proj.s - behind_m;
  const float s_max = current_proj.s + forward_m;

  std::vector<QPointF> current_pts = clip_centerline_by_s(current, s_min, s_max);
  current_pts = smooth_polyline(current_pts);
  if (current_pts.size() >= 2) {
    scene.polylines.push_back({current_pts, true, false});
  }

  const uint64_t current_way_id = current_segment.getWayId();
  const auto nearby = live_map.getNearbyRoadSegments();
  int stub_count = 0;

  for (size_t i = 0; i < nearby.size(); ++i) {
    const auto seg = nearby[i];
    if (current_way_id != 0 && seg.getWayId() == current_way_id) continue;

    CenterlineData cl = build_centerline(seg, ego_lat, ego_lon);
    if (cl.pts.size() < 2) continue;

    const Projection proj = project_onto_centerline(cl);
    if (!std::isfinite(proj.dist) || proj.dist > kStubGateM) continue;

    const float half_len = kStubLengthM * 0.5f;
    std::vector<QPointF> stub_pts = clip_centerline_by_s(cl, proj.s - half_len, proj.s + half_len);
    stub_pts = smooth_polyline(stub_pts);
    if (stub_pts.size() < 2) continue;

    scene.polylines.push_back({stub_pts, false, true});
    if (++stub_count >= kMaxStubs) break;
  }

  scene.valid = !scene.polylines.empty();
  return scene;
}
