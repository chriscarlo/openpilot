/**
 * RTI Arrow - Directional threat indicator for Real-Time Intelligence
 * Provides bearing calculation and arrow rendering for threat awareness
 */

#pragma once

#include <QWidget>
#include <QPainter>
#include <QPixmap>
#include <cmath>

class RTIArrow : public QWidget {
  Q_OBJECT

public:
  explicit RTIArrow(QWidget *parent = nullptr);
  
  // Static methods for bearing and distance calculation
  static double calculateRelativeBearing(double ego_lat, double ego_lon, 
                                        double threat_lat, double threat_lon, 
                                        double ego_heading_deg);
  
  static double calculateDistance(double lat1, double lon1, 
                                 double lat2, double lon2);
  
  // Update the arrow rotation based on relative bearing
  void setRotation(double bearing_deg);
  
  // Set threat type for color coding
  void setThreatType(int type);
  
  // Enable/disable the arrow display
  void setVisible(bool visible);

protected:
  void paintEvent(QPaintEvent *event) override;

private:
  // Arrow state
  double rotation_angle = 0.0;  // Current rotation in degrees
  int threat_type = 0;
  bool is_visible = false;
  
  // Cached arrow pixmap for performance
  QPixmap arrow_pixmap;
  bool pixmap_cached = false;
  
  // Create the arrow pixmap (called once)
  void createArrowPixmap();
  
  // Get color based on threat distance
  QColor getThreatColor(double distance_m) const;
  
  // Constants
  static constexpr double kEarthRadiusM = 6371000.0;
  static constexpr int kArrowSize = 64;  // Arrow icon size in pixels
};

// Inline implementations for performance-critical calculations
inline double RTIArrow::calculateRelativeBearing(double ego_lat, double ego_lon, 
                                                double threat_lat, double threat_lon, 
                                                double ego_heading_deg) {
  // Handle same location edge case
  if (std::abs(ego_lat - threat_lat) < 1e-9 && 
      std::abs(ego_lon - threat_lon) < 1e-9) {
    return 0.0;  // Default to ahead
  }
  
  // Convert to radians
  double phi1 = ego_lat * M_PI / 180.0;
  double phi2 = threat_lat * M_PI / 180.0;
  double lam1 = ego_lon * M_PI / 180.0;
  double lam2 = threat_lon * M_PI / 180.0;
  
  // Calculate differences
  double dphi = phi2 - phi1;
  double dlam = lam2 - lam1;
  
  // Local tangent plane approximation (accurate for < 10km)
  double avg_lat = (phi1 + phi2) / 2.0;
  double north = dphi * kEarthRadiusM;
  double east = dlam * kEarthRadiusM * std::cos(avg_lat);
  
  // Calculate world bearing (0° = north, clockwise positive)
  double world_bearing = std::atan2(east, north) * 180.0 / M_PI;
  if (world_bearing < 0) world_bearing += 360.0;
  
  // Calculate relative bearing
  double rel_bearing = world_bearing - ego_heading_deg;
  
  // Normalize to [-180, 180]
  while (rel_bearing > 180.0) rel_bearing -= 360.0;
  while (rel_bearing < -180.0) rel_bearing += 360.0;
  
  return rel_bearing;
}

inline double RTIArrow::calculateDistance(double lat1, double lon1, 
                                         double lat2, double lon2) {
  // Haversine formula for distance
  double phi1 = lat1 * M_PI / 180.0;
  double phi2 = lat2 * M_PI / 180.0;
  double dphi = phi2 - phi1;
  double dlam = (lon2 - lon1) * M_PI / 180.0;
  
  double a = std::sin(dphi/2) * std::sin(dphi/2) +
             std::cos(phi1) * std::cos(phi2) *
             std::sin(dlam/2) * std::sin(dlam/2);
  
  double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
  
  return kEarthRadiusM * c;
}