/**
 * RTI Arrow Bearing Calculation - Unit Tests
 * Tests the relative bearing calculation for RTI threat directional arrows
 */

#include <gtest/gtest.h>
#include <cmath>
#include "selfdrive/ui/sunnypilot/qt/onroad/rti_arrow.h"

class RTIArrowBearingTest : public ::testing::Test {
protected:
  // Test constants
  static constexpr double kEpsilon = 0.01;  // Tolerance for floating point comparisons
  
  // Helper to normalize angle to [-180, 180]
  double normalizeAngle(double angle) {
    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;
    return angle;
  }
};

// Test: Threat directly ahead (0°)
TEST_F(RTIArrowBearingTest, ThreatDirectlyAhead) {
  // Ego at origin, heading north
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 0.0;  // North
  
  // Threat 500m north
  double threat_lat = 37.7794;  // ~500m north
  double threat_lon = -122.4194;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 0.0, kEpsilon);  // Should point straight up
}

// Test: Threat directly behind (180°)
TEST_F(RTIArrowBearingTest, ThreatDirectlyBehind) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 0.0;  // North
  
  // Threat 500m south
  double threat_lat = 37.7704;  // ~500m south
  double threat_lon = -122.4194;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(std::abs(bearing), 180.0, kEpsilon);  // Should point straight down
}

// Test: Threat to the right (90°)
TEST_F(RTIArrowBearingTest, ThreatToRight) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 0.0;  // North
  
  // Threat 500m east
  double threat_lat = 37.7749;
  double threat_lon = -122.4139;  // ~500m east
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 90.0, kEpsilon);  // Should point right
}

// Test: Threat to the left (-90°)
TEST_F(RTIArrowBearingTest, ThreatToLeft) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 0.0;  // North
  
  // Threat 500m west
  double threat_lat = 37.7749;
  double threat_lon = -122.4249;  // ~500m west
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, -90.0, kEpsilon);  // Should point left
}

// Test: Ego heading east, threat ahead
TEST_F(RTIArrowBearingTest, EgoHeadingEastThreatAhead) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 90.0;  // East
  
  // Threat 500m east (ahead relative to ego)
  double threat_lat = 37.7749;
  double threat_lon = -122.4139;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 0.0, kEpsilon);  // Should point ahead (up)
}

// Test: Ego heading south, threat to right
TEST_F(RTIArrowBearingTest, EgoHeadingSouthThreatRight) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 180.0;  // South
  
  // Threat west (right when facing south)
  double threat_lat = 37.7749;
  double threat_lon = -122.4249;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 90.0, kEpsilon);  // Should point right
}

// Test: Corner case - same location
TEST_F(RTIArrowBearingTest, SameLocation) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 45.0;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, ego_lat, ego_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 0.0, kEpsilon);  // Default to ahead when at same location
}

// Test: Diagonal bearings
TEST_F(RTIArrowBearingTest, DiagonalBearings) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 0.0;  // North
  
  // Threat northeast (45°)
  double threat_lat = 37.7785;  // ~400m north
  double threat_lon = -122.4149;  // ~400m east
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  EXPECT_NEAR(bearing, 45.0, 5.0);  // Should be approximately 45°
}

// Test: Wrap-around cases
TEST_F(RTIArrowBearingTest, WrapAroundPositive) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  double ego_heading = 350.0;  // Almost north
  
  // Threat slightly east
  double threat_lat = 37.7749;
  double threat_lon = -122.4180;
  
  double bearing = RTIArrow::calculateRelativeBearing(
    ego_lat, ego_lon, threat_lat, threat_lon, ego_heading
  );
  
  // Bearing should be normalized to [-180, 180]
  EXPECT_GE(bearing, -180.0);
  EXPECT_LE(bearing, 180.0);
}

// Test: Distance calculation
TEST_F(RTIArrowBearingTest, DistanceCalculation) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  
  // Known distance: Golden Gate Bridge to SF City Hall (~7.5km)
  double threat_lat = 37.8199;  // GG Bridge
  double threat_lon = -122.4783;
  
  double distance = RTIArrow::calculateDistance(
    ego_lat, ego_lon, threat_lat, threat_lon
  );
  
  // Should be approximately 7500m
  EXPECT_NEAR(distance, 7500.0, 500.0);
}

// Test: Very small distances (edge case)
TEST_F(RTIArrowBearingTest, VerySmallDistance) {
  double ego_lat = 37.7749;
  double ego_lon = -122.4194;
  
  // Threat 10m away
  double threat_lat = 37.77499;  // ~10m north
  double threat_lon = -122.4194;
  
  double distance = RTIArrow::calculateDistance(
    ego_lat, ego_lon, threat_lat, threat_lon
  );
  
  EXPECT_NEAR(distance, 10.0, 2.0);
}