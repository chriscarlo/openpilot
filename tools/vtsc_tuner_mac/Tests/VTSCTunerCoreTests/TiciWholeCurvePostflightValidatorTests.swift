import Foundation
import Testing
@testable import VTSCTunerCore

@Test func wholeCurvePostflightValidatorAcceptsTheRuntimeProfileContractNearRealGPS() throws {
  let now = Date(timeIntervalSince1970: 1_800_000_000)
  let profile = try wholeCurveProfileData(now: now)
  let gps = try JSONSerialization.data(withJSONObject: [
    "latitude": 37.0,
    "longitude": -122.0,
    "bearing": 90.0,
  ])

  let status = TiciWholeCurvePostflightValidator.inspect(profileData: profile, gpsData: gps, now: now)
  #expect(status.gpsStatus == "valid")
  #expect(status.validationStatus == "valid")
  #expect(status.estimatorVersion == "whole-curve-v3")
  #expect(status.fresh)
  #expect(status.valuesFinite)
  #expect(status.pointCount == 5)
  #expect(status.eventCount == 1)
}

@Test func wholeCurvePostflightValidatorRejectsFingerprintAndRouteDistanceDrift() throws {
  let now = Date(timeIntervalSince1970: 1_800_000_000)
  var root = try JSONSerialization.jsonObject(with: wholeCurveProfileData(now: now)) as! [String: Any]
  root["routeFingerprint"] = String(repeating: "0", count: 64)
  let badFingerprint = try JSONSerialization.data(withJSONObject: root)
  let gps = try JSONSerialization.data(withJSONObject: [
    "latitude": 37.0,
    "longitude": -122.0,
    "bearing": 90.0,
  ])
  let fingerprintStatus = TiciWholeCurvePostflightValidator.inspect(
    profileData: badFingerprint,
    gpsData: gps,
    now: now
  )
  #expect(fingerprintStatus.validationStatus == "rejected:fingerprint_mismatch")
  #expect(!fingerprintStatus.fresh)

  let farGPS = try JSONSerialization.data(withJSONObject: [
    "latitude": 38.0,
    "longitude": -122.0,
    "bearing": 90.0,
  ])
  let farStatus = TiciWholeCurvePostflightValidator.inspect(
    profileData: try wholeCurveProfileData(now: now),
    gpsData: farGPS,
    now: now
  )
  #expect(farStatus.gpsStatus == "valid")
  #expect(farStatus.validationStatus == "rejected:route_not_near_real_gps")
}

@Test func wholeCurvePostflightValidatorWaitsForRealGPSWithoutRelaxingProfileChecks() throws {
  let now = Date(timeIntervalSince1970: 1_800_000_000)
  let status = TiciWholeCurvePostflightValidator.inspect(
    profileData: try wholeCurveProfileData(now: now),
    gpsData: nil,
    now: now
  )
  #expect(status.gpsStatus == "pending")
  #expect(status.validationStatus == "pending_real_gps")
  #expect(!status.fresh)
}

@Test func wholeCurvePostflightValidatorTreatsEmptyObjectAsPendingProfile() throws {
  let now = Date(timeIntervalSince1970: 1_800_000_000)
  let gps = try JSONSerialization.data(withJSONObject: [
    "latitude": 37.0,
    "longitude": -122.0,
    "bearing": 90.0,
  ])
  let status = TiciWholeCurvePostflightValidator.inspect(
    profileData: Data("{}".utf8),
    gpsData: gps,
    now: now
  )
  #expect(status.gpsStatus == "valid")
  #expect(status.validationStatus == "profile_pending")
  #expect(status.estimatorVersion.isEmpty)
  #expect(status.sigmoidHash.isEmpty)
}

@Test func wholeCurvePostflightValidatorUsesTheTiciClockForFreshness() throws {
  let ticiNow = Date(timeIntervalSince1970: 1_800_000_000)
  let profile = try wholeCurveProfileData(now: ticiNow)
  let gps = try JSONSerialization.data(withJSONObject: [
    "latitude": 37.0,
    "longitude": -122.0,
    "bearing": 90.0,
  ])

  let skewedMacStatus = TiciWholeCurvePostflightValidator.inspect(
    profileData: profile,
    gpsData: gps,
    now: ticiNow.addingTimeInterval(60)
  )
  #expect(skewedMacStatus.validationStatus == "rejected:stale")

  let ticiClockStatus = TiciWholeCurvePostflightValidator.inspect(
    profileData: profile,
    gpsData: gps,
    now: ticiNow
  )
  #expect(ticiClockStatus.validationStatus == "valid")
  #expect(ticiClockStatus.fresh)
}

@Test func wholeCurveFingerprintMatchesThePythonCrossLanguageVector() throws {
  let value = try TiciWholeCurvePostflightValidator.routeFingerprint(generation: 7, sigmoidHash: "85a608e68945", points: [
    (37.0, -122.0, 0.0, 0.0, 1.0, 70.0, ""),
    (37.000045, -122.0, 5.0, -0.0123456785, 1.25, 11.25, "0123456789abcdefabcd-a"),
    (37.000090, -122.0, 10.0, 0.0123456785, 1.5, 10.5, "0123456789abcdefabcd-b"),
  ])
  #expect(value == "a0d4823aa1f16ae4bd70f3d80a1d379f02fd944ac7804ad193b9875ec40710db")
}

private func wholeCurveProfileData(now: Date) throws -> Data {
  let stepDegrees = 5.0 / 6_371_007.2 * 180 / Double.pi
  let eventID = "0123456789abcdefabcd-a"
  let curvatures = [0.0, 0.003, 0.006, 0.003, 0.0]
  let coefficients = [1.0, 0.5, 1.0, 0.5, 1.0]
  let baseSpeeds = [70.0, 20.0, 14.0, 20.0, 70.0]
  let points: [[String: Any]] = (0..<5).map { index in
    let pointEventID = (1...3).contains(index) ? eventID : ""
    return [
      "latitude": 37.0 + Double(index) * stepDegrees,
      "longitude": -122.0,
      "distanceMeters": Double(index) * 5,
      "curvature": curvatures[index],
      "curvatureCoefficient": coefficients[index],
      "baseSafeSpeedMPS": baseSpeeds[index],
      "eventID": pointEventID,
      "confidence": 1.0,
      "flags": [],
    ]
  }
  let sigmoidHash = "85a608e68945"
  let fingerprint = try TiciWholeCurvePostflightValidator.routeFingerprint(generation: 7, sigmoidHash: sigmoidHash, points: points.map {
    ($0["latitude"] as! Double, $0["longitude"] as! Double, $0["distanceMeters"] as! Double,
     $0["curvature"] as! Double, $0["curvatureCoefficient"] as! Double,
     $0["baseSafeSpeedMPS"] as! Double, $0["eventID"] as! String)
  })
  return try JSONSerialization.data(withJSONObject: [
    "estimatorVersion": "whole-curve-v3",
    "generatedAtUnixMillis": now.timeIntervalSince1970 * 1_000,
    "routeFingerprint": fingerprint,
    "sigmoidHash": sigmoidHash,
    "generation": 7,
    "points": points,
    "events": [[
      "eventID": eventID,
      "startIndex": 1,
      "endIndex": 3,
      "apexIndex": 2,
      "profileApexIndex": 2,
      "controllingCurvature": 0.006,
      "maximumApexCoefficient": 1.0,
      "confidence": 1.0,
      "flags": [],
    ]],
    "fatalAmbiguity": false,
  ])
}
