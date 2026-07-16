import CoreFoundation
import Foundation

/// Host-side validation result for the mapd whole-curve runtime contract. The
/// tici only returns the raw shared-memory/persistent JSON; Swift keeps the
/// same freshness, shape, fingerprint, and real-GPS checks that the former
/// Python postflight performed by importing the controller.
struct TiciWholeCurvePostflightStatus: Equatable, Sendable {
  var estimatorVersion: String
  var routeFingerprint: String
  var sigmoidHash: String
  var fresh: Bool
  var valuesFinite: Bool
  var gpsStatus: String
  var validationStatus: String
  var pointCount: Int
  var eventCount: Int
}

enum TiciWholeCurvePostflightValidator {
  private static let estimatorVersion = "whole-curve-v3"
  private static let maximumProfileBytes = 512 * 1_024
  private static let maximumPoints = 512
  private static let maximumEvents = 128
  private static let maximumDistanceMeters = 1_600.0
  private static let maximumEgoDistanceMeters = 75.0
  private static let earthRadiusMeters = 6_371_007.2
  private static let postflightEarthRadiusMeters = 6_371_000.0

  private struct GPSPose: Equatable, Sendable {
    var latitude: Double
    var longitude: Double
    var bearingDegrees: Double
  }

  private struct Point: Equatable, Sendable {
    var latitude: Double
    var longitude: Double
    var distanceMeters: Double
    var curvature: Double
    var curvatureCoefficient: Double
    var baseSafeSpeedMPS: Double
    var eventID: String
  }

  private struct Profile: Equatable, Sendable {
    var estimatorVersion: String
    var routeFingerprint: String
    var sigmoidHash: String
    var points: [Point]
    var eventCount: Int
  }

  /// Mirrors the Python postflight's three outcomes: an absent/unusable GPS
  /// waits for a real fix; an invalid profile reports its rejection; a valid
  /// profile must also be close enough to the current real route position.
  static func inspect(
    profileData: Data?,
    gpsData: Data?,
    now: Date = Date()
  ) -> TiciWholeCurvePostflightStatus {
    let metadata = profileMetadata(profileData)
    let gps: GPSPose?
    let gpsStatus: String
    do {
      gps = try parseGPS(gpsData)
      gpsStatus = "valid"
    } catch let error as ValidationError {
      gps = nil
      gpsStatus = error.isPendingGPS ? "pending" : "invalid:\(error.code)"
    } catch {
      gps = nil
      gpsStatus = "invalid:unknown"
    }

    guard let profileData, !profileData.isEmpty, !isEmptyPlaceholderProfile(profileData) else {
      return TiciWholeCurvePostflightStatus(
        estimatorVersion: metadata.estimatorVersion,
        routeFingerprint: metadata.routeFingerprint,
        sigmoidHash: metadata.sigmoidHash,
        fresh: false,
        valuesFinite: false,
        gpsStatus: gpsStatus,
        validationStatus: "profile_pending",
        pointCount: metadata.pointCount,
        eventCount: metadata.eventCount
      )
    }
    guard let gps else {
      return TiciWholeCurvePostflightStatus(
        estimatorVersion: metadata.estimatorVersion,
        routeFingerprint: metadata.routeFingerprint,
        sigmoidHash: metadata.sigmoidHash,
        fresh: false,
        valuesFinite: false,
        gpsStatus: gpsStatus,
        validationStatus: "pending_real_gps",
        pointCount: metadata.pointCount,
        eventCount: metadata.eventCount
      )
    }

    do {
      let profile = try parseProfile(profileData, now: now)
      let distances = profile.points.map {
        haversineMeters(
          latitude1: gps.latitude,
          longitude1: gps.longitude,
          latitude2: $0.latitude,
          longitude2: $0.longitude,
          radiusMeters: postflightEarthRadiusMeters
        )
      }
      guard let nearestIndex = distances.indices.min(by: { distances[$0] < distances[$1] }),
            distances[nearestIndex].isFinite,
            distances[nearestIndex] <= maximumEgoDistanceMeters
      else { throw ValidationError.routeNotNearRealGPS }
      guard nearestIndex <= profile.points.count - 3 else {
        throw ValidationError.insufficientForwardContext
      }
      return TiciWholeCurvePostflightStatus(
        estimatorVersion: profile.estimatorVersion,
        routeFingerprint: profile.routeFingerprint,
        sigmoidHash: profile.sigmoidHash,
        fresh: true,
        valuesFinite: true,
        gpsStatus: gpsStatus,
        validationStatus: "valid",
        pointCount: profile.points.count,
        eventCount: profile.eventCount
      )
    } catch let error as ValidationError {
      return TiciWholeCurvePostflightStatus(
        estimatorVersion: metadata.estimatorVersion,
        routeFingerprint: metadata.routeFingerprint,
        sigmoidHash: metadata.sigmoidHash,
        fresh: false,
        valuesFinite: false,
        gpsStatus: gpsStatus,
        validationStatus: "rejected:\(error.code)",
        pointCount: metadata.pointCount,
        eventCount: metadata.eventCount
      )
    } catch {
      return TiciWholeCurvePostflightStatus(
        estimatorVersion: metadata.estimatorVersion,
        routeFingerprint: metadata.routeFingerprint,
        sigmoidHash: metadata.sigmoidHash,
        fresh: false,
        valuesFinite: false,
        gpsStatus: gpsStatus,
        validationStatus: "rejected:unknown",
        pointCount: metadata.pointCount,
        eventCount: metadata.eventCount
      )
    }
  }

  /// The cross-language v3 fingerprint contract is intentionally exposed to
  /// tests and to any future snapshot verifier.
  static func routeFingerprint(
    generation: Int,
    sigmoidHash: String,
    points: [(Double, Double, Double, Double, Double, Double, String)]
  ) throws -> String {
    var text = "MapWholeCurveProfile|\(estimatorVersion)|\(generation)|\(sigmoidHash)\n"
    for point in points {
      let latitude = try roundHalfAwayFromZero(point.0, scale: 1e7)
      let longitude = try roundHalfAwayFromZero(point.1, scale: 1e7)
      let distance = try roundHalfAwayFromZero(point.2, scale: 1e3)
      let curvature = try roundHalfAwayFromZero(point.3, scale: 1e9)
      let coefficient = try roundHalfAwayFromZero(point.4, scale: 1e6)
      let baseSafeSpeed = try roundHalfAwayFromZero(point.5, scale: 1e6)
      text += "\(latitude),\(longitude),\(distance),\(curvature),\(coefficient),\(baseSafeSpeed),\(point.6)\n"
    }
    return TuneDeploymentIdentity.sha256Hex(Data(text.utf8))
  }

  private enum ValidationError: Error, Equatable {
    case gpsMissing
    case gpsMalformed
    case gpsNonFinite
    case gpsOutOfRange
    case gpsPending
    case profileTooLarge
    case malformedJSON
    case rootNotObject
    case versionMismatch
    case generatedAtInvalid
    case timestampInFuture
    case stale
    case invalidGeneration
    case invalidFingerprint
    case invalidSigmoidHash
    case fatalAmbiguityInvalid
    case fatalAmbiguity
    case invalidPointCount
    case pointInvalidObject(Int)
    case pointUnknownField(Int)
    case pointNumberInvalid(Int, String)
    case pointCoordinateRange(Int)
    case pointDistanceRange(Int)
    case pointCurvatureRange(Int)
    case pointCurvatureCoefficientRange(Int)
    case pointBaseSafeSpeedRange(Int)
    case pointEventID(Int)
    case pointConfidence(Int)
    case pointFlags(Int)
    case firstDistanceNotZero
    case nonmonotonicDistance(Int)
    case duplicateCoordinate(Int)
    case distanceMismatch(Int)
    case fingerprintMismatch
    case invalidEvents
    case eventNotObject(Int)
    case eventInvalidValue(Int)
    case eventConflictingID(Int)
    case eventInvalidID(Int)
    case eventIDMismatch(Int)
    case eventIndexRange(Int, String)
    case eventBoundaryOrder(Int)
    case eventApexRange(Int)
    case eventControllingCurvature(Int)
    case eventPhysicalID(Int)
    case eventPointSetMismatch
    case routeNotNearRealGPS
    case insufficientForwardContext

    var isPendingGPS: Bool {
      self == .gpsMissing || self == .gpsPending
    }

    var code: String {
      switch self {
      case .gpsMissing: "gps_missing"
      case .gpsMalformed: "gps_malformed"
      case .gpsNonFinite: "gps_non_finite"
      case .gpsOutOfRange: "gps_out_of_range"
      case .gpsPending: "gps_pending"
      case .profileTooLarge: "profile_too_large"
      case .malformedJSON: "malformed_json"
      case .rootNotObject: "root_not_object"
      case .versionMismatch: "version_mismatch"
      case .generatedAtInvalid: "generated_at_not_number"
      case .timestampInFuture: "timestamp_in_future"
      case .stale: "stale"
      case .invalidGeneration: "invalid_generation"
      case .invalidFingerprint: "invalid_fingerprint"
      case .invalidSigmoidHash: "invalid_sigmoid_hash"
      case .fatalAmbiguityInvalid: "invalid_fatal_ambiguity"
      case .fatalAmbiguity: "fatal_ambiguity"
      case .invalidPointCount: "invalid_point_count"
      case let .pointInvalidObject(index): "point_\(index)_invalid_object"
      case let .pointUnknownField(index): "point_\(index)_unknown_field"
      case let .pointNumberInvalid(index, field): "point_\(index)_\(field)_not_finite"
      case let .pointCoordinateRange(index): "point_\(index)_coordinate_range"
      case let .pointDistanceRange(index): "point_\(index)_distance_range"
      case let .pointCurvatureRange(index): "point_\(index)_curvature_range"
      case let .pointCurvatureCoefficientRange(index): "point_\(index)_curvature_coefficient_range"
      case let .pointBaseSafeSpeedRange(index): "point_\(index)_base_safe_speed_range"
      case let .pointEventID(index): "point_\(index)_invalid_event_id"
      case let .pointConfidence(index): "point_\(index)_confidence_value"
      case let .pointFlags(index): "point_\(index)_invalid_flags"
      case .firstDistanceNotZero: "first_distance_not_zero"
      case let .nonmonotonicDistance(index): "point_\(index)_nonmonotonic_distance"
      case let .duplicateCoordinate(index): "point_\(index)_duplicate_coordinate"
      case let .distanceMismatch(index): "point_\(index)_distance_mismatch"
      case .fingerprintMismatch: "fingerprint_mismatch"
      case .invalidEvents: "invalid_events"
      case let .eventNotObject(index): "event_\(index)_not_object"
      case let .eventInvalidValue(index): "event_\(index)_invalid_value"
      case let .eventConflictingID(index): "event_\(index)_conflicting_id"
      case let .eventInvalidID(index): "event_\(index)_invalid_id"
      case let .eventIDMismatch(index): "event_\(index)_id_mismatch"
      case let .eventIndexRange(index, field): "event_\(index)_\(field)_range"
      case let .eventBoundaryOrder(index): "event_\(index)_boundary_order"
      case let .eventApexRange(index): "event_\(index)_apex_range"
      case let .eventControllingCurvature(index): "event_\(index)_controlling_curvature_range"
      case let .eventPhysicalID(index): "event_\(index)_invalid_physical_id"
      case .eventPointSetMismatch: "event_point_set_mismatch"
      case .routeNotNearRealGPS: "route_not_near_real_gps"
      case .insufficientForwardContext: "insufficient_forward_context"
      }
    }
  }

  private static func parseGPS(_ data: Data?) throws -> GPSPose {
    guard let data, !data.isEmpty else { throw ValidationError.gpsMissing }
    guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw ValidationError.gpsMalformed
    }
    let latitude = try gpsNumber(object["latitude"])
    let longitude = try gpsNumber(object["longitude"])
    let bearing = try gpsNumber(object["bearing"] ?? object["bearingDeg"])
    guard latitude.isFinite, longitude.isFinite, bearing.isFinite else {
      throw ValidationError.gpsNonFinite
    }
    guard (-90...90).contains(latitude), (-180...180).contains(longitude), (0..<360).contains(bearing) else {
      throw ValidationError.gpsOutOfRange
    }
    guard latitude != 0 || longitude != 0 else { throw ValidationError.gpsPending }
    return GPSPose(latitude: latitude, longitude: longitude, bearingDegrees: bearing)
  }

  private static func gpsNumber(_ value: Any?) throws -> Double {
    guard let value else { throw ValidationError.gpsMalformed }
    if let string = value as? String, let number = Double(string) { return number }
    guard let number = number(value) else { throw ValidationError.gpsMalformed }
    return number
  }

  private static func parseProfile(_ data: Data, now: Date) throws -> Profile {
    guard data.count <= maximumProfileBytes else { throw ValidationError.profileTooLarge }
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw ValidationError.malformedJSON
    }
    guard root["estimatorVersion"] as? String == estimatorVersion else { throw ValidationError.versionMismatch }
    let generatedAt = try finiteNumber(root["generatedAtUnixMillis"], error: .generatedAtInvalid)
    let age = now.timeIntervalSince1970 - generatedAt / 1_000
    guard age >= -1 else { throw ValidationError.timestampInFuture }
    guard age <= 3 else { throw ValidationError.stale }
    guard let generation = strictInteger(root["generation"]), generation >= 0 else {
      throw ValidationError.invalidGeneration
    }
    guard let routeFingerprint = root["routeFingerprint"] as? String,
          isHex(routeFingerprint, exactLength: 64), routeFingerprint == routeFingerprint.lowercased()
    else { throw ValidationError.invalidFingerprint }
    guard let sigmoidHash = root["sigmoidHash"] as? String,
          isHex(sigmoidHash, exactLength: 12), sigmoidHash == sigmoidHash.lowercased()
    else { throw ValidationError.invalidSigmoidHash }
    guard let fatalAmbiguity = bool(root["fatalAmbiguity"]) else { throw ValidationError.fatalAmbiguityInvalid }
    guard !fatalAmbiguity else { throw ValidationError.fatalAmbiguity }

    guard let rawPoints = root["points"] as? [Any], (3...maximumPoints).contains(rawPoints.count) else {
      throw ValidationError.invalidPointCount
    }
    var points: [Point] = []
    var previous: Point?
    for (index, rawPoint) in rawPoints.enumerated() {
      guard let point = rawPoint as? [String: Any],
            ["latitude", "longitude", "distanceMeters", "curvature", "curvatureCoefficient", "baseSafeSpeedMPS"]
              .allSatisfy(point.keys.contains)
      else { throw ValidationError.pointInvalidObject(index) }
      let allowed = Set([
        "latitude", "longitude", "distanceMeters", "curvature", "curvatureCoefficient",
        "baseSafeSpeedMPS", "eventID", "confidence", "flags",
      ])
      guard point.keys.allSatisfy(allowed.contains) else { throw ValidationError.pointUnknownField(index) }
      let latitude = try finiteNumber(point["latitude"], error: .pointNumberInvalid(index, "latitude"))
      let longitude = try finiteNumber(point["longitude"], error: .pointNumberInvalid(index, "longitude"))
      let distance = try finiteNumber(point["distanceMeters"], error: .pointNumberInvalid(index, "distance"))
      let curvature = try finiteNumber(point["curvature"], error: .pointNumberInvalid(index, "curvature"))
      let curvatureCoefficient = try finiteNumber(
        point["curvatureCoefficient"], error: .pointNumberInvalid(index, "curvature_coefficient")
      )
      let baseSafeSpeed = try finiteNumber(
        point["baseSafeSpeedMPS"], error: .pointNumberInvalid(index, "base_safe_speed")
      )
      guard (-90...90).contains(latitude), (-180...180).contains(longitude) else {
        throw ValidationError.pointCoordinateRange(index)
      }
      guard (0...maximumDistanceMeters).contains(distance) else { throw ValidationError.pointDistanceRange(index) }
      guard abs(curvature) <= 1 else { throw ValidationError.pointCurvatureRange(index) }
      guard curvatureCoefficient > 0, curvatureCoefficient <= 4 else {
        throw ValidationError.pointCurvatureCoefficientRange(index)
      }
      guard (0...70).contains(baseSafeSpeed) else {
        throw ValidationError.pointBaseSafeSpeedRange(index)
      }
      let eventID: String
      if point["eventID"] == nil || point["eventID"] is NSNull { eventID = "" }
      else if let value = point["eventID"] as? String, isEventID(value) { eventID = value }
      else { throw ValidationError.pointEventID(index) }
      try validateConfidence(point["confidence"], index: index)
      try validateFlags(point["flags"], index: index)

      let parsed = Point(
        latitude: latitude,
        longitude: longitude,
        distanceMeters: distance,
        curvature: curvature,
        curvatureCoefficient: curvatureCoefficient,
        baseSafeSpeedMPS: baseSafeSpeed,
        eventID: eventID
      )
      if index == 0 {
        guard distance <= 0.01 else { throw ValidationError.firstDistanceNotZero }
      } else if let previous {
        let delta = distance - previous.distanceMeters
        guard delta > 0.05, delta <= 25 else { throw ValidationError.nonmonotonicDistance(index) }
        let geometry = haversineMeters(
          latitude1: previous.latitude,
          longitude1: previous.longitude,
          latitude2: latitude,
          longitude2: longitude,
          radiusMeters: earthRadiusMeters
        )
        guard geometry.isFinite, geometry > 0.05 else { throw ValidationError.duplicateCoordinate(index) }
        guard abs(geometry - delta) <= max(1, 0.2 * delta) else { throw ValidationError.distanceMismatch(index) }
      }
      points.append(parsed)
      previous = parsed
    }

    let expectedFingerprint = try Self.routeFingerprint(
      generation: generation,
      sigmoidHash: sigmoidHash,
      points: points.map {
        ($0.latitude, $0.longitude, $0.distanceMeters, $0.curvature,
         $0.curvatureCoefficient, $0.baseSafeSpeedMPS, $0.eventID)
      }
    )
    guard routeFingerprint == expectedFingerprint else { throw ValidationError.fingerprintMismatch }

    guard let rawEvents = root["events"] as? [Any], rawEvents.count <= maximumEvents else {
      throw ValidationError.invalidEvents
    }
    let pointEventIDs = Set(points.map(\.eventID).filter { !$0.isEmpty })
    var seenEventIDs: Set<String> = []
    for (index, rawEvent) in rawEvents.enumerated() {
      guard let event = rawEvent as? [String: Any] else { throw ValidationError.eventNotObject(index) }
      guard validateEventValue(event) else { throw ValidationError.eventInvalidValue(index) }
      let eventID = (event["eventID"] ?? event["id"]) as? String
      if event["eventID"] != nil, event["id"] != nil, !isEqualJSONScalar(event["eventID"], event["id"]) {
        throw ValidationError.eventConflictingID(index)
      }
      guard let eventID, !eventID.isEmpty, isEventID(eventID) else { throw ValidationError.eventInvalidID(index) }
      guard !seenEventIDs.contains(eventID), pointEventIDs.contains(eventID) else { throw ValidationError.eventIDMismatch(index) }
      seenEventIDs.insert(eventID)
      for key in ["startIndex", "endIndex", "apexIndex", "profileApexIndex"] {
        if let value = event[key] {
          guard let integer = strictInteger(value), (0..<points.count).contains(integer) else {
            throw ValidationError.eventIndexRange(index, key)
          }
        }
      }
      if let start = strictInteger(event["startIndex"]), let end = strictInteger(event["endIndex"]), start > end {
        throw ValidationError.eventBoundaryOrder(index)
      }
      if let apex = strictInteger(event["apexIndex"]),
         let start = strictInteger(event["startIndex"]),
         let end = strictInteger(event["endIndex"]),
         !(start...end).contains(apex) {
        throw ValidationError.eventApexRange(index)
      }
      if let value = event["controllingCurvature"] {
        let curvature = try finiteNumber(value, error: .eventControllingCurvature(index))
        guard abs(curvature) > 0, abs(curvature) <= 1 else { throw ValidationError.eventControllingCurvature(index) }
      }
      if let physicalID = event["physicalID"] {
        guard let physicalID = physicalID as? String, isHex(physicalID, exactLength: 20) else {
          throw ValidationError.eventPhysicalID(index)
        }
      }
    }
    guard seenEventIDs == pointEventIDs else { throw ValidationError.eventPointSetMismatch }

    return Profile(
      estimatorVersion: estimatorVersion,
      routeFingerprint: routeFingerprint,
      sigmoidHash: sigmoidHash,
      points: points,
      eventCount: rawEvents.count
    )
  }

  private static func profileMetadata(
    _ data: Data?
  ) -> (estimatorVersion: String, routeFingerprint: String, sigmoidHash: String, pointCount: Int, eventCount: Int) {
    guard let data,
          let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return ("", "", "", 0, 0) }
    return (
      root["estimatorVersion"] as? String ?? "",
      root["routeFingerprint"] as? String ?? "",
      root["sigmoidHash"] as? String ?? "",
      (root["points"] as? [Any])?.count ?? 0,
      (root["events"] as? [Any])?.count ?? 0
    )
  }

  private static func isEmptyPlaceholderProfile(_ data: Data) -> Bool {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
    return root.isEmpty
  }

  private static func finiteNumber(_ value: Any?, error: ValidationError) throws -> Double {
    guard let value, let result = number(value), result.isFinite else { throw error }
    return result
  }

  private static func number(_ value: Any) -> Double? {
    guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else { return nil }
    return value.doubleValue
  }

  private static func strictInteger(_ value: Any?) -> Int? {
    guard let value, let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          number.int64Value >= 0,
          number.int64Value <= Int64(Int.max)
    else { return nil }
    return Int(number.int64Value)
  }

  private static func bool(_ value: Any?) -> Bool? {
    guard let value = value as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return nil }
    return value.boolValue
  }

  private static func validateConfidence(_ value: Any?, index: Int) throws {
    guard let value, !(value is NSNull) else { return }
    if let string = value as? String, ["high", "review", "low"].contains(string) { return }
    guard let numeric = number(value), numeric.isFinite, (0...1).contains(numeric) else {
      throw ValidationError.pointConfidence(index)
    }
  }

  private static func validateFlags(_ value: Any?, index: Int) throws {
    guard let value else { return }
    guard let flags = value as? [Any],
          flags.count <= 16,
          flags.allSatisfy({ ($0 as? String).map(isFlag) ?? false })
    else { throw ValidationError.pointFlags(index) }
  }

  private static func validateEventValue(_ value: Any, depth: Int = 0) -> Bool {
    guard depth <= 4 else { return false }
    if value is NSNull { return true }
    if let string = value as? String { return string.count <= 256 }
    if let boolean = value as? NSNumber, CFGetTypeID(boolean) == CFBooleanGetTypeID() { return true }
    if let number = number(value) { return number.isFinite }
    if let values = value as? [Any] {
      return values.count <= maximumPoints && values.allSatisfy { validateEventValue($0, depth: depth + 1) }
    }
    if let object = value as? [String: Any] {
      return object.count <= 64 && object.allSatisfy { key, item in
        key.count <= 96 && validateEventValue(item, depth: depth + 1)
      }
    }
    return false
  }

  private static func isEqualJSONScalar(_ lhs: Any?, _ rhs: Any?) -> Bool {
    switch (lhs, rhs) {
    case let (lhs as String, rhs as String): lhs == rhs
    case let (lhs as NSNumber, rhs as NSNumber): lhs == rhs
    default: false
    }
  }

  private static func isEventID(_ value: String) -> Bool {
    value.isEmpty || value.range(of: #"^[0-9a-f]{20}-[ab]$"#, options: .regularExpression) != nil
  }

  private static func isFlag(_ value: String) -> Bool {
    value.range(of: #"^[A-Za-z0-9_.:-]{1,96}$"#, options: .regularExpression) != nil
  }

  private static func isHex(_ value: String, exactLength: Int) -> Bool {
    value.count == exactLength && value.range(of: #"^[0-9a-f]+$"#, options: .regularExpression) != nil
  }

  private static func roundHalfAwayFromZero(_ value: Double, scale: Double) throws -> Int64 {
    guard value.isFinite else { throw ValidationError.generatedAtInvalid }
    let magnitude = floor(abs(value) * scale + 0.5)
    guard magnitude <= Double(Int64.max) else { throw ValidationError.generatedAtInvalid }
    return value < 0 ? -Int64(magnitude) : Int64(magnitude)
  }

  private static func haversineMeters(
    latitude1: Double,
    longitude1: Double,
    latitude2: Double,
    longitude2: Double,
    radiusMeters: Double
  ) -> Double {
    let phi1 = latitude1 * .pi / 180
    let phi2 = latitude2 * .pi / 180
    let deltaPhi = (latitude2 - latitude1) * .pi / 180
    let deltaLambda = (longitude2 - longitude1) * .pi / 180
    let a = sin(deltaPhi / 2) * sin(deltaPhi / 2) +
      cos(phi1) * cos(phi2) * sin(deltaLambda / 2) * sin(deltaLambda / 2)
    return 2 * radiusMeters * asin(min(1, sqrt(max(0, a))))
  }
}
