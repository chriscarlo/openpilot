import Foundation

enum MapPreviewPurpose: String, CaseIterable, Identifiable {
  case calibration = "Calibration"
  case wholeCurveStudy = "Whole-Curve Study"

  var id: String { rawValue }
}

enum MapWholeCurveDisplayMode: String, CaseIterable, Identifiable {
  case currentMapd = "Today’s mapd"
  case wholeCurve = "Whole curve"
  case difference = "Difference"

  var id: String { rawValue }

  var legendCaption: String {
    switch self {
    case .currentMapd:
      "current five-node mapd result along each curve event"
    case .wholeCurve:
      "continuous local-curvature speed along each complete curve event"
    case .difference:
      "whole-curve preview minus today’s mapd result"
    }
  }
}

struct MapWholeCurvePoint: Sendable, Equatable {
  var latitude: Double
  var longitude: Double
  var distanceMeters: Double
  var currentMapdSpeedMPH: Double
  var wholeCurveSpeedMPH: Double
  var curvature60: Double?
  var curvature100: Double?
  var curvature160: Double?
}

struct MapWholeCurveEvent: Identifiable, Sendable, Equatable {
  var id: String
  var roadName: String
  var reference: String
  var travelDirection: String
  var bendDirection: String
  var points: [MapWholeCurvePoint]
  var sourceKeys: Set<String>
  var lengthMeters: Double
  var turnDegrees: Double
  var apexLatitude: Double
  var apexLongitude: Double
  var curvature60: Double?
  var curvature100: Double?
  var curvature160: Double?
  var controllingCurvature: Double
  var wholeCurveSpeedMPH: Double
  var currentMinimumSpeedMPH: Double
  var currentMaximumSpeedMPH: Double
  var confidenceLabel: String
  var flags: [String]
  var bankSampleNumbers: [Int]
  var bankTargetsMPH: [Double]

  var displayName: String {
    let road = !roadName.isEmpty ? roadName : (!reference.isEmpty ? reference : "Unnamed mapd road")
    return "\(road) · \(travelDirection) · \(bendDirection) bend"
  }

  var targetMedianMPH: Double? {
    guard !bankTargetsMPH.isEmpty else { return nil }
    let sorted = bankTargetsMPH.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
      return 0.5 * (sorted[middle - 1] + sorted[middle])
    }
    return sorted[middle]
  }

  var targetSpreadMPH: Double {
    guard let minimum = bankTargetsMPH.min(), let maximum = bankTargetsMPH.max() else { return 0 }
    return maximum - minimum
  }

  var targetIsCoherent: Bool { targetSpreadMPH <= 2.0 }

  var lateralAccelerationMPS2: Double {
    let speedMPS = wholeCurveSpeedMPH / 2.236_936_292_054_4
    return abs(controllingCurvature) * speedMPS * speedMPS
  }
}
