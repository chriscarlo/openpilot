import CryptoKit
import Foundation

public struct MapRuntimeCurvatureEstimate: Equatable, Sendable {
  public var curvature: Double
  /// Physical span from the first through fifth unique route node used by the estimate.
  public var supportMeters: Double

  public init(curvature: Double, supportMeters: Double) {
    self.curvature = curvature
    self.supportMeters = supportMeters
  }
}

public enum MapBakeMath {
  public static let earthRadiusMeters = 6_373_000.0
  public static let defaultMaximumSpeedMPS = 70.0
  public static let mergeOrSplitCurvature = 0.0015
  public static let mergeOrSplitRadiusMeters = 15.0

  public static func distanceMeters(from first: MapTileNode, to second: MapTileNode) -> Double {
    let latitudeA = first.latitude * .pi / 180.0
    let longitudeA = first.longitude * .pi / 180.0
    let latitudeB = second.latitude * .pi / 180.0
    let longitudeB = second.longitude * .pi / 180.0
    let latitudeTerm = Foundation.sin((latitudeB - latitudeA) / 2.0)
    let longitudeTerm = Foundation.sin((longitudeB - longitudeA) / 2.0)
    let haversine = latitudeTerm * latitudeTerm +
      Foundation.cos(latitudeA) * Foundation.cos(latitudeB) * longitudeTerm * longitudeTerm
    let centralAngle = 2.0 * Foundation.atan2(Foundation.sqrt(haversine), Foundation.sqrt(1.0 - haversine))
    return earthRadiusMeters * centralAngle
  }

  public static func curvature(
    previous: MapTileNode,
    current: MapTileNode,
    next: MapTileNode
  ) -> Double {
    curvatureMeasurement(previous: previous, current: current, next: next).curvature
  }

  /// The raw three-node circumcircle measurement used by mapd.  The arc
  /// length is retained because production `GetAverageCurvatures` weights
  /// three adjacent measurements by their represented road distance.
  public static func curvatureMeasurement(
    previous: MapTileNode,
    current: MapTileNode,
    next: MapTileNode
  ) -> (curvature: Double, arcLengthMeters: Double) {
    let lengthA = distanceMeters(from: previous, to: current)
    let lengthB = distanceMeters(from: previous, to: next)
    let lengthC = distanceMeters(from: current, to: next)
    guard lengthA * lengthB * lengthC != 0 else { return (0, 0) }
    let semiperimeter = (lengthA + lengthB + lengthC) / 2.0
    let area = Foundation.sqrt(
      max(0, semiperimeter *
        (semiperimeter - lengthA) *
        (semiperimeter - lengthB) *
        (semiperimeter - lengthC))
    )
    let curvature = (4.0 * area) / (lengthA * lengthB * lengthC)
    guard curvature.isFinite, curvature > 0 else { return (0, 0) }

    let radius = 1.0 / curvature
    let cosine = ((2.0 * radius * radius) - (lengthB * lengthB)) /
      (2.0 * radius * radius)
    let angle = Foundation.acos(cosine.clamped(to: -1.0 ... 1.0))
    let arcLength = radius * angle
    return (curvature, arcLength.isFinite ? arcLength : 0)
  }

  /// Mirrors mapd's production curvature at one route node: three raw
  /// triplet measurements (five source nodes) averaged by represented arc
  /// length.  `nil` means adjacent-way context is required.
  public static func runtimeSmoothedCurvature(
    nodes: [MapTileNode],
    nodeIndex: Int,
    mergeOrSplitNodeIndices: Set<Int> = []
  ) -> Double? {
    runtimeCurvatureEstimate(
      nodes: nodes,
      nodeIndex: nodeIndex,
      mergeOrSplitNodeIndices: mergeOrSplitNodeIndices
    )?.curvature
  }

  public static func runtimeCurvatureEstimate(
    nodes: [MapTileNode],
    nodeIndex: Int,
    mergeOrSplitNodeIndices: Set<Int> = []
  ) -> MapRuntimeCurvatureEstimate? {
    guard nodeIndex >= 2, nodeIndex + 2 < nodes.count else { return nil }
    var measurements = (1 ..< nodes.count - 1).map { center in
      curvatureMeasurement(
        previous: nodes[center - 1],
        current: nodes[center],
        next: nodes[center + 1]
      )
    }

    // Exact mapd parity: lane merges/splits are clamped before the three
    // adjacent raw measurements are averaged.  The index arithmetic mirrors
    // openpilot-mapd/math.go:GetStateCurvatures, including its use of the raw
    // curvature-array index for the 15 m proximity checks.
    for mergeNodeIndex in mergeOrSplitNodeIndices.sorted()
      where nodes.indices.contains(mergeNodeIndex) {
      if mergeNodeIndex >= 2 {
        for index in [mergeNodeIndex - 2, mergeNodeIndex - 1]
          where measurements.indices.contains(index) {
          measurements[index].curvature = mergeOrSplitCurvature
        }
      }
      if mergeNodeIndex >= 3 {
        for index in stride(from: mergeNodeIndex - 3, through: 0, by: -1) {
          guard measurements.indices.contains(index) else { continue }
          if distanceMeters(from: nodes[mergeNodeIndex], to: nodes[index]) > mergeOrSplitRadiusMeters {
            break
          }
          measurements[index].curvature = mergeOrSplitCurvature
        }
      }
      if mergeNodeIndex < measurements.count {
        for index in mergeNodeIndex ..< measurements.count {
          if distanceMeters(from: nodes[mergeNodeIndex], to: nodes[index]) > mergeOrSplitRadiusMeters {
            break
          }
          measurements[index].curvature = mergeOrSplitCurvature
        }
      }
    }

    let firstMeasurement = nodeIndex - 2
    let localMeasurements = measurements[firstMeasurement ... firstMeasurement + 2]
    let totalArcLength = localMeasurements.reduce(0) { $0 + $1.arcLengthMeters }
    guard totalArcLength.isFinite, totalArcLength > 0 else {
      return MapRuntimeCurvatureEstimate(curvature: 0, supportMeters: 0)
    }
    let curvature = localMeasurements.reduce(0) {
      $0 + $1.curvature * $1.arcLengthMeters
    } / totalArcLength
    let firstSupportNode = nodeIndex - 2
    let lastSupportNode = nodeIndex + 2
    let routeSpanMeters = (firstSupportNode ..< lastSupportNode).reduce(0.0) {
      $0 + distanceMeters(from: nodes[$1], to: nodes[$1 + 1])
    }
    return MapRuntimeCurvatureEstimate(
      curvature: curvature,
      supportMeters: routeSpanMeters
    )
  }

  public static func runtimeSmoothedCurvatures(nodes: [MapTileNode]) -> [Double?] {
    nodes.indices.map { runtimeSmoothedCurvature(nodes: nodes, nodeIndex: $0) }
  }

  public static func bakedSpeedMPS(
    curvature: Double,
    parameters: SigmoidParameters,
    maximumSpeedMPS: Double = defaultMaximumSpeedMPS
  ) -> Double {
    let magnitude = abs(curvature)
    guard magnitude >= 1.0e-7 else { return maximumSpeedMPS }
    return min(maximumSpeedMPS, Foundation.sqrt(VTSCMath.evaluate(parameters, curvature: magnitude) / magnitude))
  }

  public static func bakedSpeeds(
    nodes: [MapTileNode],
    parameters: SigmoidParameters,
    maximumSpeedMPS: Double = defaultMaximumSpeedMPS
  ) -> [Double] {
    guard !nodes.isEmpty else { return [] }
    guard nodes.count >= 3 else { return Array(repeating: maximumSpeedMPS, count: nodes.count) }
    var speeds = Array(repeating: maximumSpeedMPS, count: nodes.count)
    for index in 1 ..< nodes.count - 1 {
      speeds[index] = bakedSpeedMPS(
        curvature: curvature(previous: nodes[index - 1], current: nodes[index], next: nodes[index + 1]),
        parameters: parameters,
        maximumSpeedMPS: maximumSpeedMPS
      )
    }
    return speeds
  }

  public static func sigmoidHash(
    parameters: SigmoidParameters,
    maximumSpeedMPS: Double = defaultMaximumSpeedMPS
  ) -> String {
    let canonical = String(
      format: "%.6f|%.6f|%.6f|%.6f|%.4f|%.4f|%.2f",
      locale: Locale(identifier: "en_US_POSIX"),
      parameters.a,
      parameters.b,
      parameters.c,
      parameters.d,
      parameters.minLat,
      parameters.maxLat,
      maximumSpeedMPS
    )
    let digest = SHA256.hash(data: Data(canonical.utf8))
    return digest.prefix(6).map { String(format: "%02x", $0) }.joined()
  }
}
