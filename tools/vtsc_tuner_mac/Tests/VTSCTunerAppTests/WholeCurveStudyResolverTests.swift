import Foundation
import Testing
@testable import VTSCTunerApp
import VTSCTunerCore

private func resolverStudyWay(
  id: String,
  coordinates: [(Double, Double)]
) -> MapRenderedWay {
  MapRenderedWay(
    id: id,
    name: "Resolver Study Road",
    reference: "US 50",
    nodes: coordinates.map {
      MapRenderedNode(
        latitude: $0.0,
        longitude: $0.1,
        curvature: 0,
        bakedSpeedMPS: nil,
        proposedSpeedMPS: 70
      )
    },
    tilePath: "/tmp/\(id)",
    tileHash: "tile-\(id)",
    schemaVersion: 1,
    maxSpeedMPS: 30,
    advisorySpeedMPS: 0,
    maxSpeedForwardMPS: 30,
    maxSpeedBackwardMPS: 0,
    lanes: 2,
    oneWay: true,
    hazard: "",
    windingForwardLevel: 0,
    windingBackwardLevel: 0,
    windingForwardScore: 0,
    windingBackwardScore: 0,
    windingForwardConfidence: 0,
    windingBackwardConfidence: 0
  )
}

private func resolverStudyPath() -> [(Double, Double)] {
  let latitude = 38.73
  let metersPerDegreeLatitude = MapBakeMath.earthRadiusMeters * .pi / 180
  let metersPerDegreeLongitude = metersPerDegreeLatitude * cos(latitude * .pi / 180)
  var local: [(x: Double, y: Double)] = [(0, 0)]
  var x = 0.0
  var y = 0.0
  var heading = 0.0
  for (length, curvature) in [(160.0, 0.0), (320.0, 0.004), (160.0, 0.0)] {
    for _ in 0 ..< Int(length / 20) {
      heading += curvature * 10
      x += cos(heading) * 20
      y += sin(heading) * 20
      heading += curvature * 10
      local.append((x, y))
    }
  }
  return local.map {
    (latitude + $0.y / metersPerDegreeLatitude, -120.75 + $0.x / metersPerDegreeLongitude)
  }
}

@Test func calibrationResolutionUsesTheWholeCurveV2ProfileApex() throws {
  let coordinates = resolverStudyPath()
  let ways = MapRuntimeCurvatureResolver.resolve(
    ways: [
      resolverStudyWay(id: "straight-before", coordinates: Array(coordinates[0 ... 8])),
      resolverStudyWay(id: "curve", coordinates: Array(coordinates[8 ... 24])),
      resolverStudyWay(id: "straight-after", coordinates: Array(coordinates[24 ... 32])),
    ],
    parameters: .checkoutFallback
  )
  let curve = try #require(ways.first { $0.id == "curve" })
  let sampleNode = curve.nodes[8]
  let sample = MapCalibrationSample(
    sourceKey: "curve:8",
    roadName: curve.name,
    reference: curve.reference,
    latitude: sampleNode.latitude,
    longitude: sampleNode.longitude,
    curvature: sampleNode.curvature,
    proposedSpeedMPH: 55,
    effectiveSpeedMPH: 55,
    desiredSpeedMPH: 55
  )
  let route = try #require(MapRuntimeCurvatureResolver.wholeCurveStudyRoutes(
    ways: ways,
    focusSourceKeys: [sample.sourceKey]
  ).first)
  let estimate = WholeCurveEstimator.estimate(route: route.points.map {
    WholeCurveInputPoint(latitude: $0.node.latitude, longitude: $0.node.longitude, sourceKey: $0.sourceKey)
  })
  let event = try #require(estimate.events.first { $0.sourceKeys.contains(sample.sourceKey) })
  let expected = abs(estimate.points[event.profileApexIndex].profileCurvature)

  let resolution = try #require(MapWholeCurveStudyResolver.calibrationResolutions(
    ways: ways,
    calibrationSamples: [sample]
  )[sample.sourceKey])

  #expect(MapRuntimeCurvatureResolver.estimatorVersion == 6)
  #expect(resolution.curvature == expected)
  #expect(resolution.supportMeters == event.lengthMeters)
  #expect(resolution.eventID == event.directionalID)
}

@Test func wholeCurveStudyRejectsNearbySampleWithoutEventProvenance() throws {
  let coordinates = resolverStudyPath()
  let ways = [
    resolverStudyWay(id: "straight-before", coordinates: Array(coordinates[0 ... 8])),
    resolverStudyWay(id: "curve", coordinates: Array(coordinates[8 ... 24])),
    resolverStudyWay(id: "straight-after", coordinates: Array(coordinates[24 ... 32])),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let straight = try #require(resolved.first { $0.id == "straight-before" })
  let sampleNode = straight.nodes[3]
  let sourceKey = "straight-before:3"
  let sample = MapCalibrationSample(
    sourceKey: sourceKey,
    roadName: straight.name,
    reference: straight.reference,
    latitude: sampleNode.latitude,
    longitude: sampleNode.longitude,
    curvature: sampleNode.curvature,
    proposedSpeedMPH: 65,
    effectiveSpeedMPH: 65,
    desiredSpeedMPH: 65
  )

  // Prove the fixture contains a real curve close enough that the old
  // straight-line fallback would have attached this unrelated straight node.
  let route = try #require(MapRuntimeCurvatureResolver.wholeCurveStudyRoutes(
    ways: resolved,
    focusSourceKeys: [sourceKey]
  ).first)
  let estimate = WholeCurveEstimator.estimate(route: route.points.map {
    WholeCurveInputPoint(
      latitude: $0.node.latitude,
      longitude: $0.node.longitude,
      sourceKey: $0.sourceKey
    )
  })
  let event = try #require(estimate.events.first)
  #expect(!event.sourceKeys.contains(sourceKey))
  let sampleMapNode = MapTileNode(latitude: sample.latitude, longitude: sample.longitude)
  let nearestStraightLineDistance = estimate.points[event.startIndex ... event.endIndex].map {
    MapBakeMath.distanceMeters(
      from: sampleMapNode,
      to: MapTileNode(latitude: $0.latitude, longitude: $0.longitude)
    )
  }.min() ?? .infinity
  #expect(nearestStraightLineDistance < 120)

  let events = MapWholeCurveStudyResolver.resolve(
    ways: resolved,
    calibrationSamples: [sample],
    parameters: .checkoutFallback,
    bands: []
  )
  #expect(events.isEmpty)
}

private func mergeCandidate(
  id: String,
  confidence: WholeCurveConfidence,
  flags: [String],
  sampleNumber: Int
) -> MapWholeCurveEvent {
  MapWholeCurveEvent(
    id: id,
    roadName: "Resolver Study Road",
    reference: "US 50",
    travelDirection: "Eastbound",
    bendDirection: "Left",
    points: [
      MapWholeCurvePoint(
        latitude: 38.0,
        longitude: -121.0,
        distanceMeters: 0,
        currentMapdSpeedMPH: 60,
        wholeCurveSpeedMPH: 62,
        curvature60: 0.005,
        curvature100: 0.005,
        curvature160: 0.005
      ),
      MapWholeCurvePoint(
        latitude: 38.001,
        longitude: -120.999,
        distanceMeters: 120,
        currentMapdSpeedMPH: 60,
        wholeCurveSpeedMPH: 62,
        curvature60: 0.005,
        curvature100: 0.005,
        curvature160: 0.005
      ),
    ],
    sourceKeys: ["curve:1", "curve:2"],
    lengthMeters: 120,
    turnDegrees: 35,
    apexLatitude: 38.0005,
    apexLongitude: -120.9995,
    curvature60: 0.005,
    curvature100: 0.005,
    curvature160: 0.005,
    controllingCurvature: 0.005,
    wholeCurveSpeedMPH: 62,
    currentMinimumSpeedMPH: 58,
    currentMaximumSpeedMPH: 65,
    confidenceLabel: confidence.rawValue,
    flags: flags,
    bankSampleNumbers: [sampleNumber],
    bankTargetsMPH: [62]
  )
}

@Test func wholeCurveMergeKeepsWeakestConfidenceAndUnionedFlags() {
  let high = mergeCandidate(
    id: "preferred-high",
    confidence: .high,
    flags: [],
    sampleNumber: 1
  )
  let low = mergeCandidate(
    id: "lower-quality",
    confidence: .low,
    flags: [
      WholeCurveFlag.sparseSourceGeometry.rawValue,
      "Route branch is ambiguous after this curve",
    ],
    sampleNumber: 2
  )
  let merged = MapWholeCurveStudyResolver.merge(high, low)

  #expect(merged.id == high.id)
  #expect(merged.confidenceLabel == WholeCurveConfidence.low.rawValue)
  #expect(merged.flags == low.flags.sorted())
  #expect(merged.bankSampleNumbers == [1, 2])

  var inconsistentFlaggedHigh = high
  inconsistentFlaggedHigh.id = "flagged-high"
  inconsistentFlaggedHigh.flags = [WholeCurveFlag.incompleteStraightShoulder.rawValue]
  let capped = MapWholeCurveStudyResolver.merge(high, inconsistentFlaggedHigh)
  #expect(capped.confidenceLabel == WholeCurveConfidence.review.rawValue)
}
