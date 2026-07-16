import Foundation
import Testing
@testable import VTSCTunerCore

private let studyLatitude = 38.73

private func geographicRoute(
  _ localPoints: [(x: Double, y: Double)],
  keyPrefix: String = "p"
) -> [WholeCurveInputPoint] {
  let metersPerDegreeLatitude = MapBakeMath.earthRadiusMeters * .pi / 180
  let metersPerDegreeLongitude = metersPerDegreeLatitude * cos(studyLatitude * .pi / 180)
  return localPoints.enumerated().map { index, point in
    WholeCurveInputPoint(
      latitude: studyLatitude + point.y / metersPerDegreeLatitude,
      longitude: -120.75 + point.x / metersPerDegreeLongitude,
      sourceKey: "\(keyPrefix):\(index)"
    )
  }
}

private func constantArc(
  radius: Double,
  arcLength: Double,
  sourceSpacing: Double,
  clockwise: Bool = false
) -> [WholeCurveInputPoint] {
  let count = Int(ceil(arcLength / sourceSpacing))
  let direction = clockwise ? -1.0 : 1.0
  let points = (0 ... count).map { index -> (x: Double, y: Double) in
    let distance = min(arcLength, Double(index) * sourceSpacing)
    let angle = direction * distance / radius
    return (radius * sin(angle), direction * radius * (1 - cos(angle)))
  }
  return geographicRoute(points)
}

private func integratedRoute(
  sections: [(length: Double, curvature: Double)],
  spacing: Double = 5
) -> [WholeCurveInputPoint] {
  var x = 0.0
  var y = 0.0
  var heading = 0.0
  var points: [(x: Double, y: Double)] = [(x, y)]
  for section in sections {
    let steps = Int(ceil(section.length / spacing))
    for step in 0 ..< steps {
      let distance = min(spacing, section.length - Double(step) * spacing)
      heading += 0.5 * section.curvature * distance
      x += cos(heading) * distance
      y += sin(heading) * distance
      heading += 0.5 * section.curvature * distance
      points.append((x, y))
    }
  }
  return geographicRoute(points)
}

private func localPoint(_ point: WholeCurveResampledPoint) -> (x: Double, y: Double) {
  let metersPerDegreeLatitude = MapBakeMath.earthRadiusMeters * .pi / 180
  let metersPerDegreeLongitude = metersPerDegreeLatitude * cos(studyLatitude * .pi / 180)
  return (
    (point.longitude + 120.75) * metersPerDegreeLongitude,
    (point.latitude - studyLatitude) * metersPerDegreeLatitude
  )
}

private func distanceToSegment(
  point: (x: Double, y: Double),
  first: (x: Double, y: Double),
  second: (x: Double, y: Double)
) -> Double {
  let dx = second.x - first.x
  let dy = second.y - first.y
  let lengthSquared = dx * dx + dy * dy
  guard lengthSquared > 0 else { return hypot(point.x - first.x, point.y - first.y) }
  let fraction = (
    (point.x - first.x) * dx + (point.y - first.y) * dy
  ) / lengthSquared
  let clamped = fraction.clamped(to: 0 ... 1)
  return hypot(
    point.x - (first.x + dx * clamped),
    point.y - (first.y + dy * clamped)
  )
}

private func testMedian(_ values: [Double]) -> Double {
  let sorted = values.sorted()
  guard !sorted.isEmpty else { return 0 }
  let middle = sorted.count / 2
  return sorted.count.isMultiple(of: 2)
    ? 0.5 * (sorted[middle - 1] + sorted[middle])
    : sorted[middle]
}

@Test func wholeCurveEstimatorLeavesStraightRoadAlone() {
  let route = geographicRoute((0 ... 120).map { (Double($0) * 5, 0) })
  let estimate = WholeCurveEstimator.estimate(route: route)

  #expect(estimate.points.count > 100)
  #expect(estimate.events.isEmpty)
  #expect(estimate.points.compactMap(\.curvature100).allSatisfy { abs($0) < 1.0e-9 })
}

@Test func wholeCurveEstimatorRecoversAConstantRadiusAcrossSourceSpacings() throws {
  for spacing in [5.0, 20.0, 80.0] {
    let estimate = WholeCurveEstimator.estimate(
      route: constantArc(radius: 250, arcLength: 520, sourceSpacing: spacing)
    )
    let event = try #require(estimate.events.max(by: { $0.lengthMeters < $1.lengthMeters }))

    // With 80 m source gaps there is not enough information to reconstruct
    // the arc between map vertices. Shape-preserving chord resampling remains
    // intentionally conservative and explicitly marks that result sparse.
    let tolerance = spacing >= 60 ? (1.0 / 250.0) * 0.30 : 0.000_30
    #expect(abs(event.controllingCurvature - 1.0 / 250.0) < tolerance)
    #expect(event.lengthMeters >= 350)
    #expect(event.signedTurnRadians > 0)
    if spacing >= 60 {
      #expect(event.flags.contains(.sparseSourceGeometry))
      #expect(event.confidence != .high)
    }
  }
}

@Test func wholeCurveEstimatorMergesCompoundContinuationBeforeQualification() throws {
  var configuration = WholeCurveConfiguration()
  configuration.shortSpanMeters = 20
  configuration.nominalSpanMeters = 40
  configuration.longSpanMeters = 80
  configuration.enterCurvature = 0.004
  configuration.exitCurvature = 0.001
  configuration.sustainedDistanceMeters = 20
  configuration.sameSignMergeGapMeters = 30
  configuration.minimumEventLengthMeters = 10
  let route = integratedRoute(sections: [
    (100, 0),
    (60, 0.010),
    (25, 0),
    (15, 0.010),
    (100, 0),
  ])

  let event = try #require(
    WholeCurveEstimator.estimate(route: route, configuration: configuration).events.first
  )

  // The 15 m continuation cannot satisfy the 20 m sustained-enter rule by
  // itself. It must nevertheless remain attached to the qualifying main bend.
  #expect(event.lengthMeters >= 95)
  #expect(event.endIndex > event.apexIndex)
}

@Test func wholeCurveEstimatorResamplingCannotOvershootNonuniformSourceSegments() {
  let local: [(x: Double, y: Double)] = [
    (0, 0),
    (150, 0),
    (154, 28),
    (161, -6),
    (320, -6),
  ]
  let estimate = WholeCurveEstimator.estimate(route: geographicRoute(local, keyPrefix: "node"))

  #expect(!estimate.points.isEmpty)
  for sample in estimate.points {
    let indices = sample.sourceKeys.compactMap { key -> Int? in
      guard key.hasPrefix("node:") else { return nil }
      return Int(key.dropFirst("node:".count))
    }.sorted()
    #expect(indices.count == 2)
    guard let first = indices.first,
          let second = indices.last,
          local.indices.contains(first),
          local.indices.contains(second)
    else { continue }
    #expect(
      distanceToSegment(
        point: localPoint(sample),
        first: local[first],
        second: local[second]
      ) < 0.001
    )
  }
}

@Test func wholeCurveEstimatorDoesNotPromoteAOneNodeZigzagToCompactApex() throws {
  let route = geographicRoute([
    (0, 0),
    (30, 0),
    (60, 0),
    (90, 0),
    (120, 0),
    (150, 0),
    (180, 0),
    (210, 24),
    (240, 0),
    (270, 0),
    (300, 0),
    (330, 0),
    (360, 0),
    (390, 0),
    (420, 0),
  ], keyPrefix: "zigzag")
  let events = WholeCurveEstimator.estimate(route: route).events

  #expect(!events.isEmpty)
  for event in events {
    let nominal = testMedian(
      [event.curvature60, event.curvature100, event.curvature160].compactMap { $0 }.map(abs)
    )
    #expect(abs(abs(event.controllingCurvature) - nominal) < 1.0e-12)
  }
}

@Test func wholeCurveEstimatorIsInsensitiveToDuplicateNodes() throws {
  let baselineRoute = constantArc(radius: 300, arcLength: 520, sourceSpacing: 20)
  var duplicated: [WholeCurveInputPoint] = []
  for (index, point) in baselineRoute.enumerated() {
    duplicated.append(point)
    if index.isMultiple(of: 3) { duplicated.append(point) }
  }
  let baseline = try #require(WholeCurveEstimator.estimate(route: baselineRoute).events.first)
  let repeated = try #require(WholeCurveEstimator.estimate(route: duplicated).events.first)

  #expect(abs(baseline.controllingCurvature - repeated.controllingCurvature) < 1.0e-12)
  #expect(abs(baseline.lengthMeters - repeated.lengthMeters) < 0.1)
}

@Test func wholeCurveEstimatorReversesSignWithoutChangingMagnitude() throws {
  let route = constantArc(radius: 220, arcLength: 500, sourceSpacing: 10)
  let forward = try #require(WholeCurveEstimator.estimate(route: route).events.first)
  let reverse = try #require(WholeCurveEstimator.estimate(route: route.reversed()).events.first)

  #expect(forward.controllingCurvature * reverse.controllingCurvature < 0)
  #expect(abs(abs(forward.controllingCurvature) - abs(reverse.controllingCurvature)) < 1.0e-8)
  #expect(abs(forward.lengthMeters - reverse.lengthMeters) < 5.1)
}

@Test func wholeCurveEstimatorSplitsSShapeIntoSignedLobes() {
  let route = integratedRoute(sections: [
    (120, 0),
    (240, 0.006),
    (50, 0),
    (240, -0.006),
    (120, 0),
  ])
  let events = WholeCurveEstimator.estimate(route: route).events

  #expect(events.count == 2)
  #expect(events[0].controllingCurvature > 0)
  #expect(events[1].controllingCurvature < 0)
  #expect(events.allSatisfy { $0.lengthMeters >= 150 })
}

@Test func wholeCurveEstimatorKeepsASustainedCompactApex() throws {
  var configuration = WholeCurveConfiguration()
  configuration.shortSpanMeters = 40
  configuration.nominalSpanMeters = 120
  configuration.longSpanMeters = 200
  let route = integratedRoute(sections: [
    (150, 0),
    (120, 0.003),
    (70, 0.040),
    (120, 0.003),
    (150, 0),
  ])
  let event = try #require(
    WholeCurveEstimator.estimate(route: route, configuration: configuration).events.first
  )
  let peak60 = try #require(event.curvature60)
  let nominal = testMedian(
    [event.curvature60, event.curvature100, event.curvature160].compactMap { $0 }.map(abs)
  )
  #expect(abs(event.controllingCurvature - peak60) < 1.0e-12)
  #expect(abs(event.controllingCurvature) >= nominal * 1.20)
  #expect(event.apexIndex > event.startIndex)
  #expect(event.apexIndex < event.endIndex)
}

@Test func wholeCurveProfileTightensProgressivelyThroughADecreasingRadiusBend() throws {
  let route = integratedRoute(sections: [
    (150, 0),
    (100, 0.0035),
    (100, 0.0045),
    (100, 0.0060),
    (150, 0),
  ])
  let estimate = WholeCurveEstimator.estimate(route: route)
  let event = try #require(estimate.events.first)
  let eventPoints = Array(estimate.points[event.startIndex ... event.endIndex])
  let thirds = max(1, eventPoints.count / 3)
  let entryMaximum = eventPoints.prefix(thirds).map(\.curvatureCoefficient).max() ?? 1
  let apexMaximum = eventPoints.suffix(thirds).map(\.curvatureCoefficient).max() ?? 1

  #expect(apexMaximum > entryMaximum)
  #expect(event.maximumApexCoefficient == apexMaximum)
  #expect(abs(estimate.points[event.profileApexIndex].profileCurvature) > abs(event.controllingCurvature))
}

@Test func wholeCurveProfileRetainsTwoApexMinimaInsideOneContinuousEvent() throws {
  let route = integratedRoute(sections: [
    (150, 0),
    (80, 0.0040),
    (120, 0.0060),
    (80, 0.0040),
    (120, 0.0065),
    (80, 0.0040),
    (150, 0),
  ])
  let estimate = WholeCurveEstimator.estimate(route: route)
  let event = try #require(estimate.events.first)
  #expect(estimate.events.count == 1)

  let firstApex = estimate.points.filter { (310...380).contains($0.distanceMeters) }
    .map(\.curvatureCoefficient).max() ?? 1
  let saddle = estimate.points.filter { (400...430).contains($0.distanceMeters) }
    .map(\.curvatureCoefficient).max() ?? .infinity
  let secondApex = estimate.points.filter { (470...540).contains($0.distanceMeters) }
    .map(\.curvatureCoefficient).max() ?? 1

  #expect(firstApex > 1)
  #expect(secondApex > 1)
  #expect(firstApex > saddle)
  #expect(secondApex > saddle)
  #expect(event.maximumApexCoefficient == max(firstApex, secondApex))
}

@Test func wholeCurveEstimatorFlagsSparseSourceGeometry() throws {
  let route = constantArc(radius: 250, arcLength: 560, sourceSpacing: 80)
  let event = try #require(WholeCurveEstimator.estimate(route: route).events.first)

  #expect(event.maximumSourceGapMeters > 60)
  #expect(event.flags.contains(.sparseSourceGeometry))
  #expect(event.confidence != .high)
}
