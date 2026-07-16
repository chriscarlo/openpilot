import CryptoKit
import Foundation

public struct WholeCurveInputPoint: Equatable, Sendable {
  public var latitude: Double
  public var longitude: Double
  public var sourceKey: String

  public init(latitude: Double, longitude: Double, sourceKey: String = "") {
    self.latitude = latitude
    self.longitude = longitude
    self.sourceKey = sourceKey
  }
}

public struct WholeCurveConfiguration: Equatable, Sendable {
  public var resampleSpacingMeters = 5.0
  public var shortSpanMeters = 60.0
  public var nominalSpanMeters = 100.0
  public var longSpanMeters = 160.0
  public var enterCurvature = 0.0015
  public var exitCurvature = 0.0010
  public var sustainedDistanceMeters = 30.0
  public var sameSignMergeGapMeters = 30.0
  public var minimumEventLengthMeters = 20.0
  public var duplicatePointDistanceMeters = 0.5
  public var apexDetailGain = 1.0
  public var apexDetailMinimumRatio = 1.05
  public var apexDetailMaximumRatio = 2.0

  public init() {}
}

public enum WholeCurveConfidence: String, Equatable, Sendable {
  case high = "High confidence"
  case review = "Review"
  case low = "Low confidence"
}

public enum WholeCurveFlag: String, Equatable, Hashable, Sendable {
  case missingScaleContext = "Incomplete 60/100/160 m route context"
  case incompleteStraightShoulder = "Entry or exit lacks 30 m of straight-road context"
  case sparseSourceGeometry = "Source map geometry has a gap over 60 m"
  case unstableTurnSign = "Turn direction is not coherent across the event"
  case highScaleSpread = "Curvature changes substantially across measurement scales"
  case shortEvent = "Detected bend is shorter than 30 m"
  case invalidGeometry = "Some curve geometry could not be measured reliably"
}

public struct WholeCurveResampledPoint: Equatable, Sendable {
  public var latitude: Double
  public var longitude: Double
  public var distanceMeters: Double
  public var nearestSourceKey: String
  public var sourceKeys: Set<String>
  public var sourceGapMeters: Double
  public var curvature60: Double?
  public var curvature100: Double?
  public var curvature160: Double?
  public var profileCurvature: Double
  public var curvatureCoefficient: Double
}

public struct WholeCurveEvent: Equatable, Identifiable, Sendable {
  public var id: String { directionalID }
  public var physicalID: String
  public var directionalID: String
  public var startIndex: Int
  public var endIndex: Int
  public var apexIndex: Int
  public var profileApexIndex: Int
  public var lengthMeters: Double
  public var signedTurnRadians: Double
  public var signCoherence: Double
  public var curvature60: Double?
  public var curvature100: Double?
  public var curvature160: Double?
  public var controllingCurvature: Double
  public var maximumApexCoefficient: Double
  public var scaleSpread: Double?
  public var maximumSourceGapMeters: Double
  public var sourceKeys: Set<String>
  public var confidence: WholeCurveConfidence
  public var flags: [WholeCurveFlag]
}

public struct WholeCurveEstimate: Equatable, Sendable {
  public var points: [WholeCurveResampledPoint]
  public var events: [WholeCurveEvent]

  public init(points: [WholeCurveResampledPoint], events: [WholeCurveEvent]) {
    self.points = points
    self.events = events
  }
}

public enum WholeCurveEstimator {
  private struct PlanarPoint {
    var latitude: Double
    var longitude: Double
    var x: Double
    var y: Double
    var distance: Double
    var sourceKeys: Set<String>
  }

  private struct PlanarSample {
    var latitude: Double
    var longitude: Double
    var x: Double
    var y: Double
    var distance: Double
    var nearestSourceKey: String
    var sourceKeys: Set<String>
    var sourceGapMeters: Double
  }

  private struct SignedRun {
    var start: Int
    var end: Int
    var sign: Double
  }

  private struct ScalePeak {
    var value: Double
    var index: Int
  }

  private struct SourceTurnVertex {
    var distance: Double
    var signedCurvature: Double
    var signedTurnRadians: Double
  }

  private struct SourceTurnEvidence {
    var supportingVertexCount: Int
    var supportingSpanMeters: Double
    var directionalCoherence: Double
  }

  public static func estimate(
    route: [WholeCurveInputPoint],
    configuration: WholeCurveConfiguration = WholeCurveConfiguration()
  ) -> WholeCurveEstimate {
    guard route.count >= 3,
          configuration.resampleSpacingMeters > 0,
          configuration.shortSpanMeters > 0,
          configuration.nominalSpanMeters > 0,
          configuration.longSpanMeters > 0
    else { return WholeCurveEstimate(points: [], events: []) }

    let deduplicated = deduplicate(route, threshold: configuration.duplicatePointDistanceMeters)
    guard deduplicated.count >= 3,
          let lastDistance = deduplicated.last?.distance,
          lastDistance >= configuration.shortSpanMeters
    else { return WholeCurveEstimate(points: [], events: []) }

    let samples = resample(deduplicated, spacing: configuration.resampleSpacingMeters)
    let sourceTurns = sourceTurnVertices(deduplicated)
    var outputPoints = samples.map {
      WholeCurveResampledPoint(
        latitude: $0.latitude,
        longitude: $0.longitude,
        distanceMeters: $0.distance,
        nearestSourceKey: $0.nearestSourceKey,
        sourceKeys: $0.sourceKeys,
        sourceGapMeters: $0.sourceGapMeters,
        curvature60: nil,
        curvature100: nil,
        curvature160: nil,
        profileCurvature: 0,
        curvatureCoefficient: 1
      )
    }
    for index in samples.indices {
      outputPoints[index].curvature60 = signedCurvature(
        samples: samples,
        at: samples[index].distance,
        span: configuration.shortSpanMeters
      )
      outputPoints[index].curvature100 = signedCurvature(
        samples: samples,
        at: samples[index].distance,
        span: configuration.nominalSpanMeters
      )
      outputPoints[index].curvature160 = signedCurvature(
        samples: samples,
        at: samples[index].distance,
        span: configuration.longSpanMeters
      )
    }

    let runs = eventRuns(points: outputPoints, configuration: configuration)
    var events = runs.compactMap {
      makeEvent(
        run: $0,
        points: outputPoints,
        sourceTurns: sourceTurns,
        configuration: configuration
      )
    }
    for eventIndex in events.indices {
      var maximumCoefficient = 1.0
      var profileApexIndex = events[eventIndex].apexIndex
      let detailBaseline = apexDetailBaseline(
        points: Array(outputPoints[events[eventIndex].startIndex ... events[eventIndex].endIndex]),
        event: events[eventIndex]
      )
      for pointIndex in events[eventIndex].startIndex ... events[eventIndex].endIndex {
        let coefficient = apexDetailCoefficient(
          point: outputPoints[pointIndex],
          event: events[eventIndex],
          detailBaseline: detailBaseline,
          configuration: configuration
        )
        let curvature = events[eventIndex].controllingCurvature * coefficient
        if abs(curvature) > abs(outputPoints[pointIndex].profileCurvature) {
          outputPoints[pointIndex].profileCurvature = curvature
          outputPoints[pointIndex].curvatureCoefficient = coefficient
        }
        if coefficient > maximumCoefficient {
          maximumCoefficient = coefficient
          profileApexIndex = pointIndex
        }
      }
      events[eventIndex].profileApexIndex = profileApexIndex
      events[eventIndex].maximumApexCoefficient = maximumCoefficient
    }
    return WholeCurveEstimate(points: outputPoints, events: events)
  }

  private static func apexDetailCoefficient(
    point: WholeCurveResampledPoint,
    event: WholeCurveEvent,
    detailBaseline: Double,
    configuration: WholeCurveConfiguration
  ) -> Double {
    let baseline = abs(detailBaseline)
    guard baseline > 1.0e-12,
          configuration.apexDetailGain > 0,
          configuration.apexDetailMaximumRatio > 1
    else { return 1 }
    let detail = pointDetailMagnitude(point: point, event: event)
    guard detail > 0 else { return 1 }
    let ratio = detail / baseline
    guard ratio >= max(1, configuration.apexDetailMinimumRatio) else { return 1 }
    let amplified = 1 + configuration.apexDetailGain * (ratio - 1)
    return min(max(amplified, 1), configuration.apexDetailMaximumRatio)
  }

  private static func apexDetailBaseline(
    points: [WholeCurveResampledPoint],
    event: WholeCurveEvent
  ) -> Double {
    let values = points.map { pointDetailMagnitude(point: $0, event: event) }.filter { $0 > 0 }
    let baseline = median(values)
    return baseline > 1.0e-12 ? baseline : abs(event.controllingCurvature)
  }

  private static func pointDetailMagnitude(
    point: WholeCurveResampledPoint,
    event: WholeCurveEvent
  ) -> Double {
    [point.curvature60, point.curvature100].compactMap { $0 }
      .filter {
        $0.isFinite && abs($0) > 0 && $0.sign == event.controllingCurvature.sign
      }
      .map(abs)
      .max() ?? 0
  }

  private static func deduplicate(
    _ route: [WholeCurveInputPoint],
    threshold: Double
  ) -> [PlanarPoint] {
    guard !route.isEmpty else { return [] }
    // A route-mean origin gives forward and reversed traversals the same local
    // projection, which is important when pairing opposite-direction studies.
    let finiteRoute = route.filter { $0.latitude.isFinite && $0.longitude.isFinite }
    guard !finiteRoute.isEmpty else { return [] }
    let latitude0 = 0.5 * (
      (finiteRoute.map(\.latitude).min() ?? 0) + (finiteRoute.map(\.latitude).max() ?? 0)
    ) * .pi / 180
    let longitude0 = 0.5 * (
      (finiteRoute.map(\.longitude).min() ?? 0) + (finiteRoute.map(\.longitude).max() ?? 0)
    ) * .pi / 180
    let cosineLatitude = Foundation.cos(latitude0)
    func project(_ point: WholeCurveInputPoint) -> (x: Double, y: Double) {
      let latitude = point.latitude * .pi / 180
      let longitude = point.longitude * .pi / 180
      return (
        MapBakeMath.earthRadiusMeters * (longitude - longitude0) * cosineLatitude,
        MapBakeMath.earthRadiusMeters * (latitude - latitude0)
      )
    }

    var result: [PlanarPoint] = []
    for point in route where point.latitude.isFinite && point.longitude.isFinite {
      let projected = project(point)
      if let previous = result.last {
        let distance = hypot(projected.x - previous.x, projected.y - previous.y)
        if distance < threshold {
          if !point.sourceKey.isEmpty {
            result[result.count - 1].sourceKeys.insert(point.sourceKey)
          }
          continue
        }
        result.append(PlanarPoint(
          latitude: point.latitude,
          longitude: point.longitude,
          x: projected.x,
          y: projected.y,
          distance: previous.distance + distance,
          sourceKeys: point.sourceKey.isEmpty ? [] : [point.sourceKey]
        ))
      } else {
        result.append(PlanarPoint(
          latitude: point.latitude,
          longitude: point.longitude,
          x: projected.x,
          y: projected.y,
          distance: 0,
          sourceKeys: point.sourceKey.isEmpty ? [] : [point.sourceKey]
        ))
      }
    }
    return result
  }

  private static func resample(_ points: [PlanarPoint], spacing: Double) -> [PlanarSample] {
    guard let totalDistance = points.last?.distance, totalDistance > 0 else { return [] }
    // Recompute a nearly-5 m spacing that divides the route exactly. Besides
    // avoiding a short final bin, this makes a reversed route use the same
    // physical sample locations in reverse order.
    let segmentCount = max(1, Int((totalDistance / spacing).rounded()))
    let uniformSpacing = totalDistance / Double(segmentCount)
    let targets = (0 ... segmentCount).map { Double($0) * uniformSpacing }
    var segment = 0
    return targets.map { target in
      while segment + 1 < points.count - 1, points[segment + 1].distance < target {
        segment += 1
      }
      let first = points[segment]
      let second = points[min(segment + 1, points.count - 1)]
      let gap = second.distance - first.distance
      let fraction = gap > 0 ? ((target - first.distance) / gap).clamped(to: 0 ... 1) : 0
      let nearest = fraction < 0.5 ? first : second
      // The map input is a polyline, so resampling stays on each source chord.
      // A free cubic can overshoot nonuniform nodes, manufacture a cusp, and
      // make the estimator report curvature that is absent from the map.
      let x = first.x + (second.x - first.x) * fraction
      let y = first.y + (second.y - first.y) * fraction
      let latitude = first.latitude + (second.latitude - first.latitude) * fraction
      let longitude = first.longitude + (second.longitude - first.longitude) * fraction
      return PlanarSample(
        latitude: latitude,
        longitude: longitude,
        x: x,
        y: y,
        distance: target,
        nearestSourceKey: nearest.sourceKeys.sorted().first ?? "",
        sourceKeys: first.sourceKeys.union(second.sourceKeys),
        sourceGapMeters: gap
      )
    }
  }

  private static func sourceTurnVertices(_ points: [PlanarPoint]) -> [SourceTurnVertex] {
    guard points.count >= 3 else { return [] }
    return (1 ..< points.count - 1).compactMap { index in
      let previous = points[index - 1]
      let current = points[index]
      let next = points[index + 1]
      let inboundDistance = current.distance - previous.distance
      let outboundDistance = next.distance - current.distance
      let supportDistance = 0.5 * (inboundDistance + outboundDistance)
      guard inboundDistance > 0, outboundDistance > 0, supportDistance > 0 else { return nil }

      let inboundHeading = Foundation.atan2(current.y - previous.y, current.x - previous.x)
      let outboundHeading = Foundation.atan2(next.y - current.y, next.x - current.x)
      var turn = outboundHeading - inboundHeading
      while turn > .pi { turn -= 2 * .pi }
      while turn < -.pi { turn += 2 * .pi }
      guard turn.isFinite else { return nil }
      return SourceTurnVertex(
        distance: current.distance,
        signedCurvature: turn / supportDistance,
        signedTurnRadians: turn
      )
    }
  }

  private static func compactApexSourceEvidence(
    sourceTurns: [SourceTurnVertex],
    points: [WholeCurveResampledPoint],
    run: SignedRun,
    peak60: Double,
    configuration: WholeCurveConfiguration
  ) -> SourceTurnEvidence? {
    let threshold = abs(peak60) * 0.85
    let supportedSampleIndices = (run.start ... run.end).filter { index in
      guard let curvature = points[index].curvature60 else { return false }
      return abs(curvature) >= threshold
        && (curvature >= 0 ? 1.0 : -1.0) == run.sign
    }
    guard let firstSample = supportedSampleIndices.first,
          let lastSample = supportedSampleIndices.last
    else { return nil }

    // A 60 m measurement is influenced by source vertices within 30 m of its
    // center. Inspect that exact source envelope rather than treating repeated
    // resampled provenance keys as independent evidence.
    let halfSpan = configuration.shortSpanMeters * 0.5
    let lowerDistance = max(
      points[run.start].distanceMeters,
      points[firstSample].distanceMeters - halfSpan
    )
    let upperDistance = min(
      points[run.end].distanceMeters,
      points[lastSample].distanceMeters + halfSpan
    )
    let localTurns = sourceTurns.filter {
      $0.distance >= lowerDistance && $0.distance <= upperDistance
        && abs($0.signedTurnRadians) > 1.0e-9
    }
    guard !localTurns.isEmpty else { return nil }

    let sourceThreshold = max(configuration.enterCurvature, abs(peak60) * 0.5)
    let supporting = localTurns.filter {
      ($0.signedCurvature >= 0 ? 1.0 : -1.0) == run.sign
        && abs($0.signedCurvature) >= sourceThreshold
    }
    let supportingSpan: Double
    if let first = supporting.first, let last = supporting.last {
      supportingSpan = last.distance - first.distance
    } else {
      supportingSpan = 0
    }
    let signedTurn = localTurns.reduce(0) { $0 + $1.signedTurnRadians }
    let absoluteTurn = localTurns.reduce(0) { $0 + abs($1.signedTurnRadians) }
    return SourceTurnEvidence(
      supportingVertexCount: supporting.count,
      supportingSpanMeters: supportingSpan,
      directionalCoherence: absoluteTurn > 1.0e-9
        ? run.sign * signedTurn / absoluteTurn
        : 0
    )
  }

  private static func signedCurvature(
    samples: [PlanarSample],
    at distance: Double,
    span: Double
  ) -> Double? {
    let halfSpan = span * 0.5
    guard let firstDistance = samples.first?.distance,
          let lastDistance = samples.last?.distance,
          distance - halfSpan >= firstDistance,
          distance + halfSpan <= lastDistance,
          let first = interpolate(samples, distance: distance - halfSpan),
          let middle = interpolate(samples, distance: distance),
          let last = interpolate(samples, distance: distance + halfSpan)
    else { return nil }
    let ab = hypot(middle.x - first.x, middle.y - first.y)
    let bc = hypot(last.x - middle.x, last.y - middle.y)
    let ac = hypot(last.x - first.x, last.y - first.y)
    guard ab > 0, bc > 0, ac > 0 else { return nil }
    let cross = (middle.x - first.x) * (last.y - first.y)
      - (middle.y - first.y) * (last.x - first.x)
    let magnitude = 2 * abs(cross) / (ab * bc * ac)
    guard magnitude.isFinite else { return nil }
    if magnitude < 1.0e-9 { return 0 }
    return cross >= 0 ? magnitude : -magnitude
  }

  private static func interpolate(
    _ samples: [PlanarSample],
    distance: Double
  ) -> (x: Double, y: Double)? {
    guard let first = samples.first, let last = samples.last,
          distance >= first.distance, distance <= last.distance
    else { return nil }
    let spacing = samples.count > 1 ? samples[1].distance - samples[0].distance : 0
    guard spacing > 0 else { return (first.x, first.y) }
    let lower = min(max(Int(floor(distance / spacing)), 0), samples.count - 1)
    let upper = min(lower + 1, samples.count - 1)
    let range = samples[upper].distance - samples[lower].distance
    let fraction = range > 0 ? (distance - samples[lower].distance) / range : 0
    return (
      samples[lower].x + (samples[upper].x - samples[lower].x) * fraction,
      samples[lower].y + (samples[upper].y - samples[lower].y) * fraction
    )
  }

  private static func eventRuns(
    points: [WholeCurveResampledPoint],
    configuration: WholeCurveConfiguration
  ) -> [SignedRun] {
    var rawRuns: [SignedRun] = []
    var index = 0
    while index < points.count {
      guard let curvature = segmentationCurvature(points[index]),
            abs(curvature) >= configuration.exitCurvature
      else {
        index += 1
        continue
      }
      let sign = curvature >= 0 ? 1.0 : -1.0
      let start = index
      var end = index
      while end + 1 < points.count,
            let next = segmentationCurvature(points[end + 1]),
            abs(next) >= configuration.exitCurvature,
            (next >= 0 ? 1.0 : -1.0) == sign {
        end += 1
      }
      rawRuns.append(SignedRun(start: start, end: end, sign: sign))
      index = end + 1
    }

    // Join a coherent bend before asking it to prove that it contains a
    // sustained enter-threshold seed. Otherwise a short same-sign shoulder or
    // compound continuation is discarded before it has a chance to attach to
    // the qualifying body of the curve.
    var merged: [SignedRun] = []
    for run in rawRuns {
      guard var previous = merged.last else {
        merged.append(run)
        continue
      }
      let gap = points[run.start].distanceMeters - points[previous.end].distanceMeters
      let opposingTurn = signedTurn(points: points, start: previous.end, end: run.start)
      if previous.sign == run.sign,
         gap <= configuration.sameSignMergeGapMeters,
         abs(opposingTurn) < 3 * .pi / 180 {
        previous.end = run.end
        merged[merged.count - 1] = previous
      } else {
        merged.append(run)
      }
    }
    return merged.filter { run in
      hasSustainedSeed(
        points: points,
        start: run.start,
        end: run.end,
        sign: run.sign,
        configuration: configuration
      ) && points[run.end].distanceMeters - points[run.start].distanceMeters
        >= configuration.minimumEventLengthMeters
    }
  }

  private static func hasSustainedSeed(
    points: [WholeCurveResampledPoint],
    start: Int,
    end: Int,
    sign: Double,
    configuration: WholeCurveConfiguration
  ) -> Bool {
    var runStart: Int?
    for index in start ... end {
      let curvature = segmentationCurvature(points[index]) ?? 0
      let matches = abs(curvature) >= configuration.enterCurvature
        && (curvature >= 0 ? 1.0 : -1.0) == sign
      if matches {
        runStart = runStart ?? index
        if let runStart,
           points[index].distanceMeters - points[runStart].distanceMeters
             >= configuration.sustainedDistanceMeters {
          return true
        }
      } else {
        runStart = nil
      }
    }
    return false
  }

  private static func segmentationCurvature(_ point: WholeCurveResampledPoint) -> Double? {
    point.curvature60 ?? point.curvature100
  }

  private static func makeEvent(
    run: SignedRun,
    points: [WholeCurveResampledPoint],
    sourceTurns: [SourceTurnVertex],
    configuration: WholeCurveConfiguration
  ) -> WholeCurveEvent? {
    guard points.indices.contains(run.start), points.indices.contains(run.end), run.start < run.end else {
      return nil
    }
    let scale60 = rollingMedianPeak(
      points,
      keyPath: \.curvature60,
      run: run,
      supportMeters: configuration.sustainedDistanceMeters
    )
    let scale100 = rollingMedianPeak(
      points,
      keyPath: \.curvature100,
      run: run,
      supportMeters: configuration.sustainedDistanceMeters
    )
    let scale160 = rollingMedianPeak(
      points,
      keyPath: \.curvature160,
      run: run,
      supportMeters: configuration.sustainedDistanceMeters
    )
    let peak60 = scale60?.value
    let peak100 = scale100?.value
    let peak160 = scale160?.value
    let availableMagnitudes = [peak60, peak100, peak160].compactMap { $0 }.map(abs)
    guard !availableMagnitudes.isEmpty else { return nil }
    let nominal = median(availableMagnitudes)
    let turnStats = turnStatistics(points: points, start: run.start, end: run.end)
    let maximumSourceGap = points[run.start ... run.end].map(\.sourceGapMeters).max() ?? 0
    let compactSourceEvidence = peak60.flatMap {
      compactApexSourceEvidence(
        sourceTurns: sourceTurns,
        points: points,
        run: run,
        peak60: $0,
        configuration: configuration
      )
    }
    let compactGuardIsSupported = peak60.map {
      abs($0) >= nominal * 1.20
        && (compactSourceEvidence?.supportingVertexCount ?? 0) >= 3
        && (compactSourceEvidence?.supportingSpanMeters ?? 0)
          >= configuration.sustainedDistanceMeters
        && (compactSourceEvidence?.directionalCoherence ?? 0) >= 0.80
        && maximumSourceGap <= 40
        && turnStats.coherence >= 0.80
    } ?? false
    let compactGuard: Double
    if let peak60,
       compactGuardIsSupported,
       hasSustainedPeak(
         points: points,
         keyPath: \.curvature60,
         run: run,
         threshold: abs(peak60) * 0.85,
         minimumDistance: configuration.sustainedDistanceMeters
       ) {
      compactGuard = abs(peak60)
    } else {
      compactGuard = nominal
    }
    let controlling = run.sign * max(nominal, compactGuard)
    let apexIndex: Int
    if compactGuard > nominal, let scale60 {
      apexIndex = scale60.index
    } else if let scale100 {
      apexIndex = scale100.index
    } else {
      apexIndex = scale60?.index ?? scale160?.index ?? run.start
    }
    let scaleSpread: Double? = availableMagnitudes.count >= 2
      ? (availableMagnitudes.max() ?? 0) / max(availableMagnitudes.min() ?? 0, 1.0e-12)
      : nil
    let eventLength = points[run.end].distanceMeters - points[run.start].distanceMeters
    var flags: [WholeCurveFlag] = []
    if availableMagnitudes.count < 3 { flags.append(.missingScaleContext) }
    if !hasStraightShoulders(
      points: points,
      run: run,
      distance: configuration.sustainedDistanceMeters,
      exitCurvature: configuration.exitCurvature
    ) {
      flags.append(.incompleteStraightShoulder)
    }
    if maximumSourceGap > 60 { flags.append(.sparseSourceGeometry) }
    if turnStats.coherence < 0.75 { flags.append(.unstableTurnSign) }
    if let scaleSpread, scaleSpread > 1.8 { flags.append(.highScaleSpread) }
    if eventLength < configuration.sustainedDistanceMeters { flags.append(.shortEvent) }
    if !controlling.isFinite || controlling == 0 { flags.append(.invalidGeometry) }

    let confidence: WholeCurveConfidence
    if flags.isEmpty,
       (scaleSpread ?? .infinity) <= 1.35,
       turnStats.coherence >= 0.85,
       maximumSourceGap <= 40 {
      confidence = .high
    } else if !flags.contains(.invalidGeometry),
              !flags.contains(.unstableTurnSign),
              !flags.contains(.missingScaleContext),
              (scaleSpread ?? .infinity) <= 1.8 {
      confidence = .review
    } else {
      confidence = .low
    }

    let identifiers = eventIdentifiers(
      start: points[run.start],
      apex: points[apexIndex],
      end: points[run.end]
    )
    return WholeCurveEvent(
      physicalID: identifiers.physical,
      directionalID: identifiers.directional,
      startIndex: run.start,
      endIndex: run.end,
      apexIndex: apexIndex,
      profileApexIndex: apexIndex,
      lengthMeters: eventLength,
      signedTurnRadians: turnStats.signed,
      signCoherence: turnStats.coherence,
      curvature60: peak60,
      curvature100: peak100,
      curvature160: peak160,
      controllingCurvature: controlling,
      maximumApexCoefficient: 1,
      scaleSpread: scaleSpread,
      maximumSourceGapMeters: maximumSourceGap,
      sourceKeys: Set(points[run.start ... run.end].flatMap(\.sourceKeys)),
      confidence: confidence,
      flags: flags
    )
  }

  private static func rollingMedianPeak(
    _ points: [WholeCurveResampledPoint],
    keyPath: KeyPath<WholeCurveResampledPoint, Double?>,
    run: SignedRun,
    supportMeters: Double
  ) -> ScalePeak? {
    guard points.count >= 2 else { return nil }
    let spacing = max(
      0.1,
      points[min(run.start + 1, points.count - 1)].distanceMeters
        - points[run.start].distanceMeters
    )
    let halfWindow = max(1, Int(ceil(supportMeters / spacing / 2)))
    var candidates: [ScalePeak] = []
    for index in run.start ... run.end {
      let lower = max(run.start, index - halfWindow)
      let upper = min(run.end, index + halfWindow)
      guard points[upper].distanceMeters - points[lower].distanceMeters
        >= min(supportMeters, points[run.end].distanceMeters - points[run.start].distanceMeters)
      else { continue }
      let local = (lower ... upper).map { sampleIndex -> Double in
        guard let value = points[sampleIndex][keyPath: keyPath],
              abs(value) > 0,
              (value >= 0 ? 1.0 : -1.0) == run.sign
        else { return 0 }
        return abs(value)
      }
      let supportedCount = local.count { $0 > 0 }
      guard supportedCount * 2 > local.count else { continue }
      candidates.append(ScalePeak(value: run.sign * median(local), index: index))
    }
    return candidates.max { abs($0.value) < abs($1.value) }
  }

  private static func hasSustainedPeak(
    points: [WholeCurveResampledPoint],
    keyPath: KeyPath<WholeCurveResampledPoint, Double?>,
    run: SignedRun,
    threshold: Double,
    minimumDistance: Double
  ) -> Bool {
    var start: Int?
    for index in run.start ... run.end {
      let value = points[index][keyPath: keyPath] ?? 0
      if abs(value) >= threshold, (value >= 0 ? 1.0 : -1.0) == run.sign {
        start = start ?? index
        if let start,
           points[index].distanceMeters - points[start].distanceMeters >= minimumDistance {
          return true
        }
      } else {
        start = nil
      }
    }
    return false
  }

  private static func hasStraightShoulders(
    points: [WholeCurveResampledPoint],
    run: SignedRun,
    distance: Double,
    exitCurvature: Double
  ) -> Bool {
    let entryDistance = points[run.start].distanceMeters
    let exitDistance = points[run.end].distanceMeters
    guard entryDistance - (points.first?.distanceMeters ?? entryDistance) >= distance,
          (points.last?.distanceMeters ?? exitDistance) - exitDistance >= distance
    else { return false }
    let entryStart = entryDistance - distance
    let exitEnd = exitDistance + distance
    let entryShoulder = points.indices.filter {
      points[$0].distanceMeters >= entryStart && points[$0].distanceMeters < entryDistance
    }
    let exitShoulder = points.indices.filter {
      points[$0].distanceMeters > exitDistance && points[$0].distanceMeters <= exitEnd
    }
    guard !entryShoulder.isEmpty, !exitShoulder.isEmpty else { return false }
    return entryShoulder.allSatisfy {
      guard let curvature = segmentationCurvature(points[$0]) else { return false }
      return abs(curvature) < exitCurvature
    } && exitShoulder.allSatisfy {
      guard let curvature = segmentationCurvature(points[$0]) else { return false }
      return abs(curvature) < exitCurvature
    }
  }

  private static func signedTurn(
    points: [WholeCurveResampledPoint],
    start: Int,
    end: Int
  ) -> Double {
    turnStatistics(points: points, start: start, end: end).signed
  }

  private static func turnStatistics(
    points: [WholeCurveResampledPoint],
    start: Int,
    end: Int
  ) -> (signed: Double, coherence: Double) {
    guard end - start >= 2 else { return (0, 0) }
    var signed = 0.0
    var absolute = 0.0
    for index in (start + 1) ..< end {
      let firstHeading = heading(from: points[index - 1], to: points[index])
      let secondHeading = heading(from: points[index], to: points[index + 1])
      var change = secondHeading - firstHeading
      while change > .pi { change -= 2 * .pi }
      while change < -.pi { change += 2 * .pi }
      signed += change
      absolute += abs(change)
    }
    return (signed, absolute > 1.0e-9 ? abs(signed) / absolute : 1)
  }

  private static func heading(
    from first: WholeCurveResampledPoint,
    to second: WholeCurveResampledPoint
  ) -> Double {
    let latitude = 0.5 * (first.latitude + second.latitude) * .pi / 180
    let x = (second.longitude - first.longitude) * Foundation.cos(latitude)
    let y = second.latitude - first.latitude
    return Foundation.atan2(y, x)
  }

  private static func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
      ? 0.5 * (sorted[middle - 1] + sorted[middle])
      : sorted[middle]
  }

  private static func eventIdentifiers(
    start: WholeCurveResampledPoint,
    apex: WholeCurveResampledPoint,
    end: WholeCurveResampledPoint
  ) -> (physical: String, directional: String) {
    func coordinate(_ point: WholeCurveResampledPoint, scale: Double) -> String {
      "\(Int64((point.latitude * scale).rounded())):\(Int64((point.longitude * scale).rounded()))"
    }
    let startKey = coordinate(start, scale: 100_000)
    let endKey = coordinate(end, scale: 100_000)
    let apexKey = coordinate(apex, scale: 100_000)
    let canonicalEnds = [startKey, endKey].sorted().joined(separator: "|")
    let physicalSeed = "whole-curve-v1|\(canonicalEnds)|\(apexKey)"
    let direction = startKey <= endKey ? "a" : "b"
    let digest = SHA256.hash(data: Data(physicalSeed.utf8))
    let physical = digest.prefix(10).map { String(format: "%02x", $0) }.joined()
    return (physical, "\(physical)-\(direction)")
  }
}
