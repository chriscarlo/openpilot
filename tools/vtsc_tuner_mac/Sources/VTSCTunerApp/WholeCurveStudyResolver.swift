import Foundation
import VTSCTunerCore

struct MapWholeCurveCalibrationResolution: Equatable, Sendable {
  var curvature: Double
  var supportMeters: Double
  var eventID: String
}

enum MapWholeCurveStudyResolver {
  static func calibrationResolutions(
    ways: [MapRenderedWay],
    calibrationSamples: [MapCalibrationSample]
  ) -> [String: MapWholeCurveCalibrationResolution] {
    guard !ways.isEmpty, !calibrationSamples.isEmpty else { return [:] }
    let routes = MapRuntimeCurvatureResolver.wholeCurveStudyRoutes(
      ways: ways,
      focusSourceKeys: calibrationSamples.map(\.sourceKey)
    )
    var resolved: [String: (resolution: MapWholeCurveCalibrationResolution, distance: Double)] = [:]
    for route in routes {
      let estimate = WholeCurveEstimator.estimate(route: route.points.map {
        WholeCurveInputPoint(
          latitude: $0.node.latitude,
          longitude: $0.node.longitude,
          sourceKey: $0.sourceKey
        )
      })
      guard !estimate.events.isEmpty else { continue }
      let routeSourceKeys = Set(route.points.map(\.sourceKey))
      for sample in calibrationSamples where routeSourceKeys.contains(sample.sourceKey) {
        let directMatches = estimate.events.indices.filter {
          estimate.events[$0].sourceKeys.contains(sample.sourceKey)
        }
        guard let eventIndex = directMatches.min(by: {
          distanceMeters(from: sample, to: estimate.events[$0], points: estimate.points)
            < distanceMeters(from: sample, to: estimate.events[$1], points: estimate.points)
        }) else { continue }
        let event = estimate.events[eventIndex]
        let totalDistance = estimate.points.last?.distanceMeters ?? 0
        let startsNearAmbiguity = route.frontIsAmbiguous
          && estimate.points[event.startIndex].distanceMeters < 160
        let endsNearAmbiguity = route.backIsAmbiguous
          && totalDistance - estimate.points[event.endIndex].distanceMeters < 160
        guard !startsNearAmbiguity, !endsNearAmbiguity,
              estimate.points.indices.contains(event.profileApexIndex)
        else { continue }
        let curvature = abs(estimate.points[event.profileApexIndex].profileCurvature)
        guard curvature.isFinite, curvature >= 1.0e-7 else { continue }
        let distance = distanceMeters(from: sample, to: event, points: estimate.points)
        let candidate = MapWholeCurveCalibrationResolution(
          curvature: curvature,
          supportMeters: event.lengthMeters,
          eventID: event.directionalID
        )
        if resolved[sample.sourceKey] == nil || distance < resolved[sample.sourceKey]!.distance {
          resolved[sample.sourceKey] = (candidate, distance)
        }
      }
    }
    return resolved.mapValues(\.resolution)
  }

  static func resolve(
    ways: [MapRenderedWay],
    calibrationSamples: [MapCalibrationSample],
    parameters: SigmoidParameters,
    bands: [EQBand]
  ) -> [MapWholeCurveEvent] {
    guard !ways.isEmpty, !calibrationSamples.isEmpty else { return [] }
    let routes = MapRuntimeCurvatureResolver.wholeCurveStudyRoutes(
      ways: ways,
      focusSourceKeys: calibrationSamples.map(\.sourceKey)
    )
    let numberedSamples = calibrationSamples.enumerated().map { (number: $0 + 1, sample: $1) }
    var candidates: [MapWholeCurveEvent] = []

    for route in routes {
      let estimate = WholeCurveEstimator.estimate(route: route.points.map {
        WholeCurveInputPoint(
          latitude: $0.node.latitude,
          longitude: $0.node.longitude,
          sourceKey: $0.sourceKey
        )
      })
      guard !estimate.events.isEmpty else { continue }
      let routeSourceKeys = Set(route.points.map(\.sourceKey))
      let routeSamples = numberedSamples.filter { routeSourceKeys.contains($0.sample.sourceKey) }
      guard !routeSamples.isEmpty else { continue }
      var samplesByEventIndex: [Int: [(number: Int, sample: MapCalibrationSample)]] = [:]

      for numbered in routeSamples {
        let directMatches = estimate.events.indices.filter {
          estimate.events[$0].sourceKeys.contains(numbered.sample.sourceKey)
        }
        // A nearby event is not necessarily the event represented by this bank
        // sample. Hairpins and parallel carriageways can put unrelated route
        // sections within a few metres of each other, so only exact source-node
        // provenance is safe enough to link a saved sample to an event.
        guard let eventIndex = directMatches.min(by: {
          distanceMeters(from: numbered.sample, to: estimate.events[$0], points: estimate.points)
            < distanceMeters(from: numbered.sample, to: estimate.events[$1], points: estimate.points)
        }) else { continue }
        samplesByEventIndex[eventIndex, default: []].append(numbered)
      }

      let routePointBySourceKey = Dictionary(
        route.points.map { ($0.sourceKey, $0) },
        uniquingKeysWith: { first, _ in first }
      )
      for (eventIndex, linkedSamples) in samplesByEventIndex {
        let event = estimate.events[eventIndex]
        guard let rendered = render(
          event: event,
          estimatePoints: estimate.points,
          route: route,
          routePointBySourceKey: routePointBySourceKey,
          linkedSamples: linkedSamples,
          parameters: parameters,
          bands: bands
        ) else { continue }
        candidates.append(rendered)
      }
    }

    var grouped: [MapWholeCurveEvent] = []
    for candidate in candidates.sorted(by: eventSort) {
      if let index = grouped.firstIndex(where: { sameDirectionalEvent($0, candidate) }) {
        grouped[index] = merge(grouped[index], candidate)
      } else {
        grouped.append(candidate)
      }
    }
    return grouped.sorted(by: eventSort)
  }

  private static func render(
    event: WholeCurveEvent,
    estimatePoints: [WholeCurveResampledPoint],
    route: MapWholeCurveRoute,
    routePointBySourceKey: [String: MapWholeCurveRoutePoint],
    linkedSamples: [(number: Int, sample: MapCalibrationSample)],
    parameters: SigmoidParameters,
    bands: [EQBand]
  ) -> MapWholeCurveEvent? {
    guard estimatePoints.indices.contains(event.startIndex),
          estimatePoints.indices.contains(event.endIndex),
          estimatePoints.indices.contains(event.apexIndex),
          estimatePoints.indices.contains(event.profileApexIndex),
          event.startIndex < event.endIndex
    else { return nil }
    let profileApexCurvature = estimatePoints[event.profileApexIndex].profileCurvature
    let wholeSpeed = SigmoidFitter.predictedSpeedMPH(
      parameters: parameters,
      curvature: abs(profileApexCurvature),
      bands: bands,
      mode: .effectiveStrategic,
      modifiers: .sourceDefaults
    )
    let startDistance = estimatePoints[event.startIndex].distanceMeters
    let studyPoints = (event.startIndex ... event.endIndex).map { index -> MapWholeCurvePoint in
      let point = estimatePoints[index]
      let routePoint = routePointBySourceKey[point.nearestSourceKey]
        ?? point.sourceKeys.lazy.compactMap { routePointBySourceKey[$0] }.first
      let currentCurvature = routePoint?.currentCurvature ?? 0
      let currentSpeed = SigmoidFitter.predictedSpeedMPH(
        parameters: parameters,
        curvature: abs(currentCurvature),
        bands: bands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
      let pointWholeSpeed = SigmoidFitter.predictedSpeedMPH(
        parameters: parameters,
        curvature: abs(point.profileCurvature),
        bands: bands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
      return MapWholeCurvePoint(
        latitude: point.latitude,
        longitude: point.longitude,
        distanceMeters: point.distanceMeters - startDistance,
        currentMapdSpeedMPH: currentSpeed,
        wholeCurveSpeedMPH: pointWholeSpeed,
        curvature60: point.curvature60,
        curvature100: point.curvature100,
        curvature160: point.curvature160
      )
    }
    guard let firstPoint = studyPoints.first, let lastPoint = studyPoints.last else { return nil }
    let apex = estimatePoints[event.profileApexIndex]
    var flags = event.flags.map(\.rawValue)
    let totalDistance = estimatePoints.last?.distanceMeters ?? 0
    if route.frontIsAmbiguous, startDistance < 160 {
      flags.append("Route branch is ambiguous before this curve")
    }
    if route.backIsAmbiguous, totalDistance - estimatePoints[event.endIndex].distanceMeters < 160 {
      flags.append("Route branch is ambiguous after this curve")
    }
    let currentSpeeds = studyPoints.map(\.currentMapdSpeedMPH).filter(\.isFinite)
    let confidence = flags.contains(where: { $0.localizedCaseInsensitiveContains("ambiguous") })
      ? WholeCurveConfidence.low.rawValue
      : event.confidence.rawValue
    let roadName = linkedSamples.first?.sample.roadName ?? route.roadName
    let reference = linkedSamples.first?.sample.reference ?? route.reference
    return MapWholeCurveEvent(
      id: event.directionalID,
      roadName: roadName,
      reference: reference,
      travelDirection: cardinalDirection(from: firstPoint, to: lastPoint),
      bendDirection: event.controllingCurvature >= 0 ? "Left" : "Right",
      points: studyPoints,
      sourceKeys: event.sourceKeys,
      lengthMeters: event.lengthMeters,
      turnDegrees: event.signedTurnRadians * 180 / .pi,
      apexLatitude: apex.latitude,
      apexLongitude: apex.longitude,
      curvature60: event.curvature60,
      curvature100: event.curvature100,
      curvature160: event.curvature160,
      controllingCurvature: profileApexCurvature,
      wholeCurveSpeedMPH: wholeSpeed,
      currentMinimumSpeedMPH: currentSpeeds.min() ?? 0,
      currentMaximumSpeedMPH: currentSpeeds.max() ?? 0,
      confidenceLabel: confidence,
      flags: Array(Set(flags)).sorted(),
      bankSampleNumbers: linkedSamples.map(\.number).sorted(),
      bankTargetsMPH: linkedSamples.sorted { $0.number < $1.number }.map(\.sample.desiredSpeedMPH)
    )
  }

  private static func sameDirectionalEvent(
    _ first: MapWholeCurveEvent,
    _ second: MapWholeCurveEvent
  ) -> Bool {
    guard first.travelDirection == second.travelDirection,
          first.bendDirection == second.bendDirection
    else { return false }
    if first.id == second.id { return true }
    let sharedSources = !first.sourceKeys.isDisjoint(with: second.sourceKeys)
    let apexDistance = MapBakeMath.distanceMeters(
      from: MapTileNode(latitude: first.apexLatitude, longitude: first.apexLongitude),
      to: MapTileNode(latitude: second.apexLatitude, longitude: second.apexLongitude)
    )
    return sharedSources && apexDistance <= 120
  }

  static func merge(
    _ first: MapWholeCurveEvent,
    _ second: MapWholeCurveEvent
  ) -> MapWholeCurveEvent {
    let preferred = eventQuality(second) > eventQuality(first) ? second : first
    var merged = preferred
    merged.sourceKeys.formUnion(first.sourceKeys)
    merged.sourceKeys.formUnion(second.sourceKeys)
    var numberedTargets = Array(zip(first.bankSampleNumbers, first.bankTargetsMPH))
    numberedTargets.append(contentsOf: zip(second.bankSampleNumbers, second.bankTargetsMPH))
    let targetByNumber = Dictionary(numberedTargets, uniquingKeysWith: { first, _ in first })
    merged.bankSampleNumbers = targetByNumber.keys.sorted()
    merged.bankTargetsMPH = merged.bankSampleNumbers.compactMap { targetByNumber[$0] }
    merged.flags = Array(Set(first.flags + second.flags)).sorted()
    merged.confidenceLabel = conservativeMergedConfidence(
      first: first.confidenceLabel,
      second: second.confidenceLabel,
      flags: merged.flags
    )
    return merged
  }

  private static func conservativeMergedConfidence(
    first: String,
    second: String,
    flags: [String]
  ) -> String {
    // Geometry comes from the better candidate, but confidence describes all
    // evidence merged into the displayed event. Never discard a weaker
    // candidate's confidence just because its points were not preferred.
    var rank = min(confidenceRank(first), confidenceRank(second))
    let lowConfidenceFlags = Set([
      WholeCurveFlag.missingScaleContext.rawValue,
      WholeCurveFlag.unstableTurnSign.rawValue,
      WholeCurveFlag.highScaleSpread.rawValue,
      WholeCurveFlag.invalidGeometry.rawValue,
    ])
    if flags.contains(where: {
      lowConfidenceFlags.contains($0)
        || $0.localizedCaseInsensitiveContains("ambiguous")
    }) {
      rank = 0
    } else if !flags.isEmpty {
      // Even inconsistent input claiming high confidence cannot remain high
      // after the merged event exposes a review flag.
      rank = min(rank, 1)
    }
    switch rank {
    case 2: return WholeCurveConfidence.high.rawValue
    case 1: return WholeCurveConfidence.review.rawValue
    default: return WholeCurveConfidence.low.rawValue
    }
  }

  private static func confidenceRank(_ label: String) -> Int {
    if label.localizedCaseInsensitiveContains("high") { return 2 }
    if label.localizedCaseInsensitiveContains("review") { return 1 }
    return 0
  }

  private static func eventQuality(_ event: MapWholeCurveEvent) -> Int {
    let confidence = event.confidenceLabel.localizedCaseInsensitiveContains("high")
      ? 3
      : (event.confidenceLabel.localizedCaseInsensitiveContains("review") ? 2 : 1)
    return confidence * 10_000 - event.flags.count * 1_000 + event.points.count
  }

  private static func distanceMeters(
    from sample: MapCalibrationSample,
    to event: WholeCurveEvent,
    points: [WholeCurveResampledPoint]
  ) -> Double {
    guard points.indices.contains(event.startIndex), points.indices.contains(event.endIndex) else {
      return .infinity
    }
    let sampleNode = MapTileNode(latitude: sample.latitude, longitude: sample.longitude)
    return points[event.startIndex ... event.endIndex].map {
      MapBakeMath.distanceMeters(
        from: sampleNode,
        to: MapTileNode(latitude: $0.latitude, longitude: $0.longitude)
      )
    }.min() ?? .infinity
  }

  private static func cardinalDirection(
    from first: MapWholeCurvePoint,
    to last: MapWholeCurvePoint
  ) -> String {
    let latitude = 0.5 * (first.latitude + last.latitude) * .pi / 180
    let east = (last.longitude - first.longitude) * cos(latitude)
    let north = last.latitude - first.latitude
    if abs(east) >= abs(north) {
      return east >= 0 ? "Eastbound" : "Westbound"
    }
    return north >= 0 ? "Northbound" : "Southbound"
  }

  private static func eventSort(_ first: MapWholeCurveEvent, _ second: MapWholeCurveEvent) -> Bool {
    if first.apexLongitude != second.apexLongitude { return first.apexLongitude < second.apexLongitude }
    if first.travelDirection != second.travelDirection { return first.travelDirection < second.travelDirection }
    return first.id < second.id
  }
}
