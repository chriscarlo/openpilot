import Foundation
import Testing
@testable import VTSCTunerApp
import VTSCTunerCore

private func renderedWay(id: String, curvature: Double = 0.01) -> MapRenderedWay {
  MapRenderedWay(
    id: id,
    name: "Curve \(id)",
    reference: "CA \(id)",
    nodes: [
      MapRenderedNode(
        latitude: 38.0,
        longitude: -121.001,
        curvature: 0,
        bakedSpeedMPS: 20,
        proposedSpeedMPS: 20
      ),
      MapRenderedNode(
        latitude: 38.001,
        longitude: -121.0,
        curvature: curvature,
        bakedSpeedMPS: 18,
        proposedSpeedMPS: 19
      ),
      MapRenderedNode(
        latitude: 38.002,
        longitude: -121.001,
        curvature: 0,
        bakedSpeedMPS: 20,
        proposedSpeedMPS: 20
      ),
    ],
    tilePath: "/tmp/\(id)",
    tileHash: "tile-\(id)",
    schemaVersion: 1,
    maxSpeedMPS: 25,
    advisorySpeedMPS: 0,
    maxSpeedForwardMPS: 25,
    maxSpeedBackwardMPS: 25,
    lanes: 2,
    oneWay: false,
    hazard: "",
    windingForwardLevel: 1,
    windingBackwardLevel: 1,
    windingForwardScore: 100,
    windingBackwardScore: 100,
    windingForwardConfidence: 200,
    windingBackwardConfidence: 200
  )
}

private func curveSelection(id: String, curvature: Double = 0.01) -> MapRoadSelection {
  MapRoadSelection(way: renderedWay(id: id, curvature: curvature), nodeIndex: 1, segmentFraction: 0.5)
}

@MainActor
private func calibrationSession() -> MapPreviewSession {
  let session = MapPreviewSession(loadPersistedState: false, persistsCalibrationSamples: false)
  session.purpose = .calibration
  return session
}

private func connectedWay(
  id: String,
  coordinates: [(Double, Double)],
  name: String = "",
  reference: String = "US 50",
  lanes: UInt8 = 2,
  oneWay: Bool = true
) -> MapRenderedWay {
  MapRenderedWay(
    id: id,
    name: name,
    reference: reference,
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
    lanes: lanes,
    oneWay: oneWay,
    hazard: "",
    windingForwardLevel: 0,
    windingBackwardLevel: 0,
    windingForwardScore: 0,
    windingBackwardScore: 0,
    windingForwardConfidence: 0,
    windingBackwardConfidence: 0
  )
}

@Test func speedBandsUseFiveMPHStepsThroughSeventyFivePlus() {
  let examples: [(Double, MapSpeedBand)] = [
    (0, .under25),
    (24.999, .under25),
    (25, .from25),
    (29.999, .from25),
    (30, .from30),
    (35, .from35),
    (40, .from40),
    (45, .from45),
    (50, .from50),
    (55, .from55),
    (60, .from60),
    (65, .from65),
    (70, .from70),
    (74.999, .from70),
    (75, .from75),
    (120, .from75),
  ]

  for (speed, expectedBand) in examples {
    #expect(MapSpeedBand.band(forMPH: speed) == expectedBand)
  }
  #expect(MapSpeedBand.allCases.map(\.label) == [
    "<25", "25–29", "30–34", "35–39", "40–44", "45–49",
    "50–54", "55–59", "60–64", "65–69", "70–74", "75+",
  ])
}

@Test func runtimeCurvatureResolverStitchesAcrossWayBoundaries() throws {
  let points = [
    (38.731_895_9, -120.746_461_8),
    (38.731_805_2, -120.746_579_5),
    (38.731_738_6, -120.746_668_9),
    (38.731_675_6, -120.746_771_3),
    (38.731_610_4, -120.746_885_6),
  ]
  let ways = [
    connectedWay(id: "before", coordinates: Array(points[0...1])),
    connectedWay(id: "selected", coordinates: Array(points[1...3])),
    connectedWay(id: "after", coordinates: Array(points[3...4])),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let selected = try #require(resolved.first(where: { $0.id == "selected" }))
  let node = selected.nodes[1]

  #expect(node.curvatureContextComplete)
  #expect(abs(node.rawCurvature - 0.008_578_403_275_769_536) < 1.0e-11)
  #expect(abs(node.curvature - 0.004_181_845_727_569) < 1.0e-11)
  #expect(node.rawCurvature / node.curvature > 2.0)
  #expect(node.proposedSpeedMPS == 70)
}

@Test func runtimeCurvatureResolverAppliesProductionLaneIncreaseCorrection() throws {
  let points = [
    (38.729_059_6, -120.809_917_1),
    (38.728_602_6, -120.809_233_6),
    (38.728_499_1, -120.809_074_8),
    (38.728_414_3, -120.808_895_7),
    (38.728_334_8, -120.808_659_6),
    (38.728_267_8, -120.808_402_2),
  ]
  let ways = [
    connectedWay(id: "three-lane", coordinates: Array(points[0...2]), lanes: 3),
    connectedWay(id: "four-lane", coordinates: Array(points[2...5]), lanes: 4),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let selected = try #require(resolved.first(where: { $0.id == "four-lane" }))
  let node = selected.nodes[1]

  #expect(node.curvatureContextComplete)
  #expect(abs(node.rawCurvature - 0.006_811_893_605_425_810_5) < 1.0e-11)
  #expect(abs(node.curvature - 0.002_332_571_7) < 1.0e-9)
  #expect(node.proposedSpeedMPS == 70)
}

@Test func runtimeCurvatureResolverRejectsSameRoadForkAmbiguity() throws {
  let points = (0 ... 8).map { (38.0, -121.002 + Double($0) * 0.000_25) }
  let ways = [
    connectedWay(id: "before", coordinates: Array(points[0...3])),
    connectedWay(id: "selected", coordinates: Array(points[3...5])),
    connectedWay(id: "fork-a", coordinates: Array(points[5...8])),
    connectedWay(
      id: "fork-b",
      coordinates: [points[5], (38.000_2, points[6].1), (38.000_2, points[7].1), (38.000_2, points[8].1)]
    ),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let selected = try #require(resolved.first(where: { $0.id == "selected" }))

  #expect(!selected.nodes[1].curvatureContextComplete)
}

@Test func runtimeCurvatureResolverUsesUniqueExactReferenceAheadOfSideRoads() throws {
  let points = (0 ... 8).map { (38.0, -121.002 + Double($0) * 0.000_25) }
  let ways = [
    connectedWay(id: "before-ref", coordinates: Array(points[0...3])),
    connectedWay(id: "selected-ref", coordinates: Array(points[3...5])),
    connectedWay(id: "mainline-ref", coordinates: Array(points[5...8])),
    connectedWay(
      id: "side-road",
      coordinates: [points[5], (38.000_2, points[6].1), (38.000_2, points[7].1)],
      name: "Side Road",
      reference: "",
      oneWay: false
    ),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let selected = try #require(resolved.first(where: { $0.id == "selected-ref" }))

  #expect(selected.nodes[1].curvatureContextComplete)
  #expect(abs(selected.nodes[1].curvature) < 1.0e-6)
}

@Test func runtimeCurvatureResolverKeepsFifteenMeterTransitionMargin() throws {
  // P contributes two nodes before S, which is enough for the local five-node
  // average but not enough to discover Q→P's nearby 2→3 lane transition.
  // Production mapd propagates that clamp through P and into S.
  let q0 = (38.0, -121.000_30)
  let q1 = (38.0, -121.000_25)
  let qToP = (38.0, -121.000_20)
  let p1 = (38.0, -121.000_15)
  let pToS = (38.0, -121.000_10)
  let s1 = (38.0, -121.000_05)
  let s2 = (38.0, -121.000_00)
  let s3 = (38.0, -120.999_95)
  let ways = [
    connectedWay(id: "q", coordinates: [q0, q1, qToP], lanes: 2),
    connectedWay(id: "p", coordinates: [qToP, p1, pToS], lanes: 3),
    connectedWay(id: "s", coordinates: [pToS, s1, s2, s3], lanes: 3),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let source = try #require(resolved.first(where: { $0.id == "s" }))
  let node = source.nodes[1]

  #expect(node.curvatureContextComplete)
  #expect(abs(node.rawCurvature) < 1.0e-6)
  #expect(abs(node.curvature - MapBakeMath.mergeOrSplitCurvature) < 1.0e-12)
}

@Test func runtimeCurvatureResolverExtendsFrontPastTheFiveNodeStencil() throws {
  // P adds the two nodes needed for S node0's five-node average and its outer
  // endpoint is >15 m from S.  The Q→P transition still clamps the raw entry
  // at that endpoint, which feeds S node0, so Q must also be resolved.
  let coordinates = [
    (38.0, -121.000_50),
    (38.0, -121.000_40),
    (38.0, -121.000_30),
    (38.0, -121.000_20),
    (38.0, -121.000_10),
    (38.0, -121.000_00),
    (38.0, -120.999_90),
    (38.0, -120.999_80),
  ]
  let ways = [
    connectedWay(id: "front-q", coordinates: Array(coordinates[0...2]), lanes: 2),
    connectedWay(id: "front-p", coordinates: Array(coordinates[2...4]), lanes: 3),
    connectedWay(id: "front-s", coordinates: Array(coordinates[4...7]), lanes: 3),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let source = try #require(resolved.first(where: { $0.id == "front-s" }))
  let expected = try #require(MapBakeMath.runtimeCurvatureEstimate(
    nodes: coordinates.map { MapTileNode(latitude: $0.0, longitude: $0.1) },
    nodeIndex: 4,
    mergeOrSplitNodeIndices: [2]
  ))

  #expect(source.nodes[0].curvatureContextComplete)
  #expect(abs(source.nodes[0].curvature - expected.curvature) < 1.0e-12)
  #expect(source.nodes[0].curvature > 0.000_8)
}

@Test func runtimeCurvatureResolverExtendsBackAndUsesUniqueDifferentIdentityFallback() throws {
  // P adds two nodes and ends >15 m beyond S, but a P→Q lane increase still
  // unconditionally clamps raw index merge-2, which feeds S's last node.
  // Q deliberately changes ref: with one physical continuation, mapd reaches
  // it through its least-curvature fallback and the tuner must do the same.
  let coordinates = [
    (38.0, -121.000_50),
    (38.0, -121.000_40),
    (38.0, -121.000_30),
    (38.0, -121.000_20),
    (38.0, -121.000_10),
    (38.0, -121.000_00),
    (38.0, -120.999_90),
  ]
  let ways = [
    connectedWay(id: "back-s", coordinates: Array(coordinates[0...2]), lanes: 3),
    connectedWay(id: "back-p", coordinates: Array(coordinates[2...4]), lanes: 3),
    connectedWay(
      id: "back-q",
      coordinates: Array(coordinates[4...6]),
      reference: "Different Route",
      lanes: 4
    ),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: ways,
    parameters: .checkoutFallback
  )
  let source = try #require(resolved.first(where: { $0.id == "back-s" }))
  let expected = try #require(MapBakeMath.runtimeCurvatureEstimate(
    nodes: coordinates.map { MapTileNode(latitude: $0.0, longitude: $0.1) },
    nodeIndex: 2,
    mergeOrSplitNodeIndices: [4]
  ))

  #expect(source.nodes[2].curvatureContextComplete)
  #expect(abs(source.nodes[2].curvature - expected.curvature) < 1.0e-12)
  #expect(source.nodes[2].curvature > 0.000_4)
}

@Test func wholeCurveStudyGroupsSeveralMapPiecesIntoOneDirectionalEvent() throws {
  let latitude = 38.73
  let metersPerDegreeLatitude = MapBakeMath.earthRadiusMeters * .pi / 180
  let metersPerDegreeLongitude = metersPerDegreeLatitude * cos(latitude * .pi / 180)
  var local: [(x: Double, y: Double)] = [(0, 0)]
  var x = 0.0
  var y = 0.0
  var heading = 0.0
  for (length, curvature) in [(120.0, 0.0), (320.0, 0.004), (120.0, 0.0)] {
    for _ in 0 ..< Int(length / 20) {
      heading += curvature * 10
      x += cos(heading) * 20
      y += sin(heading) * 20
      heading += curvature * 10
      local.append((x, y))
    }
  }
  let coordinates = local.map {
    (latitude + $0.y / metersPerDegreeLatitude, -120.75 + $0.x / metersPerDegreeLongitude)
  }
  let ways = [
    connectedWay(id: "study-before", coordinates: Array(coordinates[0 ... 6])),
    connectedWay(id: "study-curve-a", coordinates: Array(coordinates[6 ... 14])),
    connectedWay(id: "study-curve-b", coordinates: Array(coordinates[14 ... 22])),
    connectedWay(id: "study-after", coordinates: Array(coordinates[22 ... 28])),
  ]
  let resolved = MapRuntimeCurvatureResolver.resolve(ways: ways, parameters: .checkoutFallback)
  let firstWay = try #require(resolved.first { $0.id == "study-curve-a" })
  let secondWay = try #require(resolved.first { $0.id == "study-curve-b" })
  let samples = [
    MapCalibrationSample(
      sourceKey: "study-curve-a:4",
      roadName: "Study Road",
      reference: "US 50",
      latitude: firstWay.nodes[4].latitude,
      longitude: firstWay.nodes[4].longitude,
      curvature: firstWay.nodes[4].curvature,
      proposedSpeedMPH: 55,
      effectiveSpeedMPH: 55,
      desiredSpeedMPH: 55
    ),
    MapCalibrationSample(
      sourceKey: "study-curve-b:3",
      roadName: "Study Road",
      reference: "US 50",
      latitude: secondWay.nodes[3].latitude,
      longitude: secondWay.nodes[3].longitude,
      curvature: secondWay.nodes[3].curvature,
      proposedSpeedMPH: 56,
      effectiveSpeedMPH: 56,
      desiredSpeedMPH: 56
    ),
  ]
  let events = MapWholeCurveStudyResolver.resolve(
    ways: resolved,
    calibrationSamples: samples,
    parameters: .checkoutFallback,
    bands: []
  )
  let event = try #require(events.first)

  #expect(events.count == 1)
  #expect(event.bankSampleNumbers == [1, 2])
  #expect(event.bankTargetsMPH == [55, 56])
  #expect(event.lengthMeters > 200)
  #expect(abs(event.controllingCurvature - 0.004) < 0.0005)
  #expect(event.wholeCurveSpeedMPH.isFinite)
  #expect(event.points.count > 30)
}

@Test func liveSavedBankWholeCurveStudyReport() async throws {
  guard ProcessInfo.processInfo.environment["VTSC_RUN_LIVE_BANK_TEST"] == "1" else { return }
  let archiveURL = try TuneStore.applicationDirectory()
    .appendingPathComponent("curve_calibration_samples.json")
  let archive = try JSONDecoder().decode(
    MapCalibrationArchive.self,
    from: Data(contentsOf: archiveURL)
  )
  let first = try #require(archive.samples.first)
  let bounds = archive.samples.dropFirst().reduce(
    MapTileBounds(
      minLatitude: first.latitude,
      minLongitude: first.longitude,
      maxLatitude: first.latitude,
      maxLongitude: first.longitude
    )
  ) { partial, sample in
    partial.union(MapTileBounds(
      minLatitude: sample.latitude,
      minLongitude: sample.longitude,
      maxLatitude: sample.latitude,
      maxLongitude: sample.longitude
    ))
  }
  let root = try MapTileRoot.defaultURL()
  let helper = try MapTileHelperLocator.bundledURL()
  let store = try MapTileStore(
    rootURL: root,
    decoder: MapTileHelperDecoder(helperURL: helper)
  )
  let tiles = try await store.tiles(intersecting: bounds, paddingDegrees: 0.02)
  let tune = try? TuneStore.load()
  let parameters = tune?.params ?? .checkoutFallback
  let bands = tune?.bands ?? []
  var deduplicated: [String: MapRenderedWay] = [:]
  for tile in tiles {
    for way in tile.ways where way.nodes.count >= 2 {
      let rendered = MapRenderedWay(tile: tile, way: way, parameters: parameters)
      if let existing = deduplicated[way.stableID] {
        let existingHasBake = existing.nodes.contains { $0.bakedSpeedMPS != nil }
        let replacementHasBake = rendered.nodes.contains { $0.bakedSpeedMPS != nil }
        if rendered.schemaVersion > existing.schemaVersion || (replacementHasBake && !existingHasBake) {
          deduplicated[way.stableID] = rendered
        }
      } else {
        deduplicated[way.stableID] = rendered
      }
    }
  }
  let resolved = MapRuntimeCurvatureResolver.resolve(
    ways: deduplicated.values.sorted { $0.id < $1.id },
    parameters: parameters
  )
  let events = MapWholeCurveStudyResolver.resolve(
    ways: resolved,
    calibrationSamples: archive.samples,
    parameters: parameters,
    bands: bands
  )
  let coveredSamples = Set(events.flatMap(\.bankSampleNumbers))
  print("LIVE WHOLE-CURVE STUDY: \(archive.samples.count) samples -> \(events.count) directional events")
  for event in events {
    print(String(
      format: "EVENT %@ %@ samples=%@ target=%@ current=%.1f-%.1f whole=%.1f k=%.6f length=%.0fm confidence=%@ flags=%@",
      event.travelDirection,
      event.bendDirection,
      event.bankSampleNumbers.map(String.init).joined(separator: ","),
      event.bankTargetsMPH.map { String(format: "%.1f", $0) }.joined(separator: ","),
      event.currentMinimumSpeedMPH,
      event.currentMaximumSpeedMPH,
      event.wholeCurveSpeedMPH,
      event.controllingCurvature,
      event.lengthMeters,
      event.confidenceLabel,
      event.flags.joined(separator: ";")
    ))
  }
  #expect(archive.samples.count == 18)
  #expect(events.count == 12)
  #expect(coveredSamples == Set(1 ... 18))
}

@MainActor
@Test func editingASelectedCurveCreatesOnlyADraft() {
  let session = calibrationSession()
  session.select(curveSelection(id: "1"))

  #expect(session.calibrationSamples.isEmpty)
  #expect(session.canQueueSelection)
  #expect(session.selectionQueueActionTitle == "Add Curve to Bank")

  session.setDraftDesiredSpeedMPH(47)

  #expect(session.draftDesiredSpeedMPH == 47)
  #expect(session.calibrationSamples.isEmpty)
}

@MainActor
@Test func addAndUpdateKeepOneNumberedBatchItem() throws {
  let session = calibrationSession()
  let selection = curveSelection(id: "2")
  session.select(selection)
  session.setDraftDesiredSpeedMPH(42)
  session.queueSelectedCurve()

  let original = try #require(session.calibrationSamples.first)
  #expect(session.calibrationSamples.count == 1)
  #expect(original.desiredSpeedMPH == 42)
  #expect(original.hasCurrentCurvatureEstimate)
  #expect(original.rawCurvature == selection.node.rawCurvature)
  #expect(session.selectionIsQueued)
  #expect(session.selectionQueuedNumber == 1)
  #expect(session.selectionQueueActionTitle == "Update Bank Item #1")

  session.setDraftDesiredSpeedMPH(52)
  session.queueSelectedCurve()

  let updated = try #require(session.calibrationSamples.first)
  #expect(session.calibrationSamples.count == 1)
  #expect(updated.id == original.id)
  #expect(updated.desiredSpeedMPH == 52)
}

@MainActor
@Test func authoritativeCurvatureAuditInvalidatesMissingAndAmbiguousSavedGeometry() {
  let session = calibrationSession()
  session.select(curveSelection(id: "missing"))
  session.setDraftDesiredSpeedMPH(45)
  session.queueSelectedCurve()
  session.select(curveSelection(id: "ambiguous"))
  session.setDraftDesiredSpeedMPH(50)
  session.queueSelectedCurve()
  #expect(session.calibrationSamples.allSatisfy { $0.hasCurrentCurvatureEstimate })

  // A partial viewport is not authoritative for offscreen samples.
  #expect(session.reconcileCalibrationCurvatures(using: []) == 0)
  #expect(session.calibrationSamples.allSatisfy { $0.hasCurrentCurvatureEstimate })

  var ambiguousWay = renderedWay(id: "ambiguous")
  ambiguousWay.nodes[1].curvatureContextComplete = false
  let changed = session.reconcileCalibrationCurvatures(
    using: [ambiguousWay],
    invalidateMissing: true
  )

  #expect(changed == 2)
  #expect(session.calibrationSamples.allSatisfy { !$0.hasCurrentCurvatureEstimate })
  #expect(session.calibrationSamples.allSatisfy { $0.curvatureContextComplete == false })
  #expect(session.calibrationSamples.allSatisfy { $0.curvatureSupportMeters == 0 })
}

@MainActor
@Test func movingToAnotherCurveDoesNotImplicitlyQueueTheDraft() {
  let session = calibrationSession()
  session.select(curveSelection(id: "3"))
  session.setDraftDesiredSpeedMPH(63)

  session.select(curveSelection(id: "4"))

  #expect(session.calibrationSamples.isEmpty)
  #expect(session.statusText.hasPrefix("Previous draft was not added."))
}

@MainActor
@Test func movingAwayWarnsAndDiscardsAnUncommittedQueuedItemEdit() throws {
  let session = calibrationSession()
  session.select(curveSelection(id: "queued-edit"))
  session.setDraftDesiredSpeedMPH(41)
  session.queueSelectedCurve()
  let original = try #require(session.calibrationSamples.first)

  session.setDraftDesiredSpeedMPH(58)
  #expect(session.selectionHasUncommittedTarget)
  #expect(session.selectionDraftBadgeText == "UNSAVED CHANGE TO #1")
  session.select(curveSelection(id: "next-curve"))

  #expect(session.calibrationSamples.first?.id == original.id)
  #expect(session.calibrationSamples.first?.desiredSpeedMPH == 41)
  #expect(session.statusText.hasPrefix("Unsaved change to bank item #1 was discarded."))
}

@MainActor
@Test func selectingAStraightAlsoWarnsAboutTheDiscardedBankEdit() {
  let session = calibrationSession()
  session.select(curveSelection(id: "curve-before-straight"))
  session.setDraftDesiredSpeedMPH(44)
  session.queueSelectedCurve()
  session.setDraftDesiredSpeedMPH(51)

  session.select(curveSelection(id: "straight", curvature: 0))

  #expect(session.calibrationSamples.first?.desiredSpeedMPH == 44)
  #expect(session.statusText.hasPrefix("Unsaved change to bank item #1 was discarded."))
  #expect(session.statusText.contains("no usable runtime-smoothed curvature"))
}

@MainActor
@Test func removingTheActiveItemKeepsItsTargetAsAnUnqueuedDraft() throws {
  let session = calibrationSession()
  session.select(curveSelection(id: "5"))
  session.setDraftDesiredSpeedMPH(57)
  session.queueSelectedCurve()
  let id = try #require(session.calibrationSamples.first?.id)

  session.removeCalibrationSample(id: id)

  #expect(session.calibrationSamples.isEmpty)
  #expect(session.draftDesiredSpeedMPH == 57)
  #expect(!session.selectionIsQueued)
  #expect(session.selectionQueueActionTitle == "Add Curve to Bank")
}

@MainActor
@Test func oneCurveBankAcceptsMoreThanFiveAndFitsEverySavedTarget() throws {
  let session = calibrationSession()
  for index in 1...12 {
    session.select(curveSelection(id: "fit-\(index)", curvature: 0.001 + 0.001 * Double(index)))
    session.setDraftDesiredSpeedMPH(Double(28 + index * 3))
    session.queueSelectedCurve()
    #expect(session.calibrationSamples.count == index)
    #expect(session.canRunFit == (index >= MapPreviewSession.minimumCalibrationSamples))
  }
  #expect(session.calibrationSamples.count == 12)

  let firstID = try #require(session.calibrationSamples.first?.id)
  session.updateCalibrationTarget(id: firstID, desiredSpeedMPH: 0)
  #expect(!session.canRunFit)

  session.updateCalibrationTarget(id: firstID, desiredSpeedMPH: 30)
  #expect(session.canRunFit)

  let result = try SigmoidFitter.fit(
    samples: session.calibrationSamples.map {
      CurveCalibrationSample(
        id: $0.id,
        label: $0.roadName,
        curvature: $0.curvature,
        desiredSpeedMPH: $0.desiredSpeedMPH
      )
    },
    currentKnobs: VTSCMath.knobs(from: .checkoutFallback),
    anchorKnobs: VTSCMath.knobs(from: .checkoutFallback)
  )
  #expect(result.diagnostics.count == 12)
}

@MainActor
@Test func generateProposalSurvivesANoOpTargetCommit() async throws {
  let session = calibrationSession()
  for index in 1...15 {
    session.select(curveSelection(id: "proposal-\(index)", curvature: 0.001 + 0.0007 * Double(index)))
    session.setDraftDesiredSpeedMPH(Double(30 + index * 2))
    session.queueSelectedCurve()
  }
  let lastSample = try #require(session.calibrationSamples.last)

  session.runFit(
    currentKnobs: VTSCMath.knobs(from: .checkoutFallback),
    anchorKnobs: VTSCMath.knobs(from: .checkoutFallback),
    bands: []
  )
  // A formatted TextField may recommit its existing numeric value when the
  // Generate button takes focus. That must not invalidate the new fit.
  session.updateCalibrationTarget(
    id: lastSample.id,
    desiredSpeedMPH: lastSample.desiredSpeedMPH
  )

  for _ in 0..<3_000 where session.isFitting {
    try await Task.sleep(for: .milliseconds(10))
  }

  #expect(session.fitErrorText == nil)
  let result = try #require(session.fitResult)
  #expect(result.diagnostics.count == 15)
  #expect(session.statusText.hasPrefix("Complete-curve fit ready:"))
  #expect(result.diagnostics.first?.label.hasPrefix("#1 ") == true)
}

@MainActor
@Test func acceptingAProposalMakesItTheActiveCurveLabTune() throws {
  let session = TunerSession()
  session.mapPreview.purpose = .calibration
  let originalKnobs = session.knobs
  let originalBand = EQBand(centerSpeedMPH: 33, gainDB: -2, q: 2)
  session.bands = [originalBand]
  session.plot.selected = .band(originalBand.id)
  let samples = (1...6).map { index in
    CurveCalibrationSample(
      label: "Acceptance curve \(index)",
      curvature: 0.002 * Double(index),
      desiredSpeedMPH: Double(28 + index * 6)
    )
  }
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: session.knobs,
    anchorKnobs: session.calibrationAnchorKnobs,
    bands: session.bands
  )
  session.workspace = .mapPreview
  session.mapPreview.fitResult = result

  session.mapPreview.acceptFit(
    result,
    currentKnobs: session.knobs,
    currentAnchorKnobs: session.calibrationAnchorKnobs,
    currentBands: session.bands,
    action: session.acceptFittedTune
  )

  #expect(session.workspace == .curveLab)
  #expect(session.knobs == result.knobs)
  #expect(session.parameters == result.parameters)
  #expect(session.bands == result.bands)
  #expect(session.plot.selected == nil)
  #expect(session.mapPreview.fitResult == nil)
  #expect(session.statusText.hasPrefix("Accepted complete fitted curve"))

  session.undo()
  #expect(session.knobs == originalKnobs)
  #expect(session.bands == [originalBand])

  session.redo()
  #expect(session.knobs == result.knobs)
  #expect(session.bands == result.bands)
}

@MainActor
@Test func staleProposalCannotOverwriteANewerTune() throws {
  let session = TunerSession()
  session.mapPreview.purpose = .calibration
  let samples = (1...6).map { index in
    CurveCalibrationSample(
      label: "Stale curve \(index)",
      curvature: 0.002 * Double(index),
      desiredSpeedMPH: Double(30 + index * 5)
    )
  }
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: session.knobs,
    anchorKnobs: session.calibrationAnchorKnobs,
    bands: session.bands
  )
  session.mapPreview.fitResult = result
  session.knobs.sharpness += 0.25
  let newerKnobs = session.knobs
  var accepted = false

  session.mapPreview.acceptFit(
    result,
    currentKnobs: session.knobs,
    currentAnchorKnobs: session.calibrationAnchorKnobs,
    currentBands: session.bands
  ) { _ in
    accepted = true
  }

  #expect(!accepted)
  #expect(session.knobs == newerKnobs)
  #expect(session.mapPreview.fitResult == result)
  #expect(session.mapPreview.statusIsError)
  #expect(session.mapPreview.statusText.hasPrefix("That fit is stale."))
}

@MainActor
@Test func proposalAnchoredToAnotherCalibrationBaselineIsRejected() throws {
  let session = TunerSession()
  session.mapPreview.purpose = .calibration
  let samples = (1...6).map { index in
    CurveCalibrationSample(
      label: "Anchor curve \(index)",
      curvature: 0.002 * Double(index),
      desiredSpeedMPH: 70.0 - Double(index * 5)
    )
  }
  let originalAnchor = session.calibrationAnchorKnobs
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: session.knobs,
    anchorKnobs: originalAnchor,
    bands: session.bands
  )
  session.mapPreview.fitResult = result
  var changedAnchor = originalAnchor
  changedAnchor.transitionSpeedMPH += 1.0
  var accepted = false

  session.mapPreview.acceptFit(
    result,
    currentKnobs: session.knobs,
    currentAnchorKnobs: changedAnchor,
    currentBands: session.bands
  ) { _ in
    accepted = true
  }

  #expect(!accepted)
  #expect(session.mapPreview.fitResult == result)
  #expect(session.mapPreview.statusIsError)
  #expect(session.mapPreview.statusText.hasPrefix("That fit is stale."))
}

@MainActor
@Test func calibrationArchivePersistsTheOriginalFitAnchor() throws {
  let original = VTSCMath.knobs(from: .checkoutFallback)
  let archive = MapCalibrationArchive(samples: [], anchorKnobs: original)
  let encoded = try JSONEncoder().encode(archive)
  let decoded = try JSONDecoder().decode(MapCalibrationArchive.self, from: encoded)
  #expect(decoded.schema == 3)
  let decodedAnchor = try #require(decoded.anchorKnobs)
  #expect(abs(decodedAnchor.tightCurveAcceleration - original.tightCurveAcceleration) < 1.0e-12)
  #expect(abs(decodedAnchor.straightRoadAcceleration - original.straightRoadAcceleration) < 1.0e-12)
  #expect(abs(decodedAnchor.transitionSpeedMPH - original.transitionSpeedMPH) < 1.0e-12)
  #expect(abs(decodedAnchor.sharpness - original.sharpness) < 1.0e-12)

  let legacy = try JSONDecoder().decode(
    MapCalibrationArchive.self,
    from: #"{"schema":1,"samples":[]}"#.data(using: .utf8)!
  )
  #expect(legacy.anchorKnobs == nil)

  let session = MapPreviewSession(
    loadPersistedState: false,
    persistsCalibrationSamples: false,
    persistsMapPreferences: false
  )
  session.purpose = .calibration
  let first = session.ensureCalibrationAnchor(original)
  var moved = original
  moved.transitionSpeedMPH += 10
  #expect(session.ensureCalibrationAnchor(moved) == first)
  #expect(session.calibrationAnchorKnobs == first)
}

@Test func bandMarkerAndDragMathUseTheCompleteRoundedQCurve() throws {
  let parameters = SigmoidParameters.checkoutFallback
  let targetID = UUID()
  var bands = [
    EQBand(id: targetID, centerSpeedMPH: 50, gainDB: 0, q: 4),
    EQBand(id: UUID(), centerSpeedMPH: 53, gainDB: 3, q: 3),
  ]
  let baselineMarker = try #require(
    CurveBandEditingMath.marker(bandID: targetID, bands: bands, parameters: parameters)
  )
  let targetAcceleration = baselineMarker.accelerationMPS2 * 1.15
  let gain = try #require(
    CurveBandEditingMath.gainDB(
      bandID: targetID,
      targetAccelerationMPS2: targetAcceleration,
      bands: bands,
      parameters: parameters
    )
  )
  bands[0].gainDB = gain
  let fittedMarker = try #require(
    CurveBandEditingMath.marker(bandID: targetID, bands: bands, parameters: parameters)
  )

  #expect(fittedMarker.speedMPH == bands[0].centerSpeedMPH)
  #expect(abs(fittedMarker.accelerationMPS2 - targetAcceleration) < 0.002)
  #expect(gain > 0)
}

@MainActor
@Test func appleRoadTextIsDefaultAndAChosenPlaceCanBecomeTheOpeningLocation() {
  let session = MapPreviewSession(
    loadPersistedState: false,
    persistsCalibrationSamples: false,
    persistsMapPreferences: false
  )
  #expect(session.roadLabelSize == .appleOnly)
  #expect(session.openingLocation == nil)

  let place = MapPlaceResult(
    title: "Madison Ave",
    subtitle: "Sacramento, CA",
    latitude: 38.661_087_3,
    longitude: -121.368_930_2,
    latitudeDelta: 0.02,
    longitudeDelta: 0.02
  )
  session.choosePlace(place)
  #expect(session.canSetOpeningLocation)

  session.setCurrentPlaceAsOpeningLocation()
  #expect(session.openingLocation?.title == "Madison Ave")
  session.goToOpeningLocation()
  #expect(session.cameraDestination?.latitude == place.latitude)
  #expect(session.cameraDestination?.longitude == place.longitude)

  session.clearOpeningLocation()
  #expect(session.openingLocation == nil)
}
