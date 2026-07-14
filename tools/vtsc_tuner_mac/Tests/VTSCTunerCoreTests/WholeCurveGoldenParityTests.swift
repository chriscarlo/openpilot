import Foundation
import Testing
@testable import VTSCTunerCore

private struct WholeCurveGoldenCorpus: Decodable {
  var schemaVersion: Int
  var estimatorVersion: String
  var tolerances: [String: Double]
  var cases: [GoldenCase]
  var savedBank: SavedBank

  struct GoldenCase: Decodable {
    var name: String
    var features: [String]
    var route: [GoldenPoint]
    var expected: GoldenResult
  }

  struct GoldenPoint: Decodable {
    var latitude: Double
    var longitude: Double
    var sourceKey: String?
  }

  struct GoldenResult: Decodable {
    var pointCount: Int
    var events: [GoldenEvent]
  }

  struct GoldenEvent: Decodable {
    var physicalID: String
    var id: String
    var startIndex: Int
    var endIndex: Int
    var apexIndex: Int
    var lengthMeters: Double
    var signedTurnRadians: Double
    var signCoherence: Double
    var curvature60: Double?
    var curvature100: Double?
    var curvature160: Double?
    var controllingCurvature: Double
    var scaleSpread: Double?
    var maximumSourceGapMeters: Double
    var confidence: String
    var flags: [String]
  }

  struct SavedBank: Decodable {
    var sampleCount: Int
    var directionalEventCount: Int
    var samples: [SavedSample]
    var cases: [SavedCase]
  }

  struct SavedSample: Decodable {
    var number: Int
    var sourceKey: String
    var latitude: Double
    var longitude: Double
    var desiredSpeedMPH: Double
  }

  struct SavedCase: Decodable {
    var name: String
    var sampleNumbers: [Int]
    var route: [GoldenPoint]
    var expectedPointCount: Int
    var expectedEvent: GoldenEvent
    var expectedStudy: ExpectedStudy
  }

  struct ExpectedStudy: Decodable {
    var travelDirection: String
    var bendDirection: String
    var wholeCurveSpeedMPH: Double
    var currentMinimumSpeedMPH: Double
    var currentMaximumSpeedMPH: Double
    var bankTargetsMPH: [Double]
  }
}

@Suite("Swift and Go whole-curve golden parity")
struct WholeCurveGoldenParityTests {
  @Test("Shared checked-in geometry corpus matches the Swift estimator")
  func sharedCorpus() throws {
    let corpus = try loadCorpus()
    #expect(corpus.schemaVersion == 1)
    #expect(corpus.estimatorVersion == "whole-curve-v1")

    let requiredFeatures: Set<String> = [
      "straight", "nonuniform_node_spacing", "duplicate_nodes", "route_reversal",
      "s_curve", "compact_genuine_apex", "sparse_geometry", "compound_bend",
      "interpolation_overshoot_resistance", "one_node_zigzag", "cross_way_curve",
      "forward_direction", "reverse_direction", "ambiguous_fork",
      "tile_context_truncation", "current_way_rollover_mid_event",
    ]
    let coveredFeatures = Set(corpus.cases.flatMap(\.features))
    #expect(requiredFeatures.isSubset(of: coveredFeatures))

    for golden in corpus.cases {
      let estimate = WholeCurveEstimator.estimate(route: golden.route.map {
        WholeCurveInputPoint(
          latitude: $0.latitude,
          longitude: $0.longitude,
          sourceKey: $0.sourceKey ?? ""
        )
      })
      #expect(estimate.points.count == golden.expected.pointCount, "\(golden.name): point count")
      #expect(estimate.events.count == golden.expected.events.count, "\(golden.name): event count")
      guard estimate.events.count == golden.expected.events.count else { continue }
      for (event, expected) in zip(estimate.events, golden.expected.events) {
        #expect(event.physicalID == expected.physicalID, "\(golden.name): physical ID")
        #expect(event.directionalID == expected.id, "\(golden.name): directional ID")
        #expect(event.startIndex == expected.startIndex, "\(golden.name): start index")
        #expect(event.endIndex == expected.endIndex, "\(golden.name): end index")
        #expect(event.apexIndex == expected.apexIndex, "\(golden.name): apex index")
        expectClose(event.lengthMeters, expected.lengthMeters, corpus.tolerances["distanceMeters"]!, golden.name, "length")
        expectClose(event.signedTurnRadians, expected.signedTurnRadians, corpus.tolerances["turnRadians"]!, golden.name, "turn")
        expectClose(event.signCoherence, expected.signCoherence, corpus.tolerances["coherence"]!, golden.name, "coherence")
        expectClose(event.curvature60, expected.curvature60, corpus.tolerances["curvature"]!, golden.name, "curvature60")
        expectClose(event.curvature100, expected.curvature100, corpus.tolerances["curvature"]!, golden.name, "curvature100")
        expectClose(event.curvature160, expected.curvature160, corpus.tolerances["curvature"]!, golden.name, "curvature160")
        expectClose(event.controllingCurvature, expected.controllingCurvature, corpus.tolerances["curvature"]!, golden.name, "controlling curvature")
        expectClose(event.scaleSpread, expected.scaleSpread, corpus.tolerances["curvature"]!, golden.name, "scale spread")
        expectClose(event.maximumSourceGapMeters, expected.maximumSourceGapMeters, corpus.tolerances["distanceMeters"]!, golden.name, "source gap")
        #expect(confidenceCode(event.confidence) == expected.confidence, "\(golden.name): confidence")
        #expect(event.flags.map(flagCode) == expected.flags, "\(golden.name): flags")
      }
    }

    let bank = corpus.savedBank
    #expect(bank.sampleCount == 18)
    #expect(bank.directionalEventCount == 12)
    #expect(bank.samples.count == 18)
    #expect(bank.cases.count == 12)
    let samplesByNumber = Dictionary(uniqueKeysWithValues: bank.samples.map { ($0.number, $0) })
    var coveredSamples: Set<Int> = []
    var sample16Speed: Double?
    for golden in bank.cases {
      let estimate = WholeCurveEstimator.estimate(route: golden.route.map {
        WholeCurveInputPoint(latitude: $0.latitude, longitude: $0.longitude, sourceKey: $0.sourceKey ?? "")
      })
      #expect(estimate.points.count == golden.expectedPointCount, "\(golden.name): bank point count")
      guard let event = estimate.events.first(where: { $0.directionalID == golden.expectedEvent.id }) else {
        Issue.record("\(golden.name): missing saved-bank event \(golden.expectedEvent.id)")
        continue
      }
      expectEvent(event, golden.expectedEvent, corpus.tolerances, golden.name)
      for number in golden.sampleNumbers {
        coveredSamples.insert(number)
        guard let sample = samplesByNumber[number] else {
          Issue.record("\(golden.name): missing saved sample \(number)")
          continue
        }
        #expect(event.sourceKeys.contains(sample.sourceKey), "\(golden.name): exact provenance for sample \(number)")
        if number == 16 { sample16Speed = golden.expectedStudy.wholeCurveSpeedMPH }
      }
      #expect(golden.expectedStudy.wholeCurveSpeedMPH.isFinite && golden.expectedStudy.wholeCurveSpeedMPH > 0)
    }
    #expect(coveredSamples == Set(1 ... 18))
    #expect(sample16Speed != nil && abs(sample16Speed! - 97.9109081492345) <= 1e-9)
  }

  private func loadCorpus() throws -> WholeCurveGoldenCorpus {
    let packageDirectory = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let fixtureURL = packageDirectory
      .deletingLastPathComponent()
      .appendingPathComponent("vtsc/fixtures/whole_curve_v1.json")
    return try JSONDecoder().decode(
      WholeCurveGoldenCorpus.self,
      from: Data(contentsOf: fixtureURL)
    )
  }

  private func expectClose(
    _ actual: Double,
    _ expected: Double,
    _ tolerance: Double,
    _ caseName: String,
    _ field: String
  ) {
    #expect(actual.isFinite && abs(actual - expected) <= tolerance, "\(caseName): \(field)")
  }

  private func expectClose(
    _ actual: Double?,
    _ expected: Double?,
    _ tolerance: Double,
    _ caseName: String,
    _ field: String
  ) {
    guard let actual, let expected else {
      #expect(actual == nil && expected == nil, "\(caseName): \(field) optional presence")
      return
    }
    expectClose(actual, expected, tolerance, caseName, field)
  }

  private func confidenceCode(_ confidence: WholeCurveConfidence) -> String {
    switch confidence {
    case .high: "high"
    case .review: "review"
    case .low: "low"
    }
  }

  private func expectEvent(
    _ event: WholeCurveEvent,
    _ expected: WholeCurveGoldenCorpus.GoldenEvent,
    _ tolerances: [String: Double],
    _ caseName: String
  ) {
    #expect(event.physicalID == expected.physicalID, "\(caseName): physical ID")
    #expect(event.directionalID == expected.id, "\(caseName): directional ID")
    #expect(event.startIndex == expected.startIndex, "\(caseName): start index")
    #expect(event.endIndex == expected.endIndex, "\(caseName): end index")
    #expect(event.apexIndex == expected.apexIndex, "\(caseName): apex index")
    expectClose(event.lengthMeters, expected.lengthMeters, tolerances["distanceMeters"]!, caseName, "length")
    expectClose(event.signedTurnRadians, expected.signedTurnRadians, tolerances["turnRadians"]!, caseName, "turn")
    expectClose(event.signCoherence, expected.signCoherence, tolerances["coherence"]!, caseName, "coherence")
    expectClose(event.curvature60, expected.curvature60, tolerances["curvature"]!, caseName, "curvature60")
    expectClose(event.curvature100, expected.curvature100, tolerances["curvature"]!, caseName, "curvature100")
    expectClose(event.curvature160, expected.curvature160, tolerances["curvature"]!, caseName, "curvature160")
    expectClose(event.controllingCurvature, expected.controllingCurvature, tolerances["curvature"]!, caseName, "controlling curvature")
    expectClose(event.scaleSpread, expected.scaleSpread, tolerances["curvature"]!, caseName, "scale spread")
    expectClose(event.maximumSourceGapMeters, expected.maximumSourceGapMeters, tolerances["distanceMeters"]!, caseName, "source gap")
    #expect(confidenceCode(event.confidence) == expected.confidence, "\(caseName): confidence")
    #expect(event.flags.map(flagCode) == expected.flags, "\(caseName): flags")
  }

  private func flagCode(_ flag: WholeCurveFlag) -> String {
    switch flag {
    case .missingScaleContext: "missing_scale_context"
    case .incompleteStraightShoulder: "incomplete_straight_shoulder"
    case .sparseSourceGeometry: "sparse_source_geometry"
    case .unstableTurnSign: "unstable_turn_sign"
    case .highScaleSpread: "high_scale_spread"
    case .shortEvent: "short_event"
    case .invalidGeometry: "invalid_geometry"
    }
  }
}
