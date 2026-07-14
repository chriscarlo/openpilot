import Foundation
import Testing
@testable import VTSCTunerCore

@Test func sigmoidMatchesReferenceAtMidpoint() {
  let parameters = SigmoidParameters.factoryBaseline
  let expected = (parameters.a / 2.0 + parameters.d)
    .clamped(to: parameters.minLat ... parameters.maxLat)
  #expect(abs(VTSCMath.evaluate(parameters, curvature: parameters.c) - expected) < 1.0e-9)
}

@Test func plainKnobRoundTripIsStable() {
  let knobs = VTSCMath.knobs(from: .factoryBaseline)
  let roundTripped = VTSCMath.knobs(from: VTSCMath.parameters(from: knobs))
  #expect(abs(knobs.tightCurveAcceleration - roundTripped.tightCurveAcceleration) < 1.0e-6)
  #expect(abs(knobs.straightRoadAcceleration - roundTripped.straightRoadAcceleration) < 1.0e-6)
  #expect(abs(knobs.transitionSpeedMPH - roundTripped.transitionSpeedMPH) < 0.1)
  #expect(abs(knobs.sharpness - roundTripped.sharpness) < 1.0e-6)
}

@Test func currentWholeCurveProductionTunePersistsExactKnobsAndRoundsExactly() throws {
  let knobs = PlainKnobs(
    tightCurveAcceleration: 2.448_138,
    straightRoadAcceleration: 4.107_103,
    transitionSpeedMPH: 55.127_268,
    sharpness: 3.815_305
  )
  let expected = SigmoidParameters(
    a: -1.658_965,
    b: -1_395.055_546,
    c: 0.005_397,
    d: 4.107_103,
    minLat: 2.448_1,
    maxLat: 4.107_1
  )

  let rounded = VTSCMath.sourceRoundedParameters(VTSCMath.parameters(from: knobs))
  #expect(rounded == expected)

  let tune = Tune(params: rounded, knobs: knobs)
  let reloaded = try JSONDecoder().decode(Tune.self, from: JSONEncoder().encode(tune))
  let reloadedKnobs = try #require(reloaded.knobs)
  #expect(VTSCMath.sourceRoundedParameters(VTSCMath.parameters(from: reloadedKnobs)) == expected)
}

@Test func aggressiveBandPlotUsesMonotonicSpeedPostProcessing() {
  let band = EQBand(centerSpeedMPH: 48.0, gainDB: 12.0, q: 16.0)
  let samples = VTSCMath.sampleCurve(
    parameters: .factoryBaseline,
    bands: [band],
    count: 512,
    enforceMonotonicSpeed: true
  )
  for pair in zip(samples, samples.dropFirst()) {
    #expect(pair.1.speedMPH >= pair.0.speedMPH - 1.0e-9)
  }
}

@Test func curveSamplesShowTheExportedQCurveBeyondTheRawSigmoidRail() {
  let parameters = SigmoidParameters.factoryBaseline
  let samples = VTSCMath.sampleCurve(
    parameters: parameters,
    bands: [EQBand(centerSpeedMPH: 48.0, gainDB: 12.0, q: 16.0)],
    count: 512
  )
  #expect(samples.map(\.lateralAcceleration).max()! > parameters.maxLat)
}

@Test func rustSchemaJSONRoundTripsWithoutUIIds() throws {
  let json = #"""
  {
    "schema": 1,
    "created": "2026-04-19T12:34:56-07:00",
    "note": "",
    "params": {"a":-3.26,"b":-6270.0,"c":0.00501,"d":5.607,"min_lat":1.8,"max_lat":4.478},
    "bands": [{"center_speed_mph":45.0,"gain_db":3.0,"q":1.5,"enabled":true}]
  }
  """#.data(using: .utf8)!
  let tune = try JSONDecoder().decode(Tune.self, from: json)
  #expect(tune.schema == 1)
  #expect(tune.bands.count == 1)
  let encoded = try JSONEncoder().encode(tune)
  let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
  let bands = try #require(object["bands"] as? [[String: Any]])
  #expect(bands[0]["id"] == nil)
  #expect(bands[0]["center_speed_mph"] as? Double == 45.0)
}

@Test func sourcePatchUsesLiveFormattingAndEnabledBandsOnly() throws {
  let physics = """
  PHYSICS_A = -1.0 # keep
  PHYSICS_B = -1000.0
  PHYSICS_C = 0.01
  PHYSICS_D = 4.0
  PHYSICS_MIN_LAT_ACCEL = 1.5
  PHYSICS_MAX_LAT_ACCEL = 4.0
  """
  let qCurve = """
  Q_CURVE_ENABLED = True
  Q_CURVE_POINTS: list[tuple[float, float]] = [
    (1.0e-5, 1.0),
  ]
  Q_CURVE_META = {}
  """
  let paramsDefaults = """
  {"VisionTurnSpeedControlPhysicsAmplitude", {PERSISTENT | BACKUP, FLOAT, "-1.000000"}},
  {"VisionTurnSpeedControlPhysicsSteepness", {PERSISTENT | BACKUP, FLOAT, "-1000.000000"}},
  {"VisionTurnSpeedControlPhysicsCenter", {PERSISTENT | BACKUP, FLOAT, "0.010000"}},
  {"VisionTurnSpeedControlPhysicsBaseline", {PERSISTENT | BACKUP, FLOAT, "4.000000"}},
  {"VisionTurnSpeedControlPhysicsMinLatAccel", {PERSISTENT | BACKUP, FLOAT, "1.5000"}},
  {"VisionTurnSpeedControlPhysicsMaxLatAccel", {PERSISTENT | BACKUP, FLOAT, "4.0000"}},
  """
  let physicsPanel = """
  params.put("VisionTurnSpeedControlPhysicsAmplitude", "-1.000000");
  params.put("VisionTurnSpeedControlPhysicsSteepness", "-1000.000000");
  params.put("VisionTurnSpeedControlPhysicsCenter", "0.010000");
  params.put("VisionTurnSpeedControlPhysicsBaseline", "4.000000");
  params.put("VisionTurnSpeedControlPhysicsMinLatAccel", "1.5000");
  params.put("VisionTurnSpeedControlPhysicsMaxLatAccel", "4.0000");
  ensure("VisionTurnSpeedControlPhysicsAmplitude", "-1.000000");
  ensure("VisionTurnSpeedControlPhysicsSteepness", "-1000.000000");
  ensure("VisionTurnSpeedControlPhysicsCenter", "0.010000");
  ensure("VisionTurnSpeedControlPhysicsBaseline", "4.000000");
  ensure("VisionTurnSpeedControlPhysicsMinLatAccel", "1.5000");
  ensure("VisionTurnSpeedControlPhysicsMaxLatAccel", "4.0000");
  """
  let patch = try SourcePatcher.makePatch(
    physicsText: physics,
    qCurveText: qCurve,
    paramsDefaultsText: paramsDefaults,
    physicsPanelText: physicsPanel,
    parameters: .checkoutFallback,
    bands: [EQBand(enabled: false)]
  )
  #expect(patch.physicsText.contains("PHYSICS_A = -1.658965 # keep"))
  #expect(patch.physicsText.contains("PHYSICS_MIN_LAT_ACCEL = 2.4481"))
  #expect(patch.qCurveText.contains("Q_CURVE_ENABLED = False"))
  #expect(patch.qCurveText.contains("Q_CURVE_POINTS: list[tuple[float, float]] = []"))
  #expect(patch.paramsDefaultsText.contains(#"VisionTurnSpeedControlPhysicsBaseline", {PERSISTENT | BACKUP, FLOAT, "4.107103""#))
  #expect(patch.physicsPanelText.contains(#"ensure("VisionTurnSpeedControlPhysicsSteepness", "-1395.055546")"#))
}

@Test func sourceAuthorityRejectsAStalePersistentDefault() throws {
  let paramsDefaults = VTSCPhysicsAuthority.entries(for: .checkoutFallback).map {
    "{\"\($0.paramKey)\", {PERSISTENT | BACKUP, FLOAT, \"\($0.formattedValue)\"}},"
  }.joined(separator: "\n")
  let physicsPanel = VTSCPhysicsAuthority.entries(for: .checkoutFallback).flatMap { entry in
    [
      "params.put(\"\(entry.paramKey)\", \"\(entry.formattedValue)\");",
      "ensure(\"\(entry.paramKey)\", \"\(entry.formattedValue)\");",
    ]
  }.joined(separator: "\n")
  let stale = paramsDefaults.replacingOccurrences(of: #"FLOAT, "4.107103""#, with: #"FLOAT, "3.144734""#)
  #expect(throws: RepositoryError.self) {
    try SourcePatcher.validateAuthority(
      parameters: .checkoutFallback,
      paramsDefaultsText: stale,
      physicsPanelText: physicsPanel
    )
  }
}

@Test func qCurveExportsCurrentRustPointCount() {
  let points = VTSCMath.qCurvePoints(
    parameters: .factoryBaseline,
    bands: [EQBand(gainDB: 3.0)],
    count: 256
  )
  #expect(points.count == 256)
  #expect(abs(points.first!.kappa - 1.0e-5) < 1.0e-12)
  #expect(abs(points.last!.kappa - 1.0) < 1.0e-12)
  #expect(points.allSatisfy { $0.kappa.isFinite && $0.speedMultiplier.isFinite })
  #expect(points.allSatisfy { (0.5 ... 1.5).contains($0.speedMultiplier) })
  #expect(zip(points, points.dropFirst()).allSatisfy { $0.kappa < $1.kappa })

  let locale = Locale(identifier: "en_US_POSIX")
  #expect(points.allSatisfy {
    Double(String(format: "%.6e", locale: locale, $0.kappa)) == $0.kappa
      && Double(String(format: "%.4f", locale: locale, $0.speedMultiplier))
        == $0.speedMultiplier
  })
}
