import Foundation

public enum VTSCClip {
  public static let baseline = 2.0 ... 6.5
  public static let amplitudeMagnitude = 0.2 ... 5.0
  public static let steepnessMagnitude = 100.0 ... 100_000.0
  public static let center = 1.0e-5 ... 0.1
  public static let minimumLateralAcceleration = 1.0 ... 3.0
  public static let maximumLateralAcceleration = 2.0 ... 5.5
  public static let transitionSpeedMPH = 8.0 ... 120.0
  public static let sharpness = 0.0 ... 10.0
}

public struct SigmoidParameters: Codable, Equatable, Sendable {
  public var a: Double
  public var b: Double
  public var c: Double
  public var d: Double
  public var minLat: Double
  public var maxLat: Double

  public init(a: Double, b: Double, c: Double, d: Double, minLat: Double, maxLat: Double) {
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.minLat = minLat
    self.maxLat = maxLat
  }

  /// Historical Rust tuner baseline, retained for schema and golden-test parity.
  public static let factoryBaseline = Self(
    a: -3.260,
    b: -6_270.0,
    c: 0.005_010,
    d: 5.607,
    minLat: 1.8,
    maxLat: 4.478
  )

  /// Fallback only. The app normally reads these values from the selected checkout.
  public static let checkoutFallback = Self(
    a: -1.658_965,
    b: -1_395.055_546,
    c: 0.005_397,
    d: 4.107_103,
    minLat: 2.448_1,
    maxLat: 4.107_1
  )

  enum CodingKeys: String, CodingKey {
    case a, b, c, d
    case minLat = "min_lat"
    case maxLat = "max_lat"
  }
}

public struct PlainKnobs: Codable, Equatable, Sendable {
  public var tightCurveAcceleration: Double
  public var straightRoadAcceleration: Double
  public var transitionSpeedMPH: Double
  public var sharpness: Double

  public init(
    tightCurveAcceleration: Double,
    straightRoadAcceleration: Double,
    transitionSpeedMPH: Double,
    sharpness: Double
  ) {
    self.tightCurveAcceleration = tightCurveAcceleration
    self.straightRoadAcceleration = straightRoadAcceleration
    self.transitionSpeedMPH = transitionSpeedMPH
    self.sharpness = sharpness
  }
}

public struct EQBand: Codable, Equatable, Identifiable, Sendable {
  /// IDs are UI-only and deliberately omitted from the Rust-compatible JSON schema.
  public var id: UUID = UUID()
  public var centerSpeedMPH: Double
  public var gainDB: Double
  public var q: Double
  public var enabled: Bool

  public init(
    id: UUID = UUID(),
    centerSpeedMPH: Double = 45.0,
    gainDB: Double = 0.0,
    q: Double = 1.0,
    enabled: Bool = true
  ) {
    self.id = id
    self.centerSpeedMPH = centerSpeedMPH
    self.gainDB = gainDB
    self.q = q
    self.enabled = enabled
  }

  enum CodingKeys: String, CodingKey {
    case centerSpeedMPH = "center_speed_mph"
    case gainDB = "gain_db"
    case q, enabled
  }
}

public struct Tune: Codable, Equatable, Sendable {
  public var schema: UInt32
  public var created: String
  public var note: String
  public var params: SigmoidParameters
  public var bands: [EQBand]
  /// Exact user-facing controls. Older schema-v1 readers ignore this additive
  /// field; newer readers prefer it because the rounded raw parameters cannot
  /// losslessly reconstruct all four controls.
  public var knobs: PlainKnobs?

  public init(
    schema: UInt32 = 1,
    created: String = Date().ISO8601Format(),
    note: String = "",
    params: SigmoidParameters,
    bands: [EQBand] = [],
    knobs: PlainKnobs? = nil
  ) {
    self.schema = schema
    self.created = created
    self.note = note
    self.params = params
    self.bands = bands
    self.knobs = knobs
  }
}

public struct CurveSample: Equatable, Sendable {
  public var kappa: Double
  public var speedMPH: Double
  public var lateralAcceleration: Double

  public init(kappa: Double, speedMPH: Double, lateralAcceleration: Double) {
    self.kappa = kappa
    self.speedMPH = speedMPH
    self.lateralAcceleration = lateralAcceleration
  }
}

public struct PreparedBand: Equatable, Sendable {
  public var logCenterKappa: Double
  public var gainDB: Double
  public var sigma: Double

  public init(logCenterKappa: Double, gainDB: Double, sigma: Double) {
    self.logCenterKappa = logCenterKappa
    self.gainDB = gainDB
    self.sigma = sigma
  }
}

public struct EditableSnapshot: Equatable, Sendable {
  public var knobs: PlainKnobs
  public var bands: [EQBand]

  public init(knobs: PlainKnobs, bands: [EQBand]) {
    self.knobs = knobs
    self.bands = bands
  }
}

extension Double {
  func clamped(to range: ClosedRange<Double>) -> Double {
    Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
  }
}
