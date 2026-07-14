import Foundation

public enum VTSCMath {
  public static let metersPerSecondToMPH = 2.236_936_292_054_4
  public static let mphToMetersPerSecond = 0.447_04

  public static func evaluate(_ parameters: SigmoidParameters, curvature: Double) -> Double {
    let kappa = curvature.clamped(to: 1.0e-8 ... 1.0)
    let exponent = parameters.b * (kappa - parameters.c)
    let raw: Double
    if exponent > 700.0 {
      raw = parameters.d
    } else if exponent < -700.0 {
      raw = parameters.a + parameters.d
    } else {
      raw = parameters.a / (1.0 + Foundation.exp(exponent)) + parameters.d
    }
    return raw.clamped(to: parameters.minLat ... parameters.maxLat)
  }

  public static func kappa(speedMPH: Double, lateralAcceleration: Double) -> Double {
    let speedMPS = speedMPH * mphToMetersPerSecond
    guard speedMPS >= 1.0e-3 else { return 1.0 }
    return (lateralAcceleration / (speedMPS * speedMPS)).clamped(to: 1.0e-8 ... 1.0)
  }

  public static func parameters(from knobs: PlainKnobs) -> SigmoidParameters {
    let low = knobs.tightCurveAcceleration.clamped(to: VTSCClip.minimumLateralAcceleration)
    let high = max(
      knobs.straightRoadAcceleration.clamped(to: VTSCClip.maximumLateralAcceleration),
      low + VTSCClip.amplitudeMagnitude.lowerBound
    )
    let d = high.clamped(to: VTSCClip.baseline)
    let a = -(high - low).clamped(to: VTSCClip.amplitudeMagnitude)
    let midpointAcceleration = 0.5 * (low + high)
    let speedMPS = knobs.transitionSpeedMPH.clamped(to: VTSCClip.transitionSpeedMPH) * mphToMetersPerSecond
    let c = (midpointAcceleration / (speedMPS * speedMPS)).clamped(to: VTSCClip.center)
    let sharpness = knobs.sharpness.clamped(to: VTSCClip.sharpness)
    let b = -pow(10.0, 2.0 + 0.3 * sharpness)
    return SigmoidParameters(a: a, b: b, c: c, d: d, minLat: low, maxLat: high)
  }

  /// The apply pipeline writes A/B/C/D with six fractional digits and the two
  /// rails with four. Fit and preview code uses this representation at the
  /// runtime boundary so safety checks cannot depend on precision that will be
  /// discarded when the proposal is applied.
  public static func sourceRoundedParameters(_ parameters: SigmoidParameters) -> SigmoidParameters {
    func rounded(_ value: Double, scale: Double) -> Double {
      (value * scale).rounded() / scale
    }
    return SigmoidParameters(
      a: rounded(parameters.a, scale: 1_000_000.0),
      b: rounded(parameters.b, scale: 1_000_000.0),
      c: rounded(parameters.c, scale: 1_000_000.0),
      d: rounded(parameters.d, scale: 1_000_000.0),
      minLat: rounded(parameters.minLat, scale: 10_000.0),
      maxLat: rounded(parameters.maxLat, scale: 10_000.0)
    )
  }

  public static func knobs(from parameters: SigmoidParameters) -> PlainKnobs {
    let low = max(parameters.a + parameters.d, parameters.minLat)
    let high = min(parameters.d, parameters.maxLat)
    let midpointAcceleration = 0.5 * (low + high)
    let speedMPS = sqrt(midpointAcceleration / max(parameters.c, 1.0e-9))
    let transition = (speedMPS * metersPerSecondToMPH).clamped(to: VTSCClip.transitionSpeedMPH)
    let sharpness = (((log10(max(abs(parameters.b), 1.0)) - 2.0) / 3.0) * 10.0)
      .clamped(to: VTSCClip.sharpness)
    return PlainKnobs(
      tightCurveAcceleration: low,
      straightRoadAcceleration: high,
      transitionSpeedMPH: transition,
      sharpness: sharpness
    )
  }

  public static func centerKappa(for band: EQBand, parameters: SigmoidParameters) -> Double {
    var acceleration = 0.5 * (parameters.minLat + parameters.maxLat)
    for _ in 0 ..< 8 {
      acceleration = evaluate(
        parameters,
        curvature: kappa(speedMPH: band.centerSpeedMPH, lateralAcceleration: acceleration)
      )
    }
    return kappa(speedMPH: band.centerSpeedMPH, lateralAcceleration: acceleration)
  }

  public static func prepareBands(_ bands: [EQBand], parameters: SigmoidParameters) -> [PreparedBand] {
    bands.filter(\.enabled).map { band in
      PreparedBand(
        logCenterKappa: log10(max(centerKappa(for: band, parameters: parameters), 1.0e-12)),
        gainDB: band.gainDB,
        sigma: 0.5 / max(band.q, 0.1)
      )
    }
  }

  public static func applyBands(
    curvature: Double,
    baseLateralAcceleration: Double,
    preparedBands: [PreparedBand]
  ) -> Double {
    guard !preparedBands.isEmpty else { return baseLateralAcceleration }
    let logKappa = log10(max(curvature, 1.0e-12))
    let totalDB = preparedBands.reduce(0.0) { partial, band in
      let z = (logKappa - band.logCenterKappa) / band.sigma
      return partial + band.gainDB * exp(-0.5 * z * z)
    }
    return baseLateralAcceleration * pow(10.0, totalDB / 20.0)
  }

  public static func sampleCurve(
    parameters: SigmoidParameters,
    bands: [EQBand],
    count: Int = 512,
    kappaMin: Double = 1.0e-5,
    kappaMax: Double = 1.0,
    enforceMonotonicSpeed: Bool = false
  ) -> [CurveSample] {
    let sampleCount = max(count, 2)
    let logMin = log10(max(kappaMin, 1.0e-12))
    let logMax = log10(max(kappaMax, kappaMin * 10.0))
    let qPoints = qCurvePoints(parameters: parameters, bands: bands)
    var samples = (0 ..< sampleCount).map { index -> CurveSample in
      let t = Double(index) / Double(sampleCount - 1)
      let logKappa = logMax + (logMin - logMax) * t
      let kappa = pow(10.0, logKappa)
      let base = evaluate(parameters, curvature: kappa)
      // Q_CURVE_POINTS is a post-sigmoid speed multiplier on device. Display
      // that exported curve exactly instead of reclamping its equivalent
      // acceleration back to the raw sigmoid rails and hiding positive bands.
      let q = qSpeedMultiplierSorted(curvature: kappa, points: qPoints)
      let acceleration = base * q * q
      return CurveSample(
        kappa: kappa,
        speedMPH: sqrt(base / kappa) * metersPerSecondToMPH * q,
        lateralAcceleration: acceleration
      )
    }
    if enforceMonotonicSpeed {
      var runningMaximum = -Double.infinity
      for index in samples.indices {
        if samples[index].speedMPH < runningMaximum {
          samples[index].speedMPH = runningMaximum
        } else {
          runningMaximum = samples[index].speedMPH
        }
      }
    }
    return samples
  }

  public static func qCurvePoints(
    parameters: SigmoidParameters,
    bands: [EQBand],
    count: Int = 256
  ) -> [(kappa: Double, speedMultiplier: Double)] {
    let sampleCount = max(count, 2)
    let prepared = prepareBands(bands, parameters: parameters)
    return (0 ..< sampleCount).map { index in
      let t = Double(index) / Double(sampleCount - 1)
      let kappa = sourceRoundedKappa(pow(10.0, -5.0 + 5.0 * t))
      let base = evaluate(parameters, curvature: kappa)
      guard base >= 1.0e-6 else { return (kappa, 1.0) }
      let composed = applyBands(
        curvature: kappa,
        baseLateralAcceleration: base,
        preparedBands: prepared
      )
      let multiplier = sqrt(composed / base).clamped(to: 0.5 ... 1.5)
      return (kappa, sourceRoundedMultiplier(multiplier))
    }
  }

  /// Mirror `_q_curve_multiplier`: clamp to the exported curvature domain and
  /// linearly interpolate q in log10(curvature).
  public static func qSpeedMultiplier(
    curvature: Double,
    points: [(kappa: Double, speedMultiplier: Double)]
  ) -> Double {
    guard points.count >= 2, curvature.isFinite, curvature > 0 else { return 1.0 }
    let sorted = points.sorted { $0.kappa < $1.kappa }
    return qSpeedMultiplierSorted(curvature: curvature, points: sorted)
  }

  static func qSpeedMultiplierSorted(
    curvature: Double,
    points: [(kappa: Double, speedMultiplier: Double)]
  ) -> Double {
    guard points.count >= 2, curvature.isFinite, curvature > 0 else { return 1.0 }
    let kappa = curvature.clamped(to: points[0].kappa ... points[points.count - 1].kappa)
    let logKappa = log10(max(kappa, 1.0e-12))
    for index in 0 ..< points.count - 1 where kappa <= points[index + 1].kappa {
      let lower = points[index]
      let upper = points[index + 1]
      let logLower = log10(max(lower.kappa, 1.0e-12))
      let logUpper = log10(max(upper.kappa, 1.0e-12))
      guard logUpper > logLower else {
        return lower.speedMultiplier.clamped(to: 0.5 ... 1.5)
      }
      let t = (logKappa - logLower) / (logUpper - logLower)
      return (lower.speedMultiplier + (upper.speedMultiplier - lower.speedMultiplier) * t)
        .clamped(to: 0.5 ... 1.5)
    }
    return points[points.count - 1].speedMultiplier.clamped(to: 0.5 ... 1.5)
  }

  /// Match SourcePatcher's `%.6e` / `%.4f` representation before the values
  /// are used anywhere else, so proposal scoring, Curve Lab, and Python source
  /// all interpolate the same 256-point curve.
  private static func sourceRoundedKappa(_ value: Double) -> Double {
    guard value.isFinite, value != 0 else { return value }
    let exponent = floor(log10(abs(value)))
    let scale = pow(10.0, 6.0 - exponent)
    return (value * scale).rounded() / scale
  }

  private static func sourceRoundedMultiplier(_ value: Double) -> Double {
    (value * 10_000.0).rounded() / 10_000.0
  }

  public static func wingHalfWidth(sharpness: Double) -> Double {
    30.0 * exp(-0.4 * sharpness) + 2.0
  }

  public static func sharpness(wingHalfWidth: Double) -> Double {
    (-log(max((wingHalfWidth - 2.0) / 30.0, 1.0e-9)) / 0.4).clamped(to: VTSCClip.sharpness)
  }
}
