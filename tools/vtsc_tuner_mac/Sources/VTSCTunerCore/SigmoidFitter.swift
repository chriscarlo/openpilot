import Foundation

public struct CurveCalibrationSample: Codable, Equatable, Identifiable, Sendable {
  public var id: UUID
  public var label: String
  public var curvature: Double
  public var desiredSpeedMPH: Double
  public var weight: Double

  public init(
    id: UUID = UUID(),
    label: String,
    curvature: Double,
    desiredSpeedMPH: Double,
    weight: Double = 1.0
  ) {
    self.id = id
    self.label = label
    self.curvature = curvature
    self.desiredSpeedMPH = desiredSpeedMPH
    self.weight = weight
  }
}

public enum SpeedPredictionMode: String, Codable, Equatable, Sendable {
  case bakedTile
  case effectiveStrategic
}

public struct VTSCSpeedModifiers: Codable, Equatable, Sendable {
  public var lowSpeedBiasMPH: Double
  public var lowSpeedBiasEndMPH: Double
  public var speedIncreaseFactor: Double

  public init(
    lowSpeedBiasMPH: Double = 0.0,
    lowSpeedBiasEndMPH: Double = 50.0,
    speedIncreaseFactor: Double = 1.0
  ) {
    self.lowSpeedBiasMPH = lowSpeedBiasMPH
    self.lowSpeedBiasEndMPH = lowSpeedBiasEndMPH
    self.speedIncreaseFactor = speedIncreaseFactor
  }

  /// Offline defaults from the checked-in Params declarations. This is not a
  /// live snapshot of the connected device's learned or user-selected values.
  public static let sourceDefaults = Self()
}

public enum SigmoidFitWarning: Equatable, Sendable {
  case underconstrained(distinctCurvatureClusters: Int)
  case conflictingCurvatures(labels: [String], speedSpreadMPH: Double)
  case impossibleTarget(label: String, attainableRangeMPH: ClosedRange<Double>)
  case effectiveAccelerationAboveBaseRail(
    labels: [String], peakMPS2: Double, baseRailMPS2: Double)
}

public struct SigmoidSampleFitDiagnostic: Equatable, Sendable {
  public var id: UUID
  public var label: String
  public var curvature: Double
  public var desiredSpeedMPH: Double
  public var fittedTargetSpeedMPH: Double
  public var weight: Double
  public var attainableSpeedRangeMPH: ClosedRange<Double>
  public var beforeSpeedMPH: Double
  public var afterSpeedMPH: Double
  public var beforeBakedTileSpeedMPH: Double
  public var afterBakedTileSpeedMPH: Double
  public var beforeEffectiveStrategicSpeedMPH: Double
  public var afterEffectiveStrategicSpeedMPH: Double
  public var desiredLateralAccelerationMPS2: Double
  public var afterLateralAccelerationMPS2: Double
  public var beforeErrorMPH: Double
  public var afterErrorMPH: Double
  public var isImpossibleTarget: Bool
  public var curvatureCluster: Int
}

public struct SigmoidFitResult: Equatable, Sendable {
  public var inputKnobs: PlainKnobs
  public var inputBands: [EQBand]
  public var anchorKnobs: PlainKnobs
  public var knobs: PlainKnobs
  public var parameters: SigmoidParameters
  public var bands: [EQBand]
  public var predictionMode: SpeedPredictionMode
  public var beforeRMSEMPH: Double
  public var afterRMSEMPH: Double
  public var afterAttainableRMSEMPH: Double
  public var maximumAbsoluteErrorMPH: Double
  public var maximumProposedLateralAccelerationMPS2: Double
  public var maximumRuntimeSpeedReversalMPH: Double
  public var maximumOffBankBackboneDeltaMPH: Double
  public var maximumOutsideInfluenceRuntimeDeltaMPH: Double
  public var diagnostics: [SigmoidSampleFitDiagnostic]
  public var warnings: [SigmoidFitWarning]
  public var evaluationCount: Int
}

public enum SigmoidFitError: Error, Equatable, Sendable, LocalizedError {
  case noSamples
  case invalidSample(label: String, reason: String)
  case invalidModifiers(reason: String)
  case unsafeRuntimeProposal(reason: String)

  public var errorDescription: String? {
    switch self {
    case .noSamples:
      "Select at least one real curve before fitting."
    case .invalidSample(let label, let reason):
      "Cannot fit \(label): \(reason)"
    case .invalidModifiers(let reason):
      "Cannot predict effective VTSC speed: \(reason)"
    case .unsafeRuntimeProposal(let reason):
      "Cannot create a guarded VTSC proposal: \(reason)"
    }
  }
}

/// A deterministic, dependency-free fitter for the complete VTSC curve: four
/// plain base knobs plus a canonical bounded residual Q-band set.
///
/// The implementation is pure and has no actor isolation or shared mutable
/// state, so callers can safely run it from `Task.detached` or another
/// background executor without touching the main actor.
public enum SigmoidFitter {
  private static let maximumBakedSpeedMPS = MapBakeMath.defaultMaximumSpeedMPS
  public static let maximumRepresentableSpeedMPH =
    maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH
  private static let similarCurvatureRatio = 1.05
  private static let exportedCurvatureRange = 1.0e-5 ... 1.0
  private static let minimumCalibrationCurvature =
    VTSCClip.maximumLateralAcceleration.upperBound
    / pow(maximumRepresentableSpeedMPH * VTSCMath.mphToMetersPerSecond, 2.0)
  private static let calibrationCurvatureRange = minimumCalibrationCurvature ... 1.0
  private static let conflictSpeedSpreadMPH = 2.0
  private static let impossibleToleranceMPH = 0.25
  private static let priorPenalty = 0.0025
  private static let backbonePenalty = 0.002
  public static let maximumOffBankBackboneDeltaMPH = 2.0
  private static let offBankNumericalGuardMPH = 0.20
  private static let qSpeedMultiplierRange = 0.5 ... 1.5
  private static let residualBandGainDBRange = -12.0 ... 12.0
  private static let residualBandQ = 4.0
  /// Four Gaussian standard deviations in log10(curvature), derived from the
  /// generated band's sigma=0.5/Q. Beyond this collar residual authority is
  /// negligible and the complete curve must return to the anchored backbone.
  public static let residualInfluenceCurvatureRatio = pow(10.0, 2.0 / residualBandQ)
  private static let residualBandPruneDB = 0.025
  private static let residualBandRidge = 1.0e-4
  public static let effectiveAccelerationWarningMPS2 = 5.5
  public static let maximumRuntimeSpeedReversalMPH = 0.10

  /// Predict the value represented by a map calibration target. Effective
  /// strategic mode mirrors the post-bake bias, factor, and Q-band path used
  /// on device; baked mode returns exactly the speed stored in generated tiles.
  public static func predictedSpeedMPH(
    parameters: SigmoidParameters,
    curvature: Double,
    bands: [EQBand] = [],
    mode: SpeedPredictionMode = .effectiveStrategic,
    modifiers: VTSCSpeedModifiers = .sourceDefaults
  ) -> Double {
    let base = predictedSpeedMPH(
      parameters: parameters,
      curvature: curvature,
      preparedBands: [],
      mode: mode,
      modifiers: modifiers
    )
    guard mode == .effectiveStrategic, !bands.isEmpty else { return base }
    let qPoints = VTSCMath.qCurvePoints(parameters: parameters, bands: bands)
    let q = VTSCMath.qSpeedMultiplier(curvature: curvature, points: qPoints)
    return (base * q).clamped(
      to: 0.0 ... maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH)
  }

  public static func fit(
    samples: [CurveCalibrationSample],
    currentKnobs: PlainKnobs,
    anchorKnobs: PlainKnobs,
    bands: [EQBand] = [],
    predictionMode: SpeedPredictionMode = .effectiveStrategic,
    modifiers: VTSCSpeedModifiers = .sourceDefaults
  ) throws -> SigmoidFitResult {
    try Task.checkCancellation()
    let samples = try validatedSamples(samples)
    try validate(modifiers)

    let clusters = curvatureClusters(samples)
    var fitWeights = Array(repeating: 0.0, count: samples.count)
    for cluster in clusters {
      let rawWeight = cluster.indices.reduce(0.0) { $0 + samples[$1].weight }
      let clusterWeight = cluster.indices.map { samples[$0].weight }.max() ?? 1.0
      for index in cluster.indices {
        fitWeights[index] = clusterWeight * samples[index].weight / rawWeight
      }
    }
    let canonicalCurrent = VTSCMath.knobs(from: VTSCMath.parameters(from: currentKnobs))
    let canonicalAnchor = VTSCMath.knobs(from: VTSCMath.parameters(from: anchorKnobs))
    let anchorVector = encode(canonicalAnchor)
    let anchorParameters = runtimeParameters(from: canonicalAnchor)
    let backboneCurvatures = logSpacedCurvatures(count: 256)
    let anchorBackboneSpeeds = backboneCurvatures.map {
      predictedSpeedMPH(
        parameters: anchorParameters,
        curvature: $0,
        preparedBands: [],
        mode: predictionMode,
        modifiers: modifiers
      )
    }
    let denseBackboneCurvatures = logSpacedCurvatures(count: 4_096)
    let denseAnchorBackboneSpeeds = denseBackboneCurvatures.map {
      predictedSpeedMPH(
        parameters: anchorParameters,
        curvature: $0,
        preparedBands: [],
        mode: predictionMode,
        modifiers: modifiers
      )
    }
    let influenceGuardRange = (samples.first!.curvature / residualInfluenceCurvatureRatio)
      ... (samples.last!.curvature * residualInfluenceCurvatureRatio)
    var evaluationCount = 0

    func maximumOffBankDelta(
      parameters: SigmoidParameters,
      curvatures: [Double],
      anchorSpeeds: [Double]
    ) -> Double {
      var maximum = zip(curvatures, anchorSpeeds)
        .filter { !influenceGuardRange.contains($0.0) }
        .map {
          abs(
            predictedSpeedMPH(
              parameters: parameters,
              curvature: $0.0,
              preparedBands: [],
              mode: predictionMode,
              modifiers: modifiers
            ) - $0.1
          )
        }.max() ?? 0.0

      // A steep sigmoid can put its entire transition between two fixed
      // log-grid samples. Probe both the anchor and candidate transitions (and
      // the bank boundaries) so the off-bank limit cannot silently miss a
      // narrow speed shelf just outside the observed curvature range.
      var transitionProbes = [
        influenceGuardRange.lowerBound * (1.0 - 1.0e-7),
        influenceGuardRange.upperBound * (1.0 + 1.0e-7),
      ]
      for transition in [anchorParameters, parameters] {
        let width = 1.0 / max(abs(transition.b), 1.0)
        for offset in [-12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0] {
          transitionProbes.append(transition.c + offset * width)
        }
      }
      for curvature in transitionProbes
      where exportedCurvatureRange.contains(curvature)
        && !influenceGuardRange.contains(curvature)
      {
        let anchorSpeed = predictedSpeedMPH(
          parameters: anchorParameters,
          curvature: curvature,
          preparedBands: [],
          mode: predictionMode,
          modifiers: modifiers
        )
        let proposedSpeed = predictedSpeedMPH(
          parameters: parameters,
          curvature: curvature,
          preparedBands: [],
          mode: predictionMode,
          modifiers: modifiers
        )
        maximum = max(maximum, abs(proposedSpeed - anchorSpeed))
      }
      return maximum
    }

    func predictions(at vector: FitVector) -> [Double] {
      evaluationCount += 1
      let parameters = runtimeParameters(from: decode(vector))
      return samples.map {
        predictedSpeedMPH(
          parameters: parameters,
          curvature: $0.curvature,
          preparedBands: [],
          mode: predictionMode,
          modifiers: modifiers
        )
      }
    }

    func prediction(at vector: FitVector, sampleIndex: Int) -> Double {
      evaluationCount += 1
      let parameters = runtimeParameters(from: decode(vector))
      return predictedSpeedMPH(
        parameters: parameters,
        curvature: samples[sampleIndex].curvature,
        preparedBands: [],
        mode: predictionMode,
        modifiers: modifiers
      )
    }

    let gridVectors = coarseGrid()
    var gridEvaluations: [EvaluatedVector] = []
    gridEvaluations.reserveCapacity(gridVectors.count)

    var minimumSpeeds = Array(repeating: Double.infinity, count: samples.count)
    var maximumSpeeds = Array(repeating: -Double.infinity, count: samples.count)
    var minimumVectors = Array(repeating: anchorVector, count: samples.count)
    var maximumVectors = Array(repeating: anchorVector, count: samples.count)

    for vector in gridVectors {
      try Task.checkCancellation()
      let values = predictions(at: vector)
      gridEvaluations.append(EvaluatedVector(vector: vector, predictions: values))
      for index in samples.indices {
        if values[index] < minimumSpeeds[index] {
          minimumSpeeds[index] = values[index]
          minimumVectors[index] = vector
        }
        if values[index] > maximumSpeeds[index] {
          maximumSpeeds[index] = values[index]
          maximumVectors[index] = vector
        }
      }
    }

    // Tighten the numerical feasibility envelopes without multiplying the
    // full fit cost by the number of samples.
    for index in samples.indices {
      try Task.checkCancellation()
      let minimum = try refineExtreme(
        start: minimumVectors[index],
        minimize: true,
        prediction: { prediction(at: $0, sampleIndex: index) }
      )
      minimumVectors[index] = minimum.vector
      minimumSpeeds[index] = min(minimumSpeeds[index], minimum.value)

      let maximum = try refineExtreme(
        start: maximumVectors[index],
        minimize: false,
        prediction: { prediction(at: $0, sampleIndex: index) }
      )
      maximumVectors[index] = maximum.vector
      maximumSpeeds[index] = max(maximumSpeeds[index], maximum.value)
    }

    // Map calibration fits the complete effective curve, not only the four
    // plain-knob projection. Runtime Q_CURVE_POINTS may multiply speed by a
    // bounded 0.5...1.5 after the base sigmoid. Include that real authority in
    // feasibility so a request is clipped only when the complete exported
    // curve cannot represent it. Raw tile-bake mode intentionally stays base
    // sigmoid-only because Q is applied later on device.
    if predictionMode == .effectiveStrategic {
      let maximumMPH = maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH
      for index in samples.indices {
        minimumSpeeds[index] *= qSpeedMultiplierRange.lowerBound
        maximumSpeeds[index] = min(
          maximumMPH,
          maximumSpeeds[index] * qSpeedMultiplierRange.upperBound
        )
      }
    }

    let pointwiseTargets = samples.indices.map { index in
      samples[index].desiredSpeedMPH.clamped(to: minimumSpeeds[index]...maximumSpeeds[index])
    }
    // A single curvature-only runtime curve cannot command a higher speed for
    // a tighter curve. Project contradictory/inverted requests onto the
    // nearest weighted non-increasing sequence before fitting; diagnostics
    // still retain the user's original targets and residuals.
    let provisionalFittedTargets = monotonicNonIncreasingProjection(
      pointwiseTargets,
      weights: fitWeights
    )

    func score(vector: FitVector, values: [Double]) -> CandidateScore {
      var weightedSquaredError = 0.0
      var clusterWeightTotal = 0.0
      var maximumAbsoluteError = 0.0

      for cluster in clusters {
        let rawWeightTotal = cluster.indices.reduce(0.0) { $0 + samples[$1].weight }
        let clusterWeight = cluster.indices.map { samples[$0].weight }.max() ?? 1.0
        var clusterMSE = 0.0
        for index in cluster.indices {
          let residual = values[index] - provisionalFittedTargets[index]
          clusterMSE += (samples[index].weight / rawWeightTotal) * residual * residual
          maximumAbsoluteError = max(maximumAbsoluteError, abs(residual))
        }
        weightedSquaredError += clusterWeight * clusterMSE
        clusterWeightTotal += clusterWeight
      }

      let dataMSE = weightedSquaredError / max(clusterWeightTotal, 1.0e-12)
      var residualAuthoritySquaredError = 0.0
      var residualAuthorityWeight = 0.0
      for index in samples.indices {
        let minimum = values[index] * qSpeedMultiplierRange.lowerBound
        let maximum = min(
          maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH,
          values[index] * qSpeedMultiplierRange.upperBound
        )
        let target = provisionalFittedTargets[index]
        let violation = target < minimum ? minimum - target : max(0.0, target - maximum)
        residualAuthoritySquaredError += fitWeights[index] * violation * violation
        residualAuthorityWeight += fitWeights[index]
      }
      let residualAuthorityMSE = residualAuthoritySquaredError
        / max(residualAuthorityWeight, 1.0e-12)
      let parameters = runtimeParameters(from: decode(vector))
      let backboneDeltas = zip(backboneCurvatures, anchorBackboneSpeeds).map {
        abs(
          predictedSpeedMPH(
            parameters: parameters,
            curvature: $0.0,
            preparedBands: [],
            mode: predictionMode,
            modifiers: modifiers
          ) - $0.1
        )
      }
      let maximumOffBankDelta = maximumOffBankDelta(
        parameters: parameters,
        curvatures: backboneCurvatures,
        anchorSpeeds: anchorBackboneSpeeds
      )
      let backboneMSE = backboneDeltas.reduce(0.0) { $0 + $1 * $1 }
        / Double(max(backboneDeltas.count, 1))
      let priorDistance = vector.squaredDistance(to: anchorVector)
      return CandidateScore(
        vector: vector,
        predictions: values,
        total: maximumOffBankDelta
          <= maximumOffBankBackboneDeltaMPH - offBankNumericalGuardMPH + 1.0e-9
          ? dataMSE + 1_000.0 * residualAuthorityMSE
            + priorPenalty * priorDistance + backbonePenalty * backboneMSE
          : .infinity,
        dataMSE: dataMSE,
        maximumAbsoluteError: maximumAbsoluteError,
        priorDistance: priorDistance
      )
    }

    var candidates = gridEvaluations.map { score(vector: $0.vector, values: $0.predictions) }
    let extraVectors = [
      anchorVector,
      analyticSeed(samples: samples, fittedTargets: provisionalFittedTargets),
      .center,
    ]
    for vector in extraVectors {
      candidates.append(score(vector: vector, values: predictions(at: vector)))
    }

    let seeds = bestSeparated(candidates, count: 8, minimumDistance: 0.12)
    var refined: [CandidateScore] = []
    refined.reserveCapacity(seeds.count)
    for seed in seeds {
      try Task.checkCancellation()
      refined.append(
        patternRefine(start: seed, predictions: predictions, score: score)
      )
    }
    // The fast score uses a 256-point guard plus transition probes. Before
    // exporting a base, re-rank against the dense guard so a narrow sigmoid
    // shelf cannot slip between coarse probes. The anchored checkout is always
    // in the pool and is a deterministic safe fallback.
    let rankedCandidates = (refined + candidates).sorted(by: isBetter)
    let best = rankedCandidates.first { candidate in
      let parameters = runtimeParameters(from: decode(candidate.vector))
      return maximumOffBankDelta(
        parameters: parameters,
        curvatures: denseBackboneCurvatures,
        anchorSpeeds: denseAnchorBackboneSpeeds
      ) <= maximumOffBankBackboneDeltaMPH + 1.0e-9
    } ?? candidates.first(where: { $0.vector == anchorVector })!

    let finalKnobs = decode(best.vector)
    let finalParameters = runtimeParameters(from: finalKnobs)
    let finalBasePredictions = samples.map {
      predictedSpeedMPH(
        parameters: finalParameters,
        curvature: $0.curvature,
        preparedBands: [],
        mode: predictionMode,
        modifiers: modifiers
      )
    }
    let finalAttainableRanges: [ClosedRange<Double>] = samples.indices.map { index in
      guard predictionMode == .effectiveStrategic else {
        return minimumSpeeds[index] ... maximumSpeeds[index]
      }
      return (finalBasePredictions[index] * qSpeedMultiplierRange.lowerBound)
        ... min(
          maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH,
          finalBasePredictions[index] * qSpeedMultiplierRange.upperBound
        )
    }
    // The base search strongly prefers a backbone whose real 0.5...1.5 Q
    // authority contains every projected target. If coupled guard constraints
    // make that impossible, clamp the diagnostic/solve target to the selected
    // base's actual authority instead of claiming an unattainable value.
    let fittedTargets = provisionalFittedTargets.indices.map { index in
      provisionalFittedTargets[index].clamped(to: finalAttainableRanges[index])
    }
    let currentParameters = runtimeParameters(from: canonicalCurrent)
    let currentPreparedBands = VTSCMath.prepareBands(bands, parameters: currentParameters)
    let proposedBands: [EQBand]
    if predictionMode == .effectiveStrategic {
      proposedBands = try fitResidualBands(
        samples: samples,
        clusters: clusters,
        parameters: finalParameters,
        anchorParameters: anchorParameters,
        sampleCurvatureRange: samples.first!.curvature ... samples.last!.curvature,
        fittedTargets: fittedTargets,
        sampleWeights: fitWeights,
        modifiers: modifiers
      )
    } else {
      proposedBands = []
    }
    let finalPreparedBands = VTSCMath.prepareBands(proposedBands, parameters: finalParameters)
    let beforeSelected = samples.map {
      predictedSpeedMPH(
        parameters: currentParameters,
        curvature: $0.curvature,
        bands: bands,
        mode: predictionMode,
        modifiers: modifiers
      )
    }
    let afterSelected = samples.map {
      predictedSpeedMPH(
        parameters: finalParameters,
        curvature: $0.curvature,
        bands: proposedBands,
        mode: predictionMode,
        modifiers: modifiers
      )
    }
    let runtimeShape = runtimeCurveShape(
      parameters: finalParameters,
      bands: proposedBands,
      modifiers: modifiers,
      probeCount: 4_096
    )
    let maximumOffBankBackboneDelta = maximumOffBankDelta(
      parameters: finalParameters,
      curvatures: denseBackboneCurvatures,
      anchorSpeeds: denseAnchorBackboneSpeeds
    )
    let maximumOutsideInfluenceRuntimeDelta = maximumRuntimeDeltaOutsideInfluenceCollar(
      parameters: finalParameters,
      bands: proposedBands,
      anchorParameters: anchorParameters,
      sampleCurvatureRange: samples.first!.curvature ... samples.last!.curvature,
      modifiers: modifiers,
      probeCount: 4_096
    )
    guard maximumOffBankBackboneDelta
      <= maximumOffBankBackboneDeltaMPH + 1.0e-9,
      maximumOutsideInfluenceRuntimeDelta
        <= maximumOffBankBackboneDeltaMPH + 1.0e-9,
      runtimeShape.maximumSpeedReversalMPH <= maximumRuntimeSpeedReversalMPH + 1.0e-9
    else {
      throw SigmoidFitError.unsafeRuntimeProposal(
        reason: "the densely sampled exported curve exceeded its off-bank or monotonicity guard"
      )
    }

    var warnings: [SigmoidFitWarning] = []
    if clusters.count < 4 {
      warnings.append(.underconstrained(distinctCurvatureClusters: clusters.count))
    }
    for conflict in conflictingSampleGroups(samples) {
      warnings.append(
        .conflictingCurvatures(
          labels: conflict.indices.map { samples[$0].label },
          speedSpreadMPH: conflict.speedSpreadMPH
        )
      )
    }

    var diagnostics: [SigmoidSampleFitDiagnostic] = []
    diagnostics.reserveCapacity(samples.count)
    for index in samples.indices {
      let attainable = finalAttainableRanges[index]
      let impossible =
        samples[index].desiredSpeedMPH < attainable.lowerBound - impossibleToleranceMPH
        || samples[index].desiredSpeedMPH > attainable.upperBound + impossibleToleranceMPH
      if impossible {
        warnings.append(
          .impossibleTarget(label: samples[index].label, attainableRangeMPH: attainable)
        )
      }

      let beforeBaked = predictedSpeedMPH(
        parameters: currentParameters,
        curvature: samples[index].curvature,
        preparedBands: currentPreparedBands,
        mode: .bakedTile,
        modifiers: modifiers
      )
      let afterBaked = predictedSpeedMPH(
        parameters: finalParameters,
        curvature: samples[index].curvature,
        preparedBands: finalPreparedBands,
        mode: .bakedTile,
        modifiers: modifiers
      )
      let beforeEffective = predictedSpeedMPH(
        parameters: currentParameters,
        curvature: samples[index].curvature,
        bands: bands,
        mode: .effectiveStrategic,
        modifiers: modifiers
      )
      let afterEffective = predictedSpeedMPH(
        parameters: finalParameters,
        curvature: samples[index].curvature,
        bands: proposedBands,
        mode: .effectiveStrategic,
        modifiers: modifiers
      )

      diagnostics.append(
        SigmoidSampleFitDiagnostic(
          id: samples[index].id,
          label: samples[index].label,
          curvature: samples[index].curvature,
          desiredSpeedMPH: samples[index].desiredSpeedMPH,
          fittedTargetSpeedMPH: fittedTargets[index],
          weight: samples[index].weight,
          attainableSpeedRangeMPH: attainable,
          beforeSpeedMPH: beforeSelected[index],
          afterSpeedMPH: afterSelected[index],
          beforeBakedTileSpeedMPH: beforeBaked,
          afterBakedTileSpeedMPH: afterBaked,
          beforeEffectiveStrategicSpeedMPH: beforeEffective,
          afterEffectiveStrategicSpeedMPH: afterEffective,
          desiredLateralAccelerationMPS2: lateralAccelerationMPS2(
            curvature: samples[index].curvature,
            speedMPH: samples[index].desiredSpeedMPH
          ),
          afterLateralAccelerationMPS2: lateralAccelerationMPS2(
            curvature: samples[index].curvature,
            speedMPH: afterEffective
          ),
          beforeErrorMPH: beforeSelected[index] - samples[index].desiredSpeedMPH,
          afterErrorMPH: afterSelected[index] - samples[index].desiredSpeedMPH,
          isImpossibleTarget: impossible,
          curvatureCluster: clusters.firstIndex(where: { $0.indices.contains(index) }) ?? 0
        )
      )
    }

    let aboveRailLabels = diagnostics.compactMap { diagnostic -> String? in
      max(
        diagnostic.desiredLateralAccelerationMPS2,
        diagnostic.afterLateralAccelerationMPS2
      ) > effectiveAccelerationWarningMPS2 + 1.0e-6
        ? diagnostic.label
        : nil
    }
    if !aboveRailLabels.isEmpty
      || runtimeShape.maximumLateralAccelerationMPS2
        > effectiveAccelerationWarningMPS2 + 1.0e-6
    {
      warnings.append(
        .effectiveAccelerationAboveBaseRail(
          labels: aboveRailLabels,
          peakMPS2: max(
            runtimeShape.maximumLateralAccelerationMPS2,
            diagnostics.map(\.desiredLateralAccelerationMPS2).max() ?? 0.0
          ),
          baseRailMPS2: effectiveAccelerationWarningMPS2
        )
      )
    }

    return SigmoidFitResult(
      inputKnobs: canonicalCurrent,
      inputBands: bands,
      anchorKnobs: canonicalAnchor,
      knobs: finalKnobs,
      parameters: finalParameters,
      bands: proposedBands,
      predictionMode: predictionMode,
      beforeRMSEMPH: weightedRMSE(
        values: beforeSelected,
        targets: samples.map(\.desiredSpeedMPH),
        samples: samples,
        clusters: clusters
      ),
      afterRMSEMPH: weightedRMSE(
        values: afterSelected,
        targets: samples.map(\.desiredSpeedMPH),
        samples: samples,
        clusters: clusters
      ),
      afterAttainableRMSEMPH: weightedRMSE(
        values: afterSelected,
        targets: fittedTargets,
        samples: samples,
        clusters: clusters
      ),
      maximumAbsoluteErrorMPH: zip(afterSelected, samples).map { abs($0 - $1.desiredSpeedMPH) }
        .max() ?? 0.0,
      maximumProposedLateralAccelerationMPS2: runtimeShape.maximumLateralAccelerationMPS2,
      maximumRuntimeSpeedReversalMPH: runtimeShape.maximumSpeedReversalMPH,
      maximumOffBankBackboneDeltaMPH: maximumOffBankBackboneDelta,
      maximumOutsideInfluenceRuntimeDeltaMPH: maximumOutsideInfluenceRuntimeDelta,
      diagnostics: diagnostics,
      warnings: warnings,
      evaluationCount: evaluationCount
    )
  }
}

// MARK: - Prediction

extension SigmoidFitter {
  fileprivate static func predictedSpeedMPH(
    parameters: SigmoidParameters,
    curvature: Double,
    preparedBands: [PreparedBand],
    mode: SpeedPredictionMode,
    modifiers: VTSCSpeedModifiers
  ) -> Double {
    let baked = bakedTileSpeedMPH(parameters: parameters, curvature: curvature)
    guard mode == .effectiveStrategic else { return baked }

    var effective = baked
    if modifiers.lowSpeedBiasMPH != 0.0, effective < modifiers.lowSpeedBiasEndMPH {
      let taper = (1.0 - effective / max(modifiers.lowSpeedBiasEndMPH, 1.0e-3)).clamped(
        to: 0.0...1.0)
      effective += modifiers.lowSpeedBiasMPH * taper
    }

    let maximumMPH = maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH
    effective = (effective * modifiers.speedIncreaseFactor).clamped(to: 0.0...maximumMPH)

    let baseAcceleration = VTSCMath.evaluate(parameters, curvature: curvature)
    let composedAcceleration = VTSCMath.applyBands(
      curvature: curvature,
      baseLateralAcceleration: baseAcceleration,
      preparedBands: preparedBands
    )
    let qMultiplier =
      baseAcceleration > 1.0e-9
      ? sqrt(composedAcceleration / baseAcceleration).clamped(to: 0.5...1.5)
      : 1.0
    return (effective * qMultiplier).clamped(to: 0.0...maximumMPH)
  }

  fileprivate static func bakedTileSpeedMPH(parameters: SigmoidParameters, curvature: Double)
    -> Double
  {
    MapBakeMath.bakedSpeedMPS(curvature: curvature, parameters: parameters)
      * VTSCMath.metersPerSecondToMPH
  }

  fileprivate static func lateralAccelerationMPS2(curvature: Double, speedMPH: Double) -> Double {
    let speedMPS = speedMPH * VTSCMath.mphToMetersPerSecond
    return curvature * speedMPS * speedMPS
  }

  fileprivate static func validate(_ modifiers: VTSCSpeedModifiers) throws {
    guard modifiers.lowSpeedBiasMPH.isFinite,
      (-5.0 ... 5.0).contains(modifiers.lowSpeedBiasMPH),
      modifiers.lowSpeedBiasEndMPH.isFinite,
      (10.0 ... 80.0).contains(modifiers.lowSpeedBiasEndMPH),
      modifiers.speedIncreaseFactor.isFinite,
      (0.5 ... 1.5).contains(modifiers.speedIncreaseFactor)
    else {
      throw SigmoidFitError.invalidModifiers(
        reason:
          "runtime ranges are bias -5...5 mph, taper endpoint 10...80 mph, and speed factor 0.5...1.5"
      )
    }
  }

  struct RuntimeCurveShape {
    var maximumLateralAccelerationMPS2: Double
    var maximumSpeedReversalMPH: Double
    var squaredSpeedReversalViolationMPH2: Double
  }

  static func runtimeCurveShape(
    parameters: SigmoidParameters,
    bands: [EQBand],
    modifiers: VTSCSpeedModifiers,
    probeCount: Int = 512
  ) -> RuntimeCurveShape {
    let points = VTSCMath.qCurvePoints(parameters: parameters, bands: bands)
    return runtimeCurveShape(
      parameters: parameters,
      qPoints: points,
      modifiers: modifiers,
      probeCount: probeCount
    )
  }

  static func runtimeCurveShape(
    parameters: SigmoidParameters,
    qPoints rawPoints: [(kappa: Double, speedMultiplier: Double)],
    modifiers: VTSCSpeedModifiers,
    probeCount: Int = 512
  ) -> RuntimeCurveShape {
    let points = rawPoints.sorted { $0.kappa < $1.kappa }
    var probes = logSpacedCurvatures(count: probeCount)
    probes.append(contentsOf: points.map(\.kappa))
    let transitionWidth = 1.0 / max(abs(parameters.b), 1.0)
    for offset in [-12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0] {
      let curvature = parameters.c + offset * transitionWidth
      if exportedCurvatureRange.contains(curvature) { probes.append(curvature) }
    }
    probes.sort()

    var qIndex = 0
    func multiplier(at curvature: Double) -> Double {
      guard points.count >= 2 else { return 1.0 }
      let clamped = curvature.clamped(to: points[0].kappa ... points[points.count - 1].kappa)
      while qIndex + 1 < points.count - 1, clamped > points[qIndex + 1].kappa {
        qIndex += 1
      }
      let lower = points[qIndex]
      let upper = points[min(qIndex + 1, points.count - 1)]
      let logLower = log10(max(lower.kappa, 1.0e-12))
      let logUpper = log10(max(upper.kappa, 1.0e-12))
      guard logUpper > logLower else { return lower.speedMultiplier }
      let t = (log10(max(clamped, 1.0e-12)) - logLower) / (logUpper - logLower)
      return (lower.speedMultiplier + (upper.speedMultiplier - lower.speedMultiplier) * t)
        .clamped(to: qSpeedMultiplierRange)
    }

    var maximumAcceleration = 0.0
    var maximumReversal = 0.0
    var squaredReversal = 0.0
    var runningMinimumSpeed = Double.infinity
    for curvature in probes {
      let baseSpeed = predictedSpeedMPH(
        parameters: parameters,
        curvature: curvature,
        preparedBands: [],
        mode: .effectiveStrategic,
        modifiers: modifiers
      )
      let speed = min(
        maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH,
        baseSpeed * multiplier(at: curvature)
      )
      maximumAcceleration = max(
        maximumAcceleration,
        lateralAccelerationMPS2(curvature: curvature, speedMPH: speed)
      )
      runningMinimumSpeed = min(runningMinimumSpeed, speed)
      let reversal = max(0.0, speed - runningMinimumSpeed)
      maximumReversal = max(maximumReversal, reversal)
      let violation = max(0.0, reversal - maximumRuntimeSpeedReversalMPH)
      squaredReversal += violation * violation
    }
    return RuntimeCurveShape(
      maximumLateralAccelerationMPS2: maximumAcceleration,
      maximumSpeedReversalMPH: maximumReversal,
      squaredSpeedReversalViolationMPH2: squaredReversal
    )
  }

  fileprivate static func maximumRuntimeDeltaOutsideInfluenceCollar(
    parameters: SigmoidParameters,
    bands: [EQBand],
    anchorParameters: SigmoidParameters,
    sampleCurvatureRange: ClosedRange<Double>,
    modifiers: VTSCSpeedModifiers,
    probeCount: Int = 512
  ) -> Double {
    let qPoints = VTSCMath.qCurvePoints(parameters: parameters, bands: bands)
    return maximumRuntimeDeltaOutsideInfluenceCollar(
      parameters: parameters,
      qPoints: qPoints,
      anchorParameters: anchorParameters,
      sampleCurvatureRange: sampleCurvatureRange,
      modifiers: modifiers,
      probeCount: probeCount
    )
  }

  fileprivate static func maximumRuntimeDeltaOutsideInfluenceCollar(
    parameters: SigmoidParameters,
    qPoints: [(kappa: Double, speedMultiplier: Double)],
    anchorParameters: SigmoidParameters,
    sampleCurvatureRange: ClosedRange<Double>,
    modifiers: VTSCSpeedModifiers,
    probeCount: Int = 512
  ) -> Double {
    let lowerBoundary = max(
      exportedCurvatureRange.lowerBound,
      sampleCurvatureRange.lowerBound / residualInfluenceCurvatureRatio
    )
    let upperBoundary = min(
      exportedCurvatureRange.upperBound,
      sampleCurvatureRange.upperBound * residualInfluenceCurvatureRatio
    )
    let count = max(probeCount, 2)
    let logLower = log10(exportedCurvatureRange.lowerBound)
    let logUpper = log10(exportedCurvatureRange.upperBound)
    var probes = (0..<count).map { index in
      pow(10.0, logLower + (logUpper - logLower) * Double(index) / Double(count - 1))
    }
    probes.append(contentsOf: [lowerBoundary, upperBoundary])
    probes.append(contentsOf: qPoints.map(\.kappa))
    // Mirror the backbone guard's transition probes. A steep candidate or
    // anchor sigmoid can otherwise put a narrow shelf between adjacent log
    // samples and make the complete-curve delta look artificially safe.
    for transition in [anchorParameters, parameters] {
      let width = 1.0 / max(abs(transition.b), 1.0)
      for offset in [-12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0] {
        let curvature = transition.c + offset * width
        if exportedCurvatureRange.contains(curvature) {
          probes.append(curvature)
        }
      }
    }
    probes.sort()

    var qIndex = 0
    func multiplier(at curvature: Double) -> Double {
      guard qPoints.count >= 2 else { return 1.0 }
      let clamped = curvature.clamped(to: qPoints[0].kappa ... qPoints[qPoints.count - 1].kappa)
      while qIndex + 1 < qPoints.count - 1, clamped > qPoints[qIndex + 1].kappa {
        qIndex += 1
      }
      let lower = qPoints[qIndex]
      let upper = qPoints[min(qIndex + 1, qPoints.count - 1)]
      let logLower = log10(max(lower.kappa, 1.0e-12))
      let logUpper = log10(max(upper.kappa, 1.0e-12))
      guard logUpper > logLower else { return lower.speedMultiplier }
      let t = (log10(max(clamped, 1.0e-12)) - logLower) / (logUpper - logLower)
      return (lower.speedMultiplier + (upper.speedMultiplier - lower.speedMultiplier) * t)
        .clamped(to: qSpeedMultiplierRange)
    }

    var maximum = 0.0
    for curvature in probes where curvature <= lowerBoundary || curvature >= upperBoundary {
      let anchorSpeed = predictedSpeedMPH(
        parameters: anchorParameters,
        curvature: curvature,
        preparedBands: [],
        mode: .effectiveStrategic,
        modifiers: modifiers
      )
      let proposedBase = predictedSpeedMPH(
        parameters: parameters,
        curvature: curvature,
        preparedBands: [],
        mode: .effectiveStrategic,
        modifiers: modifiers
      )
      let q = multiplier(at: curvature)
      let proposedSpeed = min(
        maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH,
        proposedBase * q
      )
      maximum = max(maximum, abs(proposedSpeed - anchorSpeed))
    }
    return maximum
  }
}

// MARK: - Input preparation and diagnostics

extension SigmoidFitter {
  fileprivate struct CurvatureCluster {
    var indices: [Int]
  }

  fileprivate static func validatedSamples(_ rawSamples: [CurveCalibrationSample]) throws
    -> [CurveCalibrationSample]
  {
    guard !rawSamples.isEmpty else { throw SigmoidFitError.noSamples }
    var unique: [UUID: CurveCalibrationSample] = [:]
    for sample in rawSamples {
      guard sample.curvature.isFinite, calibrationCurvatureRange.contains(sample.curvature) else {
        throw SigmoidFitError.invalidSample(
          label: sample.label,
          reason:
            "curvature must be \(minimumCalibrationCurvature.formatted(.number.precision(.fractionLength(6))))...1.0 1/m so generated residual centers remain below the runtime \(maximumRepresentableSpeedMPH.formatted(.number.precision(.fractionLength(1)))) mph cap; gentler curves use mapd's fixed speed cap"
        )
      }
      guard sample.desiredSpeedMPH.isFinite, sample.desiredSpeedMPH > 0.0 else {
        throw SigmoidFitError.invalidSample(
          label: sample.label, reason: "desired speed must be finite and positive")
      }
      guard sample.weight.isFinite, sample.weight > 0.0 else {
        throw SigmoidFitError.invalidSample(
          label: sample.label, reason: "weight must be finite and positive")
      }
      unique[sample.id] = sample
    }
    return unique.values.sorted {
      if $0.curvature != $1.curvature { return $0.curvature < $1.curvature }
      return $0.id.uuidString < $1.id.uuidString
    }
  }

  fileprivate static func curvatureClusters(_ samples: [CurveCalibrationSample])
    -> [CurvatureCluster]
  {
    var clusters: [CurvatureCluster] = []
    var startIndex = 0
    while startIndex < samples.count {
      var endIndex = startIndex + 1
      while endIndex < samples.count,
        samples[endIndex].curvature / samples[startIndex].curvature <= similarCurvatureRatio
      {
        endIndex += 1
      }
      clusters.append(CurvatureCluster(indices: Array(startIndex..<endIndex)))
      startIndex = endIndex
    }
    return clusters
  }

  fileprivate struct ConflictGroup {
    var indices: [Int]
    var speedSpreadMPH: Double
  }

  /// Conflict detection is intentionally a sliding fixed-width window rather
  /// than the fixed-anchor clustering used for weights/bands. That catches a
  /// close pair straddling a cluster boundary without transitively collapsing
  /// a long, well-distributed bank into one group.
  fileprivate static func conflictingSampleGroups(_ samples: [CurveCalibrationSample])
    -> [ConflictGroup]
  {
    var candidates: [ConflictGroup] = []
    for start in samples.indices {
      var end = start + 1
      while end < samples.count,
        samples[end].curvature / samples[start].curvature <= similarCurvatureRatio
      {
        end += 1
      }
      let indices = Array(start..<end)
      guard indices.count > 1 else { continue }
      let desired = indices.map { samples[$0].desiredSpeedMPH }
      let spread = (desired.max() ?? 0.0) - (desired.min() ?? 0.0)
      if spread >= conflictSpeedSpreadMPH {
        candidates.append(ConflictGroup(indices: indices, speedSpreadMPH: spread))
      }
    }

    var result: [ConflictGroup] = []
    for candidate in candidates.sorted(by: {
      if $0.indices.count != $1.indices.count { return $0.indices.count > $1.indices.count }
      return ($0.indices.first ?? 0) < ($1.indices.first ?? 0)
    }) {
      let candidateSet = Set(candidate.indices)
      if result.contains(where: { candidateSet.isSubset(of: Set($0.indices)) }) { continue }
      result.append(candidate)
    }
    return result.sorted { ($0.indices.first ?? 0) < ($1.indices.first ?? 0) }
  }

  fileprivate static func logSpacedCurvatures(count: Int) -> [Double] {
    let sampleCount = max(count, 2)
    let lower = log10(exportedCurvatureRange.lowerBound)
    let upper = log10(exportedCurvatureRange.upperBound)
    return (0..<sampleCount).map { index in
      let t = Double(index) / Double(sampleCount - 1)
      return pow(10.0, lower + (upper - lower) * t)
    }
  }

  fileprivate static func weightedRMSE(
    values: [Double],
    targets: [Double],
    samples: [CurveCalibrationSample],
    clusters: [CurvatureCluster]
  ) -> Double {
    var total = 0.0
    var totalWeight = 0.0
    for cluster in clusters {
      let rawWeight = cluster.indices.reduce(0.0) { $0 + samples[$1].weight }
      let clusterWeight = cluster.indices.map { samples[$0].weight }.max() ?? 1.0
      var clusterMSE = 0.0
      for index in cluster.indices {
        let residual = values[index] - targets[index]
        clusterMSE += samples[index].weight / rawWeight * residual * residual
      }
      total += clusterWeight * clusterMSE
      totalWeight += clusterWeight
    }
    return sqrt(total / max(totalWeight, 1.0e-12))
  }

  fileprivate static func monotonicNonIncreasingProjection(
    _ values: [Double],
    weights: [Double]
  ) -> [Double] {
    struct Block {
      var start: Int
      var end: Int
      var weight: Double
      var weightedValue: Double

      var value: Double { weightedValue / max(weight, 1.0e-12) }
    }

    guard values.count == weights.count, !values.isEmpty else { return values }
    var blocks: [Block] = []
    for index in values.indices {
      let weight = max(weights[index], 1.0e-12)
      blocks.append(
        Block(
          start: index,
          end: index,
          weight: weight,
          weightedValue: values[index] * weight
        )
      )
      while blocks.count >= 2,
        blocks[blocks.count - 2].value < blocks[blocks.count - 1].value
      {
        let right = blocks.removeLast()
        let left = blocks.removeLast()
        blocks.append(
          Block(
            start: left.start,
            end: right.end,
            weight: left.weight + right.weight,
            weightedValue: left.weightedValue + right.weightedValue
          )
        )
      }
    }

    var projected = values
    for block in blocks {
      for index in block.start ... block.end { projected[index] = block.value }
    }
    return projected
  }

  /// Fit the residual Q curve after the four-knob sigmoid has captured the
  /// global trend. One deterministic broad band per distinct curvature
  /// cluster gives the exported curve enough local authority to honor real
  /// map targets without relaxing any PHYSICS_* clip. Gains are solved in the
  /// same log-curvature / acceleration-dB domain used by Q_CURVE_POINTS.
  fileprivate static func fitResidualBands(
    samples: [CurveCalibrationSample],
    clusters: [CurvatureCluster],
    parameters: SigmoidParameters,
    anchorParameters: SigmoidParameters,
    sampleCurvatureRange: ClosedRange<Double>,
    fittedTargets: [Double],
    sampleWeights: [Double],
    modifiers: VTSCSpeedModifiers
  ) throws -> [EQBand] {
    guard !clusters.isEmpty else { return [] }

    let proposed = clusters.map { cluster -> EQBand in
      let rawWeight = cluster.indices.reduce(0.0) { $0 + samples[$1].weight }
      let logCenter = cluster.indices.reduce(0.0) { partial, index in
        partial + samples[index].weight / rawWeight * log(max(samples[index].curvature, 1.0e-12))
      }
      let centerKappa = exp(logCenter)
      let centerSpeed = bakedTileSpeedMPH(parameters: parameters, curvature: centerKappa)
        .clamped(to: 1.0 ... maximumRepresentableSpeedMPH)
      return EQBand(
        id: samples[cluster.indices[0]].id,
        centerSpeedMPH: centerSpeed,
        gainDB: 0.0,
        q: residualBandQ,
        enabled: true
      )
    }
    let prepared = VTSCMath.prepareBands(proposed, parameters: parameters)
    guard prepared.count == proposed.count else { return [] }

    let basePredictions = samples.map {
      predictedSpeedMPH(
        parameters: parameters,
        curvature: $0.curvature,
        preparedBands: [],
        mode: .effectiveStrategic,
        modifiers: modifiers
      )
    }
    let targetDB = samples.indices.map { index in
      let multiplier = (fittedTargets[index] / max(basePredictions[index], 1.0e-9))
        .clamped(to: qSpeedMultiplierRange)
      return 40.0 * log10(multiplier)
    }
    let basis: [[Double]] = samples.map { sample in
      let logKappa = log10(max(sample.curvature, 1.0e-12))
      return prepared.map { band in
        let z = (logKappa - band.logCenterKappa) / band.sigma
        return exp(-0.5 * z * z)
      }
    }

    var gains = Array(repeating: 0.0, count: proposed.count)
    for _ in 0 ..< 96 {
      var maximumChange = 0.0
      for bandIndex in proposed.indices {
        var numerator = 0.0
        var denominator = residualBandRidge
        for sampleIndex in samples.indices {
          let coefficient = basis[sampleIndex][bandIndex]
          guard coefficient > 1.0e-12 else { continue }
          var otherDB = 0.0
          for otherIndex in proposed.indices where otherIndex != bandIndex {
            otherDB += basis[sampleIndex][otherIndex] * gains[otherIndex]
          }
          numerator += sampleWeights[sampleIndex] * coefficient
            * (targetDB[sampleIndex] - otherDB)
          denominator += sampleWeights[sampleIndex] * coefficient * coefficient
        }
        let updated = (numerator / max(denominator, 1.0e-12))
          .clamped(to: residualBandGainDBRange)
        maximumChange = max(maximumChange, abs(updated - gains[bandIndex]))
        gains[bandIndex] = updated
      }
      if maximumChange < 1.0e-6 { break }
    }

    let unconstrainedGains = gains

    func bands(with candidateGains: [Double]) -> [EQBand] {
      var candidate = proposed
      for index in candidate.indices { candidate[index].gainDB = candidateGains[index] }
      return candidate
    }

    func evaluate(_ candidateGains: [Double]) -> (score: Double, maximumError: Double) {
      let candidateBands = bands(with: candidateGains)
      let qPoints = VTSCMath.qCurvePoints(parameters: parameters, bands: candidateBands)
      let shape = runtimeCurveShape(
        parameters: parameters,
        qPoints: qPoints,
        modifiers: modifiers
      )
      let values = samples.indices.map { index in
        let q = VTSCMath.qSpeedMultiplier(
          curvature: samples[index].curvature,
          points: qPoints
        )
        return min(
          maximumBakedSpeedMPS * VTSCMath.metersPerSecondToMPH,
          basePredictions[index] * q
        )
      }
      let rmse = weightedRMSE(
        values: values,
        targets: fittedTargets,
        samples: samples,
        clusters: clusters
      )
      let ridge = candidateGains.reduce(0.0) { $0 + $1 * $1 }
        / Double(max(candidateGains.count, 1))
      let maximumError = zip(values, fittedTargets).map { abs($0 - $1) }.max() ?? 0.0
      // A soft phase lets coordinated neighboring bands move together instead
      // of getting trapped because one intermediate coordinate creates a
      // reversal. A hard raw-runtime check is still applied before returning.
      let reversalPenalty = 1_000.0 * shape.squaredSpeedReversalViolationMPH2
      let outsideDelta = maximumRuntimeDeltaOutsideInfluenceCollar(
        parameters: parameters,
        qPoints: qPoints,
        anchorParameters: anchorParameters,
        sampleCurvatureRange: sampleCurvatureRange,
        modifiers: modifiers
      )
      let outsideViolation = max(
        0.0,
        outsideDelta - maximumOffBankBackboneDeltaMPH
      )
      let outsidePenalty = 1_000.0 * outsideViolation * outsideViolation
      return (
        rmse * rmse + 0.10 * maximumError * maximumError
          + residualBandRidge * ridge + reversalPenalty + outsidePenalty,
        maximumError
      )
    }

    func isBetter(
      _ lhs: (gains: [Double], score: Double, maximumError: Double),
      than rhs: (gains: [Double], score: Double, maximumError: Double)
    ) -> Bool {
      let epsilon = 1.0e-12
      if abs(lhs.score - rhs.score) > epsilon { return lhs.score < rhs.score }
      if abs(lhs.maximumError - rhs.maximumError) > epsilon {
        return lhs.maximumError < rhs.maximumError
      }
      for index in lhs.gains.indices where lhs.gains[index] != rhs.gains[index] {
        return lhs.gains[index] < rhs.gains[index]
      }
      return false
    }

    func candidate(_ gains: [Double]) -> (gains: [Double], score: Double, maximumError: Double) {
      let evaluation = evaluate(gains)
      return (gains, evaluation.score, evaluation.maximumError)
    }

    func refine(
      _ seed: (gains: [Double], score: Double, maximumError: Double)
    ) -> (gains: [Double], score: Double, maximumError: Double) {
      var best = seed
      for step in [3.0, 1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375] {
        var changed = true
        var pass = 0
        while changed, pass < 8 {
          changed = false
          pass += 1
          for bandIndex in proposed.indices {
            var bandBest = best
            for direction in [-1.0, 1.0] {
              var candidateGains = best.gains
              candidateGains[bandIndex] = (candidateGains[bandIndex] + direction * step)
                .clamped(to: residualBandGainDBRange)
              guard candidateGains != best.gains else { continue }
              let next = candidate(candidateGains)
              if isBetter(next, than: bandBest) { bandBest = next }
            }
            if isBetter(bandBest, than: best) {
              best = bandBest
              changed = true
            }
          }
        }
      }
      return best
    }

    let zeroGains = Array(repeating: 0.0, count: proposed.count)
    let seeds = [
      candidate(zeroGains),
      candidate(unconstrainedGains),
      candidate(unconstrainedGains.map { $0 * 0.5 }),
    ]
    var best = seeds.map(refine).min { isBetter($0, than: $1) }!

    // Guarantee the final raw exported curve remains effectively monotonic.
    // If the soft solve is still outside tolerance, scale it toward the safe
    // zero-residual curve and keep the best runtime-evaluated safe candidate.
    func isRuntimeSafe(_ candidateGains: [Double], probeCount: Int = 512) -> Bool {
      let candidateBands = bands(with: candidateGains)
      let qPoints = VTSCMath.qCurvePoints(parameters: parameters, bands: candidateBands)
      return runtimeCurveShape(
        parameters: parameters,
        qPoints: qPoints,
        modifiers: modifiers,
        probeCount: probeCount
      ).maximumSpeedReversalMPH <= maximumRuntimeSpeedReversalMPH
        && maximumRuntimeDeltaOutsideInfluenceCollar(
          parameters: parameters,
          qPoints: qPoints,
          anchorParameters: anchorParameters,
          sampleCurvatureRange: sampleCurvatureRange,
          modifiers: modifiers,
          probeCount: probeCount
        ) <= maximumOffBankBackboneDeltaMPH + 1.0e-9
    }

    guard isRuntimeSafe(zeroGains, probeCount: 4_096) else {
      throw SigmoidFitError.unsafeRuntimeProposal(
        reason: "the selected sigmoid does not return to its checkout anchor outside the fit collar"
      )
    }

    // The optimization loop uses a cheaper guard. The export boundary always
    // rechecks the same rounded runtime curve at dense resolution.
    if !isRuntimeSafe(best.gains, probeCount: 4_096) {
      var safe = candidate(zeroGains)
      for scale in stride(from: 1.0, through: 0.0, by: -0.01) {
        let scaled = best.gains.map { $0 * scale }
        if isRuntimeSafe(scaled, probeCount: 4_096) {
          let next = candidate(scaled)
          if isBetter(next, than: safe) { safe = next }
        }
      }
      best = safe
    }

    let fitted = bands(with: best.gains)
      .filter { abs($0.gainDB) >= residualBandPruneDB }
    let fittedGains = fitted.map(\.gainDB)
    if fitted.count == proposed.count, isRuntimeSafe(fittedGains, probeCount: 4_096) {
      return fitted
    }
    return bands(with: best.gains)
  }
}

// MARK: - Bounded deterministic search

extension SigmoidFitter {
  fileprivate struct FitVector: Equatable {
    var x0: Double
    var x1: Double
    var x2: Double
    var x3: Double

    static let center = Self(x0: 0.5, x1: 0.5, x2: 0.5, x3: 0.5)

    subscript(index: Int) -> Double {
      get {
        switch index {
        case 0: x0
        case 1: x1
        case 2: x2
        default: x3
        }
      }
      set {
        switch index {
        case 0: x0 = newValue
        case 1: x1 = newValue
        case 2: x2 = newValue
        default: x3 = newValue
        }
      }
    }

    func adding(_ direction: FitVector, scale: Double) -> FitVector {
      FitVector(
        x0: (x0 + direction.x0 * scale).clamped(to: 0.0...1.0),
        x1: (x1 + direction.x1 * scale).clamped(to: 0.0...1.0),
        x2: (x2 + direction.x2 * scale).clamped(to: 0.0...1.0),
        x3: (x3 + direction.x3 * scale).clamped(to: 0.0...1.0)
      )
    }

    func squaredDistance(to other: FitVector) -> Double {
      let values = [x0 - other.x0, x1 - other.x1, x2 - other.x2, x3 - other.x3]
      return values.reduce(0.0) { $0 + $1 * $1 }
    }
  }

  fileprivate struct EvaluatedVector {
    var vector: FitVector
    var predictions: [Double]
  }

  fileprivate struct CandidateScore {
    var vector: FitVector
    var predictions: [Double]
    var total: Double
    var dataMSE: Double
    var maximumAbsoluteError: Double
    var priorDistance: Double
  }

  fileprivate struct ExtremeResult {
    var vector: FitVector
    var value: Double
  }

  fileprivate static func runtimeParameters(from knobs: PlainKnobs) -> SigmoidParameters {
    VTSCMath.sourceRoundedParameters(VTSCMath.parameters(from: knobs))
  }

  fileprivate static func decode(_ vector: FitVector) -> PlainKnobs {
    let low = 1.0 + 2.0 * vector.x0
    let highMinimum = max(2.0, low + 0.2)
    let high = highMinimum + vector.x1 * (5.5 - highMinimum)
    let midpointAcceleration = 0.5 * (low + high)
    let minimumTransition = max(
      8.0,
      sqrt(midpointAcceleration / VTSCClip.center.upperBound) * VTSCMath.metersPerSecondToMPH
    )
    let transition = exp(
      log(minimumTransition) + vector.x2 * (log(120.0) - log(minimumTransition))
    )
    return PlainKnobs(
      tightCurveAcceleration: low,
      straightRoadAcceleration: high,
      transitionSpeedMPH: transition,
      sharpness: 10.0 * vector.x3
    )
  }

  fileprivate static func encode(_ knobs: PlainKnobs) -> FitVector {
    let canonical = VTSCMath.knobs(from: VTSCMath.parameters(from: knobs))
    let low = canonical.tightCurveAcceleration.clamped(to: VTSCClip.minimumLateralAcceleration)
    let highMinimum = max(2.0, low + 0.2)
    let high = canonical.straightRoadAcceleration.clamped(to: highMinimum...5.5)
    let midpointAcceleration = 0.5 * (low + high)
    let minimumTransition = max(
      8.0,
      sqrt(midpointAcceleration / VTSCClip.center.upperBound) * VTSCMath.metersPerSecondToMPH
    )
    let transition = canonical.transitionSpeedMPH.clamped(to: minimumTransition...120.0)
    return FitVector(
      x0: ((low - 1.0) / 2.0).clamped(to: 0.0...1.0),
      x1: ((high - highMinimum) / max(5.5 - highMinimum, 1.0e-9)).clamped(to: 0.0...1.0),
      x2: ((log(transition) - log(minimumTransition))
        / max(log(120.0) - log(minimumTransition), 1.0e-9))
        .clamped(to: 0.0...1.0),
      x3: (canonical.sharpness / 10.0).clamped(to: 0.0...1.0)
    )
  }

  fileprivate static func coarseGrid() -> [FitVector] {
    let divisions = 4
    var result: [FitVector] = []
    result.reserveCapacity(625)
    for i0 in 0...divisions {
      for i1 in 0...divisions {
        for i2 in 0...divisions {
          for i3 in 0...divisions {
            result.append(
              FitVector(
                x0: Double(i0) / Double(divisions),
                x1: Double(i1) / Double(divisions),
                x2: Double(i2) / Double(divisions),
                x3: Double(i3) / Double(divisions)
              )
            )
          }
        }
      }
    }
    return result
  }

  fileprivate static func analyticSeed(samples: [CurveCalibrationSample], fittedTargets: [Double])
    -> FitVector
  {
    let requiredAccelerations = samples.indices.map { index in
      let speedMPS = fittedTargets[index] * VTSCMath.mphToMetersPerSecond
      return samples[index].curvature * speedMPS * speedMPS
    }
    let third = max(1, samples.count / 3)
    let gentleAcceleration = requiredAccelerations.prefix(third).reduce(0.0, +) / Double(third)
    let tightAcceleration = requiredAccelerations.suffix(third).reduce(0.0, +) / Double(third)
    let low = tightAcceleration.clamped(to: VTSCClip.minimumLateralAcceleration)
    let high = max(
      gentleAcceleration.clamped(to: VTSCClip.maximumLateralAcceleration),
      low + VTSCClip.amplitudeMagnitude.lowerBound
    ).clamped(to: VTSCClip.maximumLateralAcceleration)
    let midpoint = 0.5 * (low + high)
    let crossingIndex =
      requiredAccelerations.indices.min {
        abs(requiredAccelerations[$0] - midpoint) < abs(requiredAccelerations[$1] - midpoint)
      } ?? 0
    let transition =
      sqrt(midpoint / samples[crossingIndex].curvature) * VTSCMath.metersPerSecondToMPH
    return encode(
      PlainKnobs(
        tightCurveAcceleration: low,
        straightRoadAcceleration: high,
        transitionSpeedMPH: transition,
        sharpness: 5.0
      )
    )
  }

  fileprivate static func axialDirections() -> [FitVector] {
    var directions: [FitVector] = []
    for axis in 0..<4 {
      for sign in [-1.0, 1.0] {
        var direction = FitVector(x0: 0, x1: 0, x2: 0, x3: 0)
        direction[axis] = sign
        directions.append(direction)
      }
    }
    return directions
  }

  fileprivate static func refinementDirections() -> [FitVector] {
    var directions = axialDirections()
    let component = 1.0 / sqrt(2.0)
    for first in 0..<4 {
      for second in (first + 1)..<4 {
        for firstSign in [-1.0, 1.0] {
          for secondSign in [-1.0, 1.0] {
            var direction = FitVector(x0: 0, x1: 0, x2: 0, x3: 0)
            direction[first] = firstSign * component
            direction[second] = secondSign * component
            directions.append(direction)
          }
        }
      }
    }
    return directions
  }

  fileprivate static func refineExtreme(
    start: FitVector,
    minimize: Bool,
    prediction: (FitVector) -> Double
  ) throws -> ExtremeResult {
    var current = ExtremeResult(vector: start, value: prediction(start))
    var step = 0.125
    var iteration = 0
    let directions = axialDirections()
    while step >= 0.0005, iteration < 32 {
      try Task.checkCancellation()
      var best = current
      for direction in directions {
        let vector = current.vector.adding(direction, scale: step)
        guard vector != current.vector else { continue }
        let value = prediction(vector)
        if minimize ? value < best.value : value > best.value {
          best = ExtremeResult(vector: vector, value: value)
        }
      }
      if best.vector == current.vector {
        step *= 0.5
      } else {
        current = best
      }
      iteration += 1
    }
    return current
  }

  fileprivate static func bestSeparated(
    _ candidates: [CandidateScore],
    count: Int,
    minimumDistance: Double
  ) -> [CandidateScore] {
    let sorted = candidates.sorted(by: isBetter)
    var result: [CandidateScore] = []
    for candidate in sorted {
      if result.allSatisfy({
        sqrt($0.vector.squaredDistance(to: candidate.vector)) >= minimumDistance
      }) {
        result.append(candidate)
        if result.count == count { break }
      }
    }
    if result.count < count {
      for candidate in sorted where !result.contains(where: { $0.vector == candidate.vector }) {
        result.append(candidate)
        if result.count == count { break }
      }
    }
    return result
  }

  fileprivate static func patternRefine(
    start: CandidateScore,
    predictions: (FitVector) -> [Double],
    score: (FitVector, [Double]) -> CandidateScore
  ) -> CandidateScore {
    var current = start
    var step = 0.125
    var iteration = 0
    let directions = refinementDirections()
    while step >= 0.0005, iteration < 64 {
      var best = current
      for direction in directions {
        let vector = current.vector.adding(direction, scale: step)
        guard vector != current.vector else { continue }
        let candidate = score(vector, predictions(vector))
        if isBetter(candidate, best) { best = candidate }
      }
      if best.vector == current.vector {
        step *= 0.5
      } else {
        current = best
      }
      iteration += 1
    }
    return current
  }

  fileprivate static func isBetter(_ lhs: CandidateScore, _ rhs: CandidateScore) -> Bool {
    let epsilon = 1.0e-12
    if abs(lhs.total - rhs.total) > epsilon { return lhs.total < rhs.total }
    if abs(lhs.dataMSE - rhs.dataMSE) > epsilon { return lhs.dataMSE < rhs.dataMSE }
    if abs(lhs.maximumAbsoluteError - rhs.maximumAbsoluteError) > epsilon {
      return lhs.maximumAbsoluteError < rhs.maximumAbsoluteError
    }
    if abs(lhs.priorDistance - rhs.priorDistance) > epsilon {
      return lhs.priorDistance < rhs.priorDistance
    }
    for index in 0..<4 where lhs.vector[index] != rhs.vector[index] {
      return lhs.vector[index] < rhs.vector[index]
    }
    return false
  }
}
