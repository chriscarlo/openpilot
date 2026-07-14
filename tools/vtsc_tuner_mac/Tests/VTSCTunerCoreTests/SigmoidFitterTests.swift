import Foundation
import Testing

@testable import VTSCTunerCore

private func calibrationID(_ suffix: Int) -> UUID {
  UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", suffix))!
}

private func savedEighteenCurveBank() -> [CurveCalibrationSample] {
  let targets: [(Double, Double)] = [
    (0.008578403275769536, 72),
    (0.004352938160952768, 72),
    (0.004040138924048590, 72),
    (0.004653349341690074, 72),
    (0.0034834721665691275, 75),
    (0.006935117637501812, 65),
    (0.008420591915605598, 65),
    (0.008589635458317087, 65),
    (0.007861477957525367, 65),
    (0.011375490008222367, 65),
    (0.007447364829048697, 62),
    (0.006225350467409148, 62),
    (0.006105971448823081, 62),
    (0.0068118936054258105, 62),
    (0.006284183768126575, 62),
    (0.0053921658046673735, 75),
    (0.006428465803861694, 72),
    (0.005229287995374826, 72),
  ]
  return targets.enumerated().map { index, target in
    CurveCalibrationSample(
      id: calibrationID(100 + index),
      label: "#\(index + 1) US 50",
      curvature: target.0,
      desiredSpeedMPH: target.1
    )
  }
}

private func runtimeResolvedEighteenCurveBank() -> [CurveCalibrationSample] {
  let targets: [(Double, Double)] = [
    (0.0041818457315006335, 72),
    (0.0025117624763593610, 72),
    (0.0036276491298469730, 72),
    (0.0026076084572648290, 72),
    (0.0031665770067601030, 75),
    (0.0045565799917226335, 65),
    (0.0063721938306250830, 65),
    (0.0040015134794371420, 65),
    (0.0037632890648147970, 65),
    (0.0047702528058511160, 65),
    (0.0056098186208494190, 62),
    (0.0036603450079111690, 62), // 699c085aac2f5ebb85bbc7d19a8edae9:2
    (0.0023486176148524345, 62),
    (0.0023325717426705550, 62),
    (0.0054832790569981390, 62),
    (0.0017763812304444923, 75),
    (0.0029369013742527820, 72),
    (0.0033382622019375930, 72),
  ]
  return targets.enumerated().map { index, target in
    CurveCalibrationSample(
      id: calibrationID(200 + index),
      label: "#\(index + 1) US 50 runtime",
      curvature: target.0,
      desiredSpeedMPH: target.1
    )
  }
}

@Test func bakedPredictionUsesMapBakeMathAndExplicitEffectiveModifiers() {
  let parameters = SigmoidParameters.checkoutFallback
  let curvature = 0.01
  let baked = SigmoidFitter.predictedSpeedMPH(
    parameters: parameters,
    curvature: curvature,
    mode: .bakedTile
  )
  let expectedBaked =
    MapBakeMath.bakedSpeedMPS(curvature: curvature, parameters: parameters)
    * VTSCMath.metersPerSecondToMPH
  #expect(abs(baked - expectedBaked) < 1.0e-12)

  let effective = SigmoidFitter.predictedSpeedMPH(
    parameters: parameters,
    curvature: curvature,
    mode: .effectiveStrategic
  )
  #expect(abs(effective - baked) < 1.0e-12)

  let biased = SigmoidFitter.predictedSpeedMPH(
    parameters: parameters,
    curvature: curvature,
    mode: .effectiveStrategic,
    modifiers: VTSCSpeedModifiers(lowSpeedBiasMPH: 5, lowSpeedBiasEndMPH: 55)
  )
  #expect(biased > effective)

  let boosted = SigmoidFitter.predictedSpeedMPH(
    parameters: parameters,
    curvature: curvature,
    bands: [EQBand(centerSpeedMPH: baked, gainDB: 6.0, q: 8.0)],
    mode: .effectiveStrategic
  )
  #expect(boosted > effective)
}

@Test func completeCurveFitMakesSavedBankReachableAndKeepsRealConflictsVisible() throws {
  let samples = savedEighteenCurveBank()
  // Keep this fitter-regression fixture independent of the checkout's current
  // production tune. The target bank and thresholds were authored against
  // this historical starting curve; changing `.checkoutFallback` should not
  // silently redefine the optimization test.
  let fitFixture = SigmoidParameters(
    a: -2.125_961,
    b: -1_601.225_452,
    c: 0.007_637,
    d: 4.478,
    minLat: 2.352,
    maxLat: 4.478
  )
  let startingKnobs = VTSCMath.knobs(from: fitFixture)
  let result = try SigmoidFitter.fit(
    samples: samples, currentKnobs: startingKnobs, anchorKnobs: startingKnobs)
  let reversed = try SigmoidFitter.fit(
    samples: samples.reversed(), currentKnobs: startingKnobs, anchorKnobs: startingKnobs)

  #expect(result == reversed)
  #expect(result.beforeRMSEMPH > 15.0)
  #expect(result.afterRMSEMPH < 3.5)
  #expect(result.afterRMSEMPH < result.beforeRMSEMPH * 0.25)
  #expect(result.maximumAbsoluteErrorMPH > 5.0)
  #expect(result.maximumAbsoluteErrorMPH < 7.1)
  #expect(!result.bands.isEmpty)
  #expect(result.diagnostics.allSatisfy { !$0.isImpossibleTarget })
  #expect(
    zip(result.diagnostics, result.diagnostics.dropFirst()).allSatisfy {
      $0.fittedTargetSpeedMPH >= $1.fittedTargetSpeedMPH - 1.0e-9
    }
  )
  #expect(
    !result.warnings.contains { warning in
      if case .impossibleTarget = warning { return true }
      return false
    }
  )

  let conflictSpreads = result.warnings.compactMap { warning -> Double? in
    if case .conflictingCurvatures(_, let spread) = warning { return spread }
    return nil
  }
  #expect(conflictSpreads.contains(10.0))
  #expect(conflictSpreads.contains(7.0))
  #expect(result.bands.allSatisfy { (-12.0 ... 12.0).contains($0.gainDB) })
  #expect(result.bands.allSatisfy { $0.q == 4.0 && $0.enabled })
  #expect(result.diagnostics.allSatisfy {
    $0.attainableSpeedRangeMPH.contains($0.fittedTargetSpeedMPH)
  })
  let exportedPoints = VTSCMath.qCurvePoints(
    parameters: result.parameters,
    bands: result.bands
  )
  #expect(exportedPoints.allSatisfy { (0.5 ... 1.5).contains($0.speedMultiplier) })
  for diagnostic in result.diagnostics {
    let base = SigmoidFitter.predictedSpeedMPH(
      parameters: result.parameters,
      curvature: diagnostic.curvature,
      mode: .effectiveStrategic
    )
    let neededQ = diagnostic.fittedTargetSpeedMPH / max(base, 1.0e-9)
    #expect((0.5 ... 1.5).contains(neededQ))
    let q = VTSCMath.qSpeedMultiplier(
      curvature: diagnostic.curvature,
      points: exportedPoints
    )
    let runtimeSpeed = min(SigmoidFitter.maximumRepresentableSpeedMPH, base * q)
    #expect(abs(runtimeSpeed - diagnostic.afterSpeedMPH) < 1.0e-9)
  }
  #expect(result.maximumRuntimeSpeedReversalMPH <= 0.10)
  #expect(
    result.maximumOffBankBackboneDeltaMPH
      <= SigmoidFitter.maximumOffBankBackboneDeltaMPH + 1.0e-6
  )
  #expect(
    result.maximumOutsideInfluenceRuntimeDeltaMPH
      <= SigmoidFitter.maximumOffBankBackboneDeltaMPH + 1.0e-6
  )
  #expect(
    result.warnings.contains { warning in
      if case .effectiveAccelerationAboveBaseRail = warning { return true }
      return false
    }
  )

  let anchorParameters = VTSCMath.parameters(from: startingKnobs)
  let lowerHull = samples.map(\.curvature).min()!
    / SigmoidFitter.residualInfluenceCurvatureRatio
  let upperHull = samples.map(\.curvature).max()!
    * SigmoidFitter.residualInfluenceCurvatureRatio
  for index in 0..<1_024 {
    let t = Double(index) / 1_023.0
    let curvature = pow(10.0, -5.0 + 5.0 * t)
    guard curvature < lowerHull || curvature > upperHull else { continue }
    let anchorSpeed = SigmoidFitter.predictedSpeedMPH(
      parameters: anchorParameters,
      curvature: curvature,
      mode: .effectiveStrategic
    )
    let proposedBackboneSpeed = SigmoidFitter.predictedSpeedMPH(
      parameters: result.parameters,
      curvature: curvature,
      mode: .effectiveStrategic
    )
    #expect(
      abs(proposedBackboneSpeed - anchorSpeed)
        <= SigmoidFitter.maximumOffBankBackboneDeltaMPH + 0.1
    )
  }

  let lowerInfluenceBoundary = samples.map(\.curvature).min()!
    / SigmoidFitter.residualInfluenceCurvatureRatio
  let upperInfluenceBoundary = samples.map(\.curvature).max()!
    * SigmoidFitter.residualInfluenceCurvatureRatio
  for index in 0..<4_096 {
    let t = Double(index) / 4_095.0
    let curvature = pow(10.0, -5.0 + 5.0 * t)
    guard curvature <= lowerInfluenceBoundary || curvature >= upperInfluenceBoundary else {
      continue
    }
    let anchorSpeed = SigmoidFitter.predictedSpeedMPH(
      parameters: anchorParameters,
      curvature: curvature,
      mode: .effectiveStrategic
    )
    let proposedRuntimeSpeed = SigmoidFitter.predictedSpeedMPH(
      parameters: result.parameters,
      curvature: curvature,
      bands: result.bands,
      mode: .effectiveStrategic
    )
    #expect(
      abs(proposedRuntimeSpeed - anchorSpeed)
        <= SigmoidFitter.maximumOffBankBackboneDeltaMPH + 0.02
    )
  }

  let runtimePoints = VTSCMath.qCurvePoints(parameters: result.parameters, bands: result.bands)
  let runtimeSpeeds = runtimePoints.map {
    SigmoidFitter.predictedSpeedMPH(
      parameters: result.parameters,
      curvature: $0.kappa,
      bands: result.bands,
      mode: .effectiveStrategic
    )
  }
  #expect(zip(runtimeSpeeds, runtimeSpeeds.dropFirst()).allSatisfy { next in
    next.1 - next.0 <= 0.10 + 1.0e-9
  })

  let repeated = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: result.knobs,
    anchorKnobs: startingKnobs,
    bands: result.bands
  )
  #expect(repeated.knobs == result.knobs)
  #expect(repeated.parameters == result.parameters)
  #expect(repeated.bands == result.bands)
  #expect(repeated.afterRMSEMPH == result.afterRMSEMPH)
  let repeatedPoints = VTSCMath.qCurvePoints(
    parameters: repeated.parameters,
    bands: repeated.bands
  )
  #expect(runtimePoints.count == repeatedPoints.count)
  for (first, second) in zip(runtimePoints, repeatedPoints) {
    #expect(first.kappa == second.kappa)
    #expect(first.speedMultiplier == second.speedMultiplier)
  }
}

@Test func runtimeResolvedBankDoesNotDemandFalseEightToTenMPS2Acceleration() throws {
  let samples = runtimeResolvedEighteenCurveBank()
  let desiredAccelerations = samples.map {
    let speed = $0.desiredSpeedMPH * VTSCMath.mphToMetersPerSecond
    return $0.curvature * speed * speed
  }
  #expect(abs(desiredAccelerations.max()! - 5.380_324_378) < 1.0e-6)
  #expect(abs(desiredAccelerations[13] - 1.791_889_225) < 1.0e-6)

  let startingKnobs = VTSCMath.knobs(from: .checkoutFallback)
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: startingKnobs,
    anchorKnobs: startingKnobs
  )
  // Curvature correction removes the false acceleration demand; it does not
  // imply that the checkout tune already matches every driver-selected speed.
  #expect(result.beforeRMSEMPH > 10.0)
  #expect(result.afterRMSEMPH < result.beforeRMSEMPH)
  #expect(result.maximumProposedLateralAccelerationMPS2 < 5.5)
  #expect(result.diagnostics.count == 18)
  #expect(result.diagnostics.allSatisfy { !$0.isImpossibleTarget })
  #expect(!result.warnings.contains { warning in
    if case .impossibleTarget = warning { return true }
    return false
  })
  #expect(!result.warnings.contains { warning in
    if case .effectiveAccelerationAboveBaseRail = warning { return true }
    return false
  })
}

@Test func runtimeShapeMeasuresCumulativeReversalBetweenRoundedQKnots() {
  let parameters = VTSCMath.sourceRoundedParameters(.checkoutFallback)
  let bands = [EQBand(centerSpeedMPH: 12.0, gainDB: -6.25, q: 4.0)]
  let points = VTSCMath.qCurvePoints(parameters: parameters, bands: bands)
  let knotSpeeds = points.map {
    SigmoidFitter.predictedSpeedMPH(
      parameters: parameters,
      curvature: $0.kappa,
      bands: bands,
      mode: .effectiveStrategic
    )
  }
  let maximumAdjacentRise = zip(knotSpeeds, knotSpeeds.dropFirst())
    .map { max(0.0, $1 - $0) }
    .max() ?? 0.0
  let shape = SigmoidFitter.runtimeCurveShape(
    parameters: parameters,
    bands: bands,
    modifiers: .sourceDefaults,
    probeCount: 4_096
  )

  #expect(maximumAdjacentRise <= 0.10)
  #expect(shape.maximumSpeedReversalMPH > 0.10)
}

@Test func fitterRecoversSyntheticSpeedsDeterministically() throws {
  let targetKnobs = PlainKnobs(
    tightCurveAcceleration: 1.65,
    straightRoadAcceleration: 5.1,
    transitionSpeedMPH: 49.0,
    sharpness: 7.2
  )
  let targetParameters = VTSCMath.parameters(from: targetKnobs)
  let curvatures = [0.0018, 0.0030, 0.0048, 0.0075, 0.012, 0.024]
  let samples = curvatures.enumerated().map { index, curvature in
    CurveCalibrationSample(
      id: calibrationID(index + 1),
      label: "Curve \(index + 1)",
      curvature: curvature,
      desiredSpeedMPH: SigmoidFitter.predictedSpeedMPH(
        parameters: targetParameters,
        curvature: curvature,
        mode: .effectiveStrategic
      )
    )
  }
  let startingKnobs = VTSCMath.knobs(from: .checkoutFallback)

  let first = try SigmoidFitter.fit(
    samples: samples, currentKnobs: startingKnobs, anchorKnobs: targetKnobs)
  let reversed = try SigmoidFitter.fit(
    samples: samples.reversed(), currentKnobs: startingKnobs, anchorKnobs: targetKnobs)

  #expect(first == reversed)
  #expect(first.afterRMSEMPH < 0.35)
  #expect(first.afterRMSEMPH < first.beforeRMSEMPH)
  #expect(
    first.parameters
      == VTSCMath.sourceRoundedParameters(VTSCMath.parameters(from: first.knobs))
  )
  #expect(first.evaluationCount < 30_000)
  #expect(
    !first.warnings.contains { warning in
      if case .underconstrained = warning { return true }
      return false
    })
}

@Test func impossibleTargetIsBoundedAndReported() throws {
  let samples = [
    CurveCalibrationSample(
      id: calibrationID(11), label: "Impossible", curvature: 0.01, desiredSpeedMPH: 250.0),
    CurveCalibrationSample(
      id: calibrationID(12), label: "Sweeper", curvature: 0.003, desiredSpeedMPH: 70.0),
    CurveCalibrationSample(
      id: calibrationID(13), label: "Medium", curvature: 0.006, desiredSpeedMPH: 45.0),
    CurveCalibrationSample(
      id: calibrationID(14), label: "Tight", curvature: 0.025, desiredSpeedMPH: 22.0),
  ]
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: VTSCMath.knobs(from: .checkoutFallback),
    anchorKnobs: VTSCMath.knobs(from: .checkoutFallback),
    predictionMode: .bakedTile
  )

  let impossible = try #require(result.diagnostics.first(where: { $0.label == "Impossible" }))
  #expect(impossible.isImpossibleTarget)
  #expect(impossible.attainableSpeedRangeMPH.contains(impossible.fittedTargetSpeedMPH))
  #expect(impossible.fittedTargetSpeedMPH <= impossible.attainableSpeedRangeMPH.upperBound)
  #expect(
    result.warnings.contains { warning in
      if case .impossibleTarget(let label, _) = warning { return label == "Impossible" }
      return false
    })
  #expect(VTSCClip.minimumLateralAcceleration.contains(result.parameters.minLat))
  #expect(VTSCClip.maximumLateralAcceleration.contains(result.parameters.maxLat))
  #expect(VTSCClip.amplitudeMagnitude.contains(abs(result.parameters.a)))
  #expect(VTSCClip.steepnessMagnitude.contains(abs(result.parameters.b)))
  #expect(VTSCClip.center.contains(result.parameters.c))
}

@Test func targetBeyondCompleteQEnvelopeIsStillReported() throws {
  let samples = [
    CurveCalibrationSample(
      id: calibrationID(31), label: "Beyond Q", curvature: 0.01, desiredSpeedMPH: 250.0),
    CurveCalibrationSample(
      id: calibrationID(32), label: "Sweeper", curvature: 0.003, desiredSpeedMPH: 70.0),
    CurveCalibrationSample(
      id: calibrationID(33), label: "Medium", curvature: 0.006, desiredSpeedMPH: 45.0),
    CurveCalibrationSample(
      id: calibrationID(34), label: "Tight", curvature: 0.025, desiredSpeedMPH: 22.0),
  ]
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: VTSCMath.knobs(from: .checkoutFallback),
    anchorKnobs: VTSCMath.knobs(from: .checkoutFallback)
  )

  let impossible = try #require(result.diagnostics.first(where: { $0.label == "Beyond Q" }))
  #expect(impossible.isImpossibleTarget)
  #expect(impossible.attainableSpeedRangeMPH.upperBound < 90.0)
  #expect(impossible.attainableSpeedRangeMPH.contains(impossible.fittedTargetSpeedMPH))
  #expect(impossible.fittedTargetSpeedMPH <= impossible.attainableSpeedRangeMPH.upperBound)
}

@Test func similarCurvatureConflictAndUnderconstrainedFitAreDiagnosed() throws {
  let samples = [
    CurveCalibrationSample(
      id: calibrationID(21), label: "Northbound", curvature: 0.0100, desiredSpeedMPH: 30.0),
    CurveCalibrationSample(
      id: calibrationID(22), label: "Southbound", curvature: 0.0102, desiredSpeedMPH: 48.0),
    CurveCalibrationSample(
      id: calibrationID(23), label: "Nearby", curvature: 0.0104, desiredSpeedMPH: 39.0),
  ]
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: VTSCMath.knobs(from: .checkoutFallback),
    anchorKnobs: VTSCMath.knobs(from: .checkoutFallback)
  )

  #expect(
    result.warnings.contains { warning in
      if case .conflictingCurvatures(let labels, let spread) = warning {
        return Set(labels) == Set(["Northbound", "Southbound", "Nearby"]) && spread == 18.0
      }
      return false
    })
  #expect(
    result.warnings.contains { warning in
      if case .underconstrained(let clusterCount) = warning { return clusterCount == 1 }
      return false
    })
}

@Test func fitterRejectsStraightOrNonfiniteSamples() {
  let knobs = VTSCMath.knobs(from: .checkoutFallback)
  #expect(throws: SigmoidFitError.self) {
    try SigmoidFitter.fit(
      samples: [CurveCalibrationSample(label: "Straight", curvature: 0.0, desiredSpeedMPH: 50.0)],
      currentKnobs: knobs,
      anchorKnobs: knobs
    )
  }
  #expect(throws: SigmoidFitError.self) {
    try SigmoidFitter.fit(
      samples: [CurveCalibrationSample(label: "Bad", curvature: .nan, desiredSpeedMPH: 50.0)],
      currentKnobs: knobs,
      anchorKnobs: knobs
    )
  }
  #expect(throws: SigmoidFitError.self) {
    try SigmoidFitter.fit(
      samples: [
        CurveCalibrationSample(label: "Outside Q domain", curvature: 1.0e-6, desiredSpeedMPH: 50.0)
      ],
      currentKnobs: knobs,
      anchorKnobs: knobs
    )
  }
  #expect(throws: SigmoidFitError.self) {
    try SigmoidFitter.fit(
      samples: [
        CurveCalibrationSample(
          label: "Below residual center domain",
          curvature: 1.0e-5,
          desiredSpeedMPH: 100.0
        )
      ],
      currentKnobs: knobs,
      anchorKnobs: knobs
    )
  }
}

@Test func fixedAnchorClusteringDoesNotTransitivelyCollapseADenseBank() throws {
  let knobs = VTSCMath.knobs(from: .checkoutFallback)
  let samples = (0..<12).map { index in
    CurveCalibrationSample(
      id: calibrationID(400 + index),
      label: "Dense \(index)",
      curvature: 0.002 * pow(1.04, Double(index)),
      desiredSpeedMPH: 65.0 - Double(index)
    )
  }
  let result = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: knobs,
    anchorKnobs: knobs
  )
  #expect(
    !result.warnings.contains { warning in
      if case .underconstrained(let clusterCount) = warning { return clusterCount == 1 }
      return false
    }
  )
  #expect(result.bands.count > 1)
}

@Test func runtimeModifierValidationMatchesDeviceBounds() throws {
  let knobs = VTSCMath.knobs(from: .checkoutFallback)
  let samples = [
    CurveCalibrationSample(label: "Modifier check", curvature: 0.01, desiredSpeedMPH: 45.0)
  ]

  _ = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: knobs,
    anchorKnobs: knobs,
    modifiers: VTSCSpeedModifiers(
      lowSpeedBiasMPH: -5.0,
      lowSpeedBiasEndMPH: 10.0,
      speedIncreaseFactor: 0.5
    )
  )
  let maximumModifiers = try SigmoidFitter.fit(
    samples: samples,
    currentKnobs: knobs,
    anchorKnobs: knobs,
    modifiers: VTSCSpeedModifiers(
      lowSpeedBiasMPH: 5.0,
      lowSpeedBiasEndMPH: 80.0,
      speedIncreaseFactor: 1.5
    )
  )
  #expect(
    maximumModifiers.maximumProposedLateralAccelerationMPS2
      > SigmoidFitter.effectiveAccelerationWarningMPS2
  )
  #expect(
    maximumModifiers.warnings.contains { warning in
      if case .effectiveAccelerationAboveBaseRail = warning { return true }
      return false
    }
  )

  for invalid in [
    VTSCSpeedModifiers(lowSpeedBiasMPH: -5.01),
    VTSCSpeedModifiers(lowSpeedBiasMPH: 5.01),
    VTSCSpeedModifiers(lowSpeedBiasEndMPH: 9.99),
    VTSCSpeedModifiers(lowSpeedBiasEndMPH: 80.01),
    VTSCSpeedModifiers(speedIncreaseFactor: 0.49),
    VTSCSpeedModifiers(speedIncreaseFactor: 1.51),
  ] {
    #expect(throws: SigmoidFitError.self) {
      try SigmoidFitter.fit(
        samples: samples,
        currentKnobs: knobs,
        anchorKnobs: knobs,
        modifiers: invalid
      )
    }
  }
}

@Test func denseSimilarCurveClusterDoesNotOutvoteSparseCurvesInMonotonicProjection() throws {
  let knobs = VTSCMath.knobs(from: .checkoutFallback)
  let sparse = [
    CurveCalibrationSample(id: calibrationID(900), label: "Gentle", curvature: 0.004, desiredSpeedMPH: 70),
    CurveCalibrationSample(id: calibrationID(901), label: "Dense A", curvature: 0.006, desiredSpeedMPH: 60),
    CurveCalibrationSample(id: calibrationID(902), label: "Inversion", curvature: 0.008, desiredSpeedMPH: 70),
    CurveCalibrationSample(id: calibrationID(903), label: "Tight", curvature: 0.012, desiredSpeedMPH: 50),
  ]
  var dense = sparse
  dense.append(
    contentsOf: [
      CurveCalibrationSample(id: calibrationID(904), label: "Dense B", curvature: 0.0061, desiredSpeedMPH: 60),
      CurveCalibrationSample(id: calibrationID(905), label: "Dense C", curvature: 0.0062, desiredSpeedMPH: 60),
    ]
  )

  let sparseResult = try SigmoidFitter.fit(
    samples: sparse,
    currentKnobs: knobs,
    anchorKnobs: knobs
  )
  let denseResult = try SigmoidFitter.fit(
    samples: dense,
    currentKnobs: knobs,
    anchorKnobs: knobs
  )
  let sparsePlateau = try #require(
    sparseResult.diagnostics.first(where: { $0.label == "Dense A" })
  ).fittedTargetSpeedMPH
  let densePlateau = try #require(
    denseResult.diagnostics.first(where: { $0.label == "Dense A" })
  ).fittedTargetSpeedMPH
  let inversionPlateau = try #require(
    denseResult.diagnostics.first(where: { $0.label == "Inversion" })
  ).fittedTargetSpeedMPH

  #expect(abs(sparsePlateau - 65.0) < 1.0e-9)
  #expect(abs(densePlateau - sparsePlateau) < 1.0e-9)
  #expect(abs(inversionPlateau - sparsePlateau) < 1.0e-9)
}
