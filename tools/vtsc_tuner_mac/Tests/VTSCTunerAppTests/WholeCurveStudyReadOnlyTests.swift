import Testing
import VTSCTunerCore
@testable import VTSCTunerApp

@MainActor
@Test func wholeCurveStudyDoesNotRewriteCalibrationStateOrPermitSync() throws {
  let session = MapPreviewSession(
    loadPersistedState: false,
    persistsCalibrationSamples: false,
    persistsMapPreferences: false
  )
  session.calibrationSamples = [MapCalibrationSample(
    sourceKey: "saved-road:1",
    roadName: "Saved road",
    reference: "US 50",
    latitude: 38.0,
    longitude: -121.0,
    curvature: 0.01,
    rawCurvature: 0.012,
    curvatureSupportMeters: 75,
    curvatureEstimatorVersion: MapRuntimeCurvatureResolver.estimatorVersion,
    curvatureContextComplete: true,
    bakedSpeedMPH: 55,
    proposedSpeedMPH: 56,
    effectiveSpeedMPH: 57,
    desiredSpeedMPH: 58
  )]
  let original = try #require(session.calibrationSamples.first)

  var changedParameters = SigmoidParameters.checkoutFallback
  changedParameters.a += 0.25
  session.updateProposal(parameters: changedParameters, bands: [])
  session.updateCalibrationTarget(id: original.id, desiredSpeedMPH: 99)
  session.removeCalibrationSample(id: original.id)
  _ = session.ensureCalibrationAnchor(VTSCMath.knobs(from: .checkoutFallback))

  #expect(session.purpose == .wholeCurveStudy)
  #expect(!session.canSync)
  #expect(session.reconcileCalibrationCurvatures(using: [], invalidateMissing: true) == 0)
  #expect(session.calibrationSamples.first == original)
  #expect(session.calibrationAnchorKnobs == nil)

  session.purpose = .calibration

  #expect(session.canSync)
  #expect(session.calibrationSamples.first?.desiredSpeedMPH == original.desiredSpeedMPH)
  #expect(session.calibrationSamples.first?.proposedSpeedMPH != original.proposedSpeedMPH)
  #expect(session.calibrationSamples.first?.effectiveSpeedMPH != original.effectiveSpeedMPH)
}

@MainActor
@Test func mapStudyRejectsGlobalTuneUndoAndApply() {
  let session = TunerSession()
  session.workspace = .curveLab
  session.addBand(at: 47)
  let edited = session.snapshot
  #expect(session.canUndo)

  session.workspace = .mapPreview
  session.undo()
  #expect(session.snapshot == edited)

  session.pendingApplyAction = .local
  session.confirmApply()
  #expect(session.pendingApplyAction == nil)
  #expect(session.runningApplyAction == nil)
  #expect(session.statusIsError)
  #expect(session.statusText == "Apply is only available in Curve Lab.")
}
