import Foundation
import Testing
@testable import VTSCTunerApp
import VTSCTunerCore

@Test func curvePlotKeepsLegacyDomainWithoutAVisibleHighSpeedBand() {
  let disabledHighBand = EQBand(centerSpeedMPH: 120, enabled: false)
  let ordinaryBand = EQBand(centerSpeedMPH: 72)

  #expect(CurvePlotSpeedDomain.maximumSpeedMPH(
    bands: [ordinaryBand, disabledHighBand],
    selection: nil
  ) == 90)
  #expect(CurvePlotSpeedDomain.maximumSpeedMPH(
    bands: [ordinaryBand, disabledHighBand],
    selection: .band(ordinaryBand.id)
  ) == 90)
}

@Test func curvePlotExpandsForEnabledOrSelectedHighSpeedBands() {
  let enabledHighBand = EQBand(centerSpeedMPH: 117)
  let disabledHighBand = EQBand(centerSpeedMPH: 120, enabled: false)

  #expect(CurvePlotSpeedDomain.maximumSpeedMPH(
    bands: [enabledHighBand],
    selection: nil
  ) == 120)
  #expect(CurvePlotSpeedDomain.maximumSpeedMPH(
    bands: [disabledHighBand],
    selection: .band(disabledHighBand.id)
  ) == 130)
}

@Test func curvePlotAndBandAuthoringStopAtTheRuntimeRepresentableLimit() {
  let maximum = SigmoidFitter.maximumRepresentableSpeedMPH
  let atLimit = EQBand(centerSpeedMPH: maximum)

  #expect(CurvePlotSpeedDomain.maximumSpeedMPH(
    bands: [atLimit],
    selection: nil
  ) == maximum)
  #expect(CurvePlotSpeedDomain.clampedBandCenterSpeedMPH(maximum + 100) == maximum)
  #expect(CurvePlotSpeedDomain.clampedBandCenterSpeedMPH(-100) == 1)
}
