import Foundation
import Testing
import VTSCTunerCore
@testable import VTSCTunerApp

@Test func applyChooserUsesFourPlainLanguageDestinationsInSafeOrder() {
  let choices = ApplyAction.availableCases
  #expect(choices == [.local, .commit, .push, .pullOnTici])
  #expect(choices.map(\.label) == [
    "Update Selected Checkout",
    "Create a Local Commit",
    "Push to Git Remote",
    "Install on the Car (tici)",
  ])
  #expect(Set(choices.map(\.chooserDestination)).count == choices.count)
  #expect(choices.allSatisfy { !$0.description.isEmpty && !$0.chooserDestination.isEmpty })
  #expect(choices.filter(\.isRecommendedForRoadTesting) == [.pullOnTici])
  #expect(!choices.contains(.rebuildTilesAndReboot))
}

@MainActor
@Test func chooserOnlyOpensConfirmationAfterAnExplicitReviewChoice() {
  let session = TunerSession()
  session.workspace = .curveLab

  session.showApplyActionChooser()
  #expect(session.applyActionChooserVisible)
  #expect(session.applyChooserSelection == nil)
  #expect(session.pendingApplyAction == nil)

  session.selectApplyAction(.pullOnTici)
  #expect(session.applyChooserSelection == .pullOnTici)
  #expect(session.pendingApplyAction == nil)

  session.reviewSelectedApplyAction()
  #expect(!session.applyActionChooserVisible)
  #expect(session.pendingApplyAction == nil)
  session.applyActionChooserDidDismiss()
  #expect(session.pendingApplyAction == .pullOnTici)
  #expect(session.runningApplyAction == nil)
}

@MainActor
@Test func cancellingTheApplyChooserDoesNothing() {
  let session = TunerSession()
  session.workspace = .curveLab
  session.showApplyActionChooser()
  session.selectApplyAction(.push)

  session.cancelApplyActionChooser()
  session.applyActionChooserDidDismiss()

  #expect(!session.applyActionChooserVisible)
  #expect(session.applyChooserSelection == nil)
  #expect(session.pendingApplyAction == nil)
  #expect(session.runningApplyAction == nil)
}

@MainActor
@Test func tileDeploymentExplainsTheBoundaryAndRoutesToRuntimeOnlyDeployment() {
  let session = TunerSession()
  session.workspace = .curveLab

  session.requestApply(.rebuildTilesAndReboot)

  #expect(session.pendingApplyAction == nil)
  #expect(session.runningApplyAction == nil)
  #expect(session.tileDeploymentInfoVisible)
  #expect(session.statusText == ApplyAction.deviceTileReplacementUnavailableReason)

  session.requestRuntimeDeploymentFromTileInfo()

  #expect(!session.tileDeploymentInfoVisible)
  #expect(session.pendingApplyAction == .pullOnTici)
}

@MainActor
@Test func staleOrInjectedTileRebuildSelectionCannotStartAnApply() {
  let session = TunerSession()
  session.workspace = .curveLab
  session.pendingApplyAction = .rebuildTilesAndReboot

  session.confirmApply()

  #expect(session.pendingApplyAction == nil)
  #expect(session.runningApplyAction == nil)
  #expect(session.statusIsError)
  #expect(session.statusText == ApplyAction.deviceTileReplacementUnavailableReason)
}

@MainActor
@Test func mixedPendingJournalsBindResumeToTheExactSelectedRuntimeOnlyJournal() throws {
  let directory = FileManager.default.temporaryDirectory
    .appendingPathComponent("vtsc-mixed-pending-ui-\(UUID().uuidString)", isDirectory: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

  let runtime = pendingJournal(targetTileSetID: nil)
  let tile = pendingJournal(targetTileSetID: String(repeating: "f", count: 64))
  let runtimeURL = directory.appendingPathComponent("runtime.json")
  let tileURL = directory.appendingPathComponent("tile.json")
  try runtime.write(to: runtimeURL)
  try tile.write(to: tileURL)

  let session = TunerSession()
  session.selectedPendingDeploymentURL = tileURL

  session.refreshRecoverableRollbackState(directory: directory)

  #expect(session.hasRuntimeOnlyPendingDeployment)
  #expect(session.selectedPendingDeploymentURL == runtimeURL.standardizedFileURL)
  #expect(session.selectedPendingDeploymentIsRuntimeOnly)

  session.selectedPendingDeploymentURL = tileURL
  #expect(!session.selectedPendingDeploymentIsRuntimeOnly)
  #expect(session.resumePostflightRequest(repositoryURL: directory) == nil)
  session.selectedPendingDeploymentURL = runtimeURL
  let request = try #require(session.resumePostflightRequest(repositoryURL: directory))
  #expect(session.selectedPendingDeploymentIsRuntimeOnly)
  #expect(request.journalURL == runtimeURL.standardizedFileURL)
}

private func pendingJournal(targetTileSetID: String?) -> DeploymentRollbackJournal {
  var journal = DeploymentRollbackJournal(
    profile: "commaAdb",
    branch: "chauffeur-exp01",
    previousHead: String(repeating: "9", count: 40),
    targetHead: String(repeating: "a", count: 40),
    previousPhysicsParams: [:],
    previousQCurveSHA256: String(repeating: "b", count: 64),
    previousMapdReleaseVersion: "whole-curve-v3",
    previousMapdVersion: "whole-curve-v3",
    previousActiveMapdSHA256: String(repeating: "c", count: 64),
    previousCachedMapdPath: "",
    mapdRollbackPath: "",
    targetTileSetID: targetTileSetID,
    rebootSent: true,
    completed: false,
    resolution: .awaitingPostflight
  )
  journal.createdAt = targetTileSetID == nil ? "2026-07-16T20:00:00Z" : "2026-07-16T20:01:00Z"
  return journal
}
