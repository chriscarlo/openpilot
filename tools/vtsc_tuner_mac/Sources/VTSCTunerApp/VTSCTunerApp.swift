import SwiftUI
import VTSCTunerCore

@main
struct VTSCTunerApp: App {
  @StateObject private var session = TunerSession()

  var body: some Scene {
    WindowGroup("VTSC Sigmoid Tuner") {
      RootView(session: session)
        .frame(minWidth: 900, minHeight: 620)
    }
    .defaultSize(width: 1_300, height: 850)
    .commands {
      CommandGroup(replacing: .undoRedo) {
        Button("Undo") { session.undo() }
          .keyboardShortcut("z", modifiers: .command)
          .disabled(session.workspace != .curveLab || !session.canUndo)
        Button("Redo") { session.redo() }
          .keyboardShortcut("z", modifiers: [.command, .shift])
          .disabled(session.workspace != .curveLab || !session.canRedo)
      }
      CommandGroup(after: .newItem) {
        Button("Choose Chauffeur Repository…") { session.chooseRepository() }
          .keyboardShortcut("o", modifiers: [.command, .shift])
        Button("Choose mapd Tile Folder…") {
          session.workspace = .mapPreview
          session.mapPreview.chooseTileFolder()
        }
        if session.workspace == .curveLab {
          Button("Save Current Tune") { session.saveTune() }
            .keyboardShortcut("s", modifiers: .command)
          Button("Load Saved Tune") { session.loadTune() }
            .keyboardShortcut("l", modifiers: .command)
        }
      }
      CommandMenu("Workspace") {
        Button("Curve Lab") { session.workspace = .curveLab }
          .keyboardShortcut("1", modifiers: .command)
        Button("Map Preview") { session.workspace = .mapPreview }
          .keyboardShortcut("2", modifiers: .command)
        if session.workspace != .mapPreview || session.mapPreview.purpose == .calibration {
          Divider()
          Button("Sync Actual Tiles from tici") {
            session.workspace = .mapPreview
            session.mapPreview.syncFromTici()
          }
          .disabled(!session.mapPreview.canSync)
        }
      }
      CommandMenu("Tune") {
        if session.workspace == .curveLab {
          Button("Revert to Checkout Baseline") { session.revertToCheckoutBaseline() }
          Divider()
          Button("Save or Send Tune…") { session.showApplyActionChooser() }
          Divider()
          Menu("Deployment Tasks") {
            Button(ResumePostflightAction.label) { session.pendingResumePostflight = true }
              .disabled(!session.hasRuntimeOnlyPendingDeployment)
            Button(AbortPendingDeploymentAction.label) { session.pendingAbortPendingDeployment = true }
              .disabled(session.pendingDeployments.isEmpty)
            Button(RollbackRecoveryAction.label) { session.pendingRollbackRecovery = true }
              .disabled(!session.hasRecoverableRollback)
            Divider()
            Button("Why Is Map Tile Rebuilding Unavailable?") {
              session.tileDeploymentInfoVisible = true
            }
          }
        } else {
          Text("Whole-curve study is read-only")
        }
      }
    }
  }
}
