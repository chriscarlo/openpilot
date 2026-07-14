import AppKit
import Foundation
import SwiftUI
import VTSCTunerCore

enum BuiltInHandle: Equatable {
  case minimumRail, maximumRail, inflection, leftWing, rightWing
}

enum DragTarget: Equatable {
  case none
  case handle(BuiltInHandle)
  case band(UUID)
}

enum PlotSelection: Equatable {
  case handle(BuiltInHandle)
  case band(UUID)
}

struct PlotMenuContext: Equatable {
  var point: CGPoint
  var speedMPH: Double
  var acceleration: Double
}

struct PlotUIState: Equatable {
  var hoverPoint: CGPoint?
  var hoverReadout: (speedMPH: Double, acceleration: Double)?
  var hoveredTarget: DragTarget = .none
  var selected: PlotSelection?
  var activeDrag: DragTarget = .none
  var speedDomainMaximumDuringDrag: Double?
  var contextMenu: PlotMenuContext?

  static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.hoverPoint == rhs.hoverPoint
      && lhs.hoverReadout?.speedMPH == rhs.hoverReadout?.speedMPH
      && lhs.hoverReadout?.acceleration == rhs.hoverReadout?.acceleration
      && lhs.hoveredTarget == rhs.hoveredTarget
      && lhs.selected == rhs.selected
      && lhs.activeDrag == rhs.activeDrag
      && lhs.speedDomainMaximumDuringDrag == rhs.speedDomainMaximumDuringDrag
      && lhs.contextMenu == rhs.contextMenu
  }
}

extension ApplyAction: Identifiable {
  public var id: String { label }
}

@MainActor
final class TunerSession: ObservableObject {
  static let repositoryDefaultsKey = "VTSCTuner.repositoryPath"

  @Published var knobs: PlainKnobs
  @Published var bands: [EQBand]
  @Published var workspace: TunerWorkspace = .curveLab
  @Published var plot = PlotUIState()
  @Published var advancedVisible = false
  @Published var inspectorVisible = true
  @Published var statusText = "Ready. Drag curve anchors; Shift-click the plot to add a band."
  @Published var statusIsError = false
  @Published var repositoryURL: URL?
  @Published var pendingApplyAction: ApplyAction?
  @Published var runningApplyAction: ApplyAction?
  @Published var applySteps: [ApplyStepEvent] = []
  @Published var applySucceeded: Bool?

  let mapPreview = MapPreviewSession()

  private(set) var checkoutBaseline: SigmoidParameters

  private var historyPast: [EditableSnapshot] = []
  private var historyFuture: [EditableSnapshot] = []
  private var continuousStart: EditableSnapshot?
  private let historyLimit = 200
  private var applyTask: Task<Void, Never>?

  init() {
    let saved = UserDefaults.standard.string(forKey: Self.repositoryDefaultsKey)
    let detectedRepository = RepositoryLocator.detect(savedPath: saved)
    // Reading a checkout under ~/Documents can trigger macOS privacy access.
    // Never perform that read while SwiftUI is still constructing its first
    // window: a fresh ad-hoc signature otherwise has no window on which to
    // present the access prompt and the app appears hung at launch.
    checkoutBaseline = .checkoutFallback
    let savedTune = try? TuneStore.load()
    let tune = savedTune ?? Tune(params: .checkoutFallback)
    knobs = tune.knobs ?? VTSCMath.knobs(from: tune.params)
    bands = tune.bands
    repositoryURL = detectedRepository
    if let detectedRepository {
      let shouldAdoptBaseline = savedTune == nil
      Task { [weak self] in
        let baseline = await Task.detached(priority: .userInitiated) {
          try? SourcePatcher.readParameters(from: detectedRepository)
        }.value
        guard let self, let baseline else { return }
        checkoutBaseline = baseline
        if shouldAdoptBaseline {
          knobs = VTSCMath.knobs(from: baseline)
          bands = []
          mapPreview.updateProposal(parameters: parameters, bands: bands)
        }
      }
    }
  }

  var parameters: SigmoidParameters {
    VTSCMath.sourceRoundedParameters(VTSCMath.parameters(from: knobs))
  }
  var snapshot: EditableSnapshot { EditableSnapshot(knobs: knobs, bands: bands) }
  var canUndo: Bool { !historyPast.isEmpty }
  var canRedo: Bool { !historyFuture.isEmpty }
  var repositoryName: String { repositoryURL?.lastPathComponent ?? "Choose repository" }
  var calibrationAnchorKnobs: PlainKnobs {
    mapPreview.calibrationAnchorKnobs ?? VTSCMath.knobs(from: checkoutBaseline)
  }

  func ensureCalibrationAnchorForFit() -> PlainKnobs {
    mapPreview.ensureCalibrationAnchor(VTSCMath.knobs(from: checkoutBaseline))
  }

  func setKnob(_ keyPath: WritableKeyPath<PlainKnobs, Double>, to value: Double) {
    knobs[keyPath: keyPath] = value
  }

  func beginContinuousEdit() {
    if continuousStart == nil { continuousStart = snapshot }
  }

  func endContinuousEdit() {
    guard let start = continuousStart else { return }
    continuousStart = nil
    record(previous: start)
  }

  func performDiscreteEdit(_ edit: () -> Void) {
    let previous = snapshot
    edit()
    record(previous: previous)
  }

  private func record(previous: EditableSnapshot) {
    guard previous != snapshot else { return }
    historyPast.append(previous)
    if historyPast.count > historyLimit { historyPast.removeFirst(historyPast.count - historyLimit) }
    historyFuture.removeAll()
    objectWillChange.send()
  }

  func undo() {
    guard workspace == .curveLab else { return }
    guard let previous = historyPast.popLast() else { return }
    historyFuture.append(snapshot)
    restore(previous)
    status("Undid last change.")
  }

  func redo() {
    guard workspace == .curveLab else { return }
    guard let next = historyFuture.popLast() else { return }
    historyPast.append(snapshot)
    restore(next)
    status("Redid change.")
  }

  private func restore(_ state: EditableSnapshot) {
    knobs = state.knobs
    bands = state.bands
    if case let .band(id) = plot.selected, !bands.contains(where: { $0.id == id }) { plot.selected = nil }
  }

  func addBand(at speedMPH: Double) {
    performDiscreteEdit {
      let band = EQBand(
        centerSpeedMPH: CurvePlotSpeedDomain.clampedBandCenterSpeedMPH(speedMPH),
        q: 1.5
      )
      bands.append(band)
      plot.selected = .band(band.id)
    }
    let addedSpeed = bands.last?.centerSpeedMPH ?? speedMPH
    status("Added an EQ band at \(addedSpeed.formatted(.number.precision(.fractionLength(1)))) mph.")
  }

  func removeBand(id: UUID) {
    performDiscreteEdit { bands.removeAll { $0.id == id } }
    if plot.selected == .band(id) { plot.selected = nil }
    status("Band removed.")
  }

  func updateBand(id: UUID, _ update: (inout EQBand) -> Void) {
    guard let index = bands.firstIndex(where: { $0.id == id }) else { return }
    update(&bands[index])
  }

  func revertToCheckoutBaseline() {
    guard workspace == .curveLab else { return }
    performDiscreteEdit {
      knobs = VTSCMath.knobs(from: checkoutBaseline)
      bands.removeAll()
      plot.selected = nil
    }
    status("Reverted to the selected checkout baseline. Undo restores the previous tune.")
  }

  func acceptFittedTune(_ result: SigmoidFitResult) {
    performDiscreteEdit {
      knobs = result.knobs
      bands = result.bands
      if case let .band(id) = plot.selected,
         !bands.contains(where: { $0.id == id }) {
        plot.selected = nil
      }
    }
    workspace = .curveLab
    status(String(
      format: "Accepted complete fitted curve with %d residual bands (effective-target RMSE %.2f → %.2f mph). Review it, then choose Apply when ready.",
      result.bands.count,
      result.beforeRMSEMPH,
      result.afterRMSEMPH
    ))
  }

  func loadTune() {
    guard workspace == .curveLab else { return }
    do {
      let tune = try TuneStore.load()
      performDiscreteEdit {
        knobs = tune.knobs ?? VTSCMath.knobs(from: tune.params)
        bands = tune.bands
      }
      status("Loaded the saved tune.")
    } catch {
      status("Load failed: \(error.localizedDescription)", error: true)
    }
  }

  func saveTune() {
    guard workspace == .curveLab else { return }
    do {
      try TuneStore.save(Tune(params: parameters, bands: bands, knobs: knobs))
      status("Saved the current tune without changing source or contacting the car.")
    } catch {
      status("Save failed: \(error.localizedDescription)", error: true)
    }
  }

  func chooseRepository() {
    let panel = NSOpenPanel()
    panel.title = "Choose the Chauffeur repository"
    panel.prompt = "Choose Repository"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    panel.directoryURL = repositoryURL ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents")
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do {
      try RepositoryLocator.validate(url)
      checkoutBaseline = try SourcePatcher.readParameters(from: url)
      mapPreview.invalidateFit()
      repositoryURL = url
      UserDefaults.standard.set(url.path, forKey: Self.repositoryDefaultsKey)
      status("Using Chauffeur checkout at \(url.path).")
    } catch {
      status(error.localizedDescription, error: true)
    }
  }

  func status(_ text: String, error: Bool = false) {
    statusText = text
    statusIsError = error
  }

  func confirmApply() {
    guard workspace == .curveLab else {
      pendingApplyAction = nil
      status("Apply is only available in Curve Lab.", error: true)
      return
    }
    guard let action = pendingApplyAction else { return }
    guard let repositoryURL else {
      pendingApplyAction = nil
      status("Choose a Chauffeur repository before applying.", error: true)
      return
    }
    let request = ApplyRequest(
      action: action,
      tune: Tune(params: parameters, bands: bands, knobs: knobs),
      repositoryRoot: repositoryURL
    )
    pendingApplyAction = nil
    runningApplyAction = action
    applySteps = []
    applySucceeded = nil
    applyTask?.cancel()
    let pipeline = ApplyPipeline()
    applyTask = Task { [weak self] in
      for await event in pipeline.events(for: request) {
        guard let self else { return }
        switch event {
        case let .step(step): self.upsertStep(step)
        case let .finished(success):
          self.applySucceeded = success
          self.status(success ? "\(action.label) succeeded." : "\(action.label) failed.", error: !success)
        }
      }
    }
  }

  func cancelApply() {
    applyTask?.cancel()
    applyTask = nil
    applySucceeded = false
    status("Apply cancelled.", error: true)
  }

  func closeApply() {
    applyTask?.cancel()
    applyTask = nil
    runningApplyAction = nil
  }

  private func upsertStep(_ step: ApplyStepEvent) {
    if let index = applySteps.firstIndex(where: { $0.id == step.id }) { applySteps[index] = step }
    else { applySteps.append(step) }
  }
}
