import AppKit
import SwiftUI
import VTSCTunerCore

struct MapPreviewInspector: View {
  @ObservedObject var tuner: TunerSession
  @ObservedObject var map: MapPreviewSession
  @State private var tileSourceExpanded = false
  @State private var fitRequestPending = false

  private var remainingSampleCount: Int {
    max(0, MapPreviewSession.minimumCalibrationSamples - map.calibrationSamples.count)
  }

  var body: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 16) {
          workflowCard
          tileSourceCard

          sectionHeader("Curve Draft", systemImage: "speedometer")
            .id("curve-draft")
          if let selection = map.selection {
            SelectedCurveDraftCard(tuner: tuner, map: map, selection: selection)
          } else {
            emptySelectionCard
          }

          bankHeader
          if map.calibrationSamples.isEmpty {
            Text("The bank is empty. A curve appears here only after you choose Add Curve to Bank in the draft above.")
              .font(.body)
              .foregroundStyle(.secondary)
              .padding(12)
              .frame(maxWidth: .infinity, alignment: .leading)
              .background(Color.secondary.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
          } else {
            ForEach(Array(map.calibrationSamples.enumerated()), id: \.element.id) { index, sample in
              CalibrationBankRow(
                number: index + 1,
                sample: sample,
                map: map,
                remove: { map.removeCalibrationSample(id: sample.id) },
                focus: { map.focusCalibrationSample(sample) }
              )
            }
          }

          fitSection
        }
        .padding(16)
      }
      .onChange(of: map.selection?.id) { _, selectedID in
        guard selectedID != nil else { return }
        withAnimation(.easeInOut(duration: 0.2)) {
          proxy.scrollTo("curve-draft", anchor: .top)
        }
      }
      .onChange(of: map.fitResult) { _, result in
        guard result != nil else { return }
        Task { @MainActor in
          await Task.yield()
          withAnimation(.easeInOut(duration: 0.25)) {
            proxy.scrollTo("fit-proposal", anchor: .top)
          }
        }
      }
      .onChange(of: map.fitErrorText) { _, error in
        guard error != nil else { return }
        withAnimation(.easeInOut(duration: 0.2)) {
          proxy.scrollTo("fit-error", anchor: .center)
        }
      }
    }
  }

  private var workflowCard: some View {
    VStack(alignment: .leading, spacing: 7) {
      Label("CALIBRATION WORKFLOW", systemImage: "list.number")
        .font(.callout.weight(.bold))
        .foregroundStyle(.tint)
      Text("1  Select a curve  →  2  Set its target  →  3  Add to Bank  →  4  Fit the bank")
        .font(.body.weight(.semibold))
        .fixedSize(horizontal: false, vertical: true)
      Text("Selecting another road never adds the current draft automatically.")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .padding(12)
    .background(Color.accentColor.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
  }

  private var tileSourceCard: some View {
    DisclosureGroup(isExpanded: $tileSourceExpanded) {
      VStack(alignment: .leading, spacing: 9) {
        Text(map.tileRootURL?.path ?? "No tile folder selected")
          .font(.system(.caption, design: .monospaced))
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
        HStack {
          Button("Choose…", action: map.chooseTileFolder)
          Picker("Tici", selection: $map.ticiProfile) {
            Text("Home").tag("commaHome")
            Text("Car").tag("commaCar")
            Text("USB / ADB").tag("commaAdb")
          }
          .labelsHidden()
          Button("Sync", action: map.syncFromTici).disabled(!map.canSync)
        }
        Text("Sync is explicit. It copies the tici’s actual offline tiles; panning never contacts the car.")
          .font(.callout)
          .foregroundStyle(.secondary)
      }
      .padding(.top, 8)
    } label: {
      Label("Tile Source & Sync", systemImage: "externaldrive")
        .font(.headline)
    }
    .padding(12)
    .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
  }

  private var emptySelectionCard: some View {
    VStack(spacing: 10) {
      Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
        .font(.system(size: 28))
        .foregroundStyle(.secondary)
      Text("Select a Curve")
        .font(.title3.weight(.semibold))
      Text("Click a colored mapd road, then set the speed you want before adding it to the bank.")
        .font(.body)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity, minHeight: 130)
    .padding(12)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
  }

  private var bankHeader: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        sectionHeader("Curve Bank", systemImage: "tray.full")
        Spacer()
        Text("\(map.calibrationSamples.count) SAVED")
          .font(.caption.weight(.bold))
          .padding(.horizontal, 8)
          .padding(.vertical, 4)
          .background(Color.accentColor.opacity(0.18), in: Capsule())
      }
      ProgressView(
        value: Double(min(map.calibrationSamples.count, MapPreviewSession.minimumCalibrationSamples)),
        total: Double(MapPreviewSession.minimumCalibrationSamples)
      )
      if remainingSampleCount > 0 {
        Text("\(map.calibrationSamples.count) saved · \(MapPreviewSession.minimumCalibrationSamples) minimum · no maximum")
          .font(.body.weight(.semibold))
        Text("Add \(remainingSampleCount) more to enable fitting, then keep adding as many curves as you want.")
          .font(.callout)
          .foregroundStyle(.secondary)
      } else {
        Label("Ready to fit — keep adding curves whenever you want", systemImage: "checkmark.circle.fill")
          .font(.body.weight(.semibold))
          .foregroundStyle(.green)
      }
      if map.unresolvedCalibrationCount > 0 {
        Label(
          "\(map.unresolvedCalibrationCount) saved curve\(map.unresolvedCalibrationCount == 1 ? "" : "s") still need a runtime-curvature audit before fitting.",
          systemImage: "exclamationmark.triangle.fill"
        )
        .font(.callout.weight(.semibold))
        .foregroundStyle(.orange)
      } else if !map.calibrationSamples.isEmpty {
        Label(
          String(
            format: "Curvature audited: raw-node peak %.2f → runtime-equivalent %.2f m/s²",
            map.maximumRawRequestedLateralAccelerationMPS2,
            map.maximumRuntimeRequestedLateralAccelerationMPS2
          ),
          systemImage: "waveform.path.ecg"
        )
        .font(.callout.weight(.semibold))
        .foregroundStyle(.cyan)
      }
      Text("This is one persistent bank with no rollover. Every saved curve is used in the next fit; the exact same map point updates its existing item. Numbered cards match the map pins.")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
  }

  private var fitSection: some View {
    VStack(alignment: .leading, spacing: 10) {
      sectionHeader("Fit the Entire Bank", systemImage: "slider.horizontal.3")
      Label(
        "This generates a tune proposal only. It does not change source, push Git, rebuild tiles, or contact the car.",
        systemImage: "checkmark.shield.fill"
      )
      .font(.callout.weight(.medium))
      .foregroundStyle(.secondary)

      Button {
        requestFit()
      } label: {
        Group {
          if fitRequestPending || map.isFitting {
            HStack { ProgressView().controlSize(.small); Text("Generating Tune Proposal…") }
          } else if remainingSampleCount > 0 {
            Text("Add \(remainingSampleCount) More Curve\(remainingSampleCount == 1 ? "" : "s") to Fit")
          } else if map.unresolvedCalibrationCount > 0 {
            Text("Audit \(map.unresolvedCalibrationCount) Curve\(map.unresolvedCalibrationCount == 1 ? "" : "s") Before Fitting")
          } else if !map.canRunFit {
            Text("Fix Invalid Target Speeds")
          } else if map.fitResult != nil {
            Label(
              "Regenerate Proposal from All \(map.calibrationSamples.count) Curves",
              systemImage: "arrow.clockwise"
            )
          } else {
            Text("Generate Proposal from All \(map.calibrationSamples.count) Curves")
          }
        }
        .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .controlSize(.large)
      .disabled(!map.canRunFit || fitRequestPending)

      if let error = map.fitErrorText {
        Label(error, systemImage: "exclamationmark.triangle.fill")
          .font(.callout)
          .foregroundStyle(.red)
          .id("fit-error")
      }
      if let result = map.fitResult {
        FitProposalView(result: result) {
          map.acceptFit(
            result,
            currentKnobs: tuner.knobs,
            currentAnchorKnobs: tuner.calibrationAnchorKnobs,
            currentBands: tuner.bands,
            action: tuner.acceptFittedTune
          )
        }
        .id("fit-proposal")
      }
      Text("The proposal refits the four-knob base sigmoid, then generates bounded residual Q bands so the complete exported curve can reach the saved targets. Conflicting targets at effectively identical curvatures remain a reported compromise. Predictions use checked-in source defaults, not live device Params.")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
  }

  private func requestFit() {
    guard map.canRunFit, !fitRequestPending else { return }
    fitRequestPending = true
    let knobs = tuner.knobs
    let anchorKnobs = tuner.ensureCalibrationAnchorForFit()
    let bands = tuner.bands

    // Commit any target still being edited before the fit snapshots the bank.
    // The next run-loop turn also gives the button a visible pressed/loading state.
    if NSApp.keyWindow?.makeFirstResponder(nil) == false {
      fitRequestPending = false
      map.reportUncommittedTargetEditor()
      return
    }
    Task { @MainActor in
      await Task.yield()
      map.runFit(currentKnobs: knobs, anchorKnobs: anchorKnobs, bands: bands)
      fitRequestPending = false
    }
  }

  private func sectionHeader(_ text: String, systemImage: String) -> some View {
    Label(text.uppercased(), systemImage: systemImage)
      .font(.callout.weight(.bold))
      .foregroundStyle(.secondary)
  }
}

struct SelectedCurveDraftCard: View {
  @ObservedObject var tuner: TunerSession
  @ObservedObject var map: MapPreviewSession
  let selection: MapRoadSelection
  @State private var technicalDetailsExpanded = false

  private var node: MapRenderedNode { selection.node }
  private var bakedMPH: Double? { node.bakedSpeedMPS?.mapMPH }
  private var proposedMPH: Double { node.proposedSpeedMPS.mapMPH }
  private var rawToRuntimeRatio: Double? {
    guard node.curvature > 1.0e-9 else { return nil }
    return node.rawCurvature / node.curvature
  }
  private var targetRuntimeAcceleration: Double {
    let speed = map.draftDesiredSpeedMPH * VTSCMath.mphToMetersPerSecond
    return node.curvature * speed * speed
  }
  private var targetRawAcceleration: Double {
    let speed = map.draftDesiredSpeedMPH * VTSCMath.mphToMetersPerSecond
    return node.rawCurvature * speed * speed
  }
  private var effectiveMPH: Double {
    SigmoidFitter.predictedSpeedMPH(
      parameters: tuner.parameters,
      curvature: node.curvature,
      bands: tuner.bands,
      mode: .effectiveStrategic,
      modifiers: .sourceDefaults
    )
  }
  private var desiredSpeed: Binding<Double> {
    Binding(get: { map.draftDesiredSpeedMPH }, set: map.setDraftDesiredSpeedMPH)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 13) {
      HStack {
        VStack(alignment: .leading, spacing: 2) {
          Text(selection.way.displayName)
            .font(.title2.weight(.bold))
          if !selection.way.reference.isEmpty,
             selection.way.reference.caseInsensitiveCompare(selection.way.displayName) != .orderedSame {
            Text(selection.way.reference)
              .font(.body.weight(.medium))
              .foregroundStyle(.secondary)
          }
        }
        Spacer()
        Button { map.clearSelection() } label: { Image(systemName: "xmark.circle.fill") }
          .buttonStyle(.plain)
          .foregroundStyle(.secondary)
      }

      HStack(spacing: 10) {
        heroMetric("MODELED EFFECTIVE", String(format: "%.1f mph", effectiveMPH), color: .cyan)
        heroMetric(
          "STORED TILE BAKE",
          bakedMPH.map { String(format: "%.1f mph", $0) } ?? "Unavailable",
          color: .secondary
        )
      }

      VStack(alignment: .leading, spacing: 6) {
        HStack(spacing: 14) {
          Label(
            node.curvature > 1.0e-9 ? String(format: "%.0f m runtime radius", 1 / node.curvature) : "Straight",
            systemImage: "arrow.turn.up.right"
          )
          Text(String(format: "κ %.5f", node.curvature))
            .font(.system(.callout, design: .monospaced))
        }
        if node.curvatureContextComplete {
          Text(String(
            format: "mapd-equivalent three-triplet average · %.0f m five-node span · target %.2f m/s²",
            node.curvatureSupportMeters,
            targetRuntimeAcceleration
          ))
          .font(.callout)
          .foregroundStyle(.secondary)
        } else {
          Label("Adjacent-way context is incomplete; this point cannot be banked.", systemImage: "exclamationmark.triangle.fill")
            .font(.callout.weight(.semibold))
            .foregroundStyle(.orange)
        }
        if let ratio = rawToRuntimeRatio, ratio >= 1.25 {
          Label(
            String(
              format: "Raw OSM vertex κ %.5f is %.2f× higher (target %.2f m/s²); it is diagnostic only.",
              node.rawCurvature,
              ratio,
              targetRawAcceleration
            ),
            systemImage: "exclamationmark.triangle.fill"
          )
          .font(.callout.weight(.semibold))
          .foregroundStyle(.orange)
        }
      }

      targetEditor
      technicalDetails
    }
    .padding(13)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
  }

  private var targetEditor: some View {
    VStack(alignment: .leading, spacing: 9) {
      HStack {
        Text("TARGET FOR THIS CURVE")
          .font(.callout.weight(.bold))
        Spacer()
        Label(
          map.selectionDraftBadgeText,
          systemImage: map.selectionHasUncommittedTarget
            ? "exclamationmark.circle.fill"
            : (map.selectionIsQueued ? "checkmark.circle.fill" : "pencil.circle")
        )
        .font(.caption.weight(.bold))
        .foregroundStyle(
          map.selectionHasUncommittedTarget || !map.selectionIsQueued ? .orange : .green
        )
      }
      Text("Change to")
        .font(.body.weight(.semibold))
      HStack(alignment: .firstTextBaseline, spacing: 8) {
        TextField(
          "Target mph",
          value: desiredSpeed,
          format: .number.precision(.fractionLength(1))
        )
        .font(.system(size: 25, weight: .bold, design: .rounded))
        .frame(width: 120)
        .multilineTextAlignment(.trailing)
        .disabled(map.isStudyCurveCaptureActive && map.selectionIsQueued)
        Text("mph")
          .font(.title3.weight(.semibold))
          .foregroundStyle(.secondary)
        Spacer()
      }
      Button(action: map.queueSelectedCurve) {
        Label(
          map.selectionQueueActionTitle,
          systemImage: map.selectionIsQueued ? "arrow.triangle.2.circlepath" : "tray.and.arrow.down.fill"
        )
        .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .controlSize(.large)
      .disabled(!map.canQueueSelection)
      .help(map.selectionQueueHelp)
      Text(
        map.isStudyCurveCaptureActive && map.selectionIsQueued
          ? "This curve is already banked. Choose another road, or switch to Calibration to edit it."
          : (map.selectionIsQueued
              ? (map.selectionHasUncommittedTarget
                  ? "This change is not in the bank yet. Press Update before selecting another curve."
                  : "This edits the existing numbered bank item; it never creates a duplicate.")
              : "This speed is only a draft until you add it to the bank.")
      )
      .font(.callout)
      .foregroundStyle(.secondary)
    }
    .padding(12)
    .background(Color.accentColor.opacity(0.11), in: RoundedRectangle(cornerRadius: 10))
  }

  private var technicalDetails: some View {
    DisclosureGroup("Technical mapd details", isExpanded: $technicalDetailsExpanded) {
      VStack(alignment: .leading, spacing: 7) {
        technicalMetric("Proposed tile bake", String(format: "%.1f mph", proposedMPH))
        technicalMetric("Delta", bakedMPH.map { String(format: "%+.1f mph", proposedMPH - $0) } ?? "Unavailable")
        technicalMetric("Runtime-smoothed κ", String(format: "%.8f", node.curvature))
        technicalMetric("Raw vertex κ", String(format: "%.8f", node.rawCurvature))
        technicalMetric("Five-node route span", String(format: "%.1f m", node.curvatureSupportMeters))
        technicalMetric("Raw / runtime", rawToRuntimeRatio.map { String(format: "%.2f×", $0) } ?? "Unavailable")
        technicalMetric("Tile hash", selection.way.tileHash.isEmpty ? "Legacy / none" : selection.way.tileHash)
        technicalMetric("Current hash", MapBakeMath.sigmoidHash(parameters: tuner.parameters))
        technicalMetric("Schema", String(selection.way.schemaVersion))
        technicalMetric("Lanes / one-way", "\(selection.way.lanes) / \(selection.way.oneWay ? "yes" : "no")")
        technicalMetric(
          "Winding F / B",
          "L\(selection.way.windingForwardLevel) S\(selection.way.windingForwardScore) C\(selection.way.windingForwardConfidence) · L\(selection.way.windingBackwardLevel) S\(selection.way.windingBackwardScore) C\(selection.way.windingBackwardConfidence)"
        )
        Divider()
        Text("This is mapd’s per-node physics bake. Route matching, adjacent-way smoothing, runtime modifiers, deceleration timing, calibration, and planner policy can change the final command in the car.")
          .font(.callout)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(.top, 7)
    }
    .font(.callout.weight(.semibold))
  }

  private func heroMetric(_ label: String, _ value: String, color: Color) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(label)
        .font(.caption.weight(.bold))
        .foregroundStyle(.secondary)
      Text(value)
        .font(.system(size: 20, weight: .bold, design: .rounded))
        .foregroundStyle(color)
        .minimumScaleFactor(0.8)
        .lineLimit(1)
    }
    .padding(10)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 9))
  }

  private func technicalMetric(_ label: String, _ value: String) -> some View {
    LabeledContent(label) {
      Text(value)
        .font(.system(.caption, design: .monospaced))
        .multilineTextAlignment(.trailing)
        .textSelection(.enabled)
    }
    .font(.caption)
  }
}

private struct CalibrationBankRow: View {
  let number: Int
  let sample: MapCalibrationSample
  @ObservedObject var map: MapPreviewSession
  let remove: () -> Void
  let focus: () -> Void

  private var targetBinding: Binding<Double> {
    Binding(
      get: { sample.desiredSpeedMPH },
      set: { map.updateCalibrationTarget(id: sample.id, desiredSpeedMPH: $0) }
    )
  }
  private var runtimeTargetAcceleration: Double {
    let speed = sample.desiredSpeedMPH * VTSCMath.mphToMetersPerSecond
    return sample.curvature * speed * speed
  }
  private var rawTargetAcceleration: Double? {
    guard let raw = sample.rawCurvature else { return nil }
    let speed = sample.desiredSpeedMPH * VTSCMath.mphToMetersPerSecond
    return raw * speed * speed
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 9) {
      HStack {
        Text("\(number)")
          .font(.headline.weight(.bold))
          .foregroundStyle(.white)
          .frame(width: 28, height: 28)
          .background(Color.purple, in: Circle())
        VStack(alignment: .leading, spacing: 1) {
          Text(sample.roadName)
            .font(.headline)
            .lineLimit(1)
          if !sample.reference.isEmpty,
             sample.reference.caseInsensitiveCompare(sample.roadName) != .orderedSame {
            Text(sample.reference)
              .font(.callout)
              .foregroundStyle(.secondary)
          }
        }
        Spacer()
        Button(action: focus) { Image(systemName: "scope") }
          .buttonStyle(.borderless)
          .help("Center bank item #\(number) on the map")
        Button(role: .destructive, action: remove) { Image(systemName: "trash") }
          .buttonStyle(.borderless)
          .help("Remove bank item #\(number)")
      }

      HStack(alignment: .firstTextBaseline) {
        VStack(alignment: .leading, spacing: 3) {
          Text("TARGET SPEED")
            .font(.caption.weight(.bold))
            .foregroundStyle(.secondary)
          HStack(alignment: .firstTextBaseline, spacing: 5) {
            TextField(
              "Target mph",
              value: targetBinding,
              format: .number.precision(.fractionLength(1))
            )
            .font(.system(size: 20, weight: .bold, design: .rounded))
            .frame(width: 92)
            .multilineTextAlignment(.trailing)
            Text("mph")
              .font(.body.weight(.medium))
              .foregroundStyle(.secondary)
          }
        }
        Spacer()
        VStack(alignment: .trailing, spacing: 3) {
          Text("MODELED")
            .font(.caption.weight(.bold))
            .foregroundStyle(.secondary)
          Text(String(format: "%.1f mph", sample.effectiveSpeedMPH))
            .font(.body.weight(.semibold))
        }
      }
      if sample.hasCurrentCurvatureEstimate {
        Text(String(
          format: "runtime κ %.5f · %.0f m route span · target %.2f m/s²",
          sample.curvature,
          sample.curvatureSupportMeters ?? 0,
          runtimeTargetAcceleration
        ))
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.secondary)
        if let ratio = sample.rawToEffectiveCurvatureRatio,
           let rawAcceleration = rawTargetAcceleration,
           ratio >= 1.25 {
          Label(
            String(
              format: "Raw vertex %.2f× higher (%.2f m/s²); raw value excluded, runtime value fitted",
              ratio,
              rawAcceleration
            ),
            systemImage: "exclamationmark.triangle.fill"
          )
          .font(.caption.weight(.semibold))
          .foregroundStyle(.orange)
        }
      } else {
        Label(
          sample.curvatureEstimatorVersion == MapRuntimeCurvatureResolver.estimatorVersion
            ? "Route geometry is missing or ambiguous — excluded"
            : "Legacy raw curvature — awaiting route-context audit",
          systemImage: "hourglass"
        )
          .font(.caption.weight(.semibold))
          .foregroundStyle(.orange)
      }
      if !sample.desiredSpeedMPH.isFinite || sample.desiredSpeedMPH <= 0 {
        Label("Invalid target — enter a speed above 0 mph", systemImage: "exclamationmark.triangle.fill")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.red)
      } else if !sample.hasCurrentCurvatureEstimate {
        Label("Saved, but excluded from fitting until curvature is audited", systemImage: "exclamationmark.triangle.fill")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.orange)
      } else if let persistenceError = map.calibrationPersistenceError {
        Label("Not saved on this Mac — \(persistenceError)", systemImage: "exclamationmark.icloud.fill")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.red)
          .fixedSize(horizontal: false, vertical: true)
      } else {
        Label("Banked and saved on this Mac", systemImage: "checkmark.circle.fill")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.green)
      }
    }
    .padding(11)
    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
  }
}

private struct FitProposalView: View {
  let result: SigmoidFitResult
  let accept: () -> Void
  @State private var detailsExpanded = false
  @State private var safetyAcknowledged = false
  @State private var acceptanceArmed = false

  private var requiresSafetyAcknowledgement: Bool {
    result.maximumProposedLateralAccelerationMPS2
      > SigmoidFitter.effectiveAccelerationWarningMPS2 + 1.0e-6
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Label("TUNE PROPOSAL — NOT YET ACTIVE", systemImage: "slider.horizontal.3")
        .font(.headline)
        .foregroundStyle(.tint)
      HStack(spacing: 16) {
        fitMetric("Before RMSE", result.beforeRMSEMPH)
        Image(systemName: "arrow.right").foregroundStyle(.secondary)
        fitMetric("Proposed RMSE", result.afterRMSEMPH)
      }
      HStack(spacing: 16) {
        fitMetric("Worst miss", result.maximumAbsoluteErrorMPH)
        fitMetric("Peak accel", result.maximumProposedLateralAccelerationMPS2, unit: "m/s²")
      }
      Label(
        "Four base knobs + \(result.bands.count) bounded residual band\(result.bands.count == 1 ? "" : "s")",
        systemImage: "waveform.path.ecg"
      )
      .font(.callout.weight(.semibold))
      .foregroundStyle(.secondary)
      Text("The sigmoid and residual may reshape the saved bank through a smooth four-sigma influence collar. Outside that collar, both the backbone and complete runtime curve stay within \(SigmoidFitter.maximumOffBankBackboneDeltaMPH.formatted(.number.precision(.fractionLength(0)))) mph of the bank’s persisted checkout anchor. ‘Envelope’ warnings are pointwise bounds; conflicts and the monotonic-shape constraint can still leave a visible residual.")
        .font(.caption)
        .foregroundStyle(.secondary)
      ForEach(Array(result.warnings.enumerated()), id: \.offset) { _, warning in
        Label(warningText(warning), systemImage: "exclamationmark.triangle")
          .font(.callout)
          .foregroundStyle(.orange)
      }
      DisclosureGroup("Per-curve fit details", isExpanded: $detailsExpanded) {
        VStack(alignment: .leading, spacing: 9) {
          ForEach(result.diagnostics, id: \.id) { diagnostic in
            VStack(alignment: .leading, spacing: 3) {
              Text(diagnostic.label)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
              Text(String(
                format: "target %.1f · before %.1f (%+.1f) · proposal %.1f (%+.1f) mph",
                diagnostic.desiredSpeedMPH,
                diagnostic.beforeSpeedMPH,
                diagnostic.beforeErrorMPH,
                diagnostic.afterSpeedMPH,
                diagnostic.afterErrorMPH
              ))
              .font(.system(.caption, design: .monospaced))
              .foregroundStyle(diagnostic.isImpossibleTarget ? .orange : .secondary)
              Text(String(
                format: "implied lateral accel: target %.2f · proposal %.2f m/s²",
                diagnostic.desiredLateralAccelerationMPS2,
                diagnostic.afterLateralAccelerationMPS2
              ))
              .font(.system(.caption, design: .monospaced))
              .foregroundStyle(.secondary)
              if diagnostic.isImpossibleTarget {
                Text(String(
                  format: "Outside the complete runtime-curve envelope %.1f–%.1f mph; the fit used %.1f mph.",
                  diagnostic.attainableSpeedRangeMPH.lowerBound,
                  diagnostic.attainableSpeedRangeMPH.upperBound,
                  diagnostic.fittedTargetSpeedMPH
                ))
                .font(.caption)
                .foregroundStyle(.orange)
              }
            }
          }
        }
        .padding(.top, 7)
      }
      .font(.callout.weight(.semibold))

      if requiresSafetyAcknowledgement {
        Toggle(
          "I understand this proposed runtime curve exceeds the 5.5 m/s² safety-review threshold",
          isOn: $safetyAcknowledged
        )
        .font(.callout.weight(.semibold))
        .toggleStyle(.checkbox)
      }
      Button("Use This Tune in Curve Lab") {
        guard acceptanceArmed else { return }
        accept()
      }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .frame(maxWidth: .infinity)
        .disabled(
          !acceptanceArmed
            || (requiresSafetyAcknowledgement && !safetyAcknowledged)
        )
      Text("Updates the four base knobs and replaces Curve Lab’s EQ bands with these \(result.bands.count) generated residual bands as one undoable edit. It does not edit source, push, rebuild tiles, or contact the car; the toolbar Apply action remains separate.")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .padding(12)
    .background(Color.accentColor.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
    .task(id: result) {
      // A Release fit can finish while the Generate button's mouse is still
      // down. The proposal then scrolls under that pointer; without an explicit
      // release boundary, SwiftUI can deliver the same mouse-up to this newly
      // inserted button and accept a tune that was meant to be review-only.
      // Arm only after every initiating primary press has ended and a fresh
      // event cycle has begun.
      acceptanceArmed = false
      safetyAcknowledged = false
      while (NSEvent.pressedMouseButtons & 1) != 0, !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 16_000_000)
      }
      try? await Task.sleep(nanoseconds: 100_000_000)
      guard !Task.isCancelled else { return }
      acceptanceArmed = true
    }
  }

  private func fitMetric(_ label: String, _ value: Double, unit: String = "mph") -> some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(label).font(.caption.weight(.bold)).foregroundStyle(.secondary)
      Text(String(format: "%.2f %@", value, unit))
        .font(.system(size: 19, weight: .bold, design: .rounded))
    }
  }

  private func warningText(_ warning: SigmoidFitWarning) -> String {
    switch warning {
    case let .underconstrained(clusterCount):
      return "Only \(clusterCount) distinct curvature clusters constrain the complete curve; add more varied curves."
    case let .conflictingCurvatures(labels, spread):
      return "Similar curves request conflicting speeds (\(labels.joined(separator: ", "))), spread \(spread.formatted(.number.precision(.fractionLength(1)))) mph."
    case let .impossibleTarget(label, range):
      return "\(label) is outside the complete runtime-curve envelope \(range.lowerBound.formatted(.number.precision(.fractionLength(1))))–\(range.upperBound.formatted(.number.precision(.fractionLength(1)))) mph."
    case let .effectiveAccelerationAboveBaseRail(labels, peak, rail):
      let scope = labels.isEmpty
        ? "The proposed runtime curve"
        : "\(labels.count) saved target\(labels.count == 1 ? "" : "s") and/or the proposal"
      let verb = labels.isEmpty ? "exceeds" : "exceed"
      return "\(scope) \(verb) the \(rail.formatted(.number.precision(.fractionLength(1)))) m/s² safety-review threshold; peak request/proposal is \(peak.formatted(.number.precision(.fractionLength(2)))) m/s². A proposal above the threshold requires explicit acknowledgement before acceptance."
    }
  }
}
