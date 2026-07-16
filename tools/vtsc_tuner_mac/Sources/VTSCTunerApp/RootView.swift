import SwiftUI
import VTSCTunerCore

struct RootView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    workspaceLayout
      .background(Color(red: 0.059, green: 0.071, blue: 0.09))
      .toolbar {
        ToolbarItemGroup(placement: .navigation) {
          if session.workspace == .curveLab {
            Button(action: session.chooseRepository) {
              Label(session.repositoryName, systemImage: "folder")
            }
            .help(session.repositoryURL?.path ?? "Choose the Chauffeur checkout")
          } else {
            Button(action: session.mapPreview.chooseTileFolder) {
              Label(session.mapPreview.tileRootName, systemImage: "map")
            }
            .help(session.mapPreview.tileRootURL?.path ?? "Choose actual mapd offline tiles")
          }
        }
        ToolbarItemGroup(placement: .primaryAction) {
          if session.workspace == .curveLab {
            Button("Load", action: session.loadTune)
            Button("Revert", action: session.revertToCheckoutBaseline)
              .help("Revert to the selected checkout's parsed VTSC baseline. This is undoable.")
            Toggle("Advanced", isOn: $session.advancedVisible)
            Menu("Apply", systemImage: "paperplane") {
              ForEach(ApplyAction.allCases) { action in
                Button(action.label) { session.pendingApplyAction = action }
              }
              Divider()
              Button(ResumePostflightAction.label) { session.pendingResumePostflight = true }
            }
          } else {
            Picker(
              "Map purpose",
              selection: Binding(
                get: { session.mapPreview.purpose },
                set: { session.mapPreview.purpose = $0 }
              )
            ) {
              ForEach(MapPreviewPurpose.allCases) { purpose in
                Text(purpose.rawValue).tag(purpose)
              }
            }
            .frame(width: 235)

            if session.mapPreview.purpose == .calibration
              || session.mapPreview.isStudyCurveCaptureActive
            {
              Picker(
                "Map Colors",
                selection: Binding(
                  get: { session.mapPreview.displayMode },
                  set: { session.mapPreview.displayMode = $0 }
                )
              ) {
                ForEach(MapSpeedDisplayMode.allCases) { mode in Text(mode.rawValue).tag(mode) }
              }
              .frame(width: 190)
              if session.mapPreview.purpose == .calibration {
                Button(action: session.mapPreview.syncFromTici) {
                  Label("Sync from tici", systemImage: "arrow.triangle.2.circlepath")
                }
                .disabled(!session.mapPreview.canSync)
                .help("Explicitly copy the tici's current mapd tiles into this Mac")
              }
            } else {
              Picker(
                "Study Colors",
                selection: Binding(
                  get: { session.mapPreview.wholeCurveDisplayMode },
                  set: { session.mapPreview.wholeCurveDisplayMode = $0 }
                )
              ) {
                ForEach(MapWholeCurveDisplayMode.allCases) { mode in Text(mode.rawValue).tag(mode) }
              }
              .frame(width: 170)
            }
          }
          Button {
            session.inspectorVisible.toggle()
          } label: {
            Label("Controls", systemImage: "sidebar.right")
          }
        }
      }
      .safeAreaInset(edge: .top, spacing: 0) {
        HStack {
          Spacer()
          Picker("Workspace", selection: $session.workspace) {
            ForEach(TunerWorkspace.allCases) { workspace in
              Text(workspace.rawValue).tag(workspace)
            }
          }
          .labelsHidden()
          .pickerStyle(.segmented)
          .frame(width: 250)
          Spacer()
        }
        .frame(height: 34)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
      }
      .safeAreaInset(edge: .bottom, spacing: 0) {
        if session.workspace == .curveLab {
          StatusBarView(session: session)
        } else {
          MapPreviewStatusBar(map: session.mapPreview, parameters: session.parameters)
        }
      }
      .sheet(item: $session.pendingApplyAction) { action in
        ApplyConfirmationView(session: session, action: action)
      }
      .sheet(item: $session.runningApplyAction) { action in
        ApplyProgressView(session: session, action: action)
      }
      .sheet(isPresented: $session.pendingResumePostflight) {
        ResumePostflightConfirmationView(session: session)
      }
      .sheet(isPresented: $session.runningResumePostflight) {
        ResumePostflightProgressView(session: session)
      }
      .onExitCommand {
        if session.workspace == .mapPreview {
          if session.mapPreview.purpose == .wholeCurveStudy {
            if session.mapPreview.isStudyCurveCaptureActive {
              session.mapPreview.cancelStudyCurveCapture()
            } else {
              session.mapPreview.selectWholeCurveEvent(nil)
            }
          } else {
            session.mapPreview.clearSelection()
          }
        } else if session.plot.contextMenu != nil {
          session.plot.contextMenu = nil
        } else {
          session.plot.selected = nil
        }
      }
      .onChange(of: session.workspace) { _, workspace in
        guard workspace != .mapPreview, session.mapPreview.isStudyCurveCaptureActive else { return }
        session.mapPreview.cancelStudyCurveCapture()
      }
  }

  @ViewBuilder
  private var workspaceLayout: some View {
    if session.inspectorVisible {
      HSplitView {
        workspaceContent
          .frame(minWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
          .layoutPriority(1)

        inspectorContent
          .frame(minWidth: 350, idealWidth: 400, maxWidth: 470, maxHeight: .infinity)
          .background(.bar)
      }
    } else {
      workspaceContent
    }
  }

  @ViewBuilder
  private var workspaceContent: some View {
    switch session.workspace {
    case .curveLab:
      CurvePlotView(session: session)
    case .mapPreview:
      MapPreviewView(tuner: session, map: session.mapPreview)
    }
  }

  @ViewBuilder
  private var inspectorContent: some View {
    switch session.workspace {
    case .curveLab:
      CurveControlsInspector(session: session)
    case .mapPreview:
      if session.mapPreview.purpose == .wholeCurveStudy {
        WholeCurveStudyInspector(tuner: session, map: session.mapPreview)
      } else {
        MapPreviewInspector(tuner: session, map: session.mapPreview)
      }
    }
  }
}

struct MapPreviewStatusBar: View {
  @ObservedObject var map: MapPreviewSession
  let parameters: SigmoidParameters

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: map.statusIsError ? "exclamationmark.triangle.fill" : "circle.fill")
        .font(.system(size: 8))
        .foregroundStyle(map.statusIsError ? Color.red : Color.teal)
      Text(map.statusText).font(.caption).lineLimit(1)
      Spacer()
      if map.purpose == .wholeCurveStudy {
        Text(
          map.isStudyCurveCaptureActive
            ? "new curve capture  ·  \(map.calibrationSamples.count) saved"
            : "\(map.wholeCurveEvents.count) directional events  ·  local shadow study"
        )
          .font(.system(.caption, design: .monospaced))
          .foregroundStyle(.secondary)
      } else {
        Text("\(map.ways.count) ways  ·  proposed hash \(MapBakeMath.sigmoidHash(parameters: parameters))")
          .font(.system(.caption, design: .monospaced))
          .foregroundStyle(.secondary)
      }
    }
    .padding(.horizontal, 12)
    .frame(height: 30)
    .background(.bar)
    .overlay(alignment: .top) { Divider() }
  }
}

struct StatusBarView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    HStack(spacing: 12) {
      Image(systemName: session.statusIsError ? "exclamationmark.triangle.fill" : "circle.fill")
        .font(.system(size: 8))
        .foregroundStyle(session.statusIsError ? .red : .teal)
      Text(session.statusText)
        .font(.caption)
        .lineLimit(1)
      Spacer()
      let p = session.parameters
      Text(String(
        format: "A=%.3f  B=%7.0f  C=%.5f  D=%.3f  MIN=%.2f  MAX=%.2f",
        p.a, p.b, p.c, p.d, p.minLat, p.maxLat
      ))
      .font(.system(.caption, design: .monospaced))
      .foregroundStyle(.secondary)
    }
    .padding(.horizontal, 12)
    .frame(height: 30)
    .background(.bar)
    .overlay(alignment: .top) { Divider() }
  }
}

struct CurveControlsInspector: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        inspectorHeader("Curve Shape")
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
          knob(
            "Tight-Curve Ceiling",
            keyPath: \PlainKnobs.tightCurveAcceleration,
            range: VTSCClip.minimumLateralAcceleration,
            defaultValue: 2.448138,
            unit: "m/s²",
            precision: 2
          )
          knob(
            "Straight-Road Ceiling",
            keyPath: \PlainKnobs.straightRoadAcceleration,
            range: VTSCClip.maximumLateralAcceleration,
            defaultValue: 4.107103,
            unit: "m/s²",
            precision: 2
          )
          knob(
            "Transition Speed",
            keyPath: \PlainKnobs.transitionSpeedMPH,
            range: VTSCClip.transitionSpeedMPH,
            defaultValue: 55.127268,
            unit: "mph",
            precision: 1
          )
          knob(
            "Sharpness",
            keyPath: \PlainKnobs.sharpness,
            range: VTSCClip.sharpness,
            defaultValue: 3.815305,
            unit: "",
            precision: 1
          )
        }
        Text("Drag vertically. Hold Shift for fine adjustment. Double-click to reset.")
          .font(.caption)
          .foregroundStyle(.secondary)

        inspectorHeader("Local Shaping (EQ Bands)")
        if session.bands.isEmpty {
          ContentUnavailableView(
            "No EQ Bands",
            systemImage: "waveform.path",
            description: Text("Shift-click the plot, or right-click its curve, to add one.")
          )
          .frame(minHeight: 150)
        } else {
          ForEach(session.bands) { band in
            BandEditorView(session: session, bandID: band.id)
          }
        }

        if session.advancedVisible {
          inspectorHeader("Advanced Raw Parameters")
          AdvancedParametersView(session: session)
        }
      }
      .padding(14)
    }
  }

  @ViewBuilder
  private func knob(
    _ label: String,
    keyPath: WritableKeyPath<PlainKnobs, Double>,
    range: ClosedRange<Double>,
    defaultValue: Double,
    unit: String,
    precision: Int
  ) -> some View {
    RotaryKnob(
      label: label,
      value: Binding(
        get: { session.knobs[keyPath: keyPath] },
        set: { session.setKnob(keyPath, to: $0) }
      ),
      range: range,
      defaultValue: defaultValue,
      unit: unit,
      precision: precision,
      onEditingChanged: { editing in
        if editing { session.beginContinuousEdit() } else { session.endContinuousEdit() }
      }
    )
  }

  private func inspectorHeader(_ text: String) -> some View {
    Text(text.uppercased())
      .font(.caption.weight(.semibold))
      .foregroundStyle(.secondary)
      .frame(maxWidth: .infinity, alignment: .leading)
  }
}

struct BandEditorView: View {
  @ObservedObject var session: TunerSession
  let bandID: UUID

  private var band: EQBand? { session.bands.first { $0.id == bandID } }

  var body: some View {
    if let band {
      VStack(alignment: .leading, spacing: 9) {
        HStack {
          Circle().fill(bandColor).frame(width: 9, height: 9)
          Button("Band @ \(band.centerSpeedMPH.formatted(.number.precision(.fractionLength(1)))) mph") {
            session.plot.selected = session.plot.selected == .band(bandID) ? nil : .band(bandID)
          }
          .buttonStyle(.plain)
          Spacer()
          Toggle("", isOn: enabledBinding).labelsHidden()
          Button(role: .destructive) { session.removeBand(id: bandID) } label: {
            Image(systemName: "xmark")
          }
          .buttonStyle(.borderless)
        }
        HStack(alignment: .top, spacing: 8) {
          smallKnob("Gain", keyPath: \EQBand.gainDB, range: -12 ... 12, defaultValue: 0, unit: "dB", precision: 1)
          smallKnob("Width (Q)", keyPath: \EQBand.q, range: 0.3 ... 16, defaultValue: 1.5, unit: "", precision: 2, logarithmic: true)
          smallKnob(
            "Center",
            keyPath: \EQBand.centerSpeedMPH,
            range: CurvePlotSpeedDomain.minimumBandCenterSpeedMPH ... CurvePlotSpeedDomain.maximumBandCenterSpeedMPH,
            defaultValue: 45,
            unit: "mph",
            precision: 1
          )
        }
      }
      .padding(10)
      .background(
        RoundedRectangle(cornerRadius: 8)
          .fill(session.plot.selected == .band(bandID) ? bandColor.opacity(0.13) : Color.secondary.opacity(0.07))
      )
      .overlay {
        RoundedRectangle(cornerRadius: 8)
          .stroke(session.plot.selected == .band(bandID) ? bandColor : Color.secondary.opacity(0.2))
      }
    }
  }

  private var bandColor: Color { CurvePalette.bandColor(for: session.bands.firstIndex { $0.id == bandID } ?? 0) }

  private var enabledBinding: Binding<Bool> {
    Binding(
      get: { band?.enabled ?? false },
      set: { value in
        session.performDiscreteEdit {
          session.updateBand(id: bandID) { $0.enabled = value }
        }
      }
    )
  }

  private func binding<T>(_ keyPath: WritableKeyPath<EQBand, T>) -> Binding<T> {
    Binding(
      get: { band![keyPath: keyPath] },
      set: { value in
        session.updateBand(id: bandID) { $0[keyPath: keyPath] = value }
      }
    )
  }

  @ViewBuilder
  private func smallKnob(
    _ label: String,
    keyPath: WritableKeyPath<EQBand, Double>,
    range: ClosedRange<Double>,
    defaultValue: Double,
    unit: String,
    precision: Int,
    logarithmic: Bool = false
  ) -> some View {
    RotaryKnob(
      label: label,
      value: binding(keyPath),
      range: range,
      defaultValue: defaultValue,
      unit: unit,
      precision: precision,
      diameter: 48,
      accent: bandColor,
      logarithmic: logarithmic,
      onEditingChanged: { editing in
        if editing { session.beginContinuousEdit() } else { session.endContinuousEdit() }
      }
    )
  }
}

struct AdvancedParametersView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    VStack(spacing: 8) {
      rawRow("A (amplitude)", \SigmoidParameters.a, range: -5 ... -0.2)
      rawRow("B (steepness)", \SigmoidParameters.b, range: -100_000 ... -100)
      rawRow("C (centre κ)", \SigmoidParameters.c, range: 0.00001 ... 0.1)
      rawRow("D (baseline)", \SigmoidParameters.d, range: 2 ... 6.5)
      rawRow("MIN (floor)", \SigmoidParameters.minLat, range: 1 ... 3)
      rawRow("MAX (ceiling)", \SigmoidParameters.maxLat, range: 2 ... 5.5)
    }
  }

  private func rawRow(
    _ label: String,
    _ keyPath: WritableKeyPath<SigmoidParameters, Double>,
    range: ClosedRange<Double>
  ) -> some View {
    LabeledContent(label) {
      TextField(
        label,
        value: Binding(
          get: { session.parameters[keyPath: keyPath] },
          set: { value in
            session.performDiscreteEdit {
              var parameters = session.parameters
              parameters[keyPath: keyPath] = min(max(value, range.lowerBound), range.upperBound)
              session.knobs = VTSCMath.knobs(from: parameters)
            }
          }
        ),
        format: .number.precision(.fractionLength(5))
      )
      .labelsHidden()
      .multilineTextAlignment(.trailing)
      .font(.system(.body, design: .monospaced))
      .frame(width: 120)
    }
    .font(.caption)
  }
}

struct ApplyConfirmationView: View {
  @ObservedObject var session: TunerSession
  let action: ApplyAction

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Label(action.label, systemImage: "paperplane.fill").font(.title2.weight(.semibold))
      Text(action.description)
      Label(session.repositoryURL?.path ?? "No Chauffeur repository selected", systemImage: "folder")
        .font(.caption)
        .foregroundStyle(.secondary)
        .textSelection(.enabled)
      if action != .local {
        Text("This uses the selected checkout's current branch and commits the complete VTSC target files. Review any existing edits to those files first.")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      if action == .pullOnTici || action == .rebuildTilesAndReboot {
        Label("Run only while parked; this action reboots the tici.", systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.orange)
      }
      HStack {
        Spacer()
        Button("Cancel", role: .cancel) { session.pendingApplyAction = nil }
        Button("Run") { session.confirmApply() }
          .keyboardShortcut(.defaultAction)
      }
    }
    .padding(24)
    .frame(width: 520)
  }
}

struct ApplyProgressView: View {
  @ObservedObject var session: TunerSession
  let action: ApplyAction

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text(action.label).font(.title2.weight(.semibold))
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 12) {
          ForEach(session.applySteps, id: \.id) { step in
            HStack(alignment: .top, spacing: 10) {
              Group {
                switch step.status {
                case .running: ProgressView().controlSize(.small)
                case .succeeded: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                }
              }
              .frame(width: 18)
              VStack(alignment: .leading, spacing: 4) {
                Text(step.text)
                if !step.detail.isEmpty {
                  Text(step.detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    .textSelection(.enabled)
                }
              }
            }
          }
        }
      }
      .frame(minHeight: 220)
      HStack {
        if action == .rebuildTilesAndReboot, session.applySucceeded == nil {
          Button("Cancel", role: .destructive) { session.cancelApply() }
        }
        Spacer()
        Button("Close") { session.closeApply() }
          .disabled(session.applySucceeded == nil)
          .keyboardShortcut(.defaultAction)
      }
    }
    .padding(24)
    .frame(width: 640, height: 380)
    .interactiveDismissDisabled(session.applySucceeded == nil)
  }
}

struct ResumePostflightConfirmationView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Label(ResumePostflightAction.label, systemImage: "checkmark.shield")
        .font(.title2.weight(.semibold))
      Text(ResumePostflightAction.description)
      Label(session.repositoryURL?.path ?? "No Chauffeur repository selected", systemImage: "folder")
        .font(.caption)
        .foregroundStyle(.secondary)
        .textSelection(.enabled)
      Label(
        "Outdoors after a normal GPS/profile-producing drive: stop safely with ignition still on, then click Verify and Complete. The app first captures a fresh profile that the live controller can consume; when that succeeds, turn ignition off. It keeps that proof only in memory while polling for IsOffroad=1 and rechecking the unchanged tune/runtime identity. Failure keeps the original journal pending.",
        systemImage: "location.viewfinder"
      )
      .foregroundStyle(.orange)
      HStack {
        Spacer()
        Button("Cancel", role: .cancel) { session.pendingResumePostflight = false }
        Button("Verify and Complete") { session.confirmResumePostflight() }
          .keyboardShortcut(.defaultAction)
      }
    }
    .padding(24)
    .frame(width: 560)
  }
}

struct ResumePostflightProgressView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text(ResumePostflightAction.label).font(.title2.weight(.semibold))
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 12) {
          ForEach(session.applySteps, id: \.id) { step in
            HStack(alignment: .top, spacing: 10) {
              Group {
                switch step.status {
                case .running: ProgressView().controlSize(.small)
                case .succeeded: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                }
              }
              .frame(width: 18)
              VStack(alignment: .leading, spacing: 4) {
                Text(step.text)
                if !step.detail.isEmpty {
                  Text(step.detail)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                }
              }
            }
          }
        }
      }
      .frame(minHeight: 220)
      HStack {
        if session.applySucceeded == nil {
          Button("Cancel", role: .cancel) { session.cancelResumePostflight() }
        }
        Spacer()
        Button("Close") { session.closeResumePostflight() }
          .disabled(session.applySucceeded == nil)
          .keyboardShortcut(.defaultAction)
      }
    }
    .padding(24)
    .frame(width: 680, height: 400)
    .interactiveDismissDisabled(session.applySucceeded == nil)
  }
}
