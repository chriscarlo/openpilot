import Foundation
import SwiftUI

struct WholeCurveStudyInspector: View {
  @ObservedObject var map: MapPreviewSession

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 16) {
        safetyBanner
        summaryCard
        eventPicker

        sectionHeader("Selected Whole Curve", systemImage: "point.topleft.down.to.point.bottomright.curvepath")
        if let event = map.selectedWholeCurveEvent {
          selectedEventCard(event)
        } else {
          emptySelectionCard
        }
      }
      .padding(16)
    }
  }

  private var safetyBanner: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label("LOCAL SHADOW STUDY", systemImage: "checkmark.shield.fill")
        .font(.headline.weight(.bold))
        .foregroundStyle(.cyan)
      Text("DOES NOT CHANGE TUNE OR CAR")
        .font(.title3.weight(.heavy))
      Text("This view only compares today’s mapd result with the experimental whole-curve calculation from local tile data. It cannot save or apply tune changes, edit the curve bank, sync, rebuild, or contact the car.")
        .font(.callout)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .padding(13)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.cyan.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
    .overlay {
      RoundedRectangle(cornerRadius: 12)
        .stroke(Color.cyan.opacity(0.42), lineWidth: 1)
    }
  }

  private var summaryCard: some View {
    let highConfidenceCount = map.wholeCurveEvents.count {
      $0.confidenceLabel.localizedCaseInsensitiveContains("high")
    }
    let bankLinkedCount = map.wholeCurveEvents.count { !$0.bankSampleNumbers.isEmpty }
    let flaggedCount = map.wholeCurveEvents.count { !$0.flags.isEmpty }

    return VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .firstTextBaseline) {
        Text("VISIBLE STUDY EVENTS")
          .font(.callout.weight(.bold))
          .foregroundStyle(.secondary)
        Spacer()
        Text("\(map.wholeCurveEvents.count)")
          .font(.system(size: 28, weight: .bold, design: .rounded))
      }
      HStack(spacing: 8) {
        summaryPill("\(highConfidenceCount) high confidence", color: .green)
        summaryPill("\(bankLinkedCount) bank-linked", color: .purple)
        summaryPill("\(flaggedCount) flagged", color: flaggedCount == 0 ? .secondary : .orange)
      }
      Text("Each event is one connected bend in one travel direction, even when the map divides it into several pieces.")
        .font(.callout)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
      Button("Show All Saved Curves", systemImage: "scope") {
        map.focusWholeCurveBank()
      }
      .buttonStyle(.bordered)
    }
    .padding(12)
    .background(Color.secondary.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
  }

  private var emptySelectionCard: some View {
    VStack(spacing: 10) {
      Image(systemName: "cursorarrow.click.2")
        .font(.system(size: 27))
        .foregroundStyle(.secondary)
      Text("Select a Whole Curve")
        .font(.title3.weight(.semibold))
      Text("Click a colored whole-curve event on the map to inspect its grouping and speed profile.")
        .font(.body)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity, minHeight: 135)
    .padding(12)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
  }

  private var eventPicker: some View {
    VStack(alignment: .leading, spacing: 9) {
      sectionHeader("Saved Curve Events", systemImage: "list.bullet.rectangle")
      if map.wholeCurveEvents.isEmpty {
        Text("No saved-bank event is resolved in the current local tile view.")
          .font(.callout)
          .foregroundStyle(.secondary)
      } else {
        VStack(spacing: 5) {
          ForEach(map.wholeCurveEvents) { event in
            Button {
              map.selectWholeCurveEvent(event.id)
            } label: {
              HStack(spacing: 9) {
                Image(systemName: map.selectedWholeCurveEventID == event.id
                  ? "checkmark.circle.fill"
                  : "circle")
                  .foregroundStyle(map.selectedWholeCurveEventID == event.id ? .cyan : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                  Text(event.bankSampleNumbers.map { "#\($0)" }.joined(separator: ", "))
                    .font(.callout.weight(.bold))
                  Text("\(event.travelDirection) · \(event.bendDirection) bend · \(String(format: "%.1f", event.wholeCurveSpeedMPH)) mph")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                  .font(.caption.weight(.semibold))
                  .foregroundStyle(.tertiary)
              }
              .padding(.horizontal, 9)
              .padding(.vertical, 7)
              .contentShape(Rectangle())
              .background(
                map.selectedWholeCurveEventID == event.id
                  ? Color.cyan.opacity(0.12)
                  : Color.secondary.opacity(0.06),
                in: RoundedRectangle(cornerRadius: 8)
              )
            }
            .buttonStyle(.plain)
            .accessibilityLabel(
              "Bank samples \(event.bankSampleNumbers.map(String.init).joined(separator: ", ")), \(event.travelDirection) \(event.bendDirection) bend, \(String(format: "%.1f", event.wholeCurveSpeedMPH)) miles per hour"
            )
          }
        }
      }
    }
  }

  private func selectedEventCard(_ event: MapWholeCurveEvent) -> some View {
    VStack(alignment: .leading, spacing: 15) {
      eventHeader(event)

      HStack(spacing: 10) {
        metricTile("LENGTH", String(format: "%.0f m", event.lengthMeters), color: .primary)
        metricTile("TOTAL TURN", String(format: "%.0f°", abs(event.turnDegrees)), color: .primary)
      }

      speedComparison(event)
      curvatureScales(event)
      WholeCurveDistanceProfile(event: event)
      confidenceSection(event)
      bankSection(event)
    }
    .padding(13)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
  }

  private func eventHeader(_ event: MapWholeCurveEvent) -> some View {
    VStack(alignment: .leading, spacing: 5) {
      Text(event.displayName)
        .font(.title2.weight(.bold))
        .fixedSize(horizontal: false, vertical: true)
      if !event.reference.isEmpty,
         event.reference.caseInsensitiveCompare(event.roadName) != .orderedSame {
        Text(event.reference)
          .font(.body.weight(.medium))
          .foregroundStyle(.secondary)
      }
      Text("\(event.travelDirection) travel · \(event.bendDirection) bend")
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
    }
  }

  private func speedComparison(_ event: MapWholeCurveEvent) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      sectionHeader("Speed Comparison", systemImage: "speedometer")
      HStack(spacing: 10) {
        metricTile(
          "CURRENT CURVE-CAP RANGE",
          String(format: "%.1f–%.1f mph", event.currentMinimumSpeedMPH, event.currentMaximumSpeedMPH),
          color: .secondary
        )
        metricTile(
          "WHOLE CURVE",
          String(format: "%.1f mph", event.wholeCurveSpeedMPH),
          color: .cyan
        )
      }
      HStack {
        Text("Controlling lateral acceleration")
          .font(.callout.weight(.semibold))
        Spacer()
        Text(String(format: "%.2f m/s²", event.lateralAccelerationMPS2))
          .font(.system(.callout, design: .monospaced).weight(.bold))
      }
      .padding(.horizontal, 2)
      Text("These are map-derived curve caps, not commanded vehicle speeds. 156.6 mph means a point is effectively unrestricted by curve geometry.")
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }

  private func curvatureScales(_ event: MapWholeCurveEvent) -> some View {
    VStack(alignment: .leading, spacing: 9) {
      sectionHeader("Distance-Aware Curvature", systemImage: "ruler")
      Text("The same bend measured over three road distances. Agreement means the shape is sustained; disagreement exposes a short spike or a genuinely tightening curve.")
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
      VStack(spacing: 6) {
        curvatureRow("60 m", value: event.curvature60)
        curvatureRow("100 m", value: event.curvature100)
        curvatureRow("160 m", value: event.curvature160)
      }
    }
  }

  private func confidenceSection(_ event: MapWholeCurveEvent) -> some View {
    let tone = confidenceColor(event.confidenceLabel)
    return VStack(alignment: .leading, spacing: 8) {
      sectionHeader("Confidence & Flags", systemImage: "checkmark.seal")
      Label(event.confidenceLabel, systemImage: "circle.fill")
        .font(.body.weight(.semibold))
        .foregroundStyle(tone)
      if event.flags.isEmpty {
        Label("No estimator flags", systemImage: "checkmark.circle.fill")
          .font(.callout.weight(.medium))
          .foregroundStyle(.green)
      } else {
        ForEach(Array(event.flags.enumerated()), id: \.offset) { _, flag in
          Label(flag, systemImage: "exclamationmark.triangle.fill")
            .font(.callout.weight(.medium))
            .foregroundStyle(.orange)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
  }

  private func bankSection(_ event: MapWholeCurveEvent) -> some View {
    VStack(alignment: .leading, spacing: 9) {
      sectionHeader("Linked Curve-Bank Samples", systemImage: "tray.full")
      if event.bankSampleNumbers.isEmpty || event.bankTargetsMPH.isEmpty {
        Text("No saved calibration samples are linked to this whole-curve event.")
          .font(.callout)
          .foregroundStyle(.secondary)
      } else {
        let sampleText = event.bankSampleNumbers.map { "#\($0)" }.joined(separator: ", ")
        let targetText = event.bankTargetsMPH
          .map { String(format: "%.1f", $0) }
          .joined(separator: ", ")
        LabeledContent("Bank samples") {
          Text(sampleText)
            .font(.system(.callout, design: .monospaced).weight(.semibold))
        }
        LabeledContent("Targets") {
          Text("\(targetText) mph")
            .font(.system(.callout, design: .monospaced).weight(.semibold))
            .multilineTextAlignment(.trailing)
        }
        if let median = event.targetMedianMPH {
          LabeledContent("Median target") {
            Text(String(format: "%.1f mph", median))
              .font(.system(.callout, design: .monospaced).weight(.bold))
          }
        }
        if event.targetIsCoherent {
          Label(
            String(format: "Targets agree within %.1f mph. The study has not changed or merged them.", event.targetSpreadMPH),
            systemImage: "checkmark.circle.fill"
          )
          .font(.callout.weight(.medium))
          .foregroundStyle(.green)
        } else {
          Label(
            String(format: "Target conflict: these samples span %.1f mph. The study reports the conflict but does not resolve or change it.", event.targetSpreadMPH),
            systemImage: "exclamationmark.triangle.fill"
          )
          .font(.callout.weight(.semibold))
          .foregroundStyle(.orange)
          .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
  }

  private func curvatureRow(_ label: String, value: Double?) -> some View {
    HStack {
      Text(label)
        .font(.callout.weight(.semibold))
      Spacer()
      Text(curvatureDescription(value))
        .font(.system(.callout, design: .monospaced))
        .foregroundStyle(value == nil ? .secondary : .primary)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 7)
    .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
  }

  private func curvatureDescription(_ curvature: Double?) -> String {
    guard let curvature, curvature.isFinite else { return "Not enough route" }
    let magnitude = abs(curvature)
    guard magnitude > 1.0e-9 else { return "κ 0.000000 · straight" }
    return String(format: "κ %.6f · R %.0f m", curvature, 1 / magnitude)
  }

  private func metricTile(_ label: String, _ value: String, color: Color) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(label)
        .font(.caption.weight(.bold))
        .foregroundStyle(.secondary)
      Text(value)
        .font(.system(size: 19, weight: .bold, design: .rounded))
        .foregroundStyle(color)
        .lineLimit(1)
        .minimumScaleFactor(0.72)
    }
    .padding(10)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 9))
  }

  private func summaryPill(_ text: String, color: Color) -> some View {
    Text(text)
      .font(.caption.weight(.semibold))
      .foregroundStyle(color)
      .padding(.horizontal, 8)
      .padding(.vertical, 5)
      .background(color.opacity(0.12), in: Capsule())
  }

  private func sectionHeader(_ text: String, systemImage: String) -> some View {
    Label(text.uppercased(), systemImage: systemImage)
      .font(.callout.weight(.bold))
      .foregroundStyle(.secondary)
  }

  private func confidenceColor(_ label: String) -> Color {
    if label.localizedCaseInsensitiveContains("high") { return .green }
    if label.localizedCaseInsensitiveContains("medium") { return .yellow }
    return .orange
  }
}

private struct WholeCurveDistanceProfile: View {
  let event: MapWholeCurveEvent

  private var points: [MapWholeCurvePoint] {
    event.points
      .filter {
        $0.distanceMeters.isFinite
          && $0.currentMapdSpeedMPH.isFinite
          && $0.wholeCurveSpeedMPH.isFinite
      }
      .sorted { $0.distanceMeters < $1.distanceMeters }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Label("SPEED ALONG CURVE", systemImage: "chart.xyaxis.line")
          .font(.callout.weight(.bold))
          .foregroundStyle(.secondary)
        Spacer()
        profileLegend(color: .secondary, label: "mapd", dashed: true)
        profileLegend(color: .cyan, label: "whole", dashed: false)
      }

      if points.count >= 2 {
        Canvas { context, size in
          drawProfile(context: &context, size: size)
        }
        .frame(height: 145)
        .padding(8)
        .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 9))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Speed along the selected whole curve")
        .accessibilityValue(accessibilityValue)

        HStack {
          Text("Entry  0 m")
          Spacer()
          Text(String(format: "Exit  %.0f m", points.last?.distanceMeters ?? event.lengthMeters))
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.secondary)
      } else {
        Text("Not enough finite route points to draw the distance profile.")
          .font(.callout)
          .foregroundStyle(.secondary)
          .padding(.vertical, 20)
          .frame(maxWidth: .infinity, alignment: .center)
          .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 9))
      }
    }
  }

  private var accessibilityValue: String {
    String(
      format: "Today’s mapd ranges from %.1f to %.1f miles per hour. Whole curve is %.1f miles per hour.",
      event.currentMinimumSpeedMPH,
      event.currentMaximumSpeedMPH,
      event.wholeCurveSpeedMPH
    )
  }

  private func drawProfile(context: inout GraphicsContext, size: CGSize) {
    let chart = CGRect(x: 4, y: 4, width: max(1, size.width - 8), height: max(1, size.height - 8))
    let maximumDistance = max(points.last?.distanceMeters ?? 0, 1)
    let allSpeeds = points.flatMap { [$0.currentMapdSpeedMPH, $0.wholeCurveSpeedMPH] }
    let minimumSpeed = allSpeeds.min() ?? 0
    let maximumSpeed = allSpeeds.max() ?? minimumSpeed + 1
    let speedPadding = max(2, (maximumSpeed - minimumSpeed) * 0.12)
    let lowerSpeed = max(0, minimumSpeed - speedPadding)
    let upperSpeed = max(lowerSpeed + 4, maximumSpeed + speedPadding)

    func chartPoint(distance: Double, speed: Double) -> CGPoint {
      let x = chart.minX + chart.width * CGFloat(distance / maximumDistance)
      let normalizedSpeed = (speed - lowerSpeed) / (upperSpeed - lowerSpeed)
      let y = chart.maxY - chart.height * CGFloat(normalizedSpeed)
      return CGPoint(x: x, y: y)
    }

    for fraction in [0.25, 0.5, 0.75] {
      var grid = Path()
      let y = chart.minY + chart.height * CGFloat(fraction)
      grid.move(to: CGPoint(x: chart.minX, y: y))
      grid.addLine(to: CGPoint(x: chart.maxX, y: y))
      context.stroke(grid, with: .color(.secondary.opacity(0.16)), lineWidth: 1)
    }

    var currentPath = Path()
    var wholePath = Path()
    for (index, point) in points.enumerated() {
      let current = chartPoint(distance: point.distanceMeters, speed: point.currentMapdSpeedMPH)
      let whole = chartPoint(distance: point.distanceMeters, speed: point.wholeCurveSpeedMPH)
      if index == 0 {
        currentPath.move(to: current)
        wholePath.move(to: whole)
      } else {
        currentPath.addLine(to: current)
        wholePath.addLine(to: whole)
      }
    }
    context.stroke(
      currentPath,
      with: .color(.secondary.opacity(0.9)),
      style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round, dash: [5, 4])
    )
    context.stroke(
      wholePath,
      with: .color(.cyan),
      style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round)
    )
  }

  private func profileLegend(color: Color, label: String, dashed: Bool) -> some View {
    HStack(spacing: 4) {
      Canvas { context, size in
        var line = Path()
        line.move(to: CGPoint(x: 0, y: size.height / 2))
        line.addLine(to: CGPoint(x: size.width, y: size.height / 2))
        context.stroke(
          line,
          with: .color(color),
          style: StrokeStyle(lineWidth: 2, dash: dashed ? [4, 3] : [])
        )
      }
      .frame(width: 20, height: 8)
      Text(label)
        .font(.caption2.weight(.semibold))
        .foregroundStyle(.secondary)
    }
  }
}
