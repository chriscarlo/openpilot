import AppKit
import SwiftUI
import VTSCTunerCore

enum CurvePalette {
  static let background = Color(red: 0.059, green: 0.071, blue: 0.09)
  static let panel = Color(red: 0.102, green: 0.118, blue: 0.149)
  static let separator = Color(red: 0.18, green: 0.204, blue: 0.251)
  static let accent = Color(red: 0.298, green: 0.839, blue: 0.757)
  static let rail = Color(red: 0.839, green: 0.31, blue: 0.31)
  static let axisText = Color.white.opacity(0.88)
  static let axisTitle = Color.white.opacity(0.96)

  static func bandColor(for index: Int) -> Color {
    let colors: [Color] = [
      Color(red: 0.94, green: 0.63, blue: 0.29),
      Color(red: 0.42, green: 0.65, blue: 1),
      Color(red: 0.94, green: 0.44, blue: 0.56),
      Color(red: 0.56, green: 0.82, blue: 0.43),
      Color(red: 0.72, green: 0.52, blue: 0.96),
      Color(red: 0.94, green: 0.84, blue: 0.29),
      Color(red: 0.29, green: 0.84, blue: 0.76),
    ]
    return colors[index % colors.count]
  }
}

/// Keeps the familiar Curve Lab viewport compact until an enabled (or
/// explicitly selected) residual band needs the fitter's wider authoring
/// range. The extra one-mile look-ahead rounds the viewport to a useful grid
/// boundary and leaves room to drag a high-speed marker farther right.
enum CurvePlotSpeedDomain {
  static let legacyMaximumSpeedMPH = 90.0
  static let minimumBandCenterSpeedMPH = 1.0
  static let gridStepMPH = 10.0

  static var maximumBandCenterSpeedMPH: Double {
    SigmoidFitter.maximumRepresentableSpeedMPH
  }

  static func clampedBandCenterSpeedMPH(_ speedMPH: Double) -> Double {
    guard speedMPH.isFinite else { return legacyMaximumSpeedMPH / 2.0 }
    return min(
      max(speedMPH, minimumBandCenterSpeedMPH),
      maximumBandCenterSpeedMPH
    )
  }

  static func maximumSpeedMPH(
    bands: [EQBand],
    selection: PlotSelection?
  ) -> Double {
    var requiredMaximum = bands.lazy
      .filter(\.enabled)
      .map(\.centerSpeedMPH)
      .filter(\.isFinite)
      .max() ?? legacyMaximumSpeedMPH

    if case let .band(selectedID) = selection,
       let selectedBand = bands.first(where: { $0.id == selectedID }),
       selectedBand.centerSpeedMPH.isFinite {
      requiredMaximum = max(requiredMaximum, selectedBand.centerSpeedMPH)
    }

    guard requiredMaximum > legacyMaximumSpeedMPH else {
      return legacyMaximumSpeedMPH
    }

    let roundedWithDragRoom = ceil(
      (requiredMaximum + 1.0) / gridStepMPH
    ) * gridStepMPH
    return min(
      max(roundedWithDragRoom, legacyMaximumSpeedMPH),
      maximumBandCenterSpeedMPH
    )
  }
}

private struct PlotMapper {
  let outer: CGRect
  let plot: CGRect
  let maximumAcceleration: Double
  let maximumSpeedMPH: Double

  init(
    size: CGSize,
    safeAreaInsets: EdgeInsets = EdgeInsets(),
    maximumAcceleration: Double = 6.0,
    maximumSpeedMPH: Double = CurvePlotSpeedDomain.legacyMaximumSpeedMPH
  ) {
    outer = CGRect(origin: .zero, size: size)
    self.maximumAcceleration = max(maximumAcceleration, 1.0)
    self.maximumSpeedMPH = min(
      max(maximumSpeedMPH, CurvePlotSpeedDomain.legacyMaximumSpeedMPH),
      CurvePlotSpeedDomain.maximumBandCenterSpeedMPH
    )
    let leftGutter: CGFloat = 78 + safeAreaInsets.leading
    let topGutter: CGFloat = 20 + safeAreaInsets.top
    let rightGutter: CGFloat = 28 + safeAreaInsets.trailing
    // RootView's custom 30-point status inset is not consistently reflected in
    // GeometryProxy.safeAreaInsets on macOS, so always reserve its clearance.
    let bottomGutter: CGFloat = 82 + max(safeAreaInsets.bottom, 32)
    plot = CGRect(
      x: leftGutter,
      y: topGutter,
      width: max(size.width - leftGutter - rightGutter, 1),
      height: max(size.height - topGutter - bottomGutter, 1)
    )
  }

  func point(speedMPH: Double, acceleration: Double) -> CGPoint {
    CGPoint(
      x: plot.minX + CGFloat(speedMPH / maximumSpeedMPH) * plot.width,
      y: plot.maxY - CGFloat(acceleration / maximumAcceleration) * plot.height
    )
  }

  func speed(at x: CGFloat) -> Double {
    min(
      max(Double((x - plot.minX) / plot.width) * maximumSpeedMPH, 0),
      maximumSpeedMPH
    )
  }

  func acceleration(at y: CGFloat) -> Double {
    min(
      max(Double((plot.maxY - y) / plot.height) * maximumAcceleration, 0),
      maximumAcceleration
    )
  }
}

/// Keeps an EQ marker and its vertical drag coupled to the exact exported
/// 256-point Q curve, including overlapping bands and source rounding.
enum CurveBandEditingMath {
  static func marker(
    bandID: UUID,
    bands: [EQBand],
    parameters: SigmoidParameters
  ) -> (speedMPH: Double, accelerationMPS2: Double)? {
    guard let band = bands.first(where: { $0.id == bandID && $0.enabled }) else { return nil }
    let kappa = VTSCMath.centerKappa(for: band, parameters: parameters)
    let base = VTSCMath.evaluate(parameters, curvature: kappa)
    let q = VTSCMath.qSpeedMultiplier(
      curvature: kappa,
      points: VTSCMath.qCurvePoints(parameters: parameters, bands: bands)
    )
    return (band.centerSpeedMPH, base * q * q)
  }

  static func gainDB(
    bandID: UUID,
    targetAccelerationMPS2: Double,
    bands: [EQBand],
    parameters: SigmoidParameters
  ) -> Double? {
    guard targetAccelerationMPS2.isFinite,
      let bandIndex = bands.firstIndex(where: { $0.id == bandID && $0.enabled })
    else { return nil }

    func acceleration(at gainDB: Double) -> Double {
      var candidate = bands
      candidate[bandIndex].gainDB = gainDB
      return marker(bandID: bandID, bands: candidate, parameters: parameters)?
        .accelerationMPS2 ?? 0.0
    }

    var lowerGain = -12.0
    var upperGain = 12.0
    let lowerAcceleration = acceleration(at: lowerGain)
    let upperAcceleration = acceleration(at: upperGain)
    if targetAccelerationMPS2 <= lowerAcceleration { return lowerGain }
    if targetAccelerationMPS2 >= upperAcceleration { return upperGain }

    // The source-rounded Q curve is monotone in this band's gain but has tiny
    // quantization plateaus. Bisection followed by a closest-candidate check
    // gives the most faithful draggable value without pretending continuity.
    for _ in 0..<32 {
      let midpoint = 0.5 * (lowerGain + upperGain)
      if acceleration(at: midpoint) < targetAccelerationMPS2 {
        lowerGain = midpoint
      } else {
        upperGain = midpoint
      }
    }
    return [lowerGain, upperGain, 0.5 * (lowerGain + upperGain)].min {
      abs(acceleration(at: $0) - targetAccelerationMPS2)
        < abs(acceleration(at: $1) - targetAccelerationMPS2)
    }
  }
}

private enum PlotPointerKind { case down, dragged, up, moved, exited, secondaryDown, escape }

private struct PlotPointerEvent {
  var kind: PlotPointerKind
  var location: CGPoint
  var shift: Bool = false
}

struct CurvePlotView: View {
  @ObservedObject var session: TunerSession

  var body: some View {
    GeometryReader { geometry in
      let samples = renderedCurveSamples()
      let maximumAcceleration = plotMaximumAcceleration(samples: samples)
      let dynamicMaximumSpeedMPH = CurvePlotSpeedDomain.maximumSpeedMPH(
        bands: session.bands,
        selection: session.plot.selected
      )
      // Freeze the x mapping for the duration of a marker drag. Otherwise a
      // marker crossing a 10-mph boundary would expand the domain underneath
      // the pointer and recursively pull itself toward the runtime cap.
      let maximumSpeedMPH = session.plot.speedDomainMaximumDuringDrag
        ?? dynamicMaximumSpeedMPH
      let mapper = PlotMapper(
        size: geometry.size,
        safeAreaInsets: geometry.safeAreaInsets,
        maximumAcceleration: maximumAcceleration,
        maximumSpeedMPH: maximumSpeedMPH
      )
      let maximumSpeedDescription = maximumSpeedMPH == CurvePlotSpeedDomain.legacyMaximumSpeedMPH
        ? "90"
        : String(format: "%.1f", maximumSpeedMPH)
      ZStack(alignment: .topLeading) {
        Canvas { context, _ in
          draw(context: &context, mapper: mapper, curveSamples: samples)
        }
        .accessibilityLabel("VTSC speed and lateral acceleration curve")
        .accessibilityValue("Speed from 0 to \(maximumSpeedDescription) miles per hour; lateral acceleration from 0 to \(maximumAcceleration.formatted(.number.precision(.fractionLength(0)))) meters per second squared")

        PlotEventLayer { event in
          handle(event, mapper: mapper)
        }

        if let menu = session.plot.contextMenu {
          plotMenu(menu, mapper: mapper)
        }
      }
    }
  }

  private func plotMenu(_ menu: PlotMenuContext, mapper: PlotMapper) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Button {
        session.addBand(at: menu.speedMPH)
        session.plot.contextMenu = nil
      } label: {
        Label("Add anchor point here", systemImage: "plus")
      }
      Button {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
          String(format: "%.1f mph, %.2f m/s²", menu.speedMPH, menu.acceleration),
          forType: .string
        )
        session.plot.contextMenu = nil
      } label: {
        Label("Copy \(String(format: "%.1f mph, %.2f m/s²", menu.speedMPH, menu.acceleration))", systemImage: "doc.on.doc")
      }
    }
    .buttonStyle(.plain)
    .padding(8)
    .frame(width: 250, alignment: .leading)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.secondary.opacity(0.35)))
    .position(
      x: min(max(menu.point.x + 125, 130), max(mapper.outer.width - 130, 130)),
      y: min(max(menu.point.y + 42, 45), max(mapper.outer.height - 45, 45))
    )
    .shadow(radius: 12)
  }

  private func handle(_ event: PlotPointerEvent, mapper: PlotMapper) {
    let target = pickTarget(event.location, mapper: mapper)
    switch event.kind {
    case .down:
      session.plot.contextMenu = nil
      guard mapper.plot.contains(event.location) else { return }
      if event.shift {
        session.addBand(at: mapper.speed(at: event.location.x))
        return
      }
      if case .band = target {
        session.plot.speedDomainMaximumDuringDrag = mapper.maximumSpeedMPH
      } else {
        session.plot.speedDomainMaximumDuringDrag = nil
      }
      session.plot.activeDrag = target
      switch target {
      case let .handle(handle): session.plot.selected = .handle(handle)
      case let .band(id): session.plot.selected = .band(id)
      case .none: session.plot.selected = nil
      }
      if target != .none { session.beginContinuousEdit() }
      if target == .none { NSCursor.crosshair.set() } else { NSCursor.closedHand.set() }

    case .dragged:
      applyDrag(event.location, mapper: mapper)
      NSCursor.closedHand.set()

    case .up:
      if session.plot.activeDrag != .none { session.endContinuousEdit() }
      session.plot.activeDrag = .none
      session.plot.speedDomainMaximumDuringDrag = nil
      NSCursor.arrow.set()

    case .moved:
      if mapper.plot.contains(event.location) {
        session.plot.hoverPoint = event.location
        session.plot.hoverReadout = (mapper.speed(at: event.location.x), mapper.acceleration(at: event.location.y))
        session.plot.hoveredTarget = target
        if target == .none { NSCursor.crosshair.set() } else { NSCursor.openHand.set() }
      } else {
        session.plot.hoverPoint = nil
        session.plot.hoverReadout = nil
        session.plot.hoveredTarget = .none
        NSCursor.arrow.set()
      }

    case .exited:
      session.plot.hoverPoint = nil
      session.plot.hoverReadout = nil
      session.plot.hoveredTarget = .none
      NSCursor.arrow.set()

    case .secondaryDown:
      let points = curvePoints(mapper: mapper)
      if mapper.plot.contains(event.location), hitPolyline(event.location, points: points, tolerance: 8) {
        session.plot.contextMenu = PlotMenuContext(
          point: event.location,
          speedMPH: mapper.speed(at: event.location.x),
          acceleration: mapper.acceleration(at: event.location.y)
        )
      } else {
        session.plot.contextMenu = nil
      }

    case .escape:
      if session.plot.contextMenu != nil { session.plot.contextMenu = nil }
      else {
        session.plot.selected = nil
        if session.plot.activeDrag == .none {
          session.plot.speedDomainMaximumDuringDrag = nil
        }
      }
    }
  }

  private func applyDrag(_ point: CGPoint, mapper: PlotMapper) {
    let newY = mapper.acceleration(at: point.y)
    let newX = mapper.speed(at: point.x)
    switch session.plot.activeDrag {
    case .none: break
    case let .handle(handle):
      switch handle {
      case .minimumRail:
        session.knobs.tightCurveAcceleration = min(max(newY, 1), 3)
        if session.knobs.straightRoadAcceleration < session.knobs.tightCurveAcceleration + 0.2 {
          session.knobs.straightRoadAcceleration = session.knobs.tightCurveAcceleration + 0.2
        }
      case .maximumRail:
        session.knobs.straightRoadAcceleration = min(max(newY, 2), 5.5)
        if session.knobs.tightCurveAcceleration > session.knobs.straightRoadAcceleration - 0.2 {
          session.knobs.tightCurveAcceleration = session.knobs.straightRoadAcceleration - 0.2
        }
      case .inflection:
        session.knobs.transitionSpeedMPH = min(max(newX, 8), 120)
      case .leftWing, .rightWing:
        let halfWidth = max(abs(newX - session.knobs.transitionSpeedMPH), 0.5)
        session.knobs.sharpness = VTSCMath.sharpness(wingHalfWidth: halfWidth)
      }
    case let .band(id):
      guard let index = session.bands.firstIndex(where: { $0.id == id }) else { return }
      var candidateBands = session.bands
      candidateBands[index].centerSpeedMPH = CurvePlotSpeedDomain.clampedBandCenterSpeedMPH(newX)
      if let gain = CurveBandEditingMath.gainDB(
        bandID: id,
        targetAccelerationMPS2: newY,
        bands: candidateBands,
        parameters: session.parameters
      ) {
        candidateBands[index].gainDB = gain
      }
      let candidate = candidateBands[index]
      session.updateBand(id: id) { $0 = candidate }
    }
  }

  private func pickTarget(_ point: CGPoint, mapper: PlotMapper) -> DragTarget {
    let parameters = session.parameters
    let midpoint = 0.5 * (session.knobs.tightCurveAcceleration + session.knobs.straightRoadAcceleration)
    let inflection = mapper.point(speedMPH: session.knobs.transitionSpeedMPH, acceleration: midpoint)
    if distance(point, inflection) <= 12 { return .handle(.inflection) }

    let half = VTSCMath.wingHalfWidth(sharpness: session.knobs.sharpness)
    let left = mapper.point(speedMPH: max(session.knobs.transitionSpeedMPH - half, 1), acceleration: midpoint)
    let right = mapper.point(
      speedMPH: min(session.knobs.transitionSpeedMPH + half, mapper.maximumSpeedMPH - 1),
      acceleration: midpoint
    )
    if distance(point, left) <= 14 { return .handle(.leftWing) }
    if distance(point, right) <= 14 { return .handle(.rightWing) }

    for band in session.bands where band.enabled {
      guard let marker = CurveBandEditingMath.marker(
        bandID: band.id,
        bands: session.bands,
        parameters: parameters
      ) else { continue }
      if distance(
        point,
        mapper.point(speedMPH: marker.speedMPH, acceleration: marker.accelerationMPS2)
      ) <= 12 {
        return .band(band.id)
      }
    }

    if abs(point.y - mapper.point(speedMPH: 0, acceleration: parameters.minLat).y) < 10 {
      return .handle(.minimumRail)
    }
    if abs(point.y - mapper.point(speedMPH: 0, acceleration: parameters.maxLat).y) < 10 {
      return .handle(.maximumRail)
    }
    return .none
  }

  private func renderedCurveSamples() -> [CurveSample] {
    VTSCMath.sampleCurve(
      parameters: session.parameters,
      bands: session.bands,
      enforceMonotonicSpeed: true
    )
  }

  private func plotMaximumAcceleration(samples: [CurveSample]) -> Double {
    let maximum = samples.lazy.map(\.lateralAcceleration).filter(\.isFinite).max() ?? 6.0
    return max(6.0, ceil(maximum))
  }

  private func curvePoints(mapper: PlotMapper, samples: [CurveSample]? = nil) -> [CGPoint] {
    let visibleSpeedRange = -2.0 ... (mapper.maximumSpeedMPH + 2.0)
    return (samples ?? renderedCurveSamples())
      .filter { visibleSpeedRange.contains($0.speedMPH) }
      .map { mapper.point(speedMPH: $0.speedMPH, acceleration: $0.lateralAcceleration) }
  }

  private func draw(
    context: inout GraphicsContext,
    mapper: PlotMapper,
    curveSamples: [CurveSample]
  ) {
    context.fill(Path(roundedRect: mapper.plot, cornerRadius: 7), with: .color(CurvePalette.panel))
    context.stroke(Path(roundedRect: mapper.plot, cornerRadius: 7), with: .color(CurvePalette.separator), lineWidth: 1)
    drawGrid(context: &context, mapper: mapper)

    let visibleSpeedRange = -2.0 ... (mapper.maximumSpeedMPH + 2.0)
    let baseline = VTSCMath.sampleCurve(parameters: session.checkoutBaseline, bands: [], count: 256)
      .filter { visibleSpeedRange.contains($0.speedMPH) }
      .map { mapper.point(speedMPH: $0.speedMPH, acceleration: $0.lateralAcceleration) }
    drawPolyline(baseline, context: &context, color: .secondary.opacity(0.5), width: 1.5, dash: [7, 5])
    drawSelectedBandZone(context: &context, mapper: mapper)
    let proposed = curvePoints(mapper: mapper, samples: curveSamples)
    drawPolyline(proposed, context: &context, color: CurvePalette.accent.opacity(0.18), width: 7)
    drawPolyline(proposed, context: &context, color: CurvePalette.accent, width: 2.5)

    drawRails(context: &context, mapper: mapper)
    drawBandMarkers(context: &context, mapper: mapper)
    drawMacroHandles(context: &context, mapper: mapper)
    drawHover(context: &context, mapper: mapper)
  }

  private func drawGrid(context: inout GraphicsContext, mapper: PlotMapper) {
    let numberFont = Font.system(size: 18, weight: .semibold, design: .monospaced)
    let titleFont = Font.system(size: 17, weight: .bold, design: .rounded)

    let finalGridSpeed = Int(floor(mapper.maximumSpeedMPH / 5.0)) * 5
    for mph in stride(from: 0, through: finalGridSpeed, by: 5) {
      let x = mapper.point(speedMPH: Double(mph), acceleration: 0).x
      var path = Path(); path.move(to: CGPoint(x: x, y: mapper.plot.minY)); path.addLine(to: CGPoint(x: x, y: mapper.plot.maxY))
      context.stroke(path, with: .color(.white.opacity(mph % 10 == 0 ? 0.16 : 0.06)), lineWidth: mph % 10 == 0 ? 1.25 : 0.5)
      if mph % 10 == 0 {
        var tick = Path()
        tick.move(to: CGPoint(x: x, y: mapper.plot.maxY))
        tick.addLine(to: CGPoint(x: x, y: mapper.plot.maxY + 8))
        context.stroke(tick, with: .color(CurvePalette.axisText.opacity(0.72)), lineWidth: 1.5)
        context.draw(
          Text("\(mph)").font(numberFont).foregroundStyle(CurvePalette.axisText),
          at: CGPoint(x: x, y: mapper.plot.maxY + 22),
          anchor: .center
        )
      }
    }
    let halfStepCount = Int(ceil(mapper.maximumAcceleration * 2.0))
    for tick in 0 ... halfStepCount {
      let value = Double(tick) / 2
      let y = mapper.point(speedMPH: 0, acceleration: value).y
      var path = Path(); path.move(to: CGPoint(x: mapper.plot.minX, y: y)); path.addLine(to: CGPoint(x: mapper.plot.maxX, y: y))
      context.stroke(path, with: .color(.white.opacity(tick % 2 == 0 ? 0.16 : 0.06)), lineWidth: tick % 2 == 0 ? 1.25 : 0.5)
      if tick % 2 == 0 {
        var marker = Path()
        marker.move(to: CGPoint(x: mapper.plot.minX - 8, y: y))
        marker.addLine(to: CGPoint(x: mapper.plot.minX, y: y))
        context.stroke(marker, with: .color(CurvePalette.axisText.opacity(0.72)), lineWidth: 1.5)
        context.draw(
          Text("\(tick / 2)").font(numberFont).foregroundStyle(CurvePalette.axisText),
          at: CGPoint(x: mapper.plot.minX - 15, y: y),
          anchor: .trailing
        )
      }
    }
    context.draw(
      Text(
        mapper.maximumSpeedMPH == CurvePlotSpeedDomain.legacyMaximumSpeedMPH
          ? "Curve speed (mph)"
          : "Curve speed (mph)  ·  0–\(String(format: "%.1f", mapper.maximumSpeedMPH))"
      )
      .font(titleFont)
      .foregroundStyle(CurvePalette.axisTitle),
      at: CGPoint(x: mapper.plot.midX, y: mapper.plot.maxY + 55),
      anchor: .center
    )
    var verticalTitleContext = context
    verticalTitleContext.translateBy(x: 21, y: mapper.plot.midY)
    verticalTitleContext.rotate(by: .degrees(-90))
    verticalTitleContext.draw(
      Text("Lateral acceleration (m/s²)").font(titleFont).foregroundStyle(CurvePalette.axisTitle),
      at: .zero,
      anchor: .center
    )
  }

  private func drawRails(context: inout GraphicsContext, mapper: PlotMapper) {
    for (value, name, handle) in [
      (session.parameters.minLat, "Tight-Curve Ceiling", BuiltInHandle.minimumRail),
      (session.parameters.maxLat, "Straight-Road Ceiling", BuiltInHandle.maximumRail),
    ] {
      let y = mapper.point(speedMPH: 0, acceleration: value).y
      var line = Path(); line.move(to: CGPoint(x: mapper.plot.minX, y: y)); line.addLine(to: CGPoint(x: mapper.plot.maxX, y: y))
      context.stroke(line, with: .color(CurvePalette.rail.opacity(0.85)), style: StrokeStyle(lineWidth: 1, dash: [7, 5]))
      context.draw(
        Text("\(name)  \(String(format: "%.2f", value))")
          .font(.system(size: 14, weight: .semibold))
          .foregroundStyle(CurvePalette.rail),
        at: CGPoint(x: mapper.plot.maxX - 6, y: y - 4),
        anchor: .bottomTrailing
      )
      let center = CGPoint(x: mapper.plot.minX + 12, y: y)
      context.fill(Path(roundedRect: CGRect(x: center.x - 9, y: center.y - 7, width: 18, height: 14), cornerRadius: 7), with: .color(CurvePalette.rail))
      if session.plot.selected == .handle(handle) || session.plot.hoveredTarget == .handle(handle) {
        context.stroke(Path(ellipseIn: CGRect(x: center.x - 13, y: center.y - 13, width: 26, height: 26)), with: .color(CurvePalette.rail), lineWidth: 1.5)
      }
    }
  }

  private func drawMacroHandles(context: inout GraphicsContext, mapper: PlotMapper) {
    let midpoint = 0.5 * (session.knobs.tightCurveAcceleration + session.knobs.straightRoadAcceleration)
    let half = VTSCMath.wingHalfWidth(sharpness: session.knobs.sharpness)
    let left = mapper.point(speedMPH: max(session.knobs.transitionSpeedMPH - half, 1), acceleration: midpoint)
    let right = mapper.point(
      speedMPH: min(session.knobs.transitionSpeedMPH + half, mapper.maximumSpeedMPH - 1),
      acceleration: midpoint
    )
    let center = mapper.point(speedMPH: session.knobs.transitionSpeedMPH, acceleration: midpoint)
    var bar = Path(); bar.move(to: left); bar.addLine(to: right)
    context.stroke(bar, with: .color(CurvePalette.accent.opacity(0.65)), lineWidth: 2)
    for (point, handle) in [(left, BuiltInHandle.leftWing), (right, BuiltInHandle.rightWing)] {
      context.fill(Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)), with: .color(CurvePalette.accent))
      if session.plot.selected == .handle(handle) || session.plot.hoveredTarget == .handle(handle) {
        context.stroke(Path(ellipseIn: CGRect(x: point.x - 10, y: point.y - 10, width: 20, height: 20)), with: .color(CurvePalette.accent), lineWidth: 1.5)
      }
    }
    context.fill(Path(ellipseIn: CGRect(x: center.x - 6, y: center.y - 6, width: 12, height: 12)), with: .color(.white))
    context.stroke(Path(ellipseIn: CGRect(x: center.x - 7, y: center.y - 7, width: 14, height: 14)), with: .color(CurvePalette.accent), lineWidth: 2)
    var guide = Path(); guide.move(to: CGPoint(x: center.x, y: mapper.plot.minY)); guide.addLine(to: CGPoint(x: center.x, y: mapper.plot.maxY))
    context.stroke(guide, with: .color(.white.opacity(0.1)), lineWidth: 1)
  }

  private func drawBandMarkers(context: inout GraphicsContext, mapper: PlotMapper) {
    for (index, band) in session.bands.enumerated() where band.enabled {
      guard let marker = CurveBandEditingMath.marker(
        bandID: band.id,
        bands: session.bands,
        parameters: session.parameters
      ) else { continue }
      let point = mapper.point(
        speedMPH: marker.speedMPH,
        acceleration: marker.accelerationMPS2
      )
      let selected = session.plot.selected == .band(band.id)
      let radius: CGFloat = selected ? 7 : 5
      context.fill(Path(ellipseIn: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)), with: .color(CurvePalette.bandColor(for: index)))
      if selected || session.plot.hoveredTarget == .band(band.id) {
        context.stroke(Path(ellipseIn: CGRect(x: point.x - radius - 4, y: point.y - radius - 4, width: radius * 2 + 8, height: radius * 2 + 8)), with: .color(CurvePalette.bandColor(for: index)), lineWidth: 1.5)
      }
    }
  }

  private func drawSelectedBandZone(context: inout GraphicsContext, mapper: PlotMapper) {
    guard case let .band(id) = session.plot.selected,
          let index = session.bands.firstIndex(where: { $0.id == id }),
          session.bands[index].enabled
    else { return }
    let band = session.bands[index]
    let kappa = VTSCMath.centerKappa(for: band, parameters: session.parameters)
    let logCenter = log10(max(kappa, 1e-12))
    let sigma = 0.5 / max(band.q, 0.1)
    func speed(_ logK: Double) -> Double {
      let k = pow(10, logK)
      return sqrt(VTSCMath.evaluate(session.parameters, curvature: k) / k) * VTSCMath.metersPerSecondToMPH
    }
    let x1 = mapper.point(speedMPH: speed(logCenter - 2 * sigma), acceleration: 0).x
    let x2 = mapper.point(speedMPH: speed(logCenter + 2 * sigma), acceleration: 0).x
    let rect = CGRect(x: min(x1, x2), y: mapper.plot.minY, width: abs(x2 - x1), height: mapper.plot.height).intersection(mapper.plot)
    context.fill(Path(rect), with: .color(CurvePalette.bandColor(for: index).opacity(0.09)))
  }

  private func drawHover(context: inout GraphicsContext, mapper: PlotMapper) {
    guard let point = session.plot.hoverPoint, let readout = session.plot.hoverReadout else { return }
    var vertical = Path(); vertical.move(to: CGPoint(x: point.x, y: mapper.plot.minY)); vertical.addLine(to: CGPoint(x: point.x, y: mapper.plot.maxY))
    var horizontal = Path(); horizontal.move(to: CGPoint(x: mapper.plot.minX, y: point.y)); horizontal.addLine(to: CGPoint(x: mapper.plot.maxX, y: point.y))
    context.stroke(vertical, with: .color(.secondary.opacity(0.45)), lineWidth: 1)
    context.stroke(horizontal, with: .color(.secondary.opacity(0.45)), lineWidth: 1)
    let text = String(format: "%.1f mph  •  %.2f m/s²", readout.speedMPH, readout.acceleration)
    context.draw(
      Text(text).font(.system(size: 15, weight: .semibold, design: .monospaced)),
      at: CGPoint(x: mapper.plot.maxX - 8, y: mapper.plot.minY + 10),
      anchor: .topTrailing
    )
  }

  private func drawPolyline(
    _ points: [CGPoint],
    context: inout GraphicsContext,
    color: Color,
    width: CGFloat,
    dash: [CGFloat] = []
  ) {
    guard let first = points.first else { return }
    var path = Path(); path.move(to: first)
    points.dropFirst().forEach { path.addLine(to: $0) }
    context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round, dash: dash))
  }

  private func distance(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat { hypot(lhs.x - rhs.x, lhs.y - rhs.y) }

  private func hitPolyline(_ point: CGPoint, points: [CGPoint], tolerance: CGFloat) -> Bool {
    guard points.count > 1 else { return false }
    for index in 1 ..< points.count {
      if distanceToSegment(point, points[index - 1], points[index]) <= tolerance { return true }
    }
    return false
  }

  private func distanceToSegment(_ point: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
    let dx = b.x - a.x, dy = b.y - a.y
    let lengthSquared = dx * dx + dy * dy
    guard lengthSquared > 1e-6 else { return distance(point, a) }
    let t = min(max(((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared, 0), 1)
    return distance(point, CGPoint(x: a.x + t * dx, y: a.y + t * dy))
  }
}

private struct PlotEventLayer: NSViewRepresentable {
  var handler: (PlotPointerEvent) -> Void

  func makeNSView(context: Context) -> PlotEventNSView {
    let view = PlotEventNSView()
    view.handler = handler
    return view
  }

  func updateNSView(_ nsView: PlotEventNSView, context: Context) {
    nsView.handler = handler
  }
}

@MainActor
private final class PlotEventNSView: NSView {
  var handler: ((PlotPointerEvent) -> Void)?
  private var tracking: NSTrackingArea?

  override var isFlipped: Bool { true }
  override var acceptsFirstResponder: Bool { true }

  override func updateTrackingAreas() {
    if let tracking { removeTrackingArea(tracking) }
    let area = NSTrackingArea(
      rect: bounds,
      options: [.activeInKeyWindow, .mouseMoved, .mouseEnteredAndExited, .inVisibleRect],
      owner: self,
      userInfo: nil
    )
    addTrackingArea(area)
    tracking = area
    super.updateTrackingAreas()
  }

  override func mouseDown(with event: NSEvent) {
    window?.makeFirstResponder(self)
    send(.down, event)
  }
  override func mouseDragged(with event: NSEvent) { send(.dragged, event) }
  override func mouseUp(with event: NSEvent) { send(.up, event) }
  override func mouseMoved(with event: NSEvent) { send(.moved, event) }
  override func mouseExited(with event: NSEvent) { send(.exited, event) }
  override func rightMouseDown(with event: NSEvent) { send(.secondaryDown, event) }
  override func keyDown(with event: NSEvent) {
    if event.keyCode == 53 {
      handler?(PlotPointerEvent(kind: .escape, location: .zero))
    } else {
      super.keyDown(with: event)
    }
  }

  private func send(_ kind: PlotPointerKind, _ event: NSEvent) {
    handler?(PlotPointerEvent(
      kind: kind,
      location: convert(event.locationInWindow, from: nil),
      shift: event.modifierFlags.contains(.shift)
    ))
  }
}
