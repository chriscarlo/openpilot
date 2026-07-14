import AppKit
import MapKit
import SwiftUI
import VTSCTunerCore

private final class MapCalibrationAnnotation: NSObject, MKAnnotation {
  @objc dynamic var coordinate: CLLocationCoordinate2D
  let number: Int
  let label: String

  init(sample: MapCalibrationSample, number: Int) {
    coordinate = CLLocationCoordinate2D(latitude: sample.latitude, longitude: sample.longitude)
    self.number = number
    label = sample.roadName
  }

  var title: String? { "Sample \(number): \(label)" }
}

private final class MapRoadLabelAnnotation: NSObject, MKAnnotation {
  @objc dynamic var coordinate: CLLocationCoordinate2D
  let text: String
  let fontSize: CGFloat
  let isMajor: Bool
  let isSelectedRoad: Bool

  init(
    coordinate: CLLocationCoordinate2D,
    text: String,
    fontSize: CGFloat,
    isMajor: Bool,
    isSelectedRoad: Bool
  ) {
    self.coordinate = coordinate
    self.text = text
    self.fontSize = fontSize
    self.isMajor = isMajor
    self.isSelectedRoad = isSelectedRoad
  }
}

private final class MapSearchAnnotation: NSObject, MKAnnotation {
  @objc dynamic var coordinate: CLLocationCoordinate2D
  let text: String
  let fontSize: CGFloat

  init(coordinate: CLLocationCoordinate2D, text: String, fontSize: CGFloat) {
    self.coordinate = coordinate
    self.text = text
    self.fontSize = fontSize
  }
}

private final class MapReadableLabelView: MKAnnotationView {
  enum Accent {
    case road
    case selectedRoad
    case search
  }

  private let textField = NSTextField(labelWithString: "")

  override init(annotation: (any MKAnnotation)?, reuseIdentifier: String?) {
    super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)
    installTextField()
  }

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    installTextField()
  }

  private func installTextField() {
    wantsLayer = true
    textField.alignment = .center
    textField.isBezeled = false
    textField.isEditable = false
    textField.isSelectable = false
    textField.drawsBackground = false
    textField.lineBreakMode = .byTruncatingTail
    textField.usesSingleLineMode = true
    addSubview(textField)
    isEnabled = false
    canShowCallout = false
  }

  func configure(text: String, fontSize: CGFloat, accent: Accent) {
    textField.stringValue = text
    textField.font = .systemFont(ofSize: fontSize, weight: accent == .road ? .semibold : .bold)
    textField.textColor = .white

    let measured = (text as NSString).size(withAttributes: [.font: textField.font as Any])
    let width = ceil(min(380, max(72, measured.width + 22)))
    let height = ceil(max(32, measured.height + 12))
    frame.size = CGSize(width: width, height: height)
    textField.frame = bounds.insetBy(dx: 10, dy: 5)

    let increasedContrast = NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast
    layer?.backgroundColor = NSColor.black
      .withAlphaComponent(increasedContrast ? 0.98 : 0.86)
      .cgColor
    layer?.cornerRadius = 8
    layer?.borderWidth = increasedContrast ? 2 : 1
    let borderColor: NSColor = switch accent {
    case .road: .white
    case .selectedRoad: .systemPink
    case .search: .systemBlue
    }
    layer?.borderColor = borderColor
      .withAlphaComponent(increasedContrast ? 1 : (accent == .road ? 0.42 : 0.95))
      .cgColor
    layer?.shadowColor = NSColor.black.cgColor
    layer?.shadowOpacity = 0.72
    layer?.shadowRadius = 4
    layer?.shadowOffset = CGSize(width: 0, height: -1)
    setAccessibilityElement(true)
    setAccessibilityLabel(text)
  }

  override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

struct StrategicMapView: NSViewRepresentable {
  var ways: [MapRenderedWay]
  var purpose: MapPreviewPurpose
  var mode: MapSpeedDisplayMode
  var wholeCurveMode: MapWholeCurveDisplayMode
  var wholeCurveEvents: [MapWholeCurveEvent]
  var selectedWholeCurveEventID: String?
  var roadLabelSize: MapRoadLabelSize
  var selection: MapRoadSelection?
  var calibrationSamples: [MapCalibrationSample]
  var renderRevision: Int
  var cameraDestination: MapCameraDestination?
  var searchMarkerVisible: Bool
  var onViewportChanged: (MapTileBounds) -> Void
  var onSelection: (MapRoadSelection?) -> Void
  var onWholeCurveSelection: (String?) -> Void

  func makeCoordinator() -> Coordinator {
    Coordinator(
      onViewportChanged: onViewportChanged,
      onSelection: onSelection,
      onWholeCurveSelection: onWholeCurveSelection
    )
  }

  func makeNSView(context: Context) -> MKMapView {
    let mapView = MKMapView()
    let configuration = MKStandardMapConfiguration(elevationStyle: .flat, emphasisStyle: .default)
    configuration.pointOfInterestFilter = .excludingAll
    configuration.showsTraffic = false
    mapView.preferredConfiguration = configuration
    mapView.showsZoomControls = true
    mapView.showsCompass = true
    mapView.showsScale = true
    mapView.isRotateEnabled = false
    mapView.isPitchEnabled = false
    mapView.delegate = context.coordinator

    let click = NSClickGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.clicked(_:)))
    click.numberOfClicksRequired = 1
    click.buttonMask = 0x1
    mapView.addGestureRecognizer(click)
    context.coordinator.mapView = mapView
    context.coordinator.publishViewport(mapView)
    return mapView
  }

  func updateNSView(_ mapView: MKMapView, context: Context) {
    context.coordinator.onViewportChanged = onViewportChanged
    context.coordinator.onSelection = onSelection
    context.coordinator.onWholeCurveSelection = onWholeCurveSelection
    context.coordinator.update(
      mapView,
      ways: ways,
      purpose: purpose,
      mode: mode,
      wholeCurveMode: wholeCurveMode,
      wholeCurveEvents: wholeCurveEvents,
      selectedWholeCurveEventID: selectedWholeCurveEventID,
      roadLabelSize: roadLabelSize,
      selection: selection,
      calibrationSamples: calibrationSamples,
      revision: renderRevision
    )
    context.coordinator.moveCameraIfNeeded(mapView, destination: cameraDestination)
    context.coordinator.setSearchMarkerVisibility(mapView, visible: searchMarkerVisible)
  }

  final class Coordinator: NSObject, MKMapViewDelegate {
    weak var mapView: MKMapView?
    var onViewportChanged: (MapTileBounds) -> Void
    var onSelection: (MapRoadSelection?) -> Void
    var onWholeCurveSelection: (String?) -> Void

    private var currentRenderKey = ""
    private var currentPurpose: MapPreviewPurpose = .calibration
    private var currentCalibrationMode: MapSpeedDisplayMode = .currentlyBaked
    private var currentWholeCurveMode: MapWholeCurveDisplayMode = .wholeCurve
    private var currentCameraID: UUID?
    private var currentRoadLabelSize: MapRoadLabelSize = .appleOnly
    private var wayByOverlay: [ObjectIdentifier: MapRenderedWay] = [:]
    private var wholeCurveEventByOverlay: [ObjectIdentifier: MapWholeCurveEvent] = [:]
    private var selectedOverlayID: ObjectIdentifier?
    private var selectedAnnotation: MKPointAnnotation?
    private var searchAnnotation: MapSearchAnnotation?
    private var roadLabelAnnotations: [MapRoadLabelAnnotation] = []
    private var calibrationAnnotations: [MapCalibrationAnnotation] = []
    private var overlays: [MKPolyline] = []

    private struct RoadLabelCandidate {
      var text: String
      var coordinate: CLLocationCoordinate2D
      var distanceSquared: Double
      var visibleNodeCount: Int
      var lanes: UInt8
      var isMajor: Bool
      var isSelectedRoad: Bool
    }

    init(
      onViewportChanged: @escaping (MapTileBounds) -> Void,
      onSelection: @escaping (MapRoadSelection?) -> Void,
      onWholeCurveSelection: @escaping (String?) -> Void
    ) {
      self.onViewportChanged = onViewportChanged
      self.onSelection = onSelection
      self.onWholeCurveSelection = onWholeCurveSelection
    }

    func update(
      _ mapView: MKMapView,
      ways: [MapRenderedWay],
      purpose: MapPreviewPurpose,
      mode: MapSpeedDisplayMode,
      wholeCurveMode: MapWholeCurveDisplayMode,
      wholeCurveEvents: [MapWholeCurveEvent],
      selectedWholeCurveEventID: String?,
      roadLabelSize: MapRoadLabelSize,
      selection: MapRoadSelection?,
      calibrationSamples: [MapCalibrationSample],
      revision: Int
    ) {
      let selectedWayID = selection?.way.id
      let sampleKey = calibrationSamples.map { $0.id.uuidString }.joined(separator: ",")
      let eventKey = wholeCurveEvents.map(\.id).joined(separator: ",")
      let key = "\(purpose.rawValue)|\(mode.rawValue)|\(wholeCurveMode.rawValue)|\(roadLabelSize.rawValue)|\(selectedWayID ?? "")|\(selection?.nodeIndex ?? -1)|\(selectedWholeCurveEventID ?? "")|\(revision)|\(ways.count)|\(eventKey)|\(sampleKey)"
      guard key != currentRenderKey else { return }
      currentRenderKey = key
      currentPurpose = purpose
      currentCalibrationMode = mode
      currentWholeCurveMode = wholeCurveMode
      currentRoadLabelSize = roadLabelSize

      if !overlays.isEmpty { mapView.removeOverlays(overlays) }
      overlays.removeAll(keepingCapacity: true)
      wayByOverlay.removeAll(keepingCapacity: true)
      wholeCurveEventByOverlay.removeAll(keepingCapacity: true)
      selectedOverlayID = nil
      if let selectedAnnotation { mapView.removeAnnotation(selectedAnnotation) }
      selectedAnnotation = nil
      if !roadLabelAnnotations.isEmpty { mapView.removeAnnotations(roadLabelAnnotations) }
      roadLabelAnnotations = []
      if !calibrationAnnotations.isEmpty { mapView.removeAnnotations(calibrationAnnotations) }
      calibrationAnnotations = []

      var selectedPolyline: MKPolyline?
      for way in ways where way.nodes.count >= 2 {
        var coordinates = way.nodes.map {
          CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        let polyline = MKPolyline(coordinates: &coordinates, count: coordinates.count)
        overlays.append(polyline)
        wayByOverlay[ObjectIdentifier(polyline)] = way
        if purpose == .calibration, way.id == selectedWayID { selectedPolyline = polyline }
      }
      mapView.addOverlays(overlays, level: .aboveRoads)

      if purpose == .wholeCurveStudy {
        var eventOverlays: [MKPolyline] = []
        for event in wholeCurveEvents where event.points.count >= 2 {
          var coordinates = event.points.map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
          }
          let polyline = MKPolyline(coordinates: &coordinates, count: coordinates.count)
          eventOverlays.append(polyline)
          wholeCurveEventByOverlay[ObjectIdentifier(polyline)] = event
          if event.id == selectedWholeCurveEventID { selectedPolyline = polyline }
        }
        overlays.append(contentsOf: eventOverlays)
        mapView.addOverlays(eventOverlays, level: .aboveRoads)
      }

      if let selectedPolyline {
        var coordinates = coordinates(for: selectedPolyline)
        let highlight = MKPolyline(coordinates: &coordinates, count: coordinates.count)
        selectedOverlayID = ObjectIdentifier(highlight)
        overlays.append(highlight)
        mapView.addOverlay(highlight, level: .aboveRoads)
      }
      if purpose == .calibration, let selection {
        let annotation = MKPointAnnotation()
        annotation.coordinate = CLLocationCoordinate2D(
          latitude: selection.node.latitude,
          longitude: selection.node.longitude
        )
        annotation.title = "Selected VTSC sample"
        selectedAnnotation = annotation
        mapView.addAnnotation(annotation)
      }
      calibrationAnnotations = calibrationSamples.enumerated().map { offset, sample in
        MapCalibrationAnnotation(sample: sample, number: offset + 1)
      }
      if !calibrationAnnotations.isEmpty { mapView.addAnnotations(calibrationAnnotations) }
      roadLabelAnnotations = makeRoadLabelAnnotations(
        mapView: mapView,
        ways: ways,
        size: roadLabelSize,
        selectedWayID: purpose == .calibration ? selectedWayID : nil
      )
      if !roadLabelAnnotations.isEmpty { mapView.addAnnotations(roadLabelAnnotations) }
    }

    func moveCameraIfNeeded(_ mapView: MKMapView, destination: MapCameraDestination?) {
      guard let destination, destination.id != currentCameraID else { return }
      currentCameraID = destination.id
      let span = mapView.region.span
      let isWorldView = span.latitudeDelta > 2 || span.longitudeDelta > 2
      let targetSpan: MKCoordinateSpan
      if let latitudeDelta = destination.latitudeDelta,
         let longitudeDelta = destination.longitudeDelta {
        targetSpan = MKCoordinateSpan(
          latitudeDelta: latitudeDelta,
          longitudeDelta: longitudeDelta
        )
      } else {
        targetSpan = isWorldView
          ? MKCoordinateSpan(latitudeDelta: 0.045, longitudeDelta: 0.045)
          : span
      }
      if let searchAnnotation { mapView.removeAnnotation(searchAnnotation) }
      searchAnnotation = nil
      if destination.showsMarker {
        let requestedSize = currentRoadLabelSize.pointSize(latitudeDelta: targetSpan.latitudeDelta) ?? 24
        let preferredBody = NSFont.preferredFont(forTextStyle: .body, options: [:]).pointSize
        let marker = MapSearchAnnotation(
          coordinate: CLLocationCoordinate2D(
            latitude: destination.latitude,
            longitude: destination.longitude
          ),
          text: "⌖ \(destination.title)",
          fontSize: min(34, max(24, max(CGFloat(requestedSize), preferredBody * 1.8)))
        )
        searchAnnotation = marker
        mapView.addAnnotation(marker)
      }
      mapView.setRegion(
        MKCoordinateRegion(
          center: CLLocationCoordinate2D(
            latitude: destination.latitude,
            longitude: destination.longitude
          ),
          span: targetSpan
        ),
        animated: true
      )
    }

    func setSearchMarkerVisibility(_ mapView: MKMapView, visible: Bool) {
      guard !visible, let searchAnnotation else { return }
      mapView.removeAnnotation(searchAnnotation)
      self.searchAnnotation = nil
    }

    func mapView(_ mapView: MKMapView, regionDidChangeAnimated animated: Bool) {
      publishViewport(mapView)
    }

    func publishViewport(_ mapView: MKMapView) {
      let region = mapView.region
      let halfLatitude = region.span.latitudeDelta * 0.5
      let halfLongitude = region.span.longitudeDelta * 0.5
      let bounds = MapTileBounds(
        minLatitude: max(-90, region.center.latitude - halfLatitude),
        minLongitude: max(-180, region.center.longitude - halfLongitude),
        maxLatitude: min(90, region.center.latitude + halfLatitude),
        maxLongitude: min(180, region.center.longitude + halfLongitude)
      )
      onViewportChanged(bounds)
    }

    func mapView(_ mapView: MKMapView, rendererFor overlay: any MKOverlay) -> MKOverlayRenderer {
      guard let polyline = overlay as? MKPolyline else {
        return MKOverlayRenderer(overlay: overlay)
      }
      if ObjectIdentifier(polyline) == selectedOverlayID {
        let renderer = MKPolylineRenderer(polyline: polyline)
        renderer.strokeColor = .white
        renderer.lineWidth = 7
        renderer.lineDashPattern = [3, 5]
        renderer.alpha = 0.85
        renderer.lineJoin = .round
        renderer.lineCap = .round
        return renderer
      }
      if let event = wholeCurveEventByOverlay[ObjectIdentifier(polyline)] {
        let renderer = MKGradientPolylineRenderer(polyline: polyline)
        let values = event.points.map { point -> NSColor in
          switch currentWholeCurveMode {
          case .currentMapd:
            Self.speedColor(mph: point.currentMapdSpeedMPH)
          case .wholeCurve:
            Self.speedColor(mph: point.wholeCurveSpeedMPH)
          case .difference:
            Self.deltaColor(mph: point.wholeCurveSpeedMPH - point.currentMapdSpeedMPH)
          }
        }
        renderer.setColors(values, locations: gradientLocations(for: polyline))
        renderer.lineWidth = 7
        renderer.lineJoin = .round
        renderer.lineCap = .round
        renderer.alpha = event.flags.isEmpty ? 0.96 : 0.78
        if !event.flags.isEmpty { renderer.lineDashPattern = [10, 6] }
        return renderer
      }
      guard let way = wayByOverlay[ObjectIdentifier(polyline)] else {
        return MKPolylineRenderer(polyline: polyline)
      }
      if currentPurpose == .wholeCurveStudy {
        let renderer = MKPolylineRenderer(polyline: polyline)
        renderer.strokeColor = .tertiaryLabelColor
        renderer.lineWidth = 2
        renderer.alpha = 0.28
        renderer.lineJoin = .round
        renderer.lineCap = .round
        return renderer
      }
      let renderer = MKGradientPolylineRenderer(polyline: polyline)
      let values = way.nodes.map { node -> NSColor in
        switch currentCalibrationMode {
        case .currentlyBaked:
          guard let speed = node.bakedSpeedMPS else { return .tertiaryLabelColor }
          return Self.speedColor(mph: speed.mapMPH)
        case .proposedBake:
          return Self.speedColor(mph: node.proposedSpeedMPS.mapMPH)
        case .delta:
          guard let baked = node.bakedSpeedMPS else { return .tertiaryLabelColor }
          return Self.deltaColor(mph: (node.proposedSpeedMPS - baked).mapMPH)
        }
      }
      renderer.setColors(values, locations: gradientLocations(for: polyline))
      renderer.lineWidth = 5
      renderer.lineJoin = .round
      renderer.lineCap = .round
      renderer.alpha = 0.9
      return renderer
    }

    func mapView(_ mapView: MKMapView, viewFor annotation: any MKAnnotation) -> MKAnnotationView? {
      if let calibration = annotation as? MapCalibrationAnnotation {
        let identifier = "vtsc-calibration-sample"
        let view = (mapView.dequeueReusableAnnotationView(withIdentifier: identifier)
          as? MKMarkerAnnotationView)
          ?? MKMarkerAnnotationView(annotation: calibration, reuseIdentifier: identifier)
        view.annotation = calibration
        view.markerTintColor = .systemPurple
        view.glyphText = String(calibration.number)
        view.canShowCallout = true
        view.displayPriority = .required
        return view
      }
      if let roadLabel = annotation as? MapRoadLabelAnnotation {
        let identifier = "vtsc-readable-road-label"
        let view = (mapView.dequeueReusableAnnotationView(withIdentifier: identifier)
          as? MapReadableLabelView)
          ?? MapReadableLabelView(annotation: roadLabel, reuseIdentifier: identifier)
        view.annotation = roadLabel
        view.configure(
          text: roadLabel.text,
          fontSize: roadLabel.fontSize,
          accent: roadLabel.isSelectedRoad ? .selectedRoad : .road
        )
        view.displayPriority = roadLabel.isSelectedRoad
          ? .required
          : (roadLabel.isMajor ? .defaultHigh : .defaultLow)
        view.collisionMode = .rectangle
        return view
      }
      if let search = annotation as? MapSearchAnnotation {
        let identifier = "vtsc-map-search-label"
        let view = (mapView.dequeueReusableAnnotationView(withIdentifier: identifier)
          as? MapReadableLabelView)
          ?? MapReadableLabelView(annotation: search, reuseIdentifier: identifier)
        view.annotation = search
        view.configure(text: search.text, fontSize: search.fontSize, accent: .search)
        view.displayPriority = .required
        view.collisionMode = .rectangle
        return view
      }
      guard annotation === selectedAnnotation else { return nil }
      let identifier = "selected-vtsc-node"
      let view = mapView.dequeueReusableAnnotationView(withIdentifier: identifier)
        ?? MKAnnotationView(annotation: annotation, reuseIdentifier: identifier)
      view.annotation = annotation
      view.frame = CGRect(x: 0, y: 0, width: 18, height: 18)
      view.wantsLayer = true
      view.layer?.cornerRadius = 9
      view.layer?.backgroundColor = NSColor.systemPink.cgColor
      view.layer?.borderColor = NSColor.white.cgColor
      view.layer?.borderWidth = 3
      view.canShowCallout = false
      return view
    }

    @objc func clicked(_ recognizer: NSClickGestureRecognizer) {
      guard recognizer.state == .ended, let mapView else { return }
      let clickPoint = recognizer.location(in: mapView)
      if currentPurpose == .wholeCurveStudy {
        var bestEvent: (distance: CGFloat, eventID: String)?
        for overlay in overlays {
          guard let event = wholeCurveEventByOverlay[ObjectIdentifier(overlay)],
                event.points.count >= 2
          else { continue }
          for index in 0 ..< event.points.count - 1 {
            let start = mapView.convert(
              CLLocationCoordinate2D(
                latitude: event.points[index].latitude,
                longitude: event.points[index].longitude
              ),
              toPointTo: mapView
            )
            let end = mapView.convert(
              CLLocationCoordinate2D(
                latitude: event.points[index + 1].latitude,
                longitude: event.points[index + 1].longitude
              ),
              toPointTo: mapView
            )
            let result = Self.distanceFromPoint(clickPoint, toSegmentFrom: start, to: end)
            if result.distance <= 14,
               result.distance < (bestEvent?.distance ?? .greatestFiniteMagnitude) {
              bestEvent = (result.distance, event.id)
            }
          }
        }
        onWholeCurveSelection(bestEvent?.eventID)
        return
      }
      var best: (distance: CGFloat, way: MapRenderedWay, segment: Int, fraction: Double)?
      for overlay in overlays {
        guard let way = wayByOverlay[ObjectIdentifier(overlay)], way.nodes.count >= 2 else { continue }
        for index in 0 ..< way.nodes.count - 1 {
          let start = mapView.convert(
            CLLocationCoordinate2D(latitude: way.nodes[index].latitude, longitude: way.nodes[index].longitude),
            toPointTo: mapView
          )
          let end = mapView.convert(
            CLLocationCoordinate2D(latitude: way.nodes[index + 1].latitude, longitude: way.nodes[index + 1].longitude),
            toPointTo: mapView
          )
          let result = Self.distanceFromPoint(clickPoint, toSegmentFrom: start, to: end)
          if result.distance <= 12, result.distance < (best?.distance ?? .greatestFiniteMagnitude) {
            best = (result.distance, way, index, result.fraction)
          }
        }
      }
      guard let best else {
        onSelection(nil)
        return
      }
      let nearestNodeIndex = best.fraction < 0.5 ? best.segment : best.segment + 1
      let nodeIndex = Self.snappedApexIndex(in: best.way, near: nearestNodeIndex)
      onSelection(MapRoadSelection(way: best.way, nodeIndex: nodeIndex, segmentFraction: best.fraction))
    }

    /// A click identifies the road segment; calibration should use the nearby
    /// apex rather than an arbitrary shoulder node whose spacing depends on OSM.
    private static func snappedApexIndex(
      in way: MapRenderedWay,
      near initialIndex: Int,
      maximumAlongRoadDistanceMeters: Double = 120
    ) -> Int {
      guard !way.nodes.isEmpty else { return 0 }
      let initial = min(max(initialIndex, 0), way.nodes.count - 1)
      var candidates: [(index: Int, distance: Double)] = [(initial, 0)]

      var distance = 0.0
      if initial > 0 {
        for index in stride(from: initial - 1, through: 0, by: -1) {
          distance += nodeDistanceMeters(way.nodes[index], way.nodes[index + 1])
          if distance > maximumAlongRoadDistanceMeters { break }
          candidates.append((index, distance))
        }
      }

      distance = 0
      if initial + 1 < way.nodes.count {
        for index in (initial + 1)..<way.nodes.count {
          distance += nodeDistanceMeters(way.nodes[index - 1], way.nodes[index])
          if distance > maximumAlongRoadDistanceMeters { break }
          candidates.append((index, distance))
        }
      }

      let localPeaks = candidates.filter { candidate in
        let index = candidate.index
        guard index > 0, index + 1 < way.nodes.count else { return false }
        let curvature = way.nodes[index].curvature
        return way.nodes[index].curvatureContextComplete
          && curvature >= 1.0e-7
          && curvature >= way.nodes[index - 1].curvature
          && curvature >= way.nodes[index + 1].curvature
      }
      if let nearestPeak = localPeaks.min(by: { lhs, rhs in
        if lhs.distance != rhs.distance { return lhs.distance < rhs.distance }
        return way.nodes[lhs.index].curvature > way.nodes[rhs.index].curvature
      }) {
        return nearestPeak.index
      }
      return initial
    }

    private static func nodeDistanceMeters(_ lhs: MapRenderedNode, _ rhs: MapRenderedNode) -> Double {
      CLLocation(latitude: lhs.latitude, longitude: lhs.longitude)
        .distance(from: CLLocation(latitude: rhs.latitude, longitude: rhs.longitude))
    }

    private func makeRoadLabelAnnotations(
      mapView: MKMapView,
      ways: [MapRenderedWay],
      size: MapRoadLabelSize,
      selectedWayID: String?
    ) -> [MapRoadLabelAnnotation] {
      let latitudeDelta = mapView.region.span.latitudeDelta
      guard latitudeDelta <= 0.30,
            let minimumPointSize = size.pointSize(latitudeDelta: latitudeDelta)
      else { return [] }

      let visibleRect = mapView.visibleMapRect
      let center = MKMapPoint(mapView.region.center)
      var bestByLabel: [String: RoadLabelCandidate] = [:]

      for way in ways {
        guard let text = Self.roadLabelText(for: way) else { continue }
        var nearestCoordinate: CLLocationCoordinate2D?
        var nearestDistanceSquared = Double.greatestFiniteMagnitude
        var visibleNodeCount = 0

        for node in way.nodes {
          let coordinate = CLLocationCoordinate2D(
            latitude: node.latitude,
            longitude: node.longitude
          )
          let point = MKMapPoint(coordinate)
          guard visibleRect.contains(point) else { continue }
          visibleNodeCount += 1
          let dx = point.x - center.x
          let dy = point.y - center.y
          let distanceSquared = dx * dx + dy * dy
          if distanceSquared < nearestDistanceSquared {
            nearestDistanceSquared = distanceSquared
            nearestCoordinate = coordinate
          }
        }

        guard let nearestCoordinate else { continue }
        let candidate = RoadLabelCandidate(
          text: text,
          coordinate: nearestCoordinate,
          distanceSquared: nearestDistanceSquared,
          visibleNodeCount: visibleNodeCount,
          lanes: way.lanes,
          isMajor: !way.reference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || way.lanes >= 3,
          isSelectedRoad: way.id == selectedWayID
        )
        let key = text.folding(
          options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
          locale: .current
        )
        if let existing = bestByLabel[key] {
          if Self.roadLabelCandidate(candidate, outranks: existing) {
            bestByLabel[key] = candidate
          }
        } else {
          bestByLabel[key] = candidate
        }
      }

      let preferredBody = NSFont.preferredFont(forTextStyle: .body, options: [:]).pointSize
      let accessibilityMultiplier: CGFloat = size == .large ? 1.45 : 1.8
      let normalPointSize = min(
        34,
        max(CGFloat(minimumPointSize), preferredBody * accessibilityMultiplier)
      )
      let maximumCount = size.maximumLabelCount(latitudeDelta: latitudeDelta)
      return bestByLabel.values
        .sorted {
          if Self.roadLabelCandidate($0, outranks: $1) { return true }
          if Self.roadLabelCandidate($1, outranks: $0) { return false }
          return $0.text.localizedCaseInsensitiveCompare($1.text) == .orderedAscending
        }
        .prefix(maximumCount)
        .map { candidate in
          MapRoadLabelAnnotation(
            coordinate: candidate.coordinate,
            text: candidate.text,
            fontSize: min(36, normalPointSize + (candidate.isSelectedRoad ? 2 : 0)),
            isMajor: candidate.isMajor,
            isSelectedRoad: candidate.isSelectedRoad
          )
        }
    }

    private static func roadLabelText(for way: MapRenderedWay) -> String? {
      let name = way.name.trimmingCharacters(in: .whitespacesAndNewlines)
      let reference = way.reference.trimmingCharacters(in: .whitespacesAndNewlines)
      if name.isEmpty && reference.isEmpty { return nil }
      if name.isEmpty { return reference }
      if reference.isEmpty || name.caseInsensitiveCompare(reference) == .orderedSame { return name }
      return "\(name) · \(reference)"
    }

    private static func roadLabelCandidate(
      _ candidate: RoadLabelCandidate,
      outranks other: RoadLabelCandidate
    ) -> Bool {
      if candidate.isSelectedRoad != other.isSelectedRoad { return candidate.isSelectedRoad }
      if candidate.isMajor != other.isMajor { return candidate.isMajor }
      if candidate.lanes != other.lanes { return candidate.lanes > other.lanes }
      if candidate.visibleNodeCount != other.visibleNodeCount {
        return candidate.visibleNodeCount > other.visibleNodeCount
      }
      return candidate.distanceSquared < other.distanceSquared
    }

    private func currentMode(from key: String) -> MapSpeedDisplayMode {
      MapSpeedDisplayMode.allCases.first(where: { key.hasPrefix($0.rawValue + "|") }) ?? .currentlyBaked
    }

    private func coordinates(for polyline: MKPolyline) -> [CLLocationCoordinate2D] {
      var coordinates = Array(
        repeating: CLLocationCoordinate2D(latitude: 0, longitude: 0),
        count: polyline.pointCount
      )
      polyline.getCoordinates(&coordinates, range: NSRange(location: 0, length: polyline.pointCount))
      return coordinates
    }

    private func gradientLocations(for polyline: MKPolyline) -> [CGFloat] {
      guard polyline.pointCount > 1 else { return [0] }
      let points = polyline.points()
      var distances = Array(repeating: 0.0, count: polyline.pointCount)
      for index in 1 ..< polyline.pointCount {
        distances[index] = distances[index - 1] + points[index].distance(to: points[index - 1])
      }
      let total = max(distances.last ?? 0, 1)
      return distances.map { CGFloat($0 / total) }
    }

    private static func speedColor(mph: Double) -> NSColor {
      let band = MapSpeedBand.band(forMPH: mph)
      return NSColor(
        calibratedHue: band.hue,
        saturation: 0.84,
        brightness: 0.98,
        alpha: 1
      )
    }

    private static func deltaColor(mph: Double) -> NSColor {
      switch mph {
      case ..<(-8): .systemBlue
      case ..<(-3): .systemTeal
      case ...3: .secondaryLabelColor
      case ...8: .systemOrange
      default: .systemPink
      }
    }

    private static func distanceFromPoint(
      _ point: CGPoint,
      toSegmentFrom start: CGPoint,
      to end: CGPoint
    ) -> (distance: CGFloat, fraction: Double) {
      let dx = end.x - start.x
      let dy = end.y - start.y
      let lengthSquared = dx * dx + dy * dy
      guard lengthSquared > 0 else {
        return (hypot(point.x - start.x, point.y - start.y), 0)
      }
      let fraction = min(max(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 0), 1)
      let nearest = CGPoint(x: start.x + fraction * dx, y: start.y + fraction * dy)
      return (hypot(point.x - nearest.x, point.y - nearest.y), Double(fraction))
    }
  }
}
