import AppKit
import Foundation
import MapKit
import VTSCTunerCore

struct MapPlaceResult: Identifiable {
  var id = UUID()
  var title: String
  var subtitle: String
  var latitude: Double
  var longitude: Double
  var latitudeDelta: Double
  var longitudeDelta: Double
}

@MainActor
final class MapPreviewSession: ObservableObject {
  static let minimumCalibrationSamples = 5
  static let tileRootDefaultsKey = "VTSCTuner.mapTileRootPath"
  static let roadLabelSizeDefaultsKey = "VTSCTuner.mapRoadLabelSize"
  static let openingLocationDefaultsKey = "VTSCTuner.mapOpeningLocation"
  static let calibrationArchiveFilename = "curve_calibration_samples.json"

  @Published var tileRootURL: URL?
  @Published var ways: [MapRenderedWay] = []
  @Published var purpose: MapPreviewPurpose = .wholeCurveStudy {
    didSet {
      guard purpose != oldValue else { return }
      if purpose == .calibration {
        isStudyCurveCaptureActive = false
        studyCaptureBankedNumber = nil
        selectedWholeCurveEventID = nil
        reconcileCalibrationCurvatures(using: ways)
        refreshCalibrationBaselines()
        invalidateFit()
        refreshUncommittedDraftIfNeeded()
      } else {
        isStudyCurveCaptureActive = false
        studyCaptureBankedNumber = nil
        clearSelection()
        focusWholeCurveBank(announce: false)
      }
      renderRevision &+= 1
    }
  }
  @Published var displayMode: MapSpeedDisplayMode = .currentlyBaked
  @Published var wholeCurveDisplayMode: MapWholeCurveDisplayMode = .wholeCurve
  @Published private(set) var wholeCurveEvents: [MapWholeCurveEvent] = []
  @Published var selectedWholeCurveEventID: String?
  @Published var roadLabelSize: MapRoadLabelSize = .appleOnly {
    didSet {
      guard roadLabelSize != oldValue, persistsMapPreferences else { return }
      UserDefaults.standard.set(roadLabelSize.rawValue, forKey: Self.roadLabelSizeDefaultsKey)
      renderRevision &+= 1
    }
  }
  @Published var selection: MapRoadSelection?
  @Published private(set) var isStudyCurveCaptureActive = false
  @Published private(set) var studyCaptureBankedNumber: Int?
  @Published private(set) var draftDesiredSpeedMPH = 0.0
  @Published var renderRevision = 0
  @Published var cameraDestination: MapCameraDestination?
  @Published var statusText = "Choose an offline mapd tile folder or sync tiles from the tici."
  @Published var statusIsError = false
  @Published var isLoading = false
  @Published var isSyncing = false
  @Published var ticiProfile = "commaHome"
  @Published var searchQuery = ""
  @Published var searchResults: [MapPlaceResult] = []
  @Published var isSearching = false
  @Published var searchMarkerVisible = false
  @Published private(set) var selectedPlaceResult: MapPlaceResult?
  @Published private(set) var openingLocation: MapOpeningLocation?
  @Published var calibrationSamples: [MapCalibrationSample] = []
  @Published private(set) var calibrationAnchorKnobs: PlainKnobs?
  @Published private(set) var calibrationPersistenceError: String?
  @Published var fitResult: SigmoidFitResult?
  @Published var fitErrorText: String?
  @Published var isFitting = false

  private var tileStore: MapTileStore?
  private var viewportTask: Task<Void, Never>?
  private var searchTask: Task<Void, Never>?
  private var activeSearch: MKLocalSearch?
  private var searchGeneration: UInt64 = 0
  private var fitTask: Task<Void, Never>?
  private var fitGeneration: UInt64 = 0
  private var lastViewport: MapTileBounds?
  private var currentParameters = SigmoidParameters.checkoutFallback
  private var currentBands: [EQBand] = []
  private var tileCoverage: MapTileBounds?
  private var draftWasEdited = false
  private var draftBaselineDesiredSpeedMPH = 0.0
  private let persistsCalibrationSamples: Bool
  private let persistsMapPreferences: Bool

  init(
    loadPersistedState: Bool = true,
    persistsCalibrationSamples: Bool = true,
    persistsMapPreferences: Bool = true
  ) {
    self.persistsCalibrationSamples = persistsCalibrationSamples
    self.persistsMapPreferences = persistsMapPreferences
    if loadPersistedState,
       let saved = UserDefaults.standard.string(forKey: Self.roadLabelSizeDefaultsKey),
       let size = MapRoadLabelSize(rawValue: saved) {
      roadLabelSize = size
    }
    if loadPersistedState,
       let data = UserDefaults.standard.data(forKey: Self.openingLocationDefaultsKey),
       let saved = try? JSONDecoder().decode(MapOpeningLocation.self, from: data) {
      openingLocation = saved
      cameraDestination = saved.cameraDestination
      searchQuery = saved.title
      searchMarkerVisible = true
    }
    if loadPersistedState, let archive = try? Self.loadCalibrationArchive() {
      calibrationSamples = archive.samples
      calibrationAnchorKnobs = archive.anchorKnobs
    }
    if loadPersistedState,
       let savedPath = UserDefaults.standard.string(forKey: Self.tileRootDefaultsKey) {
      Task { await configureTileRoot(URL(fileURLWithPath: savedPath), persist: false) }
    } else if loadPersistedState,
              let defaultRoot = try? MapTileRoot.defaultURL(),
              FileManager.default.fileExists(atPath: defaultRoot.path) {
      Task { await configureTileRoot(defaultRoot, persist: false) }
    }
  }

  var tileRootName: String { tileRootURL?.lastPathComponent ?? "Choose Tiles" }
  var selectedWholeCurveEvent: MapWholeCurveEvent? {
    guard let selectedWholeCurveEventID else { return nil }
    return wholeCurveEvents.first { $0.id == selectedWholeCurveEventID }
  }
  var canSync: Bool { purpose == .calibration && !isSyncing && !isLoading }
  var canSetOpeningLocation: Bool { selectedPlaceResult != nil }
  var unresolvedCalibrationCount: Int {
    calibrationSamples.count(where: { !$0.hasCurrentCurvatureEstimate })
  }
  var maximumRawRequestedLateralAccelerationMPS2: Double {
    calibrationSamples.map {
      Self.lateralAcceleration(
        curvature: $0.rawCurvature ?? $0.curvature,
        speedMPH: $0.desiredSpeedMPH
      )
    }.filter(\.isFinite).max() ?? 0
  }
  var maximumRuntimeRequestedLateralAccelerationMPS2: Double {
    calibrationSamples.map {
      Self.lateralAcceleration(curvature: $0.curvature, speedMPH: $0.desiredSpeedMPH)
    }.filter(\.isFinite).max() ?? 0
  }
  var canEditSelectedCurveDraft: Bool {
    purpose == .calibration || isStudyCurveCaptureActive
  }
  var canQueueSelection: Bool {
    guard canEditSelectedCurveDraft,
          let selection,
          selection.node.curvatureContextComplete,
          selection.node.curvature >= 1.0e-7
    else { return false }
    guard !(isStudyCurveCaptureActive && selectionIsQueued) else { return false }
    return draftDesiredSpeedMPH.isFinite && draftDesiredSpeedMPH > 0
  }
  var selectionIsQueued: Bool {
    guard let selection else { return false }
    return calibrationSamples.contains { $0.sourceKey == calibrationSourceKey(selection) }
  }
  var selectionQueuedNumber: Int? {
    guard let selection,
          let index = calibrationSamples.firstIndex(where: {
            $0.sourceKey == calibrationSourceKey(selection)
          })
    else { return nil }
    return index + 1
  }
  var selectionQueueActionTitle: String {
    if isStudyCurveCaptureActive, selectionIsQueued { return "Already in Curve Bank" }
    if let selectionQueuedNumber { return "Update Bank Item #\(selectionQueuedNumber)" }
    return "Add Curve to Bank"
  }
  var selectionHasUncommittedTarget: Bool { draftWasEdited }
  var selectionDraftBadgeText: String {
    if let selectionQueuedNumber {
      return draftWasEdited
        ? "UNSAVED CHANGE TO #\(selectionQueuedNumber)"
        : "BANKED #\(selectionQueuedNumber)"
    }
    return "DRAFT — NOT BANKED"
  }
  var canRunFit: Bool {
    calibrationSamples.count >= Self.minimumCalibrationSamples
      && calibrationSamples.allSatisfy { $0.desiredSpeedMPH.isFinite && $0.desiredSpeedMPH > 0 }
      && unresolvedCalibrationCount == 0
      && !isFitting
  }
  var selectionQueueHelp: String {
    guard canEditSelectedCurveDraft else {
      return "Choose Add New Curve before drafting a local curve-bank sample."
    }
    guard let selection else { return "Select a mapd curve first." }
    guard selection.node.curvatureContextComplete else {
      return "This point lacks unambiguous adjacent-way context. Load more surrounding map geometry or choose another point."
    }
    guard selection.node.curvature >= 1.0e-7 else {
      return "This point has no usable runtime-smoothed curvature. Select a curve point."
    }
    guard draftDesiredSpeedMPH.isFinite && draftDesiredSpeedMPH > 0 else {
      return "Enter a positive target speed before adding this curve."
    }
    if isStudyCurveCaptureActive, selectionIsQueued {
      return "This curve is already in the bank. Choose a new curve, or switch to Calibration to edit the saved item."
    }
    return selectionIsQueued
      ? "Save the edited target back to this curve's bank item."
      : "Commit this draft target to the persistent curve bank."
  }

  func chooseTileFolder() {
    let panel = NSOpenPanel()
    panel.title = "Choose mapd offline tiles"
    panel.message = "Choose the offline folder, or its parent mapd folder."
    panel.prompt = "Use Tile Folder"
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false
    panel.directoryURL = tileRootURL ?? FileManager.default.homeDirectoryForCurrentUser
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await configureTileRoot(url, persist: true) }
  }

  func syncFromTici() {
    guard canSync else { return }
    isSyncing = true
    status("Syncing actual mapd tiles from \(ticiProfile)…")
    let profile = ticiProfile
    Task {
      do {
        let destination = try MapTileRoot.defaultURL()
        let service = TiciMapTileSyncService()
        _ = try await service.sync(profile: profile, destinationRootURL: destination)
        await configureTileRoot(destination, persist: true)
        status("Synced actual mapd tiles from \(profile). Pan or search to a curve to inspect it.")
      } catch {
        status("Tile sync failed: \(error.localizedDescription)", error: true)
      }
      isSyncing = false
    }
  }

  func viewportChanged(
    _ viewport: MapTileBounds,
    parameters: SigmoidParameters,
    bands: [EQBand]
  ) {
    let invalidatesFit = parameters != currentParameters || bands != currentBands
    lastViewport = viewport
    currentParameters = parameters
    currentBands = bands
    if invalidatesFit {
      if purpose == .calibration {
        refreshCalibrationBaselines()
        invalidateFit()
        refreshUncommittedDraftIfNeeded()
      } else if isStudyCurveCaptureActive {
        refreshUncommittedDraftIfNeeded()
      }
    }
    guard !isSyncing else { return }
    scheduleLoad(viewport: viewport, debounceNanoseconds: 220_000_000)
  }

  func updateProposal(parameters: SigmoidParameters, bands: [EQBand]) {
    let invalidatesFit = parameters != currentParameters || bands != currentBands
    currentParameters = parameters
    currentBands = bands
    if invalidatesFit {
      if purpose == .calibration {
        refreshCalibrationBaselines()
        invalidateFit()
        refreshUncommittedDraftIfNeeded()
      } else if isStudyCurveCaptureActive {
        refreshUncommittedDraftIfNeeded()
      }
    }
    guard let lastViewport else { return }
    scheduleLoad(viewport: lastViewport, debounceNanoseconds: 80_000_000)
  }

  func runPlaceSearch() {
    let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else {
      clearPlaceSearch()
      return
    }
    searchTask?.cancel()
    activeSearch?.cancel()
    activeSearch = nil
    searchGeneration &+= 1
    let generation = searchGeneration
    isSearching = true
    searchTask = Task {
      do {
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = query
        request.resultTypes = [.address, .pointOfInterest]
        if let bounds = tileCoverage {
          request.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(
              latitude: 0.5 * (bounds.minLatitude + bounds.maxLatitude),
              longitude: 0.5 * (bounds.minLongitude + bounds.maxLongitude)
            ),
            span: MKCoordinateSpan(
              latitudeDelta: min(90, max(0.05, bounds.maxLatitude - bounds.minLatitude)),
              longitudeDelta: min(180, max(0.05, bounds.maxLongitude - bounds.minLongitude))
            )
          )
        }
        let localSearch = MKLocalSearch(request: request)
        activeSearch = localSearch
        let response = try await localSearch.start()
        try Task.checkCancellation()
        guard generation == searchGeneration,
              searchQuery.trimmingCharacters(in: .whitespacesAndNewlines) == query
        else { return }
        let isSingleResult = response.mapItems.count == 1
        let responseSpan = response.boundingRegion.span
        let resolvedResults = response.mapItems.prefix(6).map { item in
          let latitudeDelta = isSingleResult
            ? min(0.25, max(0.012, responseSpan.latitudeDelta * 1.2))
            : 0.025
          let longitudeDelta = isSingleResult
            ? min(0.25, max(0.012, responseSpan.longitudeDelta * 1.2))
            : 0.025
          return MapPlaceResult(
            title: item.name ?? query,
            subtitle: item.placemark.title ?? "",
            latitude: item.placemark.coordinate.latitude,
            longitude: item.placemark.coordinate.longitude,
            latitudeDelta: latitudeDelta,
            longitudeDelta: longitudeDelta
          )
        }
        searchResults = resolvedResults
        if resolvedResults.count == 1, let result = resolvedResults.first {
          choosePlace(result)
        } else if resolvedResults.isEmpty {
          status("No MapKit place matched “\(query)”.", error: true)
        }
      } catch is CancellationError {
        // A newer search replaced this one.
      } catch {
        if generation == searchGeneration {
          status("Map search failed: \(error.localizedDescription)", error: true)
        }
      }
      if generation == searchGeneration {
        isSearching = false
        searchTask = nil
        activeSearch = nil
      }
    }
  }

  func updateSearchQuery(_ value: String) {
    guard value != searchQuery else { return }
    searchQuery = value
    searchResults = []
    selectedPlaceResult = nil
    searchMarkerVisible = false
    searchGeneration &+= 1
    searchTask?.cancel()
    searchTask = nil
    activeSearch?.cancel()
    activeSearch = nil
    isSearching = false
  }

  func clearPlaceSearch() {
    searchGeneration &+= 1
    searchTask?.cancel()
    searchTask = nil
    activeSearch?.cancel()
    activeSearch = nil
    isSearching = false
    searchQuery = ""
    searchResults = []
    selectedPlaceResult = nil
    searchMarkerVisible = false
  }

  func choosePlace(_ result: MapPlaceResult) {
    selectedPlaceResult = result
    cameraDestination = MapCameraDestination(
      latitude: result.latitude,
      longitude: result.longitude,
      title: result.title,
      latitudeDelta: result.latitudeDelta,
      longitudeDelta: result.longitudeDelta,
      showsMarker: true
    )
    searchQuery = result.title
    searchResults = []
    searchMarkerVisible = true
    status("Centered on \(result.title). Overlays still come only from your local mapd tiles.")
  }

  func setCurrentPlaceAsOpeningLocation() {
    guard let result = selectedPlaceResult else {
      status("Search for and choose a road, city, address, or place first.", error: true)
      return
    }
    let location = MapOpeningLocation(
      title: result.title,
      subtitle: result.subtitle,
      latitude: result.latitude,
      longitude: result.longitude,
      latitudeDelta: result.latitudeDelta,
      longitudeDelta: result.longitudeDelta
    )
    openingLocation = location
    if persistsMapPreferences {
      do {
        UserDefaults.standard.set(
          try JSONEncoder().encode(location),
          forKey: Self.openingLocationDefaultsKey
        )
      } catch {
        status("Could not save the opening location: \(error.localizedDescription)", error: true)
        return
      }
    }
    status("\(location.title) will be the opening map location.")
  }

  func goToOpeningLocation() {
    moveToOpeningLocation(announce: true)
  }

  func clearOpeningLocation() {
    guard let openingLocation else { return }
    self.openingLocation = nil
    if persistsMapPreferences {
      UserDefaults.standard.removeObject(forKey: Self.openingLocationDefaultsKey)
    }
    if selectedPlaceResult == nil { searchMarkerVisible = false }
    status("Cleared \(openingLocation.title) as the opening map location.")
  }

  private func moveToOpeningLocation(announce: Bool) {
    guard let openingLocation else { return }
    selectedPlaceResult = nil
    searchResults = []
    searchQuery = openingLocation.title
    cameraDestination = openingLocation.cameraDestination
    searchMarkerVisible = true
    if announce { status("Centered on opening location \(openingLocation.title).") }
  }

  func beginStudyCurveCapture() {
    guard purpose == .wholeCurveStudy else {
      status("Switch to Whole-Curve Study before starting a new curve capture.", error: true)
      return
    }
    guard !isStudyCurveCaptureActive else { return }
    isStudyCurveCaptureActive = true
    studyCaptureBankedNumber = nil
    clearSelection()
    renderRevision &+= 1
    status("Add a familiar curve: click a colored mapd road. It remains a draft until you choose Add Curve to Bank.")
  }

  func cancelStudyCurveCapture() {
    guard isStudyCurveCaptureActive else { return }
    let discardedMessage = discardedDraftMessage()
    isStudyCurveCaptureActive = false
    studyCaptureBankedNumber = nil
    selection = nil
    draftDesiredSpeedMPH = 0
    draftBaselineDesiredSpeedMPH = 0
    draftWasEdited = false
    renderRevision &+= 1
    status(discardedMessage ?? "Back to Whole-Curve Study. No new curve was added.")
  }

  func setDraftDesiredSpeedMPH(_ value: Double) {
    guard canEditSelectedCurveDraft,
          !(isStudyCurveCaptureActive && selectionIsQueued)
    else { return }
    draftDesiredSpeedMPH = value
    draftWasEdited = !Self.speedsMatch(value, draftBaselineDesiredSpeedMPH)
  }

  func queueSelectedCurve() {
    guard canEditSelectedCurveDraft else {
      status("Choose Add New Curve before adding a sample from Whole-Curve Study.", error: true)
      return
    }
    guard let selection else { return }
    guard !(isStudyCurveCaptureActive && selectionIsQueued) else {
      status("That curve is already in the bank. Choose a new curve, or switch to Calibration to edit it.", error: true)
      return
    }
    let node = selection.node
    guard node.curvatureContextComplete else {
      status("That point lacks enough unambiguous route context for production-equivalent curvature. Load surrounding tiles or choose another point.", error: true)
      return
    }
    guard node.curvature >= 1.0e-7 else {
      status("That point has no usable runtime-smoothed curvature. Select a curve point.", error: true)
      return
    }
    guard draftDesiredSpeedMPH.isFinite, draftDesiredSpeedMPH > 0 else {
      status("Enter a positive target speed before adding this curve to the bank.", error: true)
      return
    }
    let sourceKey = calibrationSourceKey(selection)
    let proposed = node.proposedSpeedMPS.mapMPH
    let effective = SigmoidFitter.predictedSpeedMPH(
      parameters: currentParameters,
      curvature: node.curvature,
      bands: currentBands,
      mode: .effectiveStrategic,
      modifiers: .sourceDefaults
    )
    let queued = MapCalibrationSample(
      sourceKey: sourceKey,
      roadName: selection.way.displayName,
      reference: selection.way.reference,
      latitude: node.latitude,
      longitude: node.longitude,
      curvature: node.curvature,
      rawCurvature: node.rawCurvature,
      curvatureSupportMeters: 0,
      curvatureEstimatorVersion: max(0, MapRuntimeCurvatureResolver.estimatorVersion - 1),
      curvatureContextComplete: false,
      bakedSpeedMPH: node.bakedSpeedMPS?.mapMPH,
      proposedSpeedMPH: proposed,
      effectiveSpeedMPH: effective,
      desiredSpeedMPH: draftDesiredSpeedMPH
    )
    if let index = calibrationSamples.firstIndex(where: { $0.sourceKey == sourceKey }) {
      var updated = queued
      updated.id = calibrationSamples[index].id
      calibrationSamples[index] = updated
      reconcileCalibrationCurvatures(using: ways)
      let saved = batchDidChange()
      draftBaselineDesiredSpeedMPH = draftDesiredSpeedMPH
      draftWasEdited = false
      if saved {
        status("Updated the banked target for \(selection.way.displayName) to \(String(format: "%.1f", draftDesiredSpeedMPH)) mph.")
      } else {
        status("Updated the target in memory, but it could not be saved to this Mac.", error: true)
      }
      return
    }
    calibrationSamples.append(queued)
    reconcileCalibrationCurvatures(using: ways)
    let saved = batchDidChange()
    if !saved, isStudyCurveCaptureActive {
      calibrationSamples.removeLast()
      status("Could not save this curve to the bank. The draft is still here; correct the problem and try Add Curve to Bank again.", error: true)
      return
    }
    draftBaselineDesiredSpeedMPH = draftDesiredSpeedMPH
    draftWasEdited = false
    if saved {
      if isStudyCurveCaptureActive {
        let bankedNumber = calibrationSamples.count
        studyCaptureBankedNumber = bankedNumber
        self.selection = nil
        draftDesiredSpeedMPH = 0
        draftBaselineDesiredSpeedMPH = 0
        renderRevision &+= 1
        refreshVisibleWholeCurveStudy()
        status("Banked #\(bankedNumber): \(selection.way.displayName) at \(String(format: "%.1f", queued.desiredSpeedMPH)) mph. Click another mapd curve to keep adding samples; no tune or car state changed.")
      } else {
        status("Added \(selection.way.displayName) to the curve bank at \(String(format: "%.1f", draftDesiredSpeedMPH)) mph (\(calibrationSamples.count) saved; \(Self.minimumCalibrationSamples) minimum, no maximum).")
      }
    } else {
      status("Added the curve in memory, but it could not be saved to this Mac.", error: true)
    }
  }

  func removeCalibrationSample(id: UUID) {
    guard purpose == .calibration else {
      status("Curve-bank editing is unavailable in the read-only whole-curve study.", error: true)
      return
    }
    let removedActiveSample = calibrationSamples.first(where: { $0.id == id }).map { sample in
      guard let selection else { return false }
      return sample.sourceKey == calibrationSourceKey(selection)
    } ?? false
    calibrationSamples.removeAll { $0.id == id }
    let saved = batchDidChange()
    if removedActiveSample {
      draftBaselineDesiredSpeedMPH = SigmoidFitter.predictedSpeedMPH(
        parameters: currentParameters,
        curvature: selection?.node.curvature ?? 0,
        bands: currentBands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
      draftWasEdited = true
      let suffix = saved ? "" : " The remaining bank is not saved to disk."
      status("Removed this curve from the bank. Its target remains in the draft until you add it again or select another curve.\(suffix)", error: !saved)
    }
  }

  func updateCalibrationTarget(id: UUID, desiredSpeedMPH: Double) {
    guard purpose == .calibration else { return }
    guard desiredSpeedMPH.isFinite,
          let index = calibrationSamples.firstIndex(where: { $0.id == id })
    else { return }
    guard calibrationSamples[index].desiredSpeedMPH != desiredSpeedMPH else { return }
    calibrationSamples[index].desiredSpeedMPH = desiredSpeedMPH
    if let selection,
       calibrationSamples[index].sourceKey == calibrationSourceKey(selection) {
      draftDesiredSpeedMPH = desiredSpeedMPH
      draftBaselineDesiredSpeedMPH = desiredSpeedMPH
      draftWasEdited = false
    }
    batchDidChange()
  }

  @discardableResult
  func ensureCalibrationAnchor(_ candidate: PlainKnobs) -> PlainKnobs {
    guard purpose == .calibration else {
      return calibrationAnchorKnobs ?? VTSCMath.knobs(from: VTSCMath.parameters(from: candidate))
    }
    if let calibrationAnchorKnobs { return calibrationAnchorKnobs }
    let canonical = VTSCMath.knobs(from: VTSCMath.parameters(from: candidate))
    calibrationAnchorKnobs = canonical
    if persistsCalibrationSamples, !calibrationSamples.isEmpty {
      do {
        try persistCalibrationSamples()
        calibrationPersistenceError = nil
      } catch {
        calibrationPersistenceError = error.localizedDescription
        status("Could not save the calibration anchor: \(error.localizedDescription)", error: true)
      }
    }
    return canonical
  }

  @discardableResult
  private func batchDidChange() -> Bool {
    let canceledActiveFit = isFitting
    invalidateFit()
    if calibrationSamples.isEmpty { calibrationAnchorKnobs = nil }
    if canceledActiveFit {
      status("Fit canceled because the curve bank changed. Generate a fresh proposal.")
    }
    guard persistsCalibrationSamples else {
      calibrationPersistenceError = nil
      return true
    }
    do {
      try persistCalibrationSamples()
      calibrationPersistenceError = nil
      return true
    } catch {
      calibrationPersistenceError = error.localizedDescription
      status("Could not save curve samples: \(error.localizedDescription)", error: true)
      return false
    }
  }

  func focusCalibrationSample(_ sample: MapCalibrationSample) {
    selectedPlaceResult = nil
    searchMarkerVisible = false
    cameraDestination = MapCameraDestination(
      latitude: sample.latitude,
      longitude: sample.longitude,
      title: sample.roadName
    )
    status("Centered on calibration sample for \(sample.roadName).")
  }

  func invalidateFit() {
    fitGeneration &+= 1
    fitTask?.cancel()
    fitTask = nil
    isFitting = false
    fitResult = nil
    fitErrorText = nil
  }

  func runFit(currentKnobs: PlainKnobs, anchorKnobs: PlainKnobs, bands: [EQBand]) {
    guard purpose == .calibration else {
      status("Tune fitting is unavailable in the read-only whole-curve study.", error: true)
      return
    }
    guard canRunFit else { return }
    let coreSamples = calibrationSamples.enumerated().map { index, sample in
      let descriptor: String
      if sample.reference.isEmpty
        || sample.reference.caseInsensitiveCompare(sample.roadName) == .orderedSame
      {
        descriptor = sample.roadName
      } else {
        descriptor = "\(sample.roadName) (\(sample.reference))"
      }
      return CurveCalibrationSample(
        id: sample.id,
        label: "#\(index + 1) \(descriptor)",
        curvature: sample.curvature,
        desiredSpeedMPH: sample.desiredSpeedMPH
      )
    }
    fitTask?.cancel()
    fitGeneration &+= 1
    let generation = fitGeneration
    isFitting = true
    fitResult = nil
    fitErrorText = nil
    status("Fitting the complete VTSC curve against \(coreSamples.count) real-curve samples…")
    fitTask = Task {
      do {
        let worker = Task.detached(priority: .userInitiated) {
          try SigmoidFitter.fit(
            samples: coreSamples,
            currentKnobs: currentKnobs,
            anchorKnobs: anchorKnobs,
            bands: bands,
            predictionMode: .effectiveStrategic,
            modifiers: .sourceDefaults
          )
        }
        let result = try await withTaskCancellationHandler {
          try await worker.value
        } onCancel: {
          worker.cancel()
        }
        try Task.checkCancellation()
        guard generation == fitGeneration else { return }
        fitResult = result
        status(String(
          format: "Complete-curve fit ready: effective-target RMSE %.2f → %.2f mph. Review before accepting.",
          result.beforeRMSEMPH,
          result.afterRMSEMPH
        ))
      } catch is CancellationError {
        return
      } catch {
        guard generation == fitGeneration else { return }
        fitErrorText = error.localizedDescription
        status("Fit failed: \(error.localizedDescription)", error: true)
      }
      if generation == fitGeneration {
        isFitting = false
        fitTask = nil
      }
    }
  }

  func reportUncommittedTargetEditor() {
    let message = "Finish entering a valid target speed before generating the proposal."
    fitErrorText = message
    status("Fit not started: \(message)", error: true)
  }

  func acceptFit(
    _ result: SigmoidFitResult,
    currentKnobs: PlainKnobs,
    currentAnchorKnobs: PlainKnobs,
    currentBands: [EQBand],
    action: (SigmoidFitResult) -> Void
  ) {
    guard purpose == .calibration else {
      status("Tune acceptance is unavailable in the read-only whole-curve study.", error: true)
      return
    }
    let canonicalCurrent = VTSCMath.knobs(from: VTSCMath.parameters(from: currentKnobs))
    let canonicalAnchor = VTSCMath.knobs(from: VTSCMath.parameters(from: currentAnchorKnobs))
    guard fitResult == result,
      !isFitting,
      result.inputKnobs == canonicalCurrent,
      result.anchorKnobs == canonicalAnchor,
      result.inputBands == currentBands
    else {
      status("That fit is stale. Run it again with the current samples and tune.", error: true)
      return
    }
    action(result)
    invalidateFit()
  }

  func clearSelection() {
    let discardedMessage = discardedDraftMessage()
    selection = nil
    draftDesiredSpeedMPH = 0
    draftBaselineDesiredSpeedMPH = 0
    draftWasEdited = false
    renderRevision &+= 1
    if let discardedMessage { status(discardedMessage) }
  }

  func selectWholeCurveEvent(_ eventID: String?) {
    selectedWholeCurveEventID = eventID
    renderRevision &+= 1
    guard let event = selectedWholeCurveEvent else {
      status("Whole-curve study selection cleared. No tune or car state changed.")
      return
    }
    status(
      String(
        format: "Selected %@: today %.1f–%.1f mph, whole-curve preview %.1f mph. Local shadow study only.",
        event.displayName,
        event.currentMinimumSpeedMPH,
        event.currentMaximumSpeedMPH,
        event.wholeCurveSpeedMPH
      )
    )
  }

  func focusWholeCurveBank(announce: Bool = true) {
    guard let first = calibrationSamples.first else { return }
    let bounds = calibrationSamples.dropFirst().reduce(
      MapTileBounds(
        minLatitude: first.latitude,
        minLongitude: first.longitude,
        maxLatitude: first.latitude,
        maxLongitude: first.longitude
      )
    ) { partial, sample in
      partial.union(MapTileBounds(
        minLatitude: sample.latitude,
        minLongitude: sample.longitude,
        maxLatitude: sample.latitude,
        maxLongitude: sample.longitude
      ))
    }
    cameraDestination = MapCameraDestination(
      latitude: 0.5 * (bounds.minLatitude + bounds.maxLatitude),
      longitude: 0.5 * (bounds.minLongitude + bounds.maxLongitude),
      title: "Saved curve bank",
      latitudeDelta: max(0.015, (bounds.maxLatitude - bounds.minLatitude) * 1.5),
      longitudeDelta: max(0.015, (bounds.maxLongitude - bounds.minLongitude) * 1.35),
      showsMarker: false
    )
    if announce {
      status("Centered the local whole-curve study on all \(calibrationSamples.count) saved bank samples.")
    }
  }

  private func refreshVisibleWholeCurveStudy() {
    guard purpose == .wholeCurveStudy, let lastViewport else { return }
    scheduleLoad(viewport: lastViewport, debounceNanoseconds: 0)
  }

  func select(_ newSelection: MapRoadSelection?) {
    guard canEditSelectedCurveDraft else { return }
    if newSelection == nil, isStudyCurveCaptureActive {
      if let selection {
        status("No mapd curve found there. The draft for \(selection.way.displayName) is still not banked.")
      } else {
        status("No mapd curve found there. Click closer to a colored road.")
      }
      return
    }
    let previousSelectionID = selection?.id
    let discardedMessage = previousSelectionID != newSelection?.id ? discardedDraftMessage() : nil
    selection = newSelection
    if previousSelectionID != newSelection?.id {
      prepareDraft(for: newSelection)
    }
    renderRevision &+= 1
    if let newSelection, !newSelection.node.curvatureContextComplete {
      let prefix = discardedMessage.map { "\($0) " } ?? ""
      status("\(prefix)This point lacks unambiguous adjacent-way context. It can be inspected, but it is excluded from calibration.")
    } else if let newSelection, newSelection.node.curvature < 1.0e-7 {
      let prefix = discardedMessage.map { "\($0) " } ?? ""
      status("\(prefix)Selected a point with no usable runtime-smoothed curvature.")
    } else if let newSelection,
              calibrationSamples.contains(where: { $0.sourceKey == calibrationSourceKey(newSelection) }) {
      let prefix = discardedMessage.map { "\($0) " } ?? ""
      if isStudyCurveCaptureActive {
        status("\(prefix)This curve is already in the bank. Choose a new curve, or switch to Calibration to edit it.")
      } else {
        status("\(prefix)This curve is already in the bank. Edit its target, then update that bank item.")
      }
    } else if let newSelection {
      let prefix = discardedMessage.map { "\($0) " } ?? ""
      let captureSuffix = isStudyCurveCaptureActive
        ? " It has not been saved."
        : ""
      status("\(prefix)Drafting \(newSelection.way.displayName). Set the target speed, then choose Add Curve to Bank.\(captureSuffix)")
    } else if let discardedMessage {
      status(discardedMessage)
    }
  }

  private func calibrationSourceKey(_ selection: MapRoadSelection) -> String {
    "\(selection.way.id):\(selection.nodeIndex)"
  }

  private func prepareDraft(for selection: MapRoadSelection?) {
    guard let selection else {
      draftDesiredSpeedMPH = 0
      draftWasEdited = false
      return
    }
    if let queued = calibrationSamples.first(where: { $0.sourceKey == calibrationSourceKey(selection) }) {
      draftDesiredSpeedMPH = queued.desiredSpeedMPH
    } else {
      draftDesiredSpeedMPH = SigmoidFitter.predictedSpeedMPH(
        parameters: currentParameters,
        curvature: selection.node.curvature,
        bands: currentBands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
    }
    draftBaselineDesiredSpeedMPH = draftDesiredSpeedMPH
    draftWasEdited = false
  }

  private func discardedDraftMessage() -> String? {
    guard draftWasEdited else { return nil }
    if let selectionQueuedNumber {
      return "Unsaved change to bank item #\(selectionQueuedNumber) was discarded."
    }
    return "Previous draft was not added."
  }

  private static func speedsMatch(_ lhs: Double, _ rhs: Double) -> Bool {
    lhs.isFinite && rhs.isFinite && abs(lhs - rhs) < 0.000_1
  }

  private static func lateralAcceleration(curvature: Double, speedMPH: Double) -> Double {
    let speedMPS = speedMPH * VTSCMath.mphToMetersPerSecond
    return curvature * speedMPS * speedMPS
  }

  private func refreshUncommittedDraftIfNeeded(force: Bool = false) {
    guard force || !draftWasEdited else { return }
    prepareDraft(for: selection)
  }

  private func refreshCalibrationBaselines() {
    for index in calibrationSamples.indices {
      let curvature = calibrationSamples[index].curvature
      calibrationSamples[index].proposedSpeedMPH =
        MapBakeMath.bakedSpeedMPS(
          curvature: calibrationSamples[index].rawCurvature ?? curvature,
          parameters: currentParameters
        ).mapMPH
      calibrationSamples[index].effectiveSpeedMPH = SigmoidFitter.predictedSpeedMPH(
        parameters: currentParameters,
        curvature: curvature,
        bands: currentBands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
    }
    if !calibrationSamples.isEmpty {
      if persistsCalibrationSamples { try? persistCalibrationSamples() }
    }
  }

  private static func calibrationArchiveURL() throws -> URL {
    try TuneStore.applicationDirectory()
      .appendingPathComponent(calibrationArchiveFilename, isDirectory: false)
  }

  private static func loadCalibrationArchive() throws -> MapCalibrationArchive {
    let url = try calibrationArchiveURL()
    guard FileManager.default.fileExists(atPath: url.path) else {
      return MapCalibrationArchive(samples: [])
    }
    let archive = try JSONDecoder().decode(MapCalibrationArchive.self, from: Data(contentsOf: url))
    guard (1 ... 3).contains(archive.schema) else { return MapCalibrationArchive(samples: []) }
    return archive
  }

  private func persistCalibrationSamples() throws {
    let url = try Self.calibrationArchiveURL()
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(
      MapCalibrationArchive(
        samples: calibrationSamples,
        anchorKnobs: calibrationAnchorKnobs
      )
    )
      .write(to: url, options: .atomic)
  }

  private func configureTileRoot(_ candidate: URL, persist: Bool) async {
    do {
      let root = try MapTileRoot.resolve(candidate)
      let helperURL = try MapTileHelperLocator.bundledURL()
      let decoder = MapTileHelperDecoder(helperURL: helperURL)
      let store = try MapTileStore(rootURL: root, decoder: decoder)
      tileStore = store
      tileRootURL = root
      if persist { UserDefaults.standard.set(root.path, forKey: Self.tileRootDefaultsKey) }
      let index = await store.index
      tileCoverage = index.bounds
      let audited = purpose == .calibration
        ? ((try? await auditPersistedCalibrationCurvatures(using: store)) ?? 0)
        : 0
      let auditStatus: String
      if unresolvedCalibrationCount > 0 {
        auditStatus = " \(unresolvedCalibrationCount) banked curve\(unresolvedCalibrationCount == 1 ? "" : "s") still need surrounding route geometry before fitting."
      } else if audited > 0 {
        auditStatus = " Audited \(audited) banked curve\(audited == 1 ? "" : "s") against current runtime curvature."
      } else {
        auditStatus = ""
      }
      status("Indexed \(index.entries.count) actual mapd tile files.\(auditStatus) Search or pan to a curve.")
      if purpose == .wholeCurveStudy, !calibrationSamples.isEmpty {
        focusWholeCurveBank(announce: false)
      } else if openingLocation != nil {
        moveToOpeningLocation(announce: false)
      } else if let bounds = index.bounds {
        cameraDestination = MapCameraDestination(
          latitude: 0.5 * (bounds.minLatitude + bounds.maxLatitude),
          longitude: 0.5 * (bounds.minLongitude + bounds.maxLongitude),
          title: "Tile coverage"
        )
      }
      if let lastViewport { scheduleLoad(viewport: lastViewport, debounceNanoseconds: 0) }
    } catch {
      tileStore = nil
      ways = []
      wholeCurveEvents = []
      selectedWholeCurveEventID = nil
      status("Could not open mapd tiles: \(error.localizedDescription)", error: true)
    }
  }

  private func scheduleLoad(viewport: MapTileBounds, debounceNanoseconds: UInt64) {
    viewportTask?.cancel()
    guard let tileStore else { return }
    guard viewport.maxLatitude - viewport.minLatitude <= 2,
          viewport.maxLongitude - viewport.minLongitude <= 2
    else {
      isLoading = false
      if !ways.isEmpty {
        ways = []
        wholeCurveEvents = []
        selectedWholeCurveEventID = nil
        renderRevision &+= 1
      }
      status("Zoom closer than 2° to load detailed mapd road geometry.")
      return
    }
    let parameters = currentParameters
    isLoading = true
    viewportTask = Task {
      do {
        if debounceNanoseconds > 0 { try await Task.sleep(nanoseconds: debounceNanoseconds) }
        let tiles = try await tileStore.tiles(intersecting: viewport, paddingDegrees: 0.02)
        try Task.checkCancellation()
        let studyWays = resolvedWays(from: tiles, parameters: parameters)
        ways = studyWays.filter { way in
          way.nodes.contains {
            viewport.expanded(by: 0.005).contains(
              latitude: $0.latitude,
              longitude: $0.longitude
            )
          }
        }
        wholeCurveEvents = MapWholeCurveStudyResolver.resolve(
          ways: studyWays,
          calibrationSamples: calibrationSamples,
          parameters: currentParameters,
          bands: currentBands
        )
        if let selectedWholeCurveEventID,
           !wholeCurveEvents.contains(where: { $0.id == selectedWholeCurveEventID }) {
          self.selectedWholeCurveEventID = nil
        }
        let audited = (purpose == .calibration || isStudyCurveCaptureActive)
          ? reconcileCalibrationCurvatures(using: ways)
          : 0
        var selectionLossMessage: String?
        if let selectedID = selection?.way.id,
           let refreshed = ways.first(where: { $0.id == selectedID }) {
          let index = min(selection?.nodeIndex ?? 0, max(0, refreshed.nodes.count - 1))
          selection = MapRoadSelection(way: refreshed, nodeIndex: index, segmentFraction: selection?.segmentFraction ?? 0)
        } else if selection != nil {
          selectionLossMessage = discardedDraftMessage()
          selection = nil
          prepareDraft(for: nil)
        }
        renderRevision &+= 1
        let prefix = selectionLossMessage.map { "\($0) " } ?? ""
        let migrationSuffix: String
        if unresolvedCalibrationCount > 0 {
          migrationSuffix = " \(unresolvedCalibrationCount) banked curve\(unresolvedCalibrationCount == 1 ? "" : "s") are excluded until route context resolves."
        } else if audited > 0 {
          migrationSuffix = " Audited \(audited) banked curve\(audited == 1 ? "" : "s") against current runtime curvature."
        } else {
          migrationSuffix = ""
        }
        let studySuffix = wholeCurveEvents.isEmpty
          ? " No bank-linked whole-curve event is fully resolved in this view."
          : " Whole-curve shadow grouped the visible bank into \(wholeCurveEvents.count) directional events."
        status("\(prefix)Showing \(ways.count) unique mapd ways from \(tiles.count) visible tile files.\(studySuffix)\(migrationSuffix)")
      } catch is CancellationError {
        // Panning or a tune edit superseded this load.
      } catch {
        ways = []
        wholeCurveEvents = []
        selectedWholeCurveEventID = nil
        status("Visible tile load failed: \(error.localizedDescription)", error: true)
      }
      isLoading = false
    }
  }

  private func resolvedWays(
    from tiles: [MapTile],
    parameters: SigmoidParameters,
    visibleBounds: MapTileBounds? = nil
  ) -> [MapRenderedWay] {
    var deduplicated: [String: MapRenderedWay] = [:]
    for tile in tiles {
      for way in tile.ways where way.nodes.count >= 2 {
        let rendered = MapRenderedWay(tile: tile, way: way, parameters: parameters)
        if let existing = deduplicated[way.stableID] {
          let existingHasBake = existing.nodes.contains { $0.bakedSpeedMPS != nil }
          let replacementHasBake = rendered.nodes.contains { $0.bakedSpeedMPS != nil }
          if rendered.schemaVersion > existing.schemaVersion ||
            (replacementHasBake && !existingHasBake) {
            deduplicated[way.stableID] = rendered
          }
        } else {
          deduplicated[way.stableID] = rendered
        }
      }
    }
    let allResolved = MapRuntimeCurvatureResolver.resolve(
      ways: deduplicated.values.sorted { $0.id < $1.id },
      parameters: parameters
    )
    guard let visibleBounds else { return allResolved }
    return allResolved.filter { way in
      way.nodes.contains {
        visibleBounds.contains(latitude: $0.latitude, longitude: $0.longitude)
      }
    }
  }

  @discardableResult
  func reconcileCalibrationCurvatures(
    using resolvedWays: [MapRenderedWay],
    invalidateMissing: Bool = false
  ) -> Int {
    guard (purpose == .calibration || isStudyCurveCaptureActive),
          !calibrationSamples.isEmpty
    else { return 0 }
    let wayByID = Dictionary(uniqueKeysWithValues: resolvedWays.map { ($0.id, $0) })
    let wholeCurveResolutions = MapWholeCurveStudyResolver.calibrationResolutions(
      ways: resolvedWays,
      calibrationSamples: calibrationSamples
    )
    var changedCount = 0
    var migratedLegacyEstimate = false

    for index in calibrationSamples.indices {
      let sample = calibrationSamples[index]
      guard let separator = sample.sourceKey.lastIndex(of: ":"),
            let nodeIndex = Int(sample.sourceKey[sample.sourceKey.index(after: separator)...])
      else {
        if invalidateCalibrationCurvature(at: index) { changedCount += 1 }
        continue
      }
      guard let way = wayByID[String(sample.sourceKey[..<separator])] else {
        if invalidateMissing,
           invalidateCalibrationCurvature(at: index) { changedCount += 1 }
        continue
      }
      guard way.nodes.indices.contains(nodeIndex) else {
        if invalidateCalibrationCurvature(at: index) { changedCount += 1 }
        continue
      }
      let node = way.nodes[nodeIndex]
      guard node.curvatureContextComplete,
            let resolution = wholeCurveResolutions[sample.sourceKey]
      else {
        if invalidateMissing,
           invalidateCalibrationCurvature(at: index) { changedCount += 1 }
        continue
      }

      let needsUpdate = !sample.hasCurrentCurvatureEstimate
        || abs(sample.curvature - resolution.curvature) > 1.0e-12
        || abs((sample.rawCurvature ?? .nan) - node.rawCurvature) > 1.0e-12
        || abs((sample.curvatureSupportMeters ?? .nan) - resolution.supportMeters) > 1.0e-9
      guard needsUpdate else { continue }

      if !sample.hasCurrentCurvatureEstimate { migratedLegacyEstimate = true }
      calibrationSamples[index].curvature = resolution.curvature
      calibrationSamples[index].rawCurvature = node.rawCurvature
      calibrationSamples[index].curvatureSupportMeters = resolution.supportMeters
      calibrationSamples[index].curvatureEstimatorVersion = MapRuntimeCurvatureResolver.estimatorVersion
      calibrationSamples[index].curvatureContextComplete = true
      calibrationSamples[index].bakedSpeedMPH = node.bakedSpeedMPS?.mapMPH
      calibrationSamples[index].proposedSpeedMPH = node.proposedSpeedMPS.mapMPH
      calibrationSamples[index].effectiveSpeedMPH = SigmoidFitter.predictedSpeedMPH(
        parameters: currentParameters,
        curvature: resolution.curvature,
        bands: currentBands,
        mode: .effectiveStrategic,
        modifiers: .sourceDefaults
      )
      changedCount += 1
    }

    guard changedCount > 0 else { return 0 }
    if migratedLegacyEstimate {
      // A prior anchor may encode a fit to raw vertex curvature.  The next
      // proposal must establish a fresh anchor against the corrected bank.
      calibrationAnchorKnobs = nil
    }
    invalidateFit()
    if persistsCalibrationSamples {
      do {
        try persistCalibrationSamples()
        calibrationPersistenceError = nil
      } catch {
        calibrationPersistenceError = error.localizedDescription
      }
    }
    return changedCount
  }

  private func invalidateCalibrationCurvature(at index: Int) -> Bool {
    let changed = calibrationSamples[index].curvatureEstimatorVersion
        != MapRuntimeCurvatureResolver.estimatorVersion
      || calibrationSamples[index].curvatureContextComplete != false
      || calibrationSamples[index].curvatureSupportMeters != 0
    calibrationSamples[index].curvatureEstimatorVersion = MapRuntimeCurvatureResolver.estimatorVersion
    calibrationSamples[index].curvatureContextComplete = false
    calibrationSamples[index].curvatureSupportMeters = 0
    return changed
  }

  private func auditPersistedCalibrationCurvatures(using store: MapTileStore) async throws -> Int {
    guard let first = calibrationSamples.first else { return 0 }
    let bounds = calibrationSamples.dropFirst().reduce(
      MapTileBounds(
        minLatitude: first.latitude,
        minLongitude: first.longitude,
        maxLatitude: first.latitude,
        maxLongitude: first.longitude
      )
    ) { partial, sample in
      partial.union(MapTileBounds(
        minLatitude: sample.latitude,
        minLongitude: sample.longitude,
        maxLatitude: sample.latitude,
        maxLongitude: sample.longitude
      ))
    }
    let tiles = try await store.tiles(intersecting: bounds, paddingDegrees: 0.02)
    let resolved = resolvedWays(from: tiles, parameters: currentParameters)
    return reconcileCalibrationCurvatures(using: resolved, invalidateMissing: true)
  }

  private func status(_ text: String, error: Bool = false) {
    statusText = text
    statusIsError = error
  }
}
