import MapKit
import SwiftUI
import VTSCTunerCore

struct MapPreviewView: View {
  @ObservedObject var tuner: TunerSession
  @ObservedObject var map: MapPreviewSession

  var body: some View {
    VStack(spacing: 0) {
      ViewThatFits(in: .horizontal) {
        mapToolbar(compact: false)
        mapToolbar(compact: true)
      }
      .padding(.horizontal, 12)
      .frame(height: 56)
      .background(.bar)
      .overlay(alignment: .bottom) { Divider() }

      ZStack(alignment: .topLeading) {
        StrategicMapView(
          ways: map.ways,
          purpose: map.purpose,
          mode: map.displayMode,
          wholeCurveMode: map.wholeCurveDisplayMode,
          wholeCurveEvents: map.wholeCurveEvents,
          selectedWholeCurveEventID: map.selectedWholeCurveEventID,
          roadLabelSize: map.roadLabelSize,
          selection: map.selection,
          calibrationSamples: map.calibrationSamples,
          renderRevision: map.renderRevision,
          cameraDestination: map.cameraDestination,
          searchMarkerVisible: map.searchMarkerVisible,
          onViewportChanged: { bounds in
            map.viewportChanged(bounds, parameters: tuner.parameters, bands: tuner.bands)
          },
          onSelection: map.select,
          onWholeCurveSelection: map.selectWholeCurveEvent
        )

        if map.tileRootURL == nil {
          tileSourcePrompt.padding(12)
        }

        if !map.searchResults.isEmpty {
          placeSearchResults
            .padding(.leading, 12)
            .padding(.top, 8)
        }

        MapSpeedLegend(
          purpose: map.purpose,
          calibrationMode: map.displayMode,
          wholeCurveMode: map.wholeCurveDisplayMode
        )
          .padding(.leading, 12)
          .padding(.bottom, 36)
          .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
          // The alignment frame spans the map. Keep this display-only legend
          // transparent to pointer events so MKMapView receives road clicks.
          .allowsHitTesting(false)
      }
    }
    // RootView owns the top workspace inset. Keep the map toolbar below it,
    // while RootView's real split pane keeps MKMapView out of the controls.
    .padding(.top, 34)
    .onChange(of: tuner.parameters) { _, value in
      map.updateProposal(parameters: value, bands: tuner.bands)
    }
    .onChange(of: tuner.bands) { _, value in
      map.updateProposal(parameters: tuner.parameters, bands: value)
    }
  }

  private func mapToolbar(compact: Bool) -> some View {
    HStack(spacing: 10) {
      placeSearch
      roadLabelMenu(compact: compact)
      openingLocationMenu(compact: compact)
      Spacer(minLength: 0)
      if map.isLoading || map.isSyncing {
        HStack(spacing: 7) {
          ProgressView().controlSize(.small)
          if !compact {
            Text(map.isSyncing ? "Syncing actual tiles…" : "Loading visible tiles…")
          }
        }
        .font(.caption.weight(.medium))
      }
    }
  }

  private var placeSearch: some View {
    HStack(spacing: 7) {
      Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
      TextField(
        "Search road, town, or address — e.g. Madison Ave, Sacramento, CA",
        text: Binding(get: { map.searchQuery }, set: map.updateSearchQuery)
      )
        .textFieldStyle(.plain)
        .font(.system(size: 16, weight: .medium))
        .onSubmit(map.runPlaceSearch)
      if map.isSearching {
        ProgressView().controlSize(.small)
      } else if !map.searchQuery.isEmpty {
        Button(action: map.clearPlaceSearch) {
          Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
      }
      Button(action: map.runPlaceSearch) { Image(systemName: "arrow.right.circle.fill") }
        .buttonStyle(.plain)
        .font(.system(size: 19, weight: .semibold))
        .disabled(map.searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
    .padding(.horizontal, 12)
    .frame(minWidth: 240, idealWidth: 500, maxWidth: 500, minHeight: 42, maxHeight: 42)
    .layoutPriority(1)
    .background(.ultraThickMaterial, in: RoundedRectangle(cornerRadius: 9))
    .shadow(radius: 5, y: 2)
    .help("Search Apple Maps for a street, full address, town, or place. Colored VTSC roads still come only from your local mapd tiles.")
  }

  private func roadLabelMenu(compact: Bool) -> some View {
    Menu {
      ForEach(MapRoadLabelSize.allCases) { size in
        Button {
          map.roadLabelSize = size
        } label: {
          if map.roadLabelSize == size {
            Label(size.rawValue, systemImage: "checkmark")
          } else {
            Text(size.rawValue)
          }
        }
      }
    } label: {
      if compact {
        Image(systemName: "textformat.size")
          .font(.system(size: 16, weight: .semibold))
          .accessibilityLabel("Road text: \(map.roadLabelSize.rawValue)")
      } else {
        Label("Road text: \(map.roadLabelSize.rawValue)", systemImage: "textformat.size")
          .font(.system(size: 15, weight: .semibold))
      }
    }
    .fixedSize()
    .help("Large labels come from the actual mapd road name/reference. Apple-only hides the app's added labels.")
  }

  private func openingLocationMenu(compact: Bool) -> some View {
    Menu {
      if let candidate = map.selectedPlaceResult {
        Button {
          map.setCurrentPlaceAsOpeningLocation()
        } label: {
          Label("Set “\(candidate.title)” as Opening Location", systemImage: "house.badge.plus")
        }
      } else {
        Button("Search and choose a place first") {}
          .disabled(true)
      }

      if let opening = map.openingLocation {
        Divider()
        Button {
          map.goToOpeningLocation()
        } label: {
          Label("Go to \(opening.title)", systemImage: "location.fill")
        }
        Button(role: .destructive) {
          map.clearOpeningLocation()
        } label: {
          Label("Clear Opening Location", systemImage: "house.slash")
        }
      }
    } label: {
      if compact {
        Image(systemName: map.openingLocation == nil ? "house" : "house.fill")
          .font(.system(size: 16, weight: .semibold))
          .accessibilityLabel("Opening location")
      } else {
        Label(
          map.canSetOpeningLocation ? "Set Opening" : "Opening",
          systemImage: map.openingLocation == nil ? "house" : "house.fill"
        )
        .font(.system(size: 15, weight: .semibold))
      }
    }
    .fixedSize()
    .help(
      map.openingLocation.map { "Opening location: \($0.title)" }
        ?? "Search for a city, road, address, or place, then set it as the opening location"
    )
  }

  private var placeSearchResults: some View {
    VStack(alignment: .leading, spacing: 3) {
      Label("APPLE MAPS RESULTS", systemImage: "map.fill")
        .font(.system(size: 13, weight: .bold))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
      ForEach(map.searchResults) { result in
        Button {
          map.choosePlace(result)
        } label: {
          VStack(alignment: .leading, spacing: 2) {
            Text(result.title)
              .font(.system(size: 17, weight: .semibold))
              .foregroundStyle(.primary)
              .lineLimit(1)
            if !result.subtitle.isEmpty {
              Text(result.subtitle)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(2)
            }
          }
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(.horizontal, 12)
          .padding(.vertical, 8)
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
      }
    }
    .padding(.vertical, 5)
    .frame(minWidth: 220, maxWidth: 500)
    .background(.ultraThickMaterial, in: RoundedRectangle(cornerRadius: 10))
    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.white.opacity(0.18)))
    .shadow(radius: 10, y: 4)
  }

  private var tileSourcePrompt: some View {
    VStack(alignment: .leading, spacing: 7) {
      Label("No local mapd tiles selected", systemImage: "map")
        .font(.callout.weight(.semibold))
      Text("Choose an offline folder or explicitly sync the tici. MapKit search only moves the camera; it never substitutes Apple road data for mapd geometry.")
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
      HStack {
        Button("Choose Folder…", action: map.chooseTileFolder)
        if map.purpose == .calibration {
          Button("Sync from tici", action: map.syncFromTici).disabled(!map.canSync)
        }
      }
    }
    .padding(12)
    .frame(width: 330)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 9))
  }
}

struct MapSpeedLegend: View {
  let purpose: MapPreviewPurpose
  let calibrationMode: MapSpeedDisplayMode
  let wholeCurveMode: MapWholeCurveDisplayMode
  private let columns = Array(repeating: GridItem(.flexible(), spacing: 8), count: 3)

  private var title: String {
    purpose == .calibration ? calibrationMode.rawValue : wholeCurveMode.rawValue
  }

  private var isDelta: Bool {
    purpose == .calibration ? calibrationMode == .delta : wholeCurveMode == .difference
  }

  private var caption: String {
    purpose == .calibration ? calibrationMode.legendCaption : wholeCurveMode.legendCaption
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 9) {
      Text(title).font(.headline)
      LazyVGrid(columns: columns, alignment: .leading, spacing: 7) {
        if isDelta {
          swatch(.blue, "≤−8")
          swatch(.teal, "−3")
          swatch(.gray, "0")
          swatch(.orange, "+3")
          swatch(.pink, "≥+8 mph")
        } else {
          ForEach(MapSpeedBand.allCases) { band in
            swatch(
              Color(hue: band.hue, saturation: 0.84, brightness: 0.98),
              band.label
            )
          }
        }
      }
      Text(caption)
        .font(.callout.weight(.medium))
        .foregroundStyle(.secondary)
    }
    .padding(12)
    .frame(width: 300, alignment: .leading)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
  }

  private func swatch(_ color: Color, _ label: String) -> some View {
    HStack(spacing: 5) {
      Capsule().fill(color).frame(width: 22, height: 7)
      Text(label)
        .font(.callout.weight(.semibold))
        .lineLimit(1)
    }
  }
}
