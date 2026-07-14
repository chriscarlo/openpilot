import Foundation
import VTSCTunerCore

enum TunerWorkspace: String, CaseIterable, Identifiable {
  case curveLab = "Curve Lab"
  case mapPreview = "Map Preview"

  var id: String { rawValue }
}

enum MapSpeedDisplayMode: String, CaseIterable, Identifiable {
  case currentlyBaked = "Currently Baked"
  case proposedBake = "Proposed Tile Bake"
  case delta = "Delta"

  var id: String { rawValue }

  var legendCaption: String {
    switch self {
    case .currentlyBaked: "mph from the selected mapd tile"
    case .proposedBake: "mph the current sigmoid would bake"
    case .delta: "proposed minus currently baked"
    }
  }
}

enum MapSpeedBand: Int, CaseIterable, Identifiable {
  case under25
  case from25
  case from30
  case from35
  case from40
  case from45
  case from50
  case from55
  case from60
  case from65
  case from70
  case from75

  var id: Int { rawValue }

  var label: String {
    switch self {
    case .under25: "<25"
    case .from25: "25–29"
    case .from30: "30–34"
    case .from35: "35–39"
    case .from40: "40–44"
    case .from45: "45–49"
    case .from50: "50–54"
    case .from55: "55–59"
    case .from60: "60–64"
    case .from65: "65–69"
    case .from70: "70–74"
    case .from75: "75+"
    }
  }

  var hue: Double {
    switch self {
    case .under25: 0.000
    case .from25: 0.025
    case .from30: 0.055
    case .from35: 0.090
    case .from40: 0.130
    case .from45: 0.170
    case .from50: 0.250
    case .from55: 0.340
    case .from60: 0.440
    case .from65: 0.520
    case .from70: 0.600
    case .from75: 0.720
    }
  }

  static func band(forMPH mph: Double) -> Self {
    switch mph {
    case ..<25: .under25
    case ..<30: .from25
    case ..<35: .from30
    case ..<40: .from35
    case ..<45: .from40
    case ..<50: .from45
    case ..<55: .from50
    case ..<60: .from55
    case ..<65: .from60
    case ..<70: .from65
    case ..<75: .from70
    default: .from75
    }
  }
}

enum MapRoadLabelSize: String, CaseIterable, Identifiable {
  case appleOnly = "Apple only"
  case large = "Large"
  case extraLarge = "Extra Large"

  var id: String { rawValue }

  func pointSize(latitudeDelta: Double) -> Double? {
    let closeZoomBonus = latitudeDelta < 0.012 ? 2.0 : (latitudeDelta < 0.04 ? 1.0 : 0.0)
    switch self {
    case .appleOnly: return nil
    case .large: return 19 + closeZoomBonus
    case .extraLarge: return 24 + closeZoomBonus
    }
  }

  func maximumLabelCount(latitudeDelta: Double) -> Int {
    let density: Int
    switch latitudeDelta {
    case 0.2...: density = 10
    case 0.08...: density = 16
    case 0.025...: density = 22
    default: density = 30
    }
    return self == .large ? density + 8 : density
  }
}

struct MapRenderedNode: Sendable {
  var latitude: Double
  var longitude: Double
  var rawCurvature: Double = 0
  var curvature: Double
  var curvatureSupportMeters: Double = 0
  var curvatureContextComplete = true
  var bakedSpeedMPS: Double?
  var proposedSpeedMPS: Double
}

struct MapRenderedWay: Identifiable, Sendable {
  var id: String
  var name: String
  var reference: String
  var nodes: [MapRenderedNode]
  var tilePath: String
  var tileHash: String
  var schemaVersion: UInt16
  var maxSpeedMPS: Double
  var advisorySpeedMPS: Double
  var maxSpeedForwardMPS: Double
  var maxSpeedBackwardMPS: Double
  var lanes: UInt8
  var oneWay: Bool
  var hazard: String
  var windingForwardLevel: UInt8
  var windingBackwardLevel: UInt8
  var windingForwardScore: UInt8
  var windingBackwardScore: UInt8
  var windingForwardConfidence: UInt8
  var windingBackwardConfidence: UInt8

  var displayName: String {
    if !name.isEmpty { return name }
    if !reference.isEmpty { return reference }
    return "Unnamed mapd way"
  }
}

struct MapRoadSelection: Identifiable, Equatable {
  var way: MapRenderedWay
  var nodeIndex: Int
  var segmentFraction: Double

  var id: String { "\(way.id):\(nodeIndex)" }
  var node: MapRenderedNode { way.nodes[nodeIndex] }

  static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.way.id == rhs.way.id && lhs.nodeIndex == rhs.nodeIndex
      && lhs.segmentFraction == rhs.segmentFraction
  }
}

struct MapWholeCurveRoutePoint: Sendable {
  var node: MapTileNode
  var sourceKey: String
  var currentCurvature: Double
  var currentCurvatureContextComplete: Bool
}

struct MapWholeCurveRoute: Sendable {
  var anchorWayID: String
  var roadName: String
  var reference: String
  var points: [MapWholeCurveRoutePoint]
  var frontContextMeters: Double
  var backContextMeters: Double
  var frontIsAmbiguous: Bool
  var backIsAmbiguous: Bool
}

struct MapCameraDestination: Equatable {
  var id = UUID()
  var latitude: Double
  var longitude: Double
  var title: String
  var latitudeDelta: Double? = nil
  var longitudeDelta: Double? = nil
  var showsMarker = false
}

struct MapOpeningLocation: Codable, Equatable, Sendable {
  var title: String
  var subtitle: String
  var latitude: Double
  var longitude: Double
  var latitudeDelta: Double
  var longitudeDelta: Double

  var cameraDestination: MapCameraDestination {
    MapCameraDestination(
      latitude: latitude,
      longitude: longitude,
      title: title,
      latitudeDelta: latitudeDelta,
      longitudeDelta: longitudeDelta,
      showsMarker: true
    )
  }
}

struct MapCalibrationSample: Codable, Identifiable, Equatable, Sendable {
  var id = UUID()
  var sourceKey: String
  var roadName: String
  var reference: String
  var latitude: Double
  var longitude: Double
  var curvature: Double
  var rawCurvature: Double? = nil
  var curvatureSupportMeters: Double? = nil
  var curvatureEstimatorVersion: Int? = nil
  var curvatureContextComplete: Bool? = nil
  var bakedSpeedMPH: Double?
  var proposedSpeedMPH: Double
  var effectiveSpeedMPH: Double
  var desiredSpeedMPH: Double

  var hasCurrentCurvatureEstimate: Bool {
    curvatureEstimatorVersion == MapRuntimeCurvatureResolver.estimatorVersion
      && curvatureContextComplete == true
      && curvature.isFinite
      && curvature >= 0
  }

  var rawToEffectiveCurvatureRatio: Double? {
    guard let rawCurvature,
          rawCurvature.isFinite,
          curvature.isFinite,
          curvature > 1.0e-9
    else { return nil }
    return rawCurvature / curvature
  }
}

struct MapCalibrationArchive: Codable, Equatable, Sendable {
  var schema = 3
  var samples: [MapCalibrationSample]
  var anchorKnobs: PlainKnobs?

  init(
    schema: Int = 3,
    samples: [MapCalibrationSample],
    anchorKnobs: PlainKnobs? = nil
  ) {
    self.schema = schema
    self.samples = samples
    self.anchorKnobs = anchorKnobs
  }
}

extension MapRenderedWay {
  init(tile: MapTile, way: MapTileWay, parameters: SigmoidParameters) {
    let proposed = MapBakeMath.bakedSpeeds(nodes: way.nodes, parameters: parameters)
    var renderedNodes: [MapRenderedNode] = []
    renderedNodes.reserveCapacity(way.nodes.count)
    for index in way.nodes.indices {
      let curvature: Double
      if index > 0, index + 1 < way.nodes.count {
        curvature = MapBakeMath.curvature(
          previous: way.nodes[index - 1],
          current: way.nodes[index],
          next: way.nodes[index + 1]
        )
      } else {
        curvature = 0
      }
      renderedNodes.append(MapRenderedNode(
        latitude: way.nodes[index].latitude,
        longitude: way.nodes[index].longitude,
        rawCurvature: curvature,
        curvature: curvature,
        curvatureContextComplete: false,
        bakedSpeedMPS: way.nodes[index].bakedSpeedMPS,
        proposedSpeedMPS: proposed.indices.contains(index)
          ? proposed[index]
          : MapBakeMath.defaultMaximumSpeedMPS
      ))
    }
    id = way.stableID
    name = way.name
    reference = way.reference
    nodes = renderedNodes
    tilePath = tile.sourcePath
    tileHash = tile.sigmoidHash
    schemaVersion = tile.schemaVersion
    maxSpeedMPS = way.maxSpeedMPS
    advisorySpeedMPS = way.advisorySpeedMPS
    maxSpeedForwardMPS = way.maxSpeedForwardMPS
    maxSpeedBackwardMPS = way.maxSpeedBackwardMPS
    lanes = way.lanes
    oneWay = way.oneWay
    hazard = way.hazard
    windingForwardLevel = way.windingForwardLevel
    windingBackwardLevel = way.windingBackwardLevel
    windingForwardScore = way.windingForwardScore
    windingBackwardScore = way.windingBackwardScore
    windingForwardConfidence = way.windingForwardConfidence
    windingBackwardConfidence = way.windingBackwardConfidence
  }
}

enum MapRuntimeCurvatureResolver {
  static let estimatorVersion = 5

  private struct CoordinateKey: Hashable {
    var latitudeE7: Int64
    var longitudeE7: Int64

    init(_ node: MapRenderedNode) {
      latitudeE7 = Int64((node.latitude * 10_000_000).rounded())
      longitudeE7 = Int64((node.longitude * 10_000_000).rounded())
    }
  }

  private struct Endpoint {
    var wayIndex: Int
    var isStart: Bool
  }

  private struct RouteContext {
    var nodes: [MapTileNode]
    var originalOffset: Int
    var mergeOrSplitNodeIndices: Set<Int>
    var ambiguousBoundaryNodeIndices: Set<Int>

    func ambiguityCanAffect(nodeIndex: Int) -> Bool {
      let measurementIndices = (nodeIndex - 2 ... nodeIndex)
        .filter(nodes.indices.contains)
      for boundary in ambiguousBoundaryNodeIndices where nodes.indices.contains(boundary) {
        for measurementIndex in measurementIndices {
          if abs(measurementIndex - boundary) <= 2 {
            return true
          }
          if MapBakeMath.distanceMeters(
            from: nodes[boundary],
            to: nodes[measurementIndex]
          ) <= MapBakeMath.mergeOrSplitRadiusMeters {
            return true
          }
        }
      }
      return false
    }
  }

  private struct ContinuationSearch {
    var continuation: (wayIndex: Int, nodes: [MapTileNode])?
    var isAmbiguous: Bool
  }

  static func resolve(
    ways: [MapRenderedWay],
    parameters: SigmoidParameters
  ) -> [MapRenderedWay] {
    guard !ways.isEmpty else { return [] }
    var endpointIndex: [CoordinateKey: [Endpoint]] = [:]
    for (wayIndex, way) in ways.enumerated() where way.nodes.count >= 2 {
      endpointIndex[CoordinateKey(way.nodes[0]), default: []]
        .append(Endpoint(wayIndex: wayIndex, isStart: true))
      endpointIndex[CoordinateKey(way.nodes[way.nodes.count - 1]), default: []]
        .append(Endpoint(wayIndex: wayIndex, isStart: false))
    }

    var resolved = ways
    for wayIndex in ways.indices {
      let context = routeContext(
        for: wayIndex,
        ways: ways,
        endpointIndex: endpointIndex
      )
      for nodeIndex in ways[wayIndex].nodes.indices {
        let contextIndex = context.originalOffset + nodeIndex
        let rawCurvature: Double
        if contextIndex > 0, contextIndex + 1 < context.nodes.count {
          rawCurvature = MapBakeMath.curvature(
            previous: context.nodes[contextIndex - 1],
            current: context.nodes[contextIndex],
            next: context.nodes[contextIndex + 1]
          )
        } else {
          rawCurvature = 0
        }
        let estimate = context.ambiguityCanAffect(nodeIndex: contextIndex)
          ? nil
          : MapBakeMath.runtimeCurvatureEstimate(
              nodes: context.nodes,
              nodeIndex: contextIndex,
              mergeOrSplitNodeIndices: context.mergeOrSplitNodeIndices
            )
        let effectiveCurvature = estimate?.curvature ?? rawCurvature
        resolved[wayIndex].nodes[nodeIndex].rawCurvature = rawCurvature
        resolved[wayIndex].nodes[nodeIndex].curvature = effectiveCurvature
        resolved[wayIndex].nodes[nodeIndex].curvatureSupportMeters = estimate?.supportMeters ?? 0
        resolved[wayIndex].nodes[nodeIndex].curvatureContextComplete = estimate != nil
      }
    }
    return resolved
  }

  /// Builds longer, provenance-preserving routes for the local whole-curve
  /// shadow study. This is deliberately separate from `resolve`: changing
  /// the study window must never change the exact five-node mapd baseline.
  static func wholeCurveStudyRoutes(
    ways: [MapRenderedWay],
    focusSourceKeys: [String],
    contextMeters: Double = 1_200
  ) -> [MapWholeCurveRoute] {
    guard !ways.isEmpty, contextMeters > 0 else { return [] }
    var endpointIndex: [CoordinateKey: [Endpoint]] = [:]
    for (wayIndex, way) in ways.enumerated() where way.nodes.count >= 2 {
      endpointIndex[CoordinateKey(way.nodes[0]), default: []]
        .append(Endpoint(wayIndex: wayIndex, isStart: true))
      endpointIndex[CoordinateKey(way.nodes[way.nodes.count - 1]), default: []]
        .append(Endpoint(wayIndex: wayIndex, isStart: false))
    }
    let indexByID = Dictionary(uniqueKeysWithValues: ways.indices.map { (ways[$0].id, $0) })
    let focusWayIndices = Set(focusSourceKeys.compactMap { sourceKey -> Int? in
      guard let separator = sourceKey.lastIndex(of: ":") else { return nil }
      return indexByID[String(sourceKey[..<separator])]
    })
    return focusWayIndices.sorted().map {
      wholeCurveStudyRoute(
        for: $0,
        ways: ways,
        endpointIndex: endpointIndex,
        contextMeters: contextMeters
      )
    }
  }

  private static func wholeCurveStudyRoute(
    for wayIndex: Int,
    ways: [MapRenderedWay],
    endpointIndex: [CoordinateKey: [Endpoint]],
    contextMeters: Double
  ) -> MapWholeCurveRoute {
    let sourceWay = ways[wayIndex]
    var path = routePoints(for: wayIndex, outwardNodes: sourceWay.nodes.map(\.tileNode), ways: ways)
    var originalOffset = 0
    var visited: Set<Int> = [wayIndex]
    var frontWayIndex = wayIndex
    var backWayIndex = wayIndex
    var frontIsAmbiguous = false
    var backIsAmbiguous = false

    var frontAttempts = 0
    while frontAttempts < 128,
          routeDistance(path, from: 0, through: originalOffset) < contextMeters {
      frontAttempts += 1
      let search = bestContinuation(
        from: path.map(\.node),
        atFront: true,
        currentWayIndex: frontWayIndex,
        ways: ways,
        endpointIndex: endpointIndex,
        visited: visited
      )
      guard !search.isAmbiguous, let outward = search.continuation else {
        frontIsAmbiguous = search.isAmbiguous
        break
      }
      visited.insert(outward.wayIndex)
      let outwardPoints = routePoints(
        for: outward.wayIndex,
        outwardNodes: outward.nodes,
        ways: ways
      )
      let added = Array(outwardPoints.dropFirst().reversed())
      guard !added.isEmpty else { break }
      path.insert(contentsOf: added, at: 0)
      originalOffset += added.count
      frontWayIndex = outward.wayIndex
    }
    if frontAttempts == 128 { frontIsAmbiguous = true }

    let originalEndOffset = originalOffset + sourceWay.nodes.count - 1
    var backAttempts = 0
    while backAttempts < 128,
          routeDistance(path, from: originalEndOffset, through: path.count - 1) < contextMeters {
      backAttempts += 1
      let search = bestContinuation(
        from: path.map(\.node),
        atFront: false,
        currentWayIndex: backWayIndex,
        ways: ways,
        endpointIndex: endpointIndex,
        visited: visited
      )
      guard !search.isAmbiguous, let outward = search.continuation else {
        backIsAmbiguous = search.isAmbiguous
        break
      }
      visited.insert(outward.wayIndex)
      let outwardPoints = routePoints(
        for: outward.wayIndex,
        outwardNodes: outward.nodes,
        ways: ways
      )
      let added = Array(outwardPoints.dropFirst())
      guard !added.isEmpty else { break }
      path.append(contentsOf: added)
      backWayIndex = outward.wayIndex
    }
    if backAttempts == 128 { backIsAmbiguous = true }

    return MapWholeCurveRoute(
      anchorWayID: sourceWay.id,
      roadName: sourceWay.name,
      reference: sourceWay.reference,
      points: path,
      frontContextMeters: routeDistance(path, from: 0, through: originalOffset),
      backContextMeters: routeDistance(
        path,
        from: originalOffset + sourceWay.nodes.count - 1,
        through: path.count - 1
      ),
      frontIsAmbiguous: frontIsAmbiguous,
      backIsAmbiguous: backIsAmbiguous
    )
  }

  private static func routePoints(
    for wayIndex: Int,
    outwardNodes: [MapTileNode],
    ways: [MapRenderedWay]
  ) -> [MapWholeCurveRoutePoint] {
    let way = ways[wayIndex]
    guard let outwardFirst = outwardNodes.first else { return [] }
    let isForward = way.nodes.first.map(CoordinateKey.init) == CoordinateKey(outwardFirst.renderedNode)
    let indexedNodes: [(Int, MapRenderedNode)] = isForward
      ? Array(way.nodes.enumerated())
      : Array(way.nodes.enumerated().reversed())
    return indexedNodes.map { index, node in
      MapWholeCurveRoutePoint(
        node: node.tileNode,
        sourceKey: "\(way.id):\(index)",
        currentCurvature: node.curvature,
        currentCurvatureContextComplete: node.curvatureContextComplete
      )
    }
  }

  private static func routeDistance(
    _ points: [MapWholeCurveRoutePoint],
    from startIndex: Int,
    through endIndex: Int
  ) -> Double {
    guard points.indices.contains(startIndex), points.indices.contains(endIndex), startIndex < endIndex else {
      return 0
    }
    return (startIndex ..< endIndex).reduce(0) {
      $0 + MapBakeMath.distanceMeters(from: points[$1].node, to: points[$1 + 1].node)
    }
  }

  private static func routeContext(
    for wayIndex: Int,
    ways: [MapRenderedWay],
    endpointIndex: [CoordinateKey: [Endpoint]]
  ) -> RouteContext {
    var path = ways[wayIndex].nodes.map(\.tileNode)
    var originalOffset = 0
    var visited: Set<Int> = [wayIndex]
    var frontWayIndex = wayIndex
    var backWayIndex = wayIndex
    var mergeOrSplitNodeIndices: Set<Int> = []
    var ambiguousBoundaryNodeIndices: Set<Int> = []

    var frontAttempts = 0
    while frontAttempts < 32,
          !frontContextIsComplete(path: path, originalOffset: originalOffset) {
      frontAttempts += 1
      let search = bestContinuation(
        from: path,
        atFront: true,
        currentWayIndex: frontWayIndex,
        ways: ways,
        endpointIndex: endpointIndex,
        visited: visited
      )
      guard !search.isAmbiguous, let outward = search.continuation else {
        if search.isAmbiguous { ambiguousBoundaryNodeIndices.insert(0) }
        break
      }
      visited.insert(outward.wayIndex)
      let added = Array(outward.nodes.dropFirst().reversed())
      guard !added.isEmpty else { break }
      mergeOrSplitNodeIndices = Set(mergeOrSplitNodeIndices.map { $0 + added.count })
      ambiguousBoundaryNodeIndices = Set(ambiguousBoundaryNodeIndices.map { $0 + added.count })
      path.insert(contentsOf: added, at: 0)
      originalOffset += added.count
      let boundaryIndex = added.count
      recordTransition(
        previous: ways[outward.wayIndex],
        next: ways[frontWayIndex],
        boundaryIndex: boundaryIndex,
        sourceWay: ways[wayIndex],
        mergeOrSplitNodeIndices: &mergeOrSplitNodeIndices,
        ambiguousBoundaryNodeIndices: &ambiguousBoundaryNodeIndices
      )
      frontWayIndex = outward.wayIndex
    }
    if frontAttempts == 32,
       !frontContextIsComplete(path: path, originalOffset: originalOffset) {
      ambiguousBoundaryNodeIndices.insert(0)
    }

    var backAttempts = 0
    while backAttempts < 32,
          !backContextIsComplete(
            path: path,
            originalOffset: originalOffset,
            originalNodeCount: ways[wayIndex].nodes.count
          ) {
      backAttempts += 1
      let search = bestContinuation(
        from: path,
        atFront: false,
        currentWayIndex: backWayIndex,
        ways: ways,
        endpointIndex: endpointIndex,
        visited: visited
      )
      guard !search.isAmbiguous, let outward = search.continuation else {
        if search.isAmbiguous { ambiguousBoundaryNodeIndices.insert(path.count - 1) }
        break
      }
      visited.insert(outward.wayIndex)
      let added = Array(outward.nodes.dropFirst())
      guard !added.isEmpty else { break }
      let boundaryIndex = path.count - 1
      path.append(contentsOf: added)
      recordTransition(
        previous: ways[backWayIndex],
        next: ways[outward.wayIndex],
        boundaryIndex: boundaryIndex,
        sourceWay: ways[wayIndex],
        mergeOrSplitNodeIndices: &mergeOrSplitNodeIndices,
        ambiguousBoundaryNodeIndices: &ambiguousBoundaryNodeIndices
      )
      backWayIndex = outward.wayIndex
    }
    if backAttempts == 32,
       !backContextIsComplete(
         path: path,
         originalOffset: originalOffset,
         originalNodeCount: ways[wayIndex].nodes.count
       ) {
      ambiguousBoundaryNodeIndices.insert(path.count - 1)
    }
    return RouteContext(
      nodes: path,
      originalOffset: originalOffset,
      mergeOrSplitNodeIndices: mergeOrSplitNodeIndices,
      ambiguousBoundaryNodeIndices: ambiguousBoundaryNodeIndices
    )
  }

  private static func frontContextIsComplete(
    path: [MapTileNode],
    originalOffset: Int
  ) -> Bool {
    guard originalOffset >= 2, path.indices.contains(originalOffset) else { return false }
    return MapBakeMath.distanceMeters(
      from: path[0],
      to: path[originalOffset - 2]
    ) > MapBakeMath.mergeOrSplitRadiusMeters
  }

  private static func backContextIsComplete(
    path: [MapTileNode],
    originalOffset: Int,
    originalNodeCount: Int
  ) -> Bool {
    let originalEndIndex = originalOffset + originalNodeCount - 1
    guard path.indices.contains(originalEndIndex),
          path.count - originalEndIndex - 1 >= 3
    else { return false }
    return MapBakeMath.distanceMeters(
      from: path[originalEndIndex],
      to: path[path.count - 1]
    ) > MapBakeMath.mergeOrSplitRadiusMeters
  }

  private static func recordTransition(
    previous: MapRenderedWay,
    next: MapRenderedWay,
    boundaryIndex: Int,
    sourceWay: MapRenderedWay,
    mergeOrSplitNodeIndices: inout Set<Int>,
    ambiguousBoundaryNodeIndices: inout Set<Int>
  ) {
    guard previous.lanes != next.lanes || previous.oneWay != next.oneWay else { return }
    // A one-way source fixes travel direction to the stored OSM node order.
    // Bidirectional ways do not tell the offline tuner which direction the car
    // will take, so direction-dependent lane transitions must fail closed.
    guard sourceWay.oneWay else {
      ambiguousBoundaryNodeIndices.insert(boundaryIndex)
      return
    }
    if previous.lanes < next.lanes ||
      (previous.lanes > next.lanes && !previous.oneWay && next.oneWay) {
      mergeOrSplitNodeIndices.insert(boundaryIndex)
    }
  }

  private static func bestContinuation(
    from path: [MapTileNode],
    atFront: Bool,
    currentWayIndex: Int,
    ways: [MapRenderedWay],
    endpointIndex: [CoordinateKey: [Endpoint]],
    visited: Set<Int>
  ) -> ContinuationSearch {
    guard path.count >= 2 else {
      return ContinuationSearch(continuation: nil, isAmbiguous: false)
    }
    let anchor = atFront ? path[0] : path[path.count - 1]
    let key = CoordinateKey(anchor.renderedNode)
    var candidates: [(wayIndex: Int, nodes: [MapTileNode], junctionCurvature: Double)] = []
    var seenWayIndices: Set<Int> = []

    for endpoint in endpointIndex[key, default: []]
      where !visited.contains(endpoint.wayIndex) && seenWayIndices.insert(endpoint.wayIndex).inserted {
      let candidate = ways[endpoint.wayIndex]
      let candidateNodes = candidate.nodes.map(\.tileNode)
      let outward: [MapTileNode]
      if atFront {
        if !endpoint.isStart {
          outward = Array(candidateNodes.reversed())
        } else if !candidate.oneWay {
          outward = candidateNodes
        } else {
          continue
        }
      } else if endpoint.isStart {
        outward = candidateNodes
      } else if !candidate.oneWay {
        outward = Array(candidateNodes.reversed())
      } else {
        continue
      }
      guard outward.count >= 2 else { continue }
      let inwardNode = atFront ? path[1] : path[path.count - 2]
      let junctionCurvature = MapBakeMath.curvature(
        previous: inwardNode,
        current: anchor,
        next: outward[1]
      )
      candidates.append((endpoint.wayIndex, outward, junctionCurvature))
    }

    let currentWay = ways[currentWayIndex]
    let gentleCandidates = candidates.filter { $0.junctionCurvature <= 0.1 }
    if !currentWay.name.isEmpty,
       let result = uniquePriorityResult(
         gentleCandidates.filter { ways[$0.wayIndex].name == currentWay.name }
       ) {
      return result
    }
    if !currentWay.reference.isEmpty,
       let result = uniquePriorityResult(
         gentleCandidates.filter { ways[$0.wayIndex].reference == currentWay.reference }
       ) {
      return result
    }
    if !currentWay.reference.isEmpty {
      let sourceRefs = Set(currentWay.reference.split(separator: ";").map(String.init))
      let partialMatches = gentleCandidates.filter { candidate in
        let candidateRefs = Set(ways[candidate.wayIndex].reference.split(separator: ";").map(String.init))
        return !sourceRefs.isDisjoint(with: candidateRefs)
      }
      if !partialMatches.isEmpty {
        return uniqueMinimumCurvatureResult(partialMatches)
      }
    }

    // A single remaining direction-feasible physical continuation is mapd's
    // final fallback when identity priorities do not match. With multiple
    // candidates, fitting has no live route evidence and stays fail-closed.
    guard candidates.count <= 1 else {
      return ContinuationSearch(continuation: nil, isAmbiguous: true)
    }
    guard let candidate = candidates.first else {
      return ContinuationSearch(continuation: nil, isAmbiguous: false)
    }
    return ContinuationSearch(
      continuation: (candidate.wayIndex, candidate.nodes),
      isAmbiguous: false
    )
  }

  private static func uniquePriorityResult(
    _ candidates: [(wayIndex: Int, nodes: [MapTileNode], junctionCurvature: Double)]
  ) -> ContinuationSearch? {
    guard !candidates.isEmpty else { return nil }
    guard candidates.count == 1, let candidate = candidates.first else {
      return ContinuationSearch(continuation: nil, isAmbiguous: true)
    }
    return ContinuationSearch(
      continuation: (candidate.wayIndex, candidate.nodes),
      isAmbiguous: false
    )
  }

  private static func uniqueMinimumCurvatureResult(
    _ candidates: [(wayIndex: Int, nodes: [MapTileNode], junctionCurvature: Double)]
  ) -> ContinuationSearch {
    let minimum = candidates.map(\.junctionCurvature).min() ?? .infinity
    let tolerance = max(1.0e-12, minimum * 1.0e-9)
    let winners = candidates.filter { abs($0.junctionCurvature - minimum) <= tolerance }
    guard winners.count == 1, let winner = winners.first else {
      return ContinuationSearch(continuation: nil, isAmbiguous: true)
    }
    return ContinuationSearch(
      continuation: (winner.wayIndex, winner.nodes),
      isAmbiguous: false
    )
  }
}

private extension MapRenderedNode {
  var tileNode: MapTileNode {
    MapTileNode(latitude: latitude, longitude: longitude, bakedSpeedMPS: bakedSpeedMPS)
  }
}

private extension MapTileNode {
  var renderedNode: MapRenderedNode {
    MapRenderedNode(
      latitude: latitude,
      longitude: longitude,
      curvature: 0,
      bakedSpeedMPS: bakedSpeedMPS,
      proposedSpeedMPS: 0
    )
  }
}

extension Double {
  var mapMPH: Double { self * VTSCMath.metersPerSecondToMPH }
}
