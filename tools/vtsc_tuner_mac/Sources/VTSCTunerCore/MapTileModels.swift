import Foundation

public struct MapTileBounds: Codable, Equatable, Hashable, Sendable {
  public var minLatitude: Double
  public var minLongitude: Double
  public var maxLatitude: Double
  public var maxLongitude: Double

  public init(
    minLatitude: Double,
    minLongitude: Double,
    maxLatitude: Double,
    maxLongitude: Double
  ) {
    self.minLatitude = minLatitude
    self.minLongitude = minLongitude
    self.maxLatitude = maxLatitude
    self.maxLongitude = maxLongitude
  }

  public func intersects(_ other: Self) -> Bool {
    minLatitude <= other.maxLatitude && maxLatitude >= other.minLatitude &&
      minLongitude <= other.maxLongitude && maxLongitude >= other.minLongitude
  }

  public func contains(latitude: Double, longitude: Double) -> Bool {
    latitude >= minLatitude && latitude <= maxLatitude &&
      longitude >= minLongitude && longitude <= maxLongitude
  }

  public func expanded(by degrees: Double) -> Self {
    Self(
      minLatitude: max(-90.0, minLatitude - degrees),
      minLongitude: max(-180.0, minLongitude - degrees),
      maxLatitude: min(90.0, maxLatitude + degrees),
      maxLongitude: min(180.0, maxLongitude + degrees)
    )
  }

  public func union(_ other: Self) -> Self {
    Self(
      minLatitude: min(minLatitude, other.minLatitude),
      minLongitude: min(minLongitude, other.minLongitude),
      maxLatitude: max(maxLatitude, other.maxLatitude),
      maxLongitude: max(maxLongitude, other.maxLongitude)
    )
  }

  enum CodingKeys: String, CodingKey {
    case minLatitude = "min_latitude"
    case minLongitude = "min_longitude"
    case maxLatitude = "max_latitude"
    case maxLongitude = "max_longitude"
  }
}

public struct MapTileNode: Codable, Equatable, Hashable, Sendable {
  public var latitude: Double
  public var longitude: Double
  public var bakedSpeedMPS: Double?

  public init(latitude: Double, longitude: Double, bakedSpeedMPS: Double? = nil) {
    self.latitude = latitude
    self.longitude = longitude
    self.bakedSpeedMPS = bakedSpeedMPS
  }

  enum CodingKeys: String, CodingKey {
    case latitude, longitude
    case bakedSpeedMPS = "baked_speed_mps"
  }
}

public struct MapTileWay: Codable, Equatable, Identifiable, Sendable {
  public var stableID: String
  public var name: String
  public var reference: String
  public var bounds: MapTileBounds
  public var nodes: [MapTileNode]
  public var maxSpeedMPS: Double
  public var maxSpeedForwardMPS: Double
  public var maxSpeedBackwardMPS: Double
  public var advisorySpeedMPS: Double
  public var lanes: UInt8
  public var hazard: String
  public var oneWay: Bool
  public var windingForwardLevel: UInt8
  public var windingBackwardLevel: UInt8
  public var windingForwardScore: UInt8
  public var windingBackwardScore: UInt8
  public var windingForwardConfidence: UInt8
  public var windingBackwardConfidence: UInt8

  public var id: String { stableID }

  public init(
    stableID: String,
    name: String = "",
    reference: String = "",
    bounds: MapTileBounds,
    nodes: [MapTileNode],
    maxSpeedMPS: Double = 0,
    maxSpeedForwardMPS: Double = 0,
    maxSpeedBackwardMPS: Double = 0,
    advisorySpeedMPS: Double = 0,
    lanes: UInt8 = 0,
    hazard: String = "",
    oneWay: Bool = false,
    windingForwardLevel: UInt8 = 0,
    windingBackwardLevel: UInt8 = 0,
    windingForwardScore: UInt8 = 0,
    windingBackwardScore: UInt8 = 0,
    windingForwardConfidence: UInt8 = 0,
    windingBackwardConfidence: UInt8 = 0
  ) {
    self.stableID = stableID
    self.name = name
    self.reference = reference
    self.bounds = bounds
    self.nodes = nodes
    self.maxSpeedMPS = maxSpeedMPS
    self.maxSpeedForwardMPS = maxSpeedForwardMPS
    self.maxSpeedBackwardMPS = maxSpeedBackwardMPS
    self.advisorySpeedMPS = advisorySpeedMPS
    self.lanes = lanes
    self.hazard = hazard
    self.oneWay = oneWay
    self.windingForwardLevel = windingForwardLevel
    self.windingBackwardLevel = windingBackwardLevel
    self.windingForwardScore = windingForwardScore
    self.windingBackwardScore = windingBackwardScore
    self.windingForwardConfidence = windingForwardConfidence
    self.windingBackwardConfidence = windingBackwardConfidence
  }

  enum CodingKeys: String, CodingKey {
    case stableID = "stable_id"
    case name
    case reference
    case bounds
    case nodes
    case maxSpeedMPS = "max_speed_mps"
    case maxSpeedForwardMPS = "max_speed_forward_mps"
    case maxSpeedBackwardMPS = "max_speed_backward_mps"
    case advisorySpeedMPS = "advisory_speed_mps"
    case lanes
    case hazard
    case oneWay = "one_way"
    case windingForwardLevel = "winding_forward_level"
    case windingBackwardLevel = "winding_backward_level"
    case windingForwardScore = "winding_forward_score"
    case windingBackwardScore = "winding_backward_score"
    case windingForwardConfidence = "winding_forward_confidence"
    case windingBackwardConfidence = "winding_backward_confidence"
  }
}

public struct MapTile: Codable, Equatable, Sendable {
  public var sourcePath: String
  public var bounds: MapTileBounds
  public var overlap: Double
  public var schemaVersion: UInt16
  public var sigmoidHash: String
  public var ways: [MapTileWay]

  public init(
    sourcePath: String,
    bounds: MapTileBounds,
    overlap: Double,
    schemaVersion: UInt16,
    sigmoidHash: String,
    ways: [MapTileWay]
  ) {
    self.sourcePath = sourcePath
    self.bounds = bounds
    self.overlap = overlap
    self.schemaVersion = schemaVersion
    self.sigmoidHash = sigmoidHash
    self.ways = ways
  }

  public var estimatedDecodedBytes: Int {
    256 + ways.reduce(0) { partial, way in
      partial + 320 + way.nodes.count * 48 + way.name.utf8.count + way.reference.utf8.count
    }
  }

  enum CodingKeys: String, CodingKey {
    case sourcePath = "source_path"
    case bounds
    case overlap
    case schemaVersion = "schema_version"
    case sigmoidHash = "sigmoid_hash"
    case ways
  }
}
