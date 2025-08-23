using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ModularAssistiveDrivingSystem {
  state @0 :ModularAssistiveDrivingSystemState;
  enabled @1 :Bool;
  active @2 :Bool;
  available @3 :Bool;

  enum ModularAssistiveDrivingSystemState {
    disabled @0;
    paused @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

# Same struct as Log.RadarState.LeadData
struct LeadData {
  dRel @0 :Float32;
  yRel @1 :Float32;
  vRel @2 :Float32;
  aRel @3 :Float32;
  vLead @4 :Float32;
  dPath @6 :Float32;
  vLat @7 :Float32;
  vLeadK @8 :Float32;
  aLeadK @9 :Float32;
  fcw @10 :Bool;
  status @11 :Bool;
  aLeadTau @12 :Float32;
  modelProb @13 :Float32;
  radar @14 :Bool;
  radarTrackId @15 :Int32 = -1;

  aLeadDEPRECATED @5 :Float32;
}

struct SelfdriveStateSP @0x81c2f05a394cf4af {
  mads @0 :ModularAssistiveDrivingSystem;
}

struct ModelManagerSP @0xaedffd8f31e7b55d {
  activeBundle @0 :ModelBundle;
  selectedBundle @1 :ModelBundle;
  availableBundles @2 :List(ModelBundle);

  struct DownloadUri {
    uri @0 :Text;
    sha256 @1 :Text;
  }

  enum DownloadStatus {
    notDownloading @0;
    downloading @1;
    downloaded @2;
    cached @3;
    failed @4;
  }

  struct DownloadProgress {
    status @0 :DownloadStatus;
    progress @1 :Float32;
    eta @2 :UInt32;
  }

  struct Artifact {
    fileName @0 :Text;
    downloadUri @1 :DownloadUri;
    downloadProgress @2 :DownloadProgress;
  }

  struct Model {
    type @0 :Type;
    artifact @1 :Artifact;  # Main artifact
    metadata @2 :Artifact;  # Metadata artifact

    enum Type {
      supercombo @0;
      navigation @1;
      vision @2;
      policy @3;
    }
  }

  enum Runner {
    snpe @0;
    tinygrad @1;
    stock @2;
  }

  struct Override {
    key @0 :Text;
    value @1 :Text;
  }

  struct ModelBundle {
    index @0 :UInt32;
    internalName @1 :Text;
    displayName @2 :Text;
    models @3 :List(Model);
    status @4 :DownloadStatus;
    generation @5 :UInt32;
    environment @6 :Text;
    runner @7 :Runner;
    is20hz @8 :Bool;
    ref @9 :Text;
    minimumSelectorVersion @10 :UInt32;
    overrides @11 :List(Override);
  }
}

struct LongitudinalPlanSP @0xf35cc4560bbf6ec2 {
  dec @0 :DynamicExperimentalControl;
  events @1 :List(OnroadEventSP.Event);
  slc @2 :SpeedLimitControl;
  visionTurnSpeedControl @3 :VisionTurnSpeedControl;
  accelPersonality @4 :AccelerationPersonality;

  struct DynamicExperimentalControl {
    state @0 :DynamicExperimentalControlState;
    enabled @1 :Bool;
    active @2 :Bool;

    enum DynamicExperimentalControlState {
      acc @0;
      blended @1;
    }
  }

  struct VisionTurnSpeedControl {
    state @0 :VisionTurnSpeedControlState;
    velocity @1 :Float32;
    currentLateralAccel @2 :Float32;
    maxPredictedLateralAccel @3 :Float32;

    enum VisionTurnSpeedControlState {
      disabled @0; # No predicted substantial turn on vision range or feature disabled.
      entering @1; # A substantial turn is predicted ahead, adapting speed to turn comfort levels.
      turning @2; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      leaving @3; # Road ahead straightens. Start to allow positive acceleration.
    }
  }

  struct SpeedLimitControl {
    state @0 :SpeedLimitControlState;
    enabled @1 :Bool;
    active @2 :Bool;
    speedLimit @3 :Float32;
    speedLimitOffset @4 :Float32;
    distToSpeedLimit @5 :Float32;
    source @6 :SlcSource;  # Selected source of current speed limit
  }

  enum SpeedLimitControlState {
    inactive @0; # No speed limit set or not enabled by parameter.
    tempInactive @1; # User wants to ignore speed limit until it changes.
    preActive @2;
    adapting @3; # Reducing speed to match new speed limit.
    active @4; # Cruising at speed limit.
  }

  enum AccelerationPersonality {
    sport @0;
    normal @1;
    eco @2;
  }

  # Source for Speed Limit Control selection
  enum SlcSource {
    none @0;
    car @1;
    map @2;
  }
}

struct OnroadEventSP @0xda96579883444c35 {
  events @0 :List(Event);

  struct Event {
    name @0 :EventName;

    # event types
    enable @1 :Bool;
    noEntry @2 :Bool;
    warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
    userDisable @4 :Bool;
    softDisable @5 :Bool;
    immediateDisable @6 :Bool;
    preEnable @7 :Bool;
    permanent @8 :Bool; # alerts presented regardless of openpilot state
    overrideLateral @10 :Bool;
    overrideLongitudinal @9 :Bool;
  }

  enum EventName {
    lkasEnable @0;
    lkasDisable @1;
    manualSteeringRequired @2;
    manualLongitudinalRequired @3;
    silentLkasEnable @4;
    silentLkasDisable @5;
    silentBrakeHold @6;
    silentWrongGear @7;
    silentReverseGear @8;
    silentDoorOpen @9;
    silentSeatbeltNotLatched @10;
    silentParkBrake @11;
    controlsMismatchLateral @12;
    hyundaiRadarTracksConfirmed @13;
    experimentalModeSwitched @14;
    wrongCarModeAlertOnly @15;
    pedalPressedAlertOnly @16;
    speedLimitPreActive @17;
    speedLimitActive @18;
    speedLimitConfirmed @19;
    speedLimitValueChange @20;
  }
}

struct CarParamsSP @0x80ae746ee2596b11 {
  flags @0 :UInt32;        # flags for car specific quirks in sunnypilot
  safetyParam @1 : Int16;  # flags for sunnypilot's custom safety flags

  neuralNetworkLateralControl @2 :NeuralNetworkLateralControl;

  struct NeuralNetworkLateralControl {
    model @0 :Model;
    fuzzyFingerprint @1 :Bool;

    struct Model {
      path @0 :Text;
      name @1 :Text;
    }
  }
}

struct CarControlSP @0xa5cd762cd951a455 {
  mads @0 :ModularAssistiveDrivingSystem;
  params @1 :List(Param);
  leadOne @2 :LeadData;
  leadTwo @3 :LeadData;

  struct Param {
    key @0 :Text;
    value @1 :Text;
  }
}

struct BackupManagerSP @0xf98d843bfd7004a3 {
  backupStatus @0 :Status;
  restoreStatus @1 :Status;
  backupProgress @2 :Float32;
  restoreProgress @3 :Float32;
  lastError @4 :Text;
  currentBackup @5 :BackupInfo;
  backupHistory @6 :List(BackupInfo);

  enum Status {
    idle @0;
    inProgress @1;
    completed @2;
    failed @3;
  }

  struct Version {
    major @0 :UInt16;
    minor @1 :UInt16;
    patch @2 :UInt16;
    build @3 :UInt16;
    branch @4 :Text;
  }

  struct MetadataEntry {
    key @0 :Text;
    value @1 :Text;
    tags @2 :List(Text);
  }

  struct BackupInfo {
    deviceId @0 :Text;
    version @1 :UInt32;
    config @2 :Text;
    isEncrypted @3 :Bool;
    createdAt @4 :Text;  # ISO timestamp
    updatedAt @5 :Text;  # ISO timestamp
    sunnypilotVersion @6 :Version;
    backupMetadata @7 :List(MetadataEntry);
  }
}

struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;  # m/s
}

struct LiveMapDataSP @0xf416ec09499d9d19 {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
  
  # Road geometry and lane information for RTI integration
  roadGeometryValid @6 :Bool;
  currentRoadSegment @7 :RoadSegment;
  nearbyRoadSegments @8 :List(RoadSegment);  # Road segments within ~500m
  
  struct RoadSegment {
    wayId @0 :UInt64;  # OSM way ID for identification
    roadClass @1 :RoadClass;  # Highway type classification
    centerline @2 :List(Coordinate);  # Road centerline geometry
    lanes @3 :List(Lane);  # Lane information
    barriers @4 :List(Barrier);  # Barriers, medians, etc.
    levelSeparation @5 :Int8;  # Bridge/underpass level (-1=under, 0=ground, 1=bridge)
    maxSpeed @6 :Float32;  # Speed limit in m/s
    roadDirection @7 :Float32;  # Road bearing at current position in degrees
    
    struct Coordinate {
      latitude @0 :Float64;
      longitude @1 :Float64;
      distanceFromStart @2 :Float32;  # Distance along road from segment start
    }
    
    struct Lane {
      laneIndex @0 :UInt8;  # Lane number (0 = rightmost)
      width @1 :Float32;  # Lane width in meters
      type @2 :LaneType;
      centerline @3 :List(Coordinate);  # Lane centerline if available
      
      enum LaneType {
        driving @0;
        bus @1;
        bicycle @2;
        parking @3;
        shoulder @4;
        median @5;
      }
    }
    
    struct Barrier {
      type @0 :BarrierType;
      coordinates @1 :List(Coordinate);
      
      enum BarrierType {
        median @0;
        guardrail @1;
        wall @2;
        fence @3;
        curb @4;
      }
    }
    
    enum RoadClass {
      motorway @0;      # Highway/freeway
      trunk @1;         # Major arterial
      primary @2;       # Primary road
      secondary @3;     # Secondary road
      tertiary @4;      # Local major road
      residential @5;   # Local residential
      service @6;       # Service road
      unclassified @7;  # Unclassified road
    }
  }
}

struct RtiStateSP @0xa1680744031fdb2d {
  # Timestamp when this state was computed (in nanoseconds since boot)
  timeStamp @0 :UInt64;
  
  # True if there's a threat ahead that requires speed reduction
  threatAhead @1 :Bool;
  
  # Distance to the nearest relevant threat in meters (0 if no threat)
  threatDistanceM @2 :Float32;
  
  # Recommended speed in m/s (0 means no recommendation)
  # Must satisfy: 0 <= recommendedSpeed <= current_speed
  recommendedSpeed @3 :Float32;
  
  # Source of traffic intelligence data
  source @4 :Text;
  
  # API connection health status
  apiStatus @5 :ApiStatus;
  
  # Detailed threat information for HUD display (up to 5 threats)
  threats @6 :List(Threat);
  
  enum ApiStatus {
    connected @0;
    disconnected @1;
    rateLimited @2;
    error @3;
    offline @4;  # Offline mode - no API calls being made
  }
  
  struct Threat {
    # Unique identifier for this threat
    id @0 :Text;
    
    # Type of threat detected
    type @1 :ThreatType;
    
    # Geographic coordinates
    latitude @2 :Float64;
    longitude @3 :Float64;
    
    # Distance from ego vehicle in meters
    distance @4 :Float32;
    
    # Direction relative to ego vehicle
    direction @5 :Direction;
    
    # Confidence in threat detection (0.0 - 1.0)
    confidence @6 :Float32;
    
    # Speed limit at threat location (m/s)
    speedLimitMs @7 :Float32;

    # True if threat is determined to be on the same road as ego
    # Published from backend ThreatDetector; used by HUD for visual indicator
    onSameRoad @8 :Bool;
  }
  
  enum ThreatType {
    police @0;
    speedTrap @1;
    speedCamera @2;
    accident @3;
    hazard @4;
    construction @5;
    jam @6;
    policeHiding @7;     # Police hiding (subtype of police)
    roadHazard @8;       # Hazard on road
    shoulderHazard @9;   # Hazard on shoulder
    roadClosed @10;      # Road closed alerts
  }
  
  enum Direction {
    ahead @0;
    behind @1;
    left @2;
    right @3;
    unknown @4;
  }
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
