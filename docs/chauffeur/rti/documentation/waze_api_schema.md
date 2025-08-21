# Waze API Schema and RTI Mapping

This document captures the expected JSON response from the Waze (RapidAPI) endpoint used by RTI and how we parse it into internal structures. It also notes field name variations we support for backward compatibility.

## Endpoint

- Base: `https://waze.p.rapidapi.com`
- Path: `/alerts-and-jams`

## Sample Response

The body below illustrates the structure and representative values:

```
{
  "status": "OK",
  "request_id": "79a5d069-bef5-42f2-ab6d-6dbe7ac2d274",
  "parameters": {
    "bottom_left": [
      "40.66615391742187",
      "-74.13732147216798"
    ],
    "top_right": [
      "40.772787404902594",
      "-73.76818084716798"
    ],
    "max_alerts": 20,
    "max_jams": 20
  },
  "data": {
    "alerts": [
      {
        "alert_id": "1198452451",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "krzycholub",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-01-05T15:41:34.000Z",
        "country": "US",
        "city": "Jersey City, NJ",
        "street": "Barrow St",
        "latitude": 40.720179,
        "longitude": -74.045089,
        "num_thumbs_up": 31,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "74084699",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "New York State Department of Transportation",
        "description": "Utility work on CR-659",
        "image": null,
        "publish_datetime_utc": "2024-06-27T15:53:26.000Z",
        "country": "US",
        "city": "Kearny, NJ",
        "street": "Pennsylvania Ave",
        "latitude": 40.739208,
        "longitude": -74.104437,
        "num_thumbs_up": 6,
        "alert_reliability": 10,
        "alert_confidence": 3,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "630228890",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "Be8el0ve",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-08-12T13:49:56.000Z",
        "country": "US",
        "city": "Brooklyn, NY",
        "street": "Gold St",
        "latitude": 40.702287,
        "longitude": -73.982807,
        "num_thumbs_up": 18,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218673202",
        "type": "HAZARD",
        "subtype": "HAZARD_ON_ROAD_EMERGENCY_VEHICLE",
        "reported_by": "HAAS Alert",
        "description": "Ambulance on scene",
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:39:44.000Z",
        "country": "US",
        "city": "Brooklyn, NY",
        "street": "9th St",
        "latitude": 40.668243,
        "longitude": -73.984384,
        "num_thumbs_up": 1,
        "alert_reliability": 7,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218551612",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "WazeClosures",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:21:39.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "to Lincoln Tun / NJ",
        "latitude": 40.758844,
        "longitude": -73.997574,
        "num_thumbs_up": 0,
        "alert_reliability": 6,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218551568",
        "type": "HAZARD",
        "subtype": "HAZARD_ON_ROAD_EMERGENCY_VEHICLE",
        "reported_by": "HAAS Alert",
        "description": "Ambulance on scene",
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:21:09.000Z",
        "country": "US",
        "city": "Brooklyn, NY",
        "street": "Bergen St",
        "latitude": 40.680726,
        "longitude": -73.974383,
        "num_thumbs_up": 0,
        "alert_reliability": 6,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218625930",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "WazeClosures",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:38:29.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "FDR Dr N",
        "latitude": 40.761158,
        "longitude": -73.955871,
        "num_thumbs_up": 0,
        "alert_reliability": 6,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218564679",
        "type": "POLICE",
        "subtype": null,
        "reported_by": null,
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:19:51.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "Walker St",
        "latitude": 40.720332,
        "longitude": -74.006221,
        "num_thumbs_up": 12,
        "alert_reliability": 10,
        "alert_confidence": 4,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218625900",
        "type": "POLICE",
        "subtype": null,
        "reported_by": null,
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:38:25.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "12th Ave",
        "latitude": 40.747817,
        "longitude": -74.007691,
        "num_thumbs_up": 3,
        "alert_reliability": 8,
        "alert_confidence": 1,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1329514637",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "dhschneider",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-03-14T18:00:15.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "E 60th St",
        "latitude": 40.75928,
        "longitude": -73.959274,
        "num_thumbs_up": 108,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218673732",
        "type": "ROAD_CLOSED",
        "subtype": null,
        "reported_by": "Busyboyy",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:41:53.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "Lexington Ave",
        "latitude": 40.760929,
        "longitude": -73.969132,
        "num_thumbs_up": 0,
        "alert_reliability": 5,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218634416",
        "type": "POLICE",
        "subtype": "POLICE_HIDING",
        "reported_by": null,
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:37:12.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "12th Ave",
        "latitude": 40.760951,
        "longitude": -74.001886,
        "num_thumbs_up": 10,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218660087",
        "type": "HAZARD",
        "subtype": "HAZARD_ON_ROAD_CONSTRUCTION",
        "reported_by": null,
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:45:14.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "E 23rd St",
        "latitude": 40.737722,
        "longitude": -73.980537,
        "num_thumbs_up": 0,
        "alert_reliability": 6,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218627297",
        "type": "POLICE",
        "subtype": null,
        "reported_by": "sulmax",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:43:21.000Z",
        "country": "US",
        "city": "Jersey City, NJ",
        "street": "John F Kennedy Blvd",
        "latitude": 40.739873,
        "longitude": -74.06176,
        "num_thumbs_up": 0,
        "alert_reliability": 5,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218528205",
        "type": "HAZARD",
        "subtype": "HAZARD_ON_ROAD_LANE_CLOSED",
        "reported_by": "HAAS Alert",
        "description": "Road Work Ahead - move RIGHT",
        "image": null,
        "publish_datetime_utc": "2024-10-18T21:00:43.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "Park Ave",
        "latitude": 40.747509,
        "longitude": -73.980912,
        "num_thumbs_up": 3,
        "alert_reliability": 10,
        "alert_confidence": 1,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1166102715",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "G_W1Z",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-09-11T22:20:26.000Z",
        "country": "US",
        "city": "Jersey City, NJ",
        "street": "Garfield Ave",
        "latitude": 40.70151,
        "longitude": -74.077726,
        "num_thumbs_up": 0,
        "alert_reliability": 6,
        "alert_confidence": 0,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1120598793",
        "type": "HAZARD",
        "subtype": "HAZARD_ON_ROAD_CONSTRUCTION",
        "reported_by": "New York State Department of Transportation",
        "description": "Construction on 94TH AVE westbound from 138TH PL (New York) to ATLANTIC AVE at VAN WYCK EXPY (New York) Subject to change without notice., Continuous Sunday September 29th, 2024 12:00 AM thru Thursday October 31st, 2024 11:59 PM 1 Right lane of 2 lanes closed",
        "image": null,
        "publish_datetime_utc": "2024-09-29T04:00:00.000Z",
        "country": "US",
        "city": "Queens, NY",
        "street": "94th Ave",
        "latitude": 40.697364,
        "longitude": -73.812302,
        "num_thumbs_up": 1076,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "790204936",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "New York State Department of Transportation",
        "description": "Construction on E 47TH ST westbound between PARK AVE (New York) and VANDERBILT AVE (New York), Continuous Wednesday July 31st, 2024 12:00 AM thru Thursday November 14th, 2024 11:59 PM All lanes closed",
        "image": null,
        "publish_datetime_utc": "2024-07-31T16:42:30.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "E 47th St",
        "latitude": 40.755261,
        "longitude": -73.975255,
        "num_thumbs_up": 44,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1218599168",
        "type": "POLICE",
        "subtype": null,
        "reported_by": "numberonedude",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-10-20T12:24:12.000Z",
        "country": "US",
        "city": "Manhattan, NY",
        "street": "11th Ave",
        "latitude": 40.74227,
        "longitude": -74.008733,
        "num_thumbs_up": 17,
        "alert_reliability": 10,
        "alert_confidence": 5,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      },
      {
        "alert_id": "1166150619",
        "type": "ROAD_CLOSED",
        "subtype": "ROAD_CLOSED_EVENT",
        "reported_by": "G_W1Z",
        "description": null,
        "image": null,
        "publish_datetime_utc": "2024-09-11T22:20:26.000Z",
        "country": "US",
        "city": "Jersey City, NJ",
        "street": "Garfield Ave",
        "latitude": 40.700187,
        "longitude": -74.079916,
        "num_thumbs_up": 3,
        "alert_reliability": 9,
        "alert_confidence": 1,
        "near_by": null,
        "comments": [],
        "num_comments": 0
      }
    ],
    "jams": [
      {
        "jam_id": "1829013778",
        "type": "NONE",
        "level": 1,
        "severity": 5,
        "line_coordinates": [
          { "lat": 40.68854, "lon": -73.923834 },
          { "lat": 40.688176, "lon": -73.927124 }
        ],
        "start_location": null,
        "end_location": null,
        "speed_kmh": 5.96,
        "length_meters": 281,
        "delay_seconds": 75,
        "block_alert_id": null,
        "block_alert_type": null,
        "block_alert_description": null,
        "block_alert_update_datetime_utc": null,
        "block_start_datetime_utc": null,
        "publish_datetime_utc": "2024-10-20T12:48:40.687Z",
        "update_datetime_utc": "2024-10-20T12:48:40.687Z",
        "country": "US",
        "city": "Brooklyn, NY",
        "street": "Monroe St"
      }
      // ... additional jams omitted for brevity ...
    ]
  }
}
```

Notes:
- `data.alerts` is an array of alert objects.
- `data.jams` is an array of jam objects.

## JSON Schema (simplified)

```
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "request_id": { "type": "string" },
    "parameters": {
      "type": "object",
      "properties": {
        "bottom_left": { "type": "array", "items": { "type": "string" } },
        "top_right": { "type": "array", "items": { "type": "string" } },
        "max_alerts": { "type": "integer" },
        "max_jams": { "type": "integer" }
      }
    },
    "data": {
      "type": "object",
      "properties": {
        "alerts": { "type": "array", "items": { "type": "object" } },
        "jams": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

## RTI Parsing and Mapping

File: `sunnypilot/rtid/waze_api_client.py`

### Alerts

- ID: `alert_id` → `WazeAlert.id`
- Type: `type` (+ optional `subtype`) → normalized via `_map_alert_type`
  - POLICE → `police`
  - POLICE + subtype `POLICE_HIDING` → `policeHiding`
  - HAZARD → `hazard` (with subtype-specific variants where applicable)
  - ROAD_CLOSED → `roadClosed`
  - SPEED_TRAP, SPEED_CAMERA → `speedTrap`, `speedCamera`
- Location: `latitude`, `longitude`
- Confidence: `alert_confidence` (integer, typically 0–5) → stored as float
- Metadata: `street`, `country`, etc. preserved in `raw_data`

### Jams

We support multiple payload variants:

- ID: `jam_id` (preferred) or `uuid` or `id` → `WazeAlert.id`
- Line coordinates for location:
  - New schema: `line_coordinates[]` with `{ lat, lon }` → first element used as jam position
  - Legacy schema: `line[]` with `{ x, y }` or `{ lon, lat }`
  - Fallback: `coordinates[]` with `{ lat, lon }`
- Confidence: `level` or `severity` (1–5) → normalized to `level / 5.0`
- Speed: `speed_kmh` (preferred) or `speedKMH` → placed in `WazeAlert.speed_limit`
- Type stored as `'jam'` internally

### Provided Schema (verbatim)

```
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "request_id": { "type": "string" },
    "parameters": {
      "type": "object",
      "properties": {
        "bottom_left": { "type": "array", "items": { "type": "string" } },
        "top_right": { "type": "array", "items": { "type": "string" } },
        "max_alerts": { "type": "integer" },
        "max_jams": { "type": "integer" }
      }
    },
    "data": {
      "type": "object",
      "properties": {
        "alerts": { "type": "array", "items": { "type": "object" } },
        "jams": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

### Ingestion Flow

1. RTI Daemon (`sunnypilot/rtid/rtid.py`) fetches via `WazeAPIClient.get_traffic_alerts`.
2. Alerts are parsed into `WazeAlert` objects (both alerts and jams become alerts internally).
3. Threat detection (`sunnypilot/rtid/threat_detector.py`) deduplicates, filters, and computes direction/same-roadness.
4. Processed threats publish via `rtiStateSP` for UI + controller.

## Compatibility Considerations

- Jams: We handle both `line_coordinates` and the older `line`/`coordinates` shapes, and both `speed_kmh` and `speedKMH` field names.
- Alert type mapping is resilient to case and unknown subtypes; unknowns are treated as `hazard`.

## Validation

- `debug_rti_api.py` prints and checks API connectivity and basic parsing.
- `docs/chauffeur/rti/testing/test_waze_api_client.py` covers parsing and fallback behavior.
