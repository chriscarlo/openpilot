# RTI Speed Limit Decision Logic - Deep Analysis

## Executive Summary

The RTI (Real-Time Intelligence) system publishes posted speed limits to the longitudinal planner **ONLY** when specific conditions are met:
1. A threat is detected **ahead** on the **same road**
2. The threat is within **0.75 miles** (1207m default)
3. The system recommends a speed **less than current speed**

When these conditions aren't met, `recommendedSpeed` is set to **0.0** (no recommendation).

## Detailed Decision Flow

### Phase 1: Threat Detection and Filtering
**Location**: `sunnypilot/rtid/threat_detector.py`

#### 1.1 Initial Threat Processing (`_process_single_threat`)
- Calculates distance from ego to threat using haversine formula
- Filters out threats beyond `detection_radius_m` (default 2 miles for HUD display)
- Determines if threat is on same road using `EnhancedRoadMatcher`
- Classifies direction: ahead, behind, left, right
- Assigns speed limit:
  - If available from threat data: uses that value
  - For police/speed traps without data: defaults to 55 mph (25 m/s)
  - For other threats: 0.0 (no speed recommendation)

#### 1.2 Road Matching Logic
- Uses distance-based heuristics:
  - Normal roads: 50m proximity threshold
  - Highways (>90 km/h): 200m proximity threshold
- Only threats passing this test are marked `on_same_road = true`

### Phase 2: Speed Recommendation Calculation
**Location**: `SpeedRecommendationEngine.calculate_recommendation()` (lines 248-309)

#### 2.1 Threat Relevance Filtering
```python
for threat in threats:
    if not threat.on_same_road:
        continue  # SKIP - not on our road
    
    # Distance thresholds by direction
    if threat.direction == 'ahead':
        max_distance = ahead_distance_threshold_m  # 1207m (0.75 miles)
    elif threat.direction == 'behind':
        max_distance = behind_distance_threshold_m  # 805m (0.5 miles)
    else:  # left or right
        max_distance = 0  # SKIP - ignore lateral threats
```

**Key Decision**: Only ahead/behind threats on same road within thresholds are considered.

#### 2.2 Speed Limit Decision
```python
# Only process if there are ahead threats
ahead_threats = [t for t in relevant_threats if t.direction == 'ahead']
if not ahead_threats:
    return 0.0, False  # NO RECOMMENDATION
```

**Critical**: If no ahead threats exist, RTI publishes `recommendedSpeed = 0.0`

#### 2.3 Speed Calculation
For the closest ahead threat:

**"posted" mode** (default):
```python
if closest_threat.speed_limit_ms > 0:
    target_speed = closest_threat.speed_limit_ms
else:
    target_speed = default_speed_limit_ms  # 25 m/s (55 mph)
```

**"custom" mode**:
```python
target_speed = current_speed_ms - speed_reduction_ms  # Default -10 mph
target_speed = max(target_speed, 10.0)  # Floor at 22 mph
```

#### 2.4 Safety Constraints
```python
# CRITICAL: Never recommend acceleration toward a threat
target_speed = min(target_speed, current_speed_ms)

# Extra caution for close threats
if closest_threat.distance < 300:  # meters
    target_speed = min(target_speed, current_speed_ms * 0.9)

# Ensure non-negative
target_speed = max(0.0, target_speed)
```

**Fundamental Safety Rule**: `recommendedSpeed` NEVER exceeds current speed.

### Phase 3: Message Publication
**Location**: `sunnypilot/rtid/rtid.py`

#### 3.1 Publishing Frequency
- RTI publishes `rtiStateSP` messages at **1Hz** (once per second)
- API fetches occur only every **30 seconds** (rate limiting)
- Uses cached data between API calls (max 5 minutes staleness)

#### 3.2 Message Contents
```python
msg.rtiStateSP.threatAhead = threat_ahead  # true/false
msg.rtiStateSP.threatDistanceM = threat_distance  # meters
msg.rtiStateSP.recommendedSpeed = recommended_speed  # m/s or 0.0
msg.rtiStateSP.threats = [...]  # Up to 5 threat details for HUD
```

### Phase 4: Longitudinal Planner Integration
**Location**: `sunnypilot/selfdrive/controls/lib/rti_controller.py`

#### 4.1 RTI Controller Activation
```python
# Must satisfy ALL conditions:
if not RTIEnabled:
    return  # Feature disabled
if v_ego < MIN_OPERATING_SPEED:  # 5 mph
    return  # Too slow
if not rti_state.threatAhead:
    return  # No threat
if threat_distance > 1000m:
    return  # Too far
```

#### 4.2 Speed Override Refinement
The RTI controller further reduces the recommended speed:

```python
# Distance-based reduction factors
if threat_distance < 100m:
    reduction = 0.75  # 75% of current speed
elif threat_distance < 300m:
    reduction = 0.85  # 85% of current speed
else:
    reduction = 0.95  # 95% of current speed

safe_speed = min(target_speed, v_ego * reduction)
```

#### 4.3 Final Integration
- RTI controller provides `speed_recommendation` property
- Only active when `safe_speed < v_cruise` (cruise setpoint)
- Longitudinal planner uses this to override cruise speed

## Key Insights

### When RTI Publishes Speed Limits
✅ **Published** (recommendedSpeed > 0):
- Threat detected ahead on same road
- Within 0.75 miles (configurable)
- Recommended speed < current speed
- Typical value: posted speed limit at threat location

❌ **Not Published** (recommendedSpeed = 0.0):
- No threats detected
- Threats only to side or behind
- Threats too far ahead (>0.75 miles)
- Would require acceleration (safety violation)

### Conservative Design Philosophy
1. **Never accelerates toward threats** - fundamental safety constraint
2. **Progressive speed reduction** - more aggressive as threat gets closer
3. **Multiple safety layers** - backend, controller, and planner all validate
4. **Rate-limited API calls** - preserves data freshness while minimizing API usage

### Configuration Parameters
- `RTIForwardSlowdownRange`: Distance to start slowing (default 1207m)
- `RTIResumeSpeedDistance`: Distance behind to resume (default 805m)
- `RTISpeedReductionMode`: "posted" or "custom"
- `RTISpeedReduction`: Speed reduction in custom mode (default 16 km/h)
- `RTIDetectionRadius`: Maximum threat detection range (default 3218m)

## Conclusion

The RTI system is designed with a clear, conservative approach to speed management. It only intervenes when there's a credible threat ahead on the same road, and it always recommends deceleration or maintaining current speed - never acceleration. The multi-layered safety checks ensure that speed recommendations are both appropriate and safe for the driving context.