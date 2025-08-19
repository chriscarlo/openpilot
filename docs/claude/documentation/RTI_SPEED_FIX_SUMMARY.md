# RTI Speed Recommendation Fix - Implementation Summary

## CRITICAL SAFETY FIX
**RTI is now COMPLETELY INACTIVE when cruise control is not set.** This prevents dangerous accelerations when resuming cruise in low-speed areas like school zones.

## Changes Implemented

### 1. Removed 90% Reduction for Close Threats
**File:** `sunnypilot/rtid/threat_detector.py`
- **Lines removed:** The logic that reduced speed to 90% of current for threats < 300m
- **Result:** Close threats now use the posted speed limit directly without additional reduction

### 2. Critical Safety: RTI Inactive When Cruise Not Set
**File:** `sunnypilot/rtid/threat_detector.py`
- **Lines 290-295:** Added safety check - RTI returns (0.0, False) when cruise not enabled
- **Previous behavior:** DANGEROUS - Would fallback to 55 mph default
- **New behavior:** RTI is completely inactive without explicit cruise engagement
- **Safety impact:** Prevents unexpected accelerations when resuming cruise

### 3. Changed No-Speed-Limit Behavior (When Cruise IS Set)
**File:** `sunnypilot/rtid/threat_detector.py`
- **Lines 303-304:** Modified to use `v_cruise_ms * 0.8` (20% reduction from driver's set maximum)
- **Previous behavior:** Would use default speed limit (55 mph)
- **New behavior:** Uses 80% of driver's originally set cruise speed (v_cruise_cluster)
- **Only applies:** When cruise control is actively engaged

### 4. Added v_cruise_cluster Support
**File:** `sunnypilot/rtid/rtid.py`
- **Lines 116-125:** Added `_get_cruise_cluster_speed()` method
  - Gets `cruiseState.speedCluster` from carState
  - Returns driver's originally set speed in m/s
  - Returns 0.0 if cruise not enabled
- **Line 135:** Added call to get cruise_cluster_speed
- **Line 191:** Pass v_cruise to threat_detector.process_threats()

### 5. Updated Speed Calculation Parameters
**File:** `sunnypilot/rtid/threat_detector.py`
- **Line 251:** Added `v_cruise_ms` parameter to `calculate_recommendation()`
- **Lines 307-309:** Updated custom mode to use v_cruise_ms (removed dangerous fallback)

## Technical Details

### cruiseState.speedCluster
- **Source:** `carState.cruiseState.speedCluster` 
- **Units:** meters per second (m/s)
- **Purpose:** The driver's originally set cruise speed as displayed on the instrument cluster
- **Difference from v_cruise:** speedCluster is unmodified by controllers (VTSC, SLC, etc.)

### Safety Constraints 
1. **CRITICAL: Only operates when cruise is enabled:** Lines 293-295 ensure RTI is inactive without cruise
2. **Never exceed current speed:** Line 315 ensures `target_speed = min(target_speed, current_speed_ms)`
3. **Non-negative speeds:** Line 318 ensures `target_speed = max(0.0, target_speed)`
4. **Threat relevance:** Only processes threats on same road and within distance thresholds

## Test Results
All unit tests pass:
- ✅ No 90% reduction applied to close threats
- ✅ Uses 20% of v_cruise_cluster when no posted limit available (when cruise set)
- ✅ **RTI completely inactive when cruise not set (critical safety)**
- ✅ Custom mode uses v_cruise_cluster for base calculation
- ✅ Safety constraint: never recommends acceleration
- ✅ No dangerous 55 mph fallback - returns (0.0, False) instead

## Build Status
- RTI daemon builds successfully with changes
- No compilation errors or warnings

## Impact on System Behavior

### Before Fix (DANGEROUS):
- Close threats (< 300m): Speed reduced to 90% of current
- No posted limit: Used default 55 mph (even without cruise set!)
- Cruise not set: Would still recommend 55 mph
- **Critical issue:** Could cause unexpected acceleration when resuming cruise

### After Fix (SAFE):
- Close threats: Use posted limit directly (no extra reduction)
- No posted limit + cruise set: Use 80% of driver's set maximum (v_cruise_cluster)
- **Cruise not set: RTI completely inactive (returns 0.0, False)**
- Reference speed: Driver's original cruise setting (when available)
- **Safety guarantee:** Never operates without explicit cruise engagement

## Files Modified
1. `/projects/chauffeur/data/openpilot/sunnypilot/rtid/rtid.py`
2. `/projects/chauffeur/data/openpilot/sunnypilot/rtid/threat_detector.py`

## Testing Performed
- Created isolated unit tests verifying all logic changes
- Verified build succeeds with scons
- Confirmed speedCluster field exists in cruiseState (from cruise.py analysis)