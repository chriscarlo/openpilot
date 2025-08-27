# Turn Desires Implementation

## Summary
Hard-coded turn desires to activate when vehicle speed is below 35 mph and a turn signal is active. This sends `turnLeft` or `turnRight` desires to the neural network model instead of normal lane change commands.

## Implementation Details

### Files Modified
1. **selfdrive/controls/lib/desire_helper.py**
   - Added `TURN_DESIRE_SPEED_MAX` constant (35 mph)
   - Added turn desire tracking variables
   - Implemented override logic in `update()` method

2. **selfdrive/modeld/modeld.py**
   - Added debug logging when turn desires are sent to model

### Key Logic (desire_helper.py:117-137)
```python
# Turn desire logic: Override desire if below turn speed threshold
below_turn_speed = v_ego < TURN_DESIRE_SPEED_MAX  # 35 mph = 15.65 m/s
if lateral_active and below_turn_speed and one_blinker:
  # Set turn desire based on blinker direction
  if carstate.leftBlinker:
    self.desire = log.Desire.turnLeft
  else:  # rightBlinker must be true
    self.desire = log.Desire.turnRight
```

### Activation Conditions
All conditions must be met:
1. **Speed**: Vehicle speed < 35 mph (15.65 m/s)
2. **Blinker**: Exactly one turn signal active (not hazards)
3. **Lateral Control**: Lateral control must be active

### Expected Behavior
- Below 35 mph: Blinker activation sends turn desires to model
- Above 35 mph: Normal lane change behavior
- Model receives desire index 1 (turnLeft) or 2 (turnRight)
- Model *should* generate turning paths at intersections

### Logging
When turn desires activate, logs will show:
```
Turn desire activated: LEFT at 25.3 mph
Sending turn desire to model: turnLeft (index 1)
```

## Testing Scenarios

### Safe Testing Environment
**CRITICAL**: Test only in safe, controlled environments with driver ready to take over.

### Test Cases
1. **Basic Left Turn**
   - Speed: 20-30 mph
   - Activate left blinker approaching intersection
   - Monitor if model attempts to turn left

2. **Basic Right Turn**
   - Speed: 20-30 mph  
   - Activate right blinker approaching intersection
   - Monitor if model attempts to turn right

3. **Speed Threshold Test**
   - Approach intersection at 40 mph with blinker
   - Slow to 34 mph (should activate turn desire)
   - Verify transition in logs

4. **Deactivation Test**
   - Activate turn desire at low speed
   - Accelerate above 35 mph
   - Verify turn desire deactivates

### Monitor for Issues
- Model trying to turn into curbs
- Model not recognizing intersections
- Unexpected path planning behavior
- Any safety-critical situations

## Model Response Unknown
**IMPORTANT**: We cannot guarantee how the model will respond to turn desires:
- Model may not be trained on turn scenarios
- Model may ignore turn desires entirely
- Model may produce unexpected outputs

This is an experimental feature to test if the neural network has latent turn capabilities.

## Verification Commands

Check logs for turn desire activation:
```bash
# Monitor desire helper logs
grep "Turn desire" /data/logs/selfdrived.log

# Monitor model logs  
grep "Sending turn desire to model" /data/logs/modeld.log
```

## Safety Notes
- Driver must remain attentive and ready to override
- Test only in low-traffic areas initially
- Disable if unexpected behavior occurs
- This is experimental functionality