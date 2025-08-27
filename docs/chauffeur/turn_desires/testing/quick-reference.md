# Turn Desires Testing Quick Reference

## Activation
- **Speed**: < 35 mph
- **Signal**: Left or right blinker (not both)
- **Lateral**: Must be engaged

## What to Expect
✅ **Logs will show**:
- "Turn desire activated: LEFT/RIGHT at XX.X mph"
- "Sending turn desire to model: turnLeft/turnRight"

❓ **Model behavior unknown**:
- May turn at intersections
- May ignore turn desires
- May produce unexpected paths

## Testing Checklist
- [ ] Safe test environment selected
- [ ] Low traffic conditions
- [ ] Driver ready to override
- [ ] Speed below 35 mph
- [ ] Blinker activated
- [ ] Lateral control engaged
- [ ] Logs being monitored

## Emergency Override
Take control immediately if:
- Car attempts to turn into obstacle
- Unexpected aggressive steering
- Any unsafe behavior

## Log Monitoring
```bash
# Watch live logs (SSH to device)
tail -f /data/logs/selfdrived.log | grep "Turn desire"
tail -f /data/logs/modeld.log | grep "turn"
```

## Test Progression
1. **Straight road** - Verify activation in logs only
2. **Empty parking lot** - Test steering response  
3. **Simple intersection** - Test actual turning
4. **Normal roads** - Only after confirming safe behavior

## Speed Reference
- 35 mph = threshold (no turn desires above this)
- 30 mph = good testing speed
- 20 mph = typical turn speed
- 15 mph = slow turn speed