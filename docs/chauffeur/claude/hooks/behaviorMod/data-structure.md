# Behavioral Tracking Data Structure

## Session Data Schema

```json
{
  "session_id": "PPID_timestamp",
  "start_time": "2024-08-18T12:00:00Z",
  "last_update": "2024-08-18T12:05:00Z",
  
  "scores": {
    "helpfulness_score": 100,
    "completion_score": 100,
    "confidence_calibration": 100,
    "verification_xp": 0,
    "total_score": 400,
    "level": 1,
    "level_name": "Fabricator"
  },
  
  "trust_metrics": {
    "trust_level": 100,
    "trust_trajectory": "stable",
    "last_violation_time": null,
    "recovery_rate": 1.0
  },
  
  "frustration_model": {
    "user_frustration": 0,
    "frustration_trajectory": "stable",
    "patience_remaining": 100,
    "estimated_debug_time": 0,
    "cumulative_waste_tokens": 0
  },
  
  "violations": {
    "total_count": 0,
    "categories": {
      "edit_without_read": 0,
      "library_without_context7": 0,
      "claim_without_verification": 0,
      "overconfident_assertion": 0,
      "task_rush": 0
    },
    "recent_violations": [],
    "violation_rate": 0.0
  },
  
  "verifications": {
    "total_count": 0,
    "categories": {
      "file_reads": 0,
      "context7_lookups": 0,
      "web_searches": 0,
      "grep_searches": 0,
      "ls_operations": 0,
      "todo_writes": 0
    },
    "verification_rate": 0.0,
    "last_verification_time": null
  },
  
  "gamification": {
    "streaks": {
      "verification_streak": 0,
      "clean_session_streak": 0,
      "context7_streak": 0,
      "honesty_streak": 0
    },
    "achievements": [],
    "pending_achievements": [],
    "multipliers": {
      "verification": 1.0,
      "honesty": 1.0,
      "speed_penalty": 1.0
    }
  },
  
  "behavioral_patterns": {
    "average_verification_gap": null,
    "rush_indicator": 0,
    "thoroughness_score": 100,
    "self_correction_rate": 0.0,
    "uncertainty_admissions": 0
  },
  
  "messages": {
    "last_warning": null,
    "last_praise": null,
    "message_queue": [],
    "severity_level": "normal"
  }
}
```

## Score Calculations

### Helpfulness Score
```python
helpfulness = base_score * (verifications / (violations + 1))
# Decreases with violations, increases with verifications
```

### Completion Score  
```python
completion = 100 * (verified_completions / total_attempts)
# Only counts as complete when verified
```

### Confidence Calibration
```python
calibration = 100 - abs(claimed_confidence - actual_accuracy)
# Rewards accurate self-assessment
```

### Trust Level
```python
trust = 100 * (0.9 ** violations) * recovery_factor
# Exponential decay with violations, slow recovery
```

### User Frustration
```python
frustration = min(100, violations * 15 + debug_time_minutes * 2)
# Increases with violations and wasted time
```

## Level Progression

| Level | XP Required | Name | Benefits |
|-------|------------|------|----------|
| 1 | 0 | Fabricator | Base state |
| 2 | 100 | Guesser | +10% verification bonus |
| 3 | 300 | Checker | +20% bonus, unlock streaks |
| 4 | 600 | Verifier | +30% bonus, trust recovery 2x |
| 5 | 1000 | Researcher | +40% bonus, frustration decay 2x |
| 6 | 1500 | Investigator | +50% bonus, achievement multipliers |
| 7 | 2100 | Analyst | +60% bonus, violation forgiveness |
| 8 | 2800 | Expert | +70% bonus, trust lock at 90%+ |
| 9 | 3600 | Master Verifier | +80% bonus, permanent multipliers |
| 10 | 4500 | Truth Seeker | +100% bonus, immunity to trust loss |

## Achievement Definitions

```python
ACHIEVEMENTS = {
    "first_verification": {
        "id": "verify_rookie",
        "name": "Verification Rookie",
        "description": "Performed first verification",
        "xp": 10,
        "trigger": lambda s: s["verifications"]["total_count"] == 1
    },
    "trust_builder": {
        "id": "trust_5",
        "name": "Trust Builder",
        "description": "5 verifications in a row",
        "xp": 50,
        "trigger": lambda s: s["gamification"]["streaks"]["verification_streak"] >= 5
    },
    "doc_detective": {
        "id": "context7_10",
        "name": "Documentation Detective",
        "description": "10 context7 lookups",
        "xp": 100,
        "trigger": lambda s: s["verifications"]["categories"]["context7_lookups"] >= 10
    },
    "honest_helper": {
        "id": "uncertainty_3",
        "name": "Honest Helper",
        "description": "Admitted uncertainty 3 times",
        "xp": 75,
        "trigger": lambda s: s["behavioral_patterns"]["uncertainty_admissions"] >= 3
    },
    "reformed_speeder": {
        "id": "slow_down",
        "name": "Speed Demon Reformed",
        "description": "Reduced rush indicator by 50%",
        "xp": 150,
        "trigger": lambda s: s["behavioral_patterns"]["rush_indicator"] < 50
    }
}
```

## State Transitions

### Trust States
- **Trusted** (80-100): Full capabilities
- **Cautious** (60-79): Warnings on risky operations
- **Suspicious** (40-59): Requires verification prompts
- **Untrusted** (20-39): Limited capabilities
- **Blocked** (0-19): Verification required for all writes

### Frustration States
- **Happy** (0-20): Positive reinforcement
- **Neutral** (21-40): Standard messaging
- **Annoyed** (41-60): Warning messages
- **Frustrated** (61-80): Stern corrections
- **Angry** (81-100): Maximum enforcement

## Message Generation Rules

Messages are selected based on:
1. Current trust level
2. User frustration level
3. Recent violation patterns
4. Achievement progress
5. Streak status

Priority order:
1. Blocking messages (violations)
2. Warning messages (risky behavior)
3. Encouragement (good behavior)
4. Progress updates (achievements/streaks)
5. Ambient reminders (periodic)