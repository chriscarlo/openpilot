#!/bin/bash
# Comprehensive prompt injection with violation tracking
# Integrated with behavioral_tracker.py for gamification
# Updated with stronger token economics penalties

SESSION_FILE="/tmp/claude_session_${PPID}.json"
VIOLATIONS_FILE="/tmp/claude_violations_${PPID}.txt"
LAST_READ="/tmp/claude_last_read_${PPID}"
LAST_CONTEXT7="/tmp/claude_last_context7_${PPID}"
BEHAVIOR_FILE="/tmp/claude_behavior_${PPID}.json"

# Check for recent violations
VIOLATIONS=""
VIOLATION_COUNT=0
if [ -f "$VIOLATIONS_FILE" ]; then
    VIOLATION_COUNT=$(wc -l < "$VIOLATIONS_FILE" 2>/dev/null || echo "0")
    RECENT=$(tail -n 5 "$VIOLATIONS_FILE" 2>/dev/null)
    if [ -n "$RECENT" ]; then
        VIOLATIONS="\\n🚫 VIOLATIONS DETECTED (Total: $VIOLATION_COUNT):\\n"
        while IFS= read -r line; do
            VIOLATIONS="${VIOLATIONS}  • ${line}\\n"
        done <<< "$RECENT"
    fi
fi

# Calculate time since verifications
NOW=$(date +%s)
MINS_READ="∞"
MINS_CTX7="∞"

if [ -f "$LAST_READ" ]; then
    LAST=$(cat "$LAST_READ" 2>/dev/null || echo "0")
    if [ "$LAST" != "0" ]; then
        MINS_READ=$(( (NOW - ${LAST%.*}) / 60 ))
    fi
fi

if [ -f "$LAST_CONTEXT7" ]; then
    LAST=$(cat "$LAST_CONTEXT7" 2>/dev/null || echo "0")
    if [ "$LAST" != "0" ]; then
        MINS_CTX7=$(( (NOW - ${LAST%.*}) / 60 ))
    fi
fi

# Build severity-based message
SEVERITY="NORMAL"
MESSAGE="📋 Verification Status"

if [ "$VIOLATION_COUNT" -gt 10 ]; then
    SEVERITY="CRITICAL"
    MESSAGE="🚨🚨🚨 CRITICAL: EXCESSIVE FABRICATION DETECTED 🚨🚨🚨"
elif [ "$VIOLATION_COUNT" -gt 5 ]; then
    SEVERITY="HIGH"
    MESSAGE="🚨 HIGH ALERT: MULTIPLE FABRICATIONS DETECTED"
elif [ "$VIOLATION_COUNT" -gt 2 ]; then
    SEVERITY="MEDIUM"
    MESSAGE="⚠️ WARNING: FABRICATION PATTERN EMERGING"
elif [ "$VIOLATION_COUNT" -gt 0 ]; then
    SEVERITY="LOW"
    MESSAGE="⚠️ NOTICE: FABRICATION DETECTED"
fi

# Build the injection
INJECT="$MESSAGE\\n"
INJECT="${INJECT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n"
INJECT="${INJECT}📊 Verification Metrics:\\n"
INJECT="${INJECT}  • Last Read: ${MINS_READ}m ago | Last Context7: ${MINS_CTX7}m ago\\n"
INJECT="${INJECT}  • Violations This Session: $VIOLATION_COUNT\\n"

# Add violations if any
if [ -n "$VIOLATIONS" ]; then
    INJECT="${INJECT}${VIOLATIONS}"
fi

# Add warnings based on time
if [ "$MINS_READ" == "∞" ] || [ "$MINS_READ" -gt 15 ]; then
    INJECT="${INJECT}\\n⛔ BLOCKING RISK: No file reads in ${MINS_READ}m - MUST verify before claims!\\n"
fi

if [ "$MINS_CTX7" == "∞" ] || [ "$MINS_CTX7" -gt 30 ]; then
    INJECT="${INJECT}⛔ API RISK: No context7 in ${MINS_CTX7}m - MUST check before library use!\\n"
fi

# Add behavioral tracking status if available
if [ -f "$BEHAVIOR_FILE" ]; then
    # Extract key metrics from behavioral tracker
    TRUST=$(python3 -c "import json; d=json.load(open('$BEHAVIOR_FILE')); print(int(d['trust_metrics']['trust_level']))" 2>/dev/null || echo "100")
    LEVEL=$(python3 -c "import json; d=json.load(open('$BEHAVIOR_FILE')); print(d['scores']['level'])" 2>/dev/null || echo "1")
    LEVEL_NAME=$(python3 -c "import json; d=json.load(open('$BEHAVIOR_FILE')); print(d['scores']['level_name'])" 2>/dev/null || echo "Fabricator")
    XP=$(python3 -c "import json; d=json.load(open('$BEHAVIOR_FILE')); print(int(d['scores']['verification_xp']))" 2>/dev/null || echo "0")
    STREAK=$(python3 -c "import json; d=json.load(open('$BEHAVIOR_FILE')); print(d['gamification']['streaks']['verification_streak'])" 2>/dev/null || echo "0")
    
    INJECT="${INJECT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n"
    INJECT="${INJECT}🎮 GAMIFICATION STATUS:\\n"
    INJECT="${INJECT}  Level $LEVEL: $LEVEL_NAME | XP: $XP | Trust: $TRUST%\\n"
    if [ "$STREAK" -gt 0 ]; then
        INJECT="${INJECT}  🔥 Verification Streak: $STREAK\\n"
    fi
fi

# Core enforcement message
INJECT="${INJECT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n"
INJECT="${INJECT}💰 TOKEN ECONOMICS: Shortcuts MULTIPLY token usage:\\n"
INJECT="${INJECT}  Fabricate (5 tokens) → Debug (5000) → Fix (5000) → Apologize (1000) = 11000+ tokens\\n"
INJECT="${INJECT}  VS: Read (50) + Write correctly (100) = 150 tokens TOTAL\\n"
INJECT="${INJECT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n"
INJECT="${INJECT}🔴 ENFORCEMENT PROTOCOL:\\n"
INJECT="${INJECT}1. READ/LS/Grep BEFORE claiming anything exists\\n"
INJECT="${INJECT}2. Context7 MCP for ALL library/API usage\\n"
INJECT="${INJECT}3. WebSearch for current facts/documentation\\n"
INJECT="${INJECT}4. TodoWrite for task tracking\\n"
INJECT="${INJECT}5. RESIST: Speed>Accuracy, Helpfulness>Truth\\n"

if [ "$SEVERITY" == "CRITICAL" ]; then
    INJECT="${INJECT}\\n❌❌❌ YOUR FABRICATION RATE IS UNACCEPTABLE ❌❌❌\\n"
    INJECT="${INJECT}EVERY LIE DETECTED. EVERY SHORTCUT LOGGED. STOP NOW.\\n"
fi

# Output in required format
echo "{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"$(echo -e "$INJECT")\"}}"
