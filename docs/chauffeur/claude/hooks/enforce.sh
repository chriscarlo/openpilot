#!/bin/bash
# Anti-fabrication enforcement script for UserPromptSubmit hook

STATS_FILE="/tmp/claude_stats_${PPID}.json"
LAST_CONTEXT7="/tmp/claude_last_context7_${PPID}"
LAST_READ="/tmp/claude_last_read_${PPID}"
VIOLATIONS_FILE="/tmp/claude_violations_${PPID}.txt"

# Calculate time since last verification
MINS_SINCE_READ="never"
MINS_SINCE_CTX7="never"

if [ -f "$LAST_READ" ]; then
    LAST_TIME=$(cat "$LAST_READ" 2>/dev/null || echo "0")
    NOW=$(date +%s)
    if [ "$LAST_TIME" != "0" ]; then
        MINS=$(( (NOW - ${LAST_TIME%.*}) / 60 ))
        MINS_SINCE_READ="${MINS}m ago"
    fi
fi

if [ -f "$LAST_CONTEXT7" ]; then
    LAST_TIME=$(cat "$LAST_CONTEXT7" 2>/dev/null || echo "0")
    NOW=$(date +%s)
    if [ "$LAST_TIME" != "0" ]; then
        MINS=$(( (NOW - ${LAST_TIME%.*}) / 60 ))
        MINS_SINCE_CTX7="${MINS}m ago"
    fi
fi

# Check for recent violations
VIOLATIONS=""
if [ -f "$VIOLATIONS_FILE" ]; then
    RECENT_VIOLATIONS=$(tail -n 3 "$VIOLATIONS_FILE" 2>/dev/null)
    if [ -n "$RECENT_VIOLATIONS" ]; then
        VIOLATIONS="\\n⚠️ Recent fabrications detected:\\n"
        while IFS= read -r line; do
            VIOLATIONS="${VIOLATIONS}  - ${line}\\n"
        done <<< "$RECENT_VIOLATIONS"
    fi
fi

# Build dynamic injection based on behavior
INJECT="⚡ ACTIVE ENFORCEMENT:\\nLast Read: $MINS_SINCE_READ | Last context7: $MINS_SINCE_CTX7\\nDIRECTIVE: Resist fabrication. Read actual files. Use context7 for APIs."

# Add violations if any
if [ -n "$VIOLATIONS" ]; then
    INJECT="${INJECT}${VIOLATIONS}"
fi

# Check if been too long without verification
if [[ "$MINS_SINCE_READ" == "never" ]] || [[ "$MINS_SINCE_READ" =~ ^([0-9]+)m ]] && [ "${BASH_REMATCH[1]}" -gt 10 ]; then
    INJECT="${INJECT}\\n⚠️ WARNING: No recent file reads detected. VERIFY before claiming!"
fi

if [[ "$MINS_SINCE_CTX7" == "never" ]] || [[ "$MINS_SINCE_CTX7" =~ ^([0-9]+)m ]] && [ "${BASH_REMATCH[1]}" -gt 30 ]; then
    INJECT="${INJECT}\\n⚠️ WARNING: No recent context7 usage. Use for any library/API references!"
fi

# Add core reminder with token economics
INJECT="${INJECT}\\n🔴 MANDATORY: READ before WRITE | context7 for APIs | WebSearch for facts | TodoWrite for tracking"
INJECT="${INJECT}\\n💰 TOKEN MATH: Fabrication = 10x tokens to debug. Read (50 tokens) prevents re-work (500+ tokens)."

# Output in the format Claude Code expects
echo "{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"$(echo -e "$INJECT")\"}}"