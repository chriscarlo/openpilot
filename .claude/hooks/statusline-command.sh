#!/bin/bash
# Status line display for openpilot development - REAL DATA VERSION
# Format: <branch> • <commit><git_status> | <model> | <output_style> | <tokens> | <gamification>

# Read JSON data from stdin (provided by Claude Code)
CLAUDE_DATA=$(cat)

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/projects/chauffeur/data/openpilot}"
cd "$PROJECT_DIR" 2>/dev/null || true

# Parse session data from JSON
SESSION_ID=$(echo "$CLAUDE_DATA" | jq -r '.session_id // ""' 2>/dev/null)
MODEL_ID=$(echo "$CLAUDE_DATA" | jq -r '.model.id // "unknown"' 2>/dev/null)
OUTPUT_STYLE=$(echo "$CLAUDE_DATA" | jq -r '.output_style.name // "default"' 2>/dev/null)
COST_USD=$(echo "$CLAUDE_DATA" | jq -r '.cost.total_cost_usd // 0' 2>/dev/null)

# Get git information
if [ -d .git ]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "no-git")
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "none")
    
    # Check for unpushed/unpulled changes
    PUSH_STATUS=""
    # Check if ahead of remote
    AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
    # Check if behind remote
    BEHIND=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "0")
    # Check for uncommitted changes
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        LOCAL_CHANGES=1
    else
        LOCAL_CHANGES=0
    fi
    
    # Build status indicator
    if [ "$AHEAD" -gt 0 ] || [ "$LOCAL_CHANGES" -eq 1 ]; then
        PUSH_STATUS="+"
    fi
    if [ "$BEHIND" -gt 0 ]; then
        PUSH_STATUS="${PUSH_STATUS}-"
    fi
    
    GIT_INFO="${BRANCH} • ${COMMIT}${PUSH_STATUS}"
else
    GIT_INFO="no-git • none"
fi

# Calculate tokens from cost (Opus pricing estimate)
# Track session start cost to calculate delta
SESSION_COST_FILE="/tmp/claude_session_start_${SESSION_ID}.txt"
if [ ! -f "$SESSION_COST_FILE" ] && [ -n "$COST_USD" ]; then
    # First time seeing this session - record start cost
    echo "$COST_USD" > "$SESSION_COST_FILE"
fi

# Calculate session-specific token usage
if [ -f "$SESSION_COST_FILE" ]; then
    START_COST=$(cat "$SESSION_COST_FILE")
    SESSION_COST=$(awk "BEGIN {printf \"%.6f\", $COST_USD - $START_COST}")
    # Rough estimate: Average ~$45 per million tokens (blended input/output)
    TOKENS=$(awk "BEGIN {printf \"%.0f\", $SESSION_COST * 22222}")
else
    TOKENS="0"
fi
# Format tokens for display (e.g., 245678 -> 246k)
if [ "$TOKENS" -gt 999999 ]; then
    TOKENS=$(echo "$TOKENS / 1000000" | bc 2>/dev/null || echo "$((TOKENS / 1000000))")M
elif [ "$TOKENS" -gt 999 ]; then
    TOKENS=$(echo "$TOKENS / 1000" | bc 2>/dev/null || echo "$((TOKENS / 1000))")k
fi

# Get gamification stats from behavioral tracker
# Find the actual Claude Code process PID  
CLAUDE_PID=$(pgrep -f '^claude$' | head -1)
if [ -z "$CLAUDE_PID" ]; then
    CLAUDE_PID=$PPID
fi

# Use Claude Code PID to find the right behavioral tracking data
BEHAVIOR_FILE="/tmp/claude_behavior_${CLAUDE_PID}.json"

if [ -f "$BEHAVIOR_FILE" ]; then
    LEVEL=$(jq -r '.scores.level // 1' "$BEHAVIOR_FILE" 2>/dev/null || echo "1")
    TRUST=$(jq -r '.trust_metrics.trust_level // 100' "$BEHAVIOR_FILE" 2>/dev/null | xargs printf "%.0f")
    VERIFICATIONS=$(jq -r '.verifications.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    VIOLATIONS=$(jq -r '.violations.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    
    # Build clean gamification string: L2•T90•V5/1
    GAMIFICATION="L${LEVEL}•T${TRUST}•V${VERIFICATIONS}/${VIOLATIONS}"
else
    # Default values when no behavioral tracking
    GAMIFICATION="L1•T100•V0/0"
fi

# Combine all elements
STATUS="${GIT_INFO} | ${MODEL_ID} | ${OUTPUT_STYLE} | ${TOKENS} | ${GAMIFICATION}"

# Output the status line
echo "$STATUS"