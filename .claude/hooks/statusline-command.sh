#!/bin/bash
# Status line display for openpilot development
# Format: <branch> : <commit><git_status> | <model> | <output_style> | <tokens> | <gamification>

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/projects/chauffeur/data/openpilot}"
cd "$PROJECT_DIR" 2>/dev/null || true

# Get git information
if [ -d .git ]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "no-git")
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "none")
    
    # Check for unpushed/unpulled changes
    PUSH_STATUS=""
    if git status 2>/dev/null | grep -q "ahead"; then
        PUSH_STATUS="+"
    fi
    if git status 2>/dev/null | grep -q "behind"; then
        PUSH_STATUS="${PUSH_STATUS}-"
    fi
    # If no push/pull status but there are local changes
    if [ -z "$PUSH_STATUS" ]; then
        if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
            PUSH_STATUS="+"
        fi
    fi
    
    GIT_INFO="${BRANCH} • ${COMMIT}${PUSH_STATUS}"
else
    GIT_INFO="no-git • none"
fi

# Get model info from input JSON or environment
if [ -n "$CLAUDE_INPUT" ]; then
    MODEL_INFO=$(echo "$CLAUDE_INPUT" 2>/dev/null | jq -r '.model // ""' 2>/dev/null)
else
    MODEL_INFO="${CLAUDE_MODEL:-}"
fi
# Default if empty
if [ -z "$MODEL_INFO" ]; then
    MODEL_INFO="claude-opus-4-1-20250805"  # Default based on current model
fi
# Map to full model names with release dates
case "$MODEL_INFO" in
    *"opus-4"*|*"opus"*) MODEL="claude-opus-4-1-20250805" ;;
    *"sonnet-3.5"*|*"sonnet"*) MODEL="claude-3.5-sonnet-20241022" ;;
    *"haiku"*) MODEL="claude-3-haiku-20240307" ;;
    *"claude-3"*) MODEL="claude-3-20240229" ;;
    *) MODEL="${MODEL_INFO}" ;;  # Use as-is if already full name
esac

# Get output style from input JSON or default
if [ -n "$CLAUDE_INPUT" ]; then
    OUTPUT_STYLE=$(echo "$CLAUDE_INPUT" 2>/dev/null | jq -r '.outputStyle // ""' 2>/dev/null)
else
    OUTPUT_STYLE=""
fi
# Default to rca if empty
if [ -z "$OUTPUT_STYLE" ]; then
    OUTPUT_STYLE="rca"
fi

# Get token count (estimate based on session)
TOKEN_FILE="/tmp/claude_tokens_$PPID.txt"
if [ -f "$TOKEN_FILE" ]; then
    TOKENS=$(cat "$TOKEN_FILE" 2>/dev/null || echo "0")
else
    # Estimate tokens from session activity
    TOKENS=$(echo "$CLAUDE_INPUT" 2>/dev/null | jq -r '.usage.totalTokens // 0' 2>/dev/null || echo "0")
fi
# Handle empty or non-numeric tokens
if [ -z "$TOKENS" ] || ! [[ "$TOKENS" =~ ^[0-9]+$ ]]; then
    TOKENS="0"
fi
# Format tokens for display (e.g., 1234 -> 1k, 12345 -> 12k)
if [ "$TOKENS" -gt 999 ]; then
    TOKENS=$((TOKENS / 1000))k
fi

# Get gamification stats (clean format)
BEHAVIOR_FILE="/tmp/claude_behavior_$PPID.json"
if [ -f "$BEHAVIOR_FILE" ]; then
    LEVEL=$(jq -r '.scores.level // 1' "$BEHAVIOR_FILE" 2>/dev/null)
    TRUST=$(jq -r '.trust_metrics.trust_level // 100' "$BEHAVIOR_FILE" 2>/dev/null | xargs printf "%.0f")
    VERIFICATIONS=$(jq -r '.verifications.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null)
    VIOLATIONS=$(jq -r '.violations.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null)
    
    # Build clean gamification string: L2•T90•V5/1
    GAMIFICATION="L${LEVEL}•T${TRUST}•V${VERIFICATIONS}/${VIOLATIONS}"
else
    GAMIFICATION="L1•T100•V0/0"
fi

# Combine all elements
STATUS="${GIT_INFO} | ${MODEL} | ${OUTPUT_STYLE} | ${TOKENS} | ${GAMIFICATION}"

# Output the status line
echo "$STATUS"