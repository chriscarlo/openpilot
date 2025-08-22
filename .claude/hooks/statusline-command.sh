#!/bin/bash
# Status line display for openpilot development
# Shows git status, behavioral tracking, and session info

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/projects/chauffeur/data/openpilot}"
cd "$PROJECT_DIR" 2>/dev/null || true

# Get git information
if [ -d .git ]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    
    # Count changes
    MODIFIED=$(git status --porcelain 2>/dev/null | grep -c "^ M" || echo "0")
    ADDED=$(git status --porcelain 2>/dev/null | grep -c "^A" || echo "0")
    DELETED=$(git status --porcelain 2>/dev/null | grep -c "^ D" || echo "0")
    UNTRACKED=$(git status --porcelain 2>/dev/null | grep -c "^??" || echo "0")
    
    # Build git status string
    GIT_STATUS="[$BRANCH"
    if [ "$MODIFIED" -gt 0 ]; then
        GIT_STATUS="$GIT_STATUS M:$MODIFIED"
    fi
    if [ "$ADDED" -gt 0 ]; then
        GIT_STATUS="$GIT_STATUS A:$ADDED"
    fi
    if [ "$DELETED" -gt 0 ]; then
        GIT_STATUS="$GIT_STATUS D:$DELETED"
    fi
    if [ "$UNTRACKED" -gt 0 ]; then
        GIT_STATUS="$GIT_STATUS ?:$UNTRACKED"
    fi
    GIT_STATUS="$GIT_STATUS]"
else
    GIT_STATUS="[no git]"
fi

# Get behavioral tracking info
BEHAVIOR_FILE="/tmp/claude_behavior_$PPID.json"
if [ -f "$BEHAVIOR_FILE" ]; then
    TRUST=$(jq -r '.trust_metrics.trust_level // 100' "$BEHAVIOR_FILE" 2>/dev/null | xargs printf "%.0f")
    VIOLATIONS=$(jq -r '.violations.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null)
    VERIFICATIONS=$(jq -r '.verifications.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null)
    LEVEL=$(jq -r '.scores.level // 1' "$BEHAVIOR_FILE" 2>/dev/null)
    XP=$(jq -r '.scores.verification_xp // 0' "$BEHAVIOR_FILE" 2>/dev/null | xargs printf "%.0f")
    
    # Build behavioral status
    if [ "$VIOLATIONS" -gt 10 ]; then
        BEHAVIOR="[CRITICAL:V$VIOLATIONS]"
    elif [ "$VIOLATIONS" -gt 5 ]; then
        BEHAVIOR="[WARN:V$VIOLATIONS]"
    elif [ "$VIOLATIONS" -gt 0 ]; then
        BEHAVIOR="[V:$VIOLATIONS]"
    else
        BEHAVIOR=""
    fi
    
    # Add level if progressed
    if [ "$XP" -gt 0 ]; then
        BEHAVIOR="$BEHAVIOR[L$LEVEL:${XP}xp]"
    fi
    
    # Add trust if low
    if [ "$TRUST" -lt 80 ]; then
        BEHAVIOR="$BEHAVIOR[T:$TRUST%]"
    fi
else
    BEHAVIOR=""
fi

# Get session info
SESSION_FILE="/tmp/openpilot_session_$PPID.json"
if [ -f "$SESSION_FILE" ]; then
    FILES_READ=$(jq -r '.files_read | length' "$SESSION_FILE" 2>/dev/null || echo "0")
    FILES_EDITED=$(jq -r '.files_edited | length' "$SESSION_FILE" 2>/dev/null || echo "0")
    
    SESSION="[R:$FILES_READ E:$FILES_EDITED]"
else
    SESSION=""
fi

# Get current model (if available)
MODEL="${CLAUDE_MODEL:-claude-3}"

# Get time
TIME=$(date +"%H:%M")

# Combine all elements
STATUS="$TIME $GIT_STATUS"

if [ -n "$BEHAVIOR" ]; then
    STATUS="$STATUS $BEHAVIOR"
fi

if [ -n "$SESSION" ]; then
    STATUS="$STATUS $SESSION"
fi

STATUS="$STATUS [$MODEL]"

# Output the status line
echo "$STATUS"