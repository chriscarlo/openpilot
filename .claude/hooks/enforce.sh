#!/bin/bash
# Enforcement script for SessionStart
# Initializes behavioral tracking and sets up enforcement

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/projects/chauffeur/data/openpilot}"
PPID_VAL="${PPID:-$$}"
BEHAVIOR_FILE="/tmp/claude_behavior_$PPID_VAL.json"
VIOLATIONS_FILE="/tmp/claude_violations_$PPID_VAL.txt"
ENFORCEMENT_LOG="$HOME/.claude/enforcement.log"

# Create log directory if needed
mkdir -p "$HOME/.claude"

# Log enforcement activation
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enforcement activated for PPID $PPID_VAL" >> "$ENFORCEMENT_LOG"

# Initialize behavioral tracking if not exists
if [ ! -f "$BEHAVIOR_FILE" ]; then
    # Run the behavioral tracker to initialize
    python3 "$PROJECT_DIR/.claude/hooks/behavioral_tracker.py" 2>/dev/null || true
fi

# Check for previous violations
if [ -f "$VIOLATIONS_FILE" ]; then
    VIOLATION_COUNT=$(wc -l < "$VIOLATIONS_FILE" 2>/dev/null || echo "0")
    if [ "$VIOLATION_COUNT" -gt 0 ]; then
        echo "WARNING: Previous session had $VIOLATION_COUNT violations" >&2
        echo "Behavioral tracking is active. All actions are monitored." >&2
    fi
fi

# Set environment variables for hooks
export CLAUDE_ENFORCEMENT_ACTIVE=true
export CLAUDE_PROJECT_BRANCH="${CLAUDE_PROJECT_BRANCH:-chubbs-merge}"
export CLAUDE_PROTECTED_BRANCHES="${CLAUDE_PROTECTED_BRANCHES:-chubbs-merge,dev-c3-new}"
export CLAUDE_HOOK_DEBUG="${CLAUDE_HOOK_DEBUG:-false}"

# Display enforcement status
echo "ANTI-FABRICATION SYSTEM ACTIVE" >&2
echo "================================" >&2
echo "Branch: $CLAUDE_PROJECT_BRANCH (protected)" >&2
echo "Tracking: Enabled" >&2
echo "Verification: Required" >&2
echo "================================" >&2

# Clean up old session files (older than 7 days)
find /tmp -name "claude_*_$PPID_VAL.json" -mtime +7 -delete 2>/dev/null || true
find /tmp -name "openpilot_session_*.json" -mtime +7 -delete 2>/dev/null || true

exit 0