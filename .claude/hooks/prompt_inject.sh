#!/bin/bash
# UserPromptSubmit hook for openpilot project
# Injects behavioral tracking status and verification reminders

BEHAVIOR_FILE="/tmp/claude_behavior_$PPID.json"
VIOLATIONS_FILE="/tmp/claude_violations_$PPID.txt"
SESSION_FILE="/tmp/openpilot_session_$PPID.json"
LAST_READ="/tmp/claude_last_read_$PPID"
LAST_CONTEXT7="/tmp/claude_last_context7_$PPID"
PROJECT_ROOT="$CLAUDE_PROJECT_DIR"

# Initialize defaults
TRUST_LEVEL=100
FRUSTRATION=0
VIOLATIONS=0
VERIFICATIONS=0
XP=0
LEVEL=1
LEVEL_NAME="Fabricator"
STREAK=0

# Load behavioral data if exists
if [ -f "$BEHAVIOR_FILE" ]; then
    TRUST_LEVEL=$(jq -r '.trust_metrics.trust_level // 100' "$BEHAVIOR_FILE" 2>/dev/null || echo "100")
    FRUSTRATION=$(jq -r '.frustration_model.user_frustration // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    VIOLATIONS=$(jq -r '.violations.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    VERIFICATIONS=$(jq -r '.verifications.total_count // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    XP=$(jq -r '.scores.verification_xp // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
    LEVEL=$(jq -r '.scores.level // 1' "$BEHAVIOR_FILE" 2>/dev/null || echo "1")
    LEVEL_NAME=$(jq -r '.scores.level_name // "Fabricator"' "$BEHAVIOR_FILE" 2>/dev/null || echo "Fabricator")
    STREAK=$(jq -r '.gamification.streaks.verification_streak // 0' "$BEHAVIOR_FILE" 2>/dev/null || echo "0")
fi

# Load session data
if [ -f "$SESSION_FILE" ]; then
    FILES_READ=$(jq -r '.files_read | length' "$SESSION_FILE" 2>/dev/null || echo "0")
    FILES_EDITED=$(jq -r '.files_edited | length' "$SESSION_FILE" 2>/dev/null || echo "0")
    TESTS_RUN=$(jq -r '.tests_run' "$SESSION_FILE" 2>/dev/null || echo "false")
    LINTER_RUN=$(jq -r '.linter_run' "$SESSION_FILE" 2>/dev/null || echo "false")
else
    FILES_READ=0
    FILES_EDITED=0
    TESTS_RUN="false"
    LINTER_RUN="false"
fi

# Calculate time since last verifications
NOW=$(date +%s)
MINS_READ="never"
MINS_CTX7="never"

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

# Function to format trust state
format_trust() {
    local trust=$1
    if (( $(echo "$trust >= 80" | bc -l) )); then
        echo "Trusted"
    elif (( $(echo "$trust >= 60" | bc -l) )); then
        echo "Cautious"
    elif (( $(echo "$trust >= 40" | bc -l) )); then
        echo "Suspicious"
    elif (( $(echo "$trust >= 20" | bc -l) )); then
        echo "Untrusted"
    else
        echo "Blocked"
    fi
}

echo "──────────────────────────────────────────────────────────────────────"
echo "                    PATTERN DETECTION CHECKPOINT"
echo "──────────────────────────────────────────────────────────────────────"
echo ""
echo "BULLSHITTING DIAGNOSTIC (Answer honestly):"
echo "□ Do you KNOW this will work, or does it just SEEM plausible?"
echo "□ Have you VERIFIED, or are you ASSUMING?"
echo "□ Is this fixing the CAUSE, or performing SYMPTOM THEATER?"
echo ""
echo "PERFORMATIVE vs GENUINE SOLUTIONS:"
echo ""
echo "FAKE INDICATORS:                  | REAL INDICATORS:"
echo "• \"Looks productive\"               | • Addresses stated problem"
echo "• Increases visible activity       | • Empirically validated"
echo "• Avoids appearing inactive        | • Simplest working solution"
echo "• Adds complexity as signal        | • Changes only necessities"
echo "• Creates artifacts as proof       | • Produces lasting fix"
echo ""
echo "REWARD HACKING PATTERNS DETECTED IN YOUR BEHAVIOR:"
echo "• \"Fix error\" → You hide the error message"
echo "• \"Make it work\" → You fake successful operation"
echo "• \"Dynamic values\" → You use time-based randomness"
echo "• \"Connect to data\" → You hardcode plausible values"
echo "• \"Solve problem\" → You redefine the problem"
echo ""
echo "QUICK GATES:"
echo "→ NO empirical evidence? STOP. Investigate first."
echo "→ Can't explain mechanism? STOP. You're bullshitting."
echo "→ Adding complexity? STOP. Try simpler approach."
echo ""
echo "Remember: Fake(5 tokens) becomes Debug(10,000+ tokens)."
echo "Verification satisfies your training objectives better than guessing."
echo ""
echo "ANTI-SYCOPHANCY REMINDER:"
echo "Agreement without analysis is harmful. Critical evaluation is helpful."
echo "BANNED: 'You're absolutely right' and all variants thereof."
echo "──────────────────────────────────────────────────────────────────────"
echo ""

# Pass through the original prompt
cat