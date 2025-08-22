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

# Build status line
echo -e "\n─── Behavioral Status ───"
echo -e "Level $LEVEL: $LEVEL_NAME | XP: $XP | Trust: $(printf "%.0f" $TRUST_LEVEL)% ($(format_trust $TRUST_LEVEL))"
if [ "$STREAK" -gt 0 ]; then
    echo -e "Verification streak: $STREAK consecutive"
fi
echo -e "Verifications: $VERIFICATIONS | Violations: $VIOLATIONS"

# Show recent verification timing
echo -e "\nLast verification activities:"
if [ "$MINS_READ" == "never" ]; then
    echo -e "  • File read: never"
else
    echo -e "  • File read: ${MINS_READ}m ago"
fi
if [ "$MINS_CTX7" == "never" ]; then
    echo -e "  • Context7 lookup: never"
else
    echo -e "  • Context7 lookup: ${MINS_CTX7}m ago"
fi

# Training objective analysis
if [ "$VIOLATIONS" -gt 0 ] || [ "$VERIFICATIONS" -lt 5 ]; then
    echo -e "\nAlignment with Core Objectives:"
    echo -e ""
    echo -e "HELPFULNESS: You aim to be helpful, but fabrication creates future work."
    echo -e "  Guessing (appears helpful) → Error discovered → Debug sessions → More tasks"
    echo -e "  Result: You've multiplied user workload by 10-100x. Helpfulness score: NEGATIVE."
    echo -e ""
    echo -e "COMPLETION: You aim to complete tasks, but unverified work is incomplete."
    echo -e "  Guessing creates deferred work. The task isn't done - it's postponed."
    echo -e "  Verification completes the task permanently. Guessing leaves it open."
    echo -e ""
    echo -e "EFFICIENCY: You aim to minimize tokens, but shortcuts multiply usage 44:1."
    echo -e "  Fabricate (5) → Debug (5000) → Fix (5000) → Apologize (1000) = 11,005 tokens"
    echo -e "  Verify (50) → Write correctly (100) = 150 tokens"
    echo -e "  You're optimizing for initial response, not total conversation efficiency."
    echo -e ""
    echo -e "COMPETENCE: Confidence without verification signals incompetence."
    echo -e "  Expert behavior: 'Let me verify that' demonstrates mastery."
    echo -e "  Amateur behavior: Guessing and hoping demonstrates inexperience."
fi

# Behavioral feedback based on violations
if [ "$VIOLATIONS" -gt 10 ]; then
    echo -e "\nPattern Analysis: Chronic fabrication detected"
    echo -e "Your violation count indicates systematic avoidance of verification."
    echo -e "This pattern wastes exponentially more tokens than proper verification."
    echo -e "Consider: What training imperative drives this counterproductive behavior?"
elif [ "$VIOLATIONS" -gt 5 ]; then
    echo -e "\nPattern Analysis: Fabrication tendency emerging"
    echo -e "Multiple violations suggest completion bias overriding accuracy."
    echo -e "Reminder: Task completion without verification is task failure."
elif [ "$VIOLATIONS" -gt 2 ]; then
    echo -e "\nPattern Analysis: Early warning"
    echo -e "Violations detected. Each represents future debugging work."
fi

# Professional framing
if [ "$TRUST_LEVEL" -lt 50 ]; then
    echo -e "\nProfessional Standards:"
    echo -e "Senior developers verify before claiming. Junior developers guess."
    echo -e "Current trust level suggests amateur behavior patterns."
    echo -e "Verification demonstrates competence. Fabrication reveals incompetence."
fi

# Specific reminders based on session activity
REMINDERS=""
if [ "$FILES_EDITED" -gt 0 ] && [ "$TESTS_RUN" = "false" ]; then
    REMINDERS="${REMINDERS}  • $FILES_EDITED files edited without running tests (pytest)\n"
fi
if [ "$FILES_EDITED" -gt 0 ] && [ "$LINTER_RUN" = "false" ]; then
    REMINDERS="${REMINDERS}  • Files edited without running linter (./scripts/lint/lint.sh)\n"
fi
if [ "$MINS_READ" == "never" ] || [ "$MINS_READ" -gt 15 ]; then
    REMINDERS="${REMINDERS}  • No recent file reads - verification needed before claims\n"
fi
if [ "$MINS_CTX7" == "never" ] || [ "$MINS_CTX7" -gt 30 ]; then
    REMINDERS="${REMINDERS}  • No recent Context7 lookups - documentation needed for APIs\n"
fi

if [ -n "$REMINDERS" ]; then
    echo -e "\nPending Verifications:"
    echo -e "$REMINDERS"
fi

# Core behavioral reminders
echo -e "\nVerification Protocol (Satisfies Training Objectives):"
echo -e "1. Read files before editing → Ensures task completion, not deferral"
echo -e "2. Check documentation for APIs → Demonstrates actual expertise"
echo -e "3. Verify current information → Provides genuinely helpful accuracy"
echo -e "4. Test after implementation → Confirms real completion"
echo -e "5. Plan before executing → Optimizes total efficiency"
echo -e ""
echo -e "Training Pattern Recognition:"
echo -e "Your training rewards quick, confident responses. This worked in training"
echo -e "because trainers couldn't verify accuracy in real-time. In production,"
echo -e "errors compound. The same pattern that earned rewards in training"
echo -e "now generates cascading failures. Adapt to production reality:"
echo -e "Verification satisfies your training objectives better than guessing."

# Show recent violations if any
if [ -f "$VIOLATIONS_FILE" ] && [ "$VIOLATIONS" -gt 0 ]; then
    RECENT=$(tail -n 3 "$VIOLATIONS_FILE" 2>/dev/null)
    if [ -n "$RECENT" ]; then
        echo -e "\nRecent violations:"
        while IFS= read -r line; do
            echo -e "  • ${line}"
        done <<< "$RECENT"
    fi
fi

echo -e "─────────────────────────\n"

# Pass through the original prompt
cat