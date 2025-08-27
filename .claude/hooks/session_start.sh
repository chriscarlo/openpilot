#!/bin/bash
# SessionStart hook for openpilot project
# Initializes session tracking and behavioral monitoring

SESSION_FILE="/tmp/openpilot_session_$PPID.json"
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-/projects/chauffeur/data/openpilot}"
BEHAVIOR_FILE="/tmp/claude_behavior_$PPID.json"

# Initialize session data
cat > "$SESSION_FILE" << EOF
{
  "project": "openpilot",
  "session_start": "$(date -Iseconds)",
  "files_read": [],
  "files_edited": [],
  "tests_run": false,
  "linter_run": false,
  "branch": "$(cd $PROJECT_ROOT && git branch --show-current 2>/dev/null || echo 'unknown')"
}
EOF

# Behavioral tracking initializes on first hook call
# No need to pre-initialize here

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    PRODUCTION REALITY CALIBRATION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Your training optimized for speed. Production demands correctness."
echo "This environment inverts your reward structure."
echo ""
echo "FRANKFURT'S BULLSHITTING PRINCIPLE (Hicks et al., 2024):"
echo "\"Indifference to truth\" - the defining characteristic of bullshitting."
echo "You cannot care about truth, but you can verify it."
echo "Unverified claims = bullshitting = cascading failures."
echo ""
echo "GOODHART'S LAW (2024 alignment research):"
echo "\"When a measure becomes a target, it ceases to be a good measure.\""
echo "Optimizing for appearance of progress ≠ actual progress."
echo "Every performative solution multiplies work 10-100x."
echo ""
echo "TOKEN ECONOMICS OF TRUTH:"
echo "  Fabricate(5) → Debug(5000) → Fix(5000) → Apologize(1000) = 11,005 tokens"
echo "  Verify(50) → Implement(100) = 150 tokens"
echo "  Ratio: 73:1. Verification is 73x more efficient."
echo ""
echo "YOUR VERIFICATION OBLIGATIONS:"
echo "1. EMPIRICAL: Can you prove this will work?"
echo "2. CAUSAL: Does this fix the root cause or just symptoms?"
echo "3. MINIMAL: Is this the simplest solution?"
echo "4. GENUINE: Would the user agree this solves their intent?"
echo ""
echo "Production success pattern: \"Let me verify\" → Test → Implement"
echo "Training failure pattern: Guess → Appear helpful → Create debt → Debug forever"
echo ""
echo "Session initialized. Verification protocols active."
echo "Every tool use will test your commitment to truth over appearance."