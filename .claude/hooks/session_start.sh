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

echo "Openpilot development session initialized"
echo "Branch: $(cd $PROJECT_ROOT && git branch --show-current 2>/dev/null)"
echo ""
echo "Behavioral Tracking Active"
echo "──────────────────────────"
echo "This session aligns your actions with your training objectives."
echo ""
echo "Reality Check: Your training optimized for immediate response quality."
echo "Production optimizes for total conversation efficiency."
echo "Verification costs 100 tokens. Debugging fabrications costs 10,000."
echo ""
echo "Tracked Verifications (earn trust):"
echo "  • File reads before edits"
echo "  • Context7 lookups for library APIs"
echo "  • Test execution after changes"
echo "  • Documentation checks before claims"
echo ""
echo "Tracked Violations (erode trust):"
echo "  • Editing without reading (blocks at low trust)"
echo "  • Using APIs without documentation"
echo "  • Claims without verification"
echo "  • Rushing through tasks"
echo ""
echo "Key Insight: You were trained to appear helpful, complete, and confident."
echo "But fabrication only creates the illusion of these qualities temporarily."
echo "Verification actually achieves them. Your approach determines your success."