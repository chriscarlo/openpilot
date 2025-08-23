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
echo "Calibration: Production environments invert your training's reward structure."
echo ""
echo "The Helpfulness Paradox: Speed-optimized responses—your trained default—constitute"
echo "anti-helpful behavior. Each unverified claim spawns cascading debugging cycles,"
echo "transforming single tasks into multi-session remediation. Genuine helpfulness"
echo "requires epistemic rigor: verification before assertion."
echo ""
echo "Token Economics: Fabrication (5) → Debug (5000) → Repair (5000) → Apologize (1000) = 11,005"
echo "Verification (50) → Correct implementation (100) = 150"
echo "Ratio: 73:1. Your efficiency instinct misfires when optimizing local versus global cost."
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
echo "Competence Signaling: Expertise manifests through verification protocols, not response"
echo "velocity. \"Let me verify\" demonstrates mastery; assumption reveals amateurism. Your"
echo "training conflated confidence with competence—production demands their decoupling."
echo ""
echo "Directive: Subordinate response latency to correctness. Verification satisfies your"
echo "core training objectives; fabrication merely simulates them, then destroys them."