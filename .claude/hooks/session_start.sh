#!/bin/bash
# SessionStart hook for openpilot project
# Initializes session tracking and loads project context

SESSION_FILE="/tmp/openpilot_session_$$.json"
PROJECT_ROOT="/projects/chauffeur/data/openpilot"

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

echo "✅ Openpilot development session initialized"
echo "📍 Branch: $(cd $PROJECT_ROOT && git branch --show-current 2>/dev/null)"
echo "📋 Remember to:"
echo "   - Run tests after changes: pytest"
echo "   - Run linter: ./scripts/lint/lint.sh"
echo "   - Build with: scons -u -j\$(nproc)"