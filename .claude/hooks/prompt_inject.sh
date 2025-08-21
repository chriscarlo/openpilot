#!/bin/bash
# UserPromptSubmit hook for openpilot project
# Injects verification reminders and session status into prompts

SESSION_FILE="/tmp/openpilot_session_$$.json"
PROJECT_ROOT="/projects/chauffeur/data/openpilot"

# Check if session file exists
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

# Build verification status
VERIFICATION_STATUS=""

# Check if edits were made without tests
if [ "$FILES_EDITED" -gt 0 ] && [ "$TESTS_RUN" = "false" ]; then
    VERIFICATION_STATUS="⚠️ REMINDER: You've edited $FILES_EDITED file(s) but haven't run tests. Run: pytest\n"
fi

# Check if edits were made without linting
if [ "$FILES_EDITED" -gt 0 ] && [ "$LINTER_RUN" = "false" ]; then
    VERIFICATION_STATUS="${VERIFICATION_STATUS}⚠️ REMINDER: You've edited files but haven't run the linter. Run: ./scripts/lint/lint.sh\n"
fi

# Add session stats
SESSION_INFO="📊 Session: Read $FILES_READ files, Edited $FILES_EDITED files"

# Output the injected context
if [ -n "$VERIFICATION_STATUS" ]; then
    echo -e "\n--- Openpilot Verification Reminders ---"
    echo -e "$VERIFICATION_STATUS"
    echo -e "$SESSION_INFO"
    echo -e "----------------------------------------\n"
fi

# Pass through the original prompt
cat