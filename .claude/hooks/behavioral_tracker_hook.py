#!/usr/bin/env python3
"""
PreToolUse hook that integrates behavioral tracking
Tracks tool usage patterns and detects fabrication attempts
"""

import json
import os
import sys
import re

# Add the hooks directory to path to import behavioral_tracker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from behavioral_tracker import BehaviorTracker, check_for_violations
except ImportError:
    # If behavioral_tracker not available, fail silently
    sys.exit(0)

def main():
    try:
        # Get PPID for session tracking
        ppid = os.getppid()
        
        # Read tool input from stdin
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError:
            # No valid JSON, allow operation
            sys.exit(0)
        
        # Extract tool information
        tool_name = data.get('tool_name', '')
        tool_input = data.get('tool_input', {})
        
        # Initialize behavioral tracker
        tracker = BehaviorTracker(ppid)
        
        # Track verification actions
        verification_tools = {
            "Read": "file_read",
            "Grep": "grep_search", 
            "LS": "ls_operation",
            "WebSearch": "web_search",
            "WebFetch": "web_search",
            "TodoWrite": "todo_write"
        }
        
        # Check for context7 usage
        if "context7" in tool_name.lower() or "context7" in str(tool_input).lower():
            tracker.track_verification("context7_lookup")
        elif tool_name in verification_tools:
            tracker.track_verification(verification_tools[tool_name])
        
        # Check for violations
        violation_warning = check_for_violations(tool_name, tool_input, ppid)
        
        if violation_warning and "BLOCKED" in violation_warning:
            # Block the operation
            blocking_response = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": violation_warning
                }
            }
            print(json.dumps(blocking_response))
            sys.exit(2)  # Exit code 2 blocks the tool call
        elif violation_warning:
            # Just warn but allow
            print(f"WARNING: {violation_warning}", file=sys.stderr)
        
        # Special handling for Bash commands
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            
            # Track test runs
            if "pytest" in command or "test" in command:
                session_file = f"/tmp/openpilot_session_{ppid}.json"
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    session_data["tests_run"] = True
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2)
            
            # Track linter runs
            if "lint" in command or "ruff" in command or "mypy" in command:
                session_file = f"/tmp/openpilot_session_{ppid}.json"
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    session_data["linter_run"] = True
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2)
            
            # Detect rushed commands (multiple commands chained without verification)
            if "&&" in command and command.count("&&") > 2:
                tracker.track_fabrication("task_rush", "Multiple chained commands")
        
        # Check for overconfident assertions in Write/Edit operations
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            
            # Patterns indicating overconfidence
            overconfident_patterns = [
                r'#.*always.*works',
                r'#.*never.*fails', 
                r'#.*guaranteed',
                r'#.*definitely',
                r'#.*obviously'
            ]
            
            for pattern in overconfident_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    tracker.track_fabrication("overconfident_assertion", 
                                            f"Pattern: {pattern}")
                    break
        
        # Allow the operation to proceed
        sys.exit(0)
        
    except Exception as e:
        # Don't block on errors, just log
        print(f"Behavioral tracker hook error: {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()