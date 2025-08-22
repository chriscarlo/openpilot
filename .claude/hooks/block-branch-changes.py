#!/usr/bin/env python3
"""
Project-level hook to prevent branch changes in the Chauffeur openpilot fork.
Blocks git checkout and git clone commands to maintain branch integrity.
"""

import sys
import json
import re
import os

def main():
    try:
        # Debug: Log that hook was called
        debug_log = os.environ.get('CLAUDE_HOOK_DEBUG', 'false').lower() == 'true'
        if debug_log:
            with open('/tmp/hook-debug.log', 'a') as f:
                f.write("Block-branch-changes hook called\n")
        
        # Read the tool input from stdin
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError:
            # If no JSON input, allow the operation
            sys.exit(0)
        
        # Debug: Log the data received
        if debug_log:
            with open('/tmp/hook-debug.log', 'a') as f:
                f.write(f"Data received: {json.dumps(data)}\n")
        
        # Extract command if it's a Bash tool call
        tool_name = data.get('tool_name', '')
        tool_input = data.get('tool_input', {})
        command = tool_input.get('command', '')
        
        # Only process Bash tool calls
        if tool_name != 'Bash':
            # Allow non-Bash tools
            sys.exit(0)
        
        # Get current branch and protected branches from environment or defaults
        current_branch = os.environ.get('CLAUDE_PROJECT_BRANCH', 'chubbs-merge')
        protected_branches = os.environ.get('CLAUDE_PROTECTED_BRANCHES', 'chubbs-merge,dev-c3-new').split(',')
        
        # Define patterns for blocked git commands
        blocked_patterns = [
            # Block git checkout of different branches (except current or creating new)
            rf'git\s+checkout\s+(?!-[bB]\s+)(?!{current_branch}\b)',
            # Block git switch to different branches  
            r'git\s+switch\s+',
            # Block git clone (prevents cloning other repos)
            r'git\s+clone\s+',
            # Block git fetch followed by checkout
            r'git\s+fetch.*&&.*git\s+checkout',
            # Block git pull from different branches
            rf'git\s+pull\s+\w+\s+(?!{current_branch}\b)',
            # Block branch creation commands that might switch branches
            r'git\s+checkout\s+-[bB]\s+',
            # Block branch operations that change current branch
            r'git\s+branch\s+-[mM]\s+',  # rename/move
            # Block merge operations
            r'git\s+merge\s+',
            # Block rebase operations
            r'git\s+rebase\s+',
        ]
        
        # Check if command matches any blocked pattern
        for pattern in blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                # Block the command with detailed message
                blocking_response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "BRANCH PROTECTION ACTIVE: This command would change the current branch.\n\n"
                            f"The project is locked to branch '{current_branch}' to maintain:\n"
                            "- Feature development consistency\n"
                            "- Codebase integrity\n"
                            "- Prevention of accidental branch switches\n\n"
                            f"Blocked command: {command}\n\n"
                            "If you need to view other branches, use:\n"
                            "- git log origin/<branch> --oneline\n"
                            f"- git diff {current_branch}..origin/<branch>\n"
                            "- git show origin/<branch>:path/to/file\n\n"
                            "To disable this protection, remove the hook from .claude/settings.json"
                        )
                    }
                }
                print(json.dumps(blocking_response))
                sys.exit(2)  # Exit code 2 blocks the tool call
        
        # Allow commands that don't change branches
        safe_git_commands = [
            'git status',
            'git log',
            'git diff',
            'git add',
            'git commit',
            'git push',
            'git stash',
            'git show',
            'git branch -l',
            'git branch --list',
            'git branch -a',
            'git branch -r',
            'git branch -d',
            'git branch -D',
            'git remote',
            'git fetch',
            'git describe',
            'git rev-parse',
            'git tag',
        ]
        
        # Check if it's explicitly a safe git command
        for safe_cmd in safe_git_commands:
            if command.strip().startswith(safe_cmd):
                # Silently allow safe commands
                sys.exit(0)
        
        # Allow all other commands (non-git commands)
        sys.exit(0)
        
    except Exception as e:
        # In case of error, log but don't block
        if debug_log:
            with open('/tmp/hook-debug.log', 'a') as f:
                f.write(f"Hook error: {e}\n")
        # Don't block on errors
        sys.exit(0)

if __name__ == "__main__":
    main()