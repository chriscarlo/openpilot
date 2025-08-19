# Claude Code Anti-Fabrication Hooks System

This directory contains a complete set of hooks and scripts for Claude Code that enforce code quality and prevent fabrication through behavioral tracking, gamification, and real-time feedback.

## Files Included

- settings.json - Main hooks configuration file
- prompt_inject.sh - UserPromptSubmit hook script
- session_start.sh - SessionStart hook script
- behavioral_tracker.py - Core behavioral tracking system (1000+ lines)
- behavioral_tracker_hook.py - PreToolUse behavioral integration
- check_edit_read.py - Edit verification hook
- track_read.py - Read operation tracking
- track_tool.py - Tool usage pattern detection
- statusline-command.sh - Status line display script
- enforce.sh - Enforcement script
- test_behavioral_tracker.py - Test suite

## Installation

Clone this repository and copy all files to ~/.claude/

Update paths in settings.json to match your home directory.

Make all scripts executable:
chmod +x ~/.claude/*.sh ~/.claude/*.py

Test with: python3 ~/.claude/test_behavioral_tracker.py

Restart Claude Code to activate hooks.

## How It Works

The system tracks all tool usage, enforces verification before claims, gamifies good behavior with XP and levels, blocks dangerous operations, and shows token economics.

## License

Provided as-is for improving Claude Code accuracy and reliability.

Version: 2.0 - Enhanced with behavioral tracking
Last updated: August 2024
