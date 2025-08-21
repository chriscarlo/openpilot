#!/usr/bin/env python3
"""
Test script to verify openpilot hooks are working correctly.
"""

import json
import os
import subprocess
import sys
import tempfile

def test_session_start():
    """Test the session_start.sh hook"""
    print("Testing session_start.sh...")
    result = subprocess.run(
        ["bash", ".claude/hooks/session_start.sh"],
        capture_output=True,
        text=True,
        cwd="/projects/chauffeur/data/openpilot"
    )
    
    if result.returncode == 0:
        print("✅ session_start.sh executed successfully")
        if "Openpilot development session initialized" in result.stdout:
            print("✅ Session initialization message found")
        else:
            print("⚠️ Expected initialization message not found")
    else:
        print(f"❌ session_start.sh failed: {result.stderr}")
    print()

def test_track_read():
    """Test the track_read.py hook"""
    print("Testing track_read.py...")
    
    # Set up environment
    os.environ['CLAUDE_TOOL_PARAMS'] = json.dumps({
        "file_path": "/projects/chauffeur/data/openpilot/README.md"
    })
    
    result = subprocess.run(
        ["python3", ".claude/hooks/track_read.py"],
        capture_output=True,
        text=True,
        cwd="/projects/chauffeur/data/openpilot"
    )
    
    if result.returncode == 0:
        print("✅ track_read.py executed successfully")
        if "Tracked read:" in result.stdout:
            print("✅ Read tracking message found")
        else:
            print("⚠️ Expected tracking message not found")
    else:
        print(f"❌ track_read.py failed: {result.stderr}")
    print()

def test_check_edit_read():
    """Test the check_edit_read.py hook"""
    print("Testing check_edit_read.py...")
    
    # Test blocking edit without read
    os.environ['CLAUDE_TOOL_PARAMS'] = json.dumps({
        "file_path": "/projects/chauffeur/data/openpilot/pyproject.toml"
    })
    
    result = subprocess.run(
        ["python3", ".claude/hooks/check_edit_read.py"],
        capture_output=True,
        text=True,
        cwd="/projects/chauffeur/data/openpilot"
    )
    
    # This should block or warn since we haven't read the file
    if "BLOCKED" in result.stderr or "must use the Read tool" in result.stderr:
        print("✅ check_edit_read.py correctly blocked unread file edit")
    elif "No session data" in result.stderr:
        print("⚠️ No session data, hook allowed edit")
    else:
        print("✅ check_edit_read.py executed")
    print()

def test_run_linters():
    """Test the run_linters.py hook"""
    print("Testing run_linters.py...")
    
    # Create a test Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def test():\n    print('test')\n")
        test_file = f.name
    
    try:
        os.environ['CLAUDE_TOOL_PARAMS'] = json.dumps({
            "file_path": test_file
        })
        
        result = subprocess.run(
            ["python3", ".claude/hooks/run_linters.py"],
            capture_output=True,
            text=True,
            cwd="/projects/chauffeur/data/openpilot"
        )
        
        if result.returncode == 0:
            print("✅ run_linters.py executed successfully")
            if "Running linters" in result.stdout:
                print("✅ Linter execution message found")
        else:
            print(f"⚠️ run_linters.py had issues: {result.stderr}")
    finally:
        os.unlink(test_file)
    print()

def test_prompt_inject():
    """Test the prompt_inject.sh hook"""
    print("Testing prompt_inject.sh...")
    
    result = subprocess.run(
        ["bash", ".claude/hooks/prompt_inject.sh"],
        input="Test prompt",
        capture_output=True,
        text=True,
        cwd="/projects/chauffeur/data/openpilot"
    )
    
    if result.returncode == 0:
        print("✅ prompt_inject.sh executed successfully")
        if "Test prompt" in result.stdout:
            print("✅ Original prompt passed through")
        else:
            print("⚠️ Original prompt not found in output")
    else:
        print(f"❌ prompt_inject.sh failed: {result.stderr}")
    print()

def main():
    print("=" * 60)
    print("Openpilot Hooks Test Suite")
    print("=" * 60)
    print()
    
    # Change to project directory
    os.chdir("/projects/chauffeur/data/openpilot")
    
    # Run tests
    test_session_start()
    test_track_read()
    test_check_edit_read()
    test_run_linters()
    test_prompt_inject()
    
    print("=" * 60)
    print("Test suite complete!")
    print("=" * 60)
    print()
    print("To activate these hooks in Claude Code:")
    print("1. Copy .claude/hooks/settings.json to ~/.claude/settings.json")
    print("2. Copy .claude/hooks/*.sh and *.py to ~/.claude/hooks/")
    print("3. Restart Claude Code")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())