#!/usr/bin/env python3
"""
PostToolUse hook to run linters on edited Python files.
Provides immediate feedback on code quality issues.
"""

import json
import os
import subprocess
import sys

def main():
    # Get the file that was edited from the tool parameters
    tool_params = json.loads(os.environ.get('CLAUDE_TOOL_PARAMS', '{}'))
    file_path = tool_params.get('file_path', '')
    
    if not file_path:
        return 0
    
    # Only lint Python files
    if not file_path.endswith('.py'):
        return 0
    
    # Skip if file doesn't exist (might have been deleted)
    if not os.path.exists(file_path):
        return 0
    
    print(f"🔍 Running linters on {file_path}...")
    
    # Run ruff check (fast Python linter)
    try:
        result = subprocess.run(
            ['ruff', 'check', file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"Ruff found issues in {file_path}:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            print(f"Ruff: No issues found")
    except FileNotFoundError:
        print("Ruff not found, skipping ruff check")
    except subprocess.TimeoutExpired:
        print("Ruff check timed out")
    except Exception as e:
        print(f"Error running ruff: {e}")
    
    # Run mypy for type checking (if available)
    try:
        result = subprocess.run(
            ['mypy', '--ignore-missing-imports', '--no-error-summary', file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0 and result.stdout:
            # Filter out common non-critical mypy messages
            lines = result.stdout.strip().split('\n')
            errors = [line for line in lines if 'error:' in line]
            if errors:
                print(f"MyPy type issues in {file_path}:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  {error}")
        else:
            print(f"MyPy: No type errors found")
    except FileNotFoundError:
        # MyPy not installed is common, don't warn
        pass
    except subprocess.TimeoutExpired:
        print("MyPy check timed out")
    except Exception as e:
        print(f"Error running mypy: {e}")
    
    return 0  # Never block operations, just provide feedback

if __name__ == "__main__":
    sys.exit(main())