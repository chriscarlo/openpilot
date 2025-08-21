#!/usr/bin/env python3
"""
Check that files are read before being edited.
Blocks edit operations on files that haven't been read in this session.
"""

import json
import os
import sys

def main():
    # Get the file being edited from the tool parameters
    tool_params = json.loads(os.environ.get('CLAUDE_TOOL_PARAMS', '{}'))
    file_path = tool_params.get('file_path', '')
    
    if not file_path:
        return 0  # No file specified, allow operation
    
    # Skip check for new files (Write tool on non-existing files)
    if not os.path.exists(file_path):
        return 0  # Allow creating new files
    
    # Find session file
    session_file = f"/tmp/openpilot_session_{os.getppid()}.json"
    
    try:
        # Load session data
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            files_read = session_data.get("files_read", [])
            
            # Check if file has been read
            if file_path not in files_read:
                print(f"BLOCKED: Cannot edit {file_path} without reading it first!", file=sys.stderr)
                print(f"You must use the Read tool on this file before editing.", file=sys.stderr)
                print(f"Files read in this session: {len(files_read)}", file=sys.stderr)
                return 1  # Block the operation
            else:
                print(f"Edit verified: {file_path} was read")
                
                # Track the edit
                if "files_edited" not in session_data:
                    session_data["files_edited"] = []
                if file_path not in session_data["files_edited"]:
                    session_data["files_edited"].append(file_path)
                    
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
        else:
            print("No session data found, allowing edit", file=sys.stderr)
    
    except Exception as e:
        print(f"Error checking read status: {e}", file=sys.stderr)
        # Don't block on errors
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())