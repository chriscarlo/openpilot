#!/usr/bin/env python3
"""
Track Read operations to maintain a list of files that have been read.
This enables verification that files are read before being edited.
"""

import json
import os
import sys
from datetime import datetime

def main():
    # Get the file being read from the tool parameters
    tool_params = json.loads(os.environ.get('CLAUDE_TOOL_PARAMS', '{}'))
    file_path = tool_params.get('file_path', '')
    
    if not file_path:
        return 0  # No file specified, nothing to track
    
    # Find session file
    session_file = f"/tmp/openpilot_session_{os.getppid()}.json"
    
    try:
        # Load existing session data
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        else:
            session_data = {
                "project": "openpilot",
                "session_start": datetime.now().isoformat(),
                "files_read": [],
                "files_edited": [],
                "tests_run": False,
                "linter_run": False
            }
        
        # Add file to read list if not already there
        if file_path not in session_data["files_read"]:
            session_data["files_read"].append(file_path)
            
            # Save updated session data
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Log the read operation
            print(f"📖 Tracked read: {file_path}")
    
    except Exception as e:
        print(f"⚠️ Failed to track read: {e}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())