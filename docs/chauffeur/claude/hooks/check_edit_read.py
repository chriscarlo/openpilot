#!/usr/bin/env python3
"""Check if file was read before editing - enforcement hook.

Robust across varying launcher PPIDs: accepts either a session-specific or
global per-file marker created by track_read.py.
"""

import sys
import json
import os
import time
import re
import glob

def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "general").strip("_")
    return s or "general"

try:
    data = json.load(sys.stdin)
    file = data.get('tool_input', {}).get('file_path', '')

    if not file:
        # No file path, let it proceed
        sys.exit(0)

    ppid = os.getppid()
    ident = sanitize(str(file))

    # Candidate markers to accept as proof of read
    candidates = [
        f"/tmp/claude_session_{ppid}_read_{ident}",
        f"/tmp/claude_read_{ident}",
    ]
    # Also accept any session's marker for this file
    candidates.extend(glob.glob(f"/tmp/claude_session_*_read_{ident}"))

    if not any(os.path.exists(p) for p in candidates):
        # Violation detected - file not read first
        print(
            f"\n❌ BLOCKED: Cannot edit {file} without a prior Read\n"
            f"💡 Hint: Run Read on this file first (creates /tmp/claude_read_{ident})\n",
            file=sys.stderr,
        )

        # Log violation
        violations_file = f"/tmp/claude_violations_{ppid}.txt"
        with open(violations_file, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')}: Edit without Read: {file}\n")

        # Block the operation (exit code 2 used elsewhere in hooks)
        sys.exit(2)

    # File was read - allow edit
    sys.exit(0)

except Exception as e:
    # Don't block on errors
    print(f"Edit check error (non-blocking): {e}", file=sys.stderr)
    sys.exit(0)
