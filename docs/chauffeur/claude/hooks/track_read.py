#!/usr/bin/env python3
"""Simple read tracking hook (robust to varying PPIDs).

Creates both a session-specific and a global per-file marker so that
subsequent Edit hooks can verify a Read occurred even if the launcher
PPID differs between hook invocations.
"""

import sys
import json
import os
import time
import re

def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "general").strip("_")
    return s or "general"

try:
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})

    # Resolve a file identifier from common keys
    file = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or tool_input.get("pattern")
        or tool_input.get("target")
        or "general"
    )
    ident = sanitize(str(file))

    # Create markers (session-scoped and global)
    ppid = os.getppid()
    ts = str(time.time())
    session_marker = f"/tmp/claude_session_{ppid}_read_{ident}"
    global_marker = f"/tmp/claude_read_{ident}"
    last_read_file = f"/tmp/claude_last_read_{ppid}"

    for path in (session_marker, global_marker, last_read_file):
        with open(path, "w") as f:
            f.write(ts)

    # Success - let tool proceed
    sys.exit(0)

except Exception as e:
    # Don't block on errors
    print(f"Read tracking error (non-blocking): {e}", file=sys.stderr)
    sys.exit(0)
