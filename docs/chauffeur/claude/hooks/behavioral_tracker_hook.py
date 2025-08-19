#!/usr/bin/env python3
"""
PreToolUse hook that integrates behavioral tracking
Monitors all tool usage and enforces behavioral modification
"""

import sys
import json
import os
from pathlib import Path

# Import the behavioral tracker
sys.path.insert(0, str(Path(__file__).parent))
from behavioral_tracker import BehaviorTracker

def main():
    try:
        # Read tool invocation data
        data = json.load(sys.stdin)
        tool_name = data.get('tool_name', '')
        tool_input = data.get('tool_input', {})
        
        # Initialize tracker
        ppid = os.getppid()
        tracker = BehaviorTracker(ppid)
        
        # Check patterns and track behavior
        response = tracker.hook_pretool_use(tool_name, tool_input)
        
        # Check for library usage without context7 (specific pattern detection)
        if tool_name in ['Write', 'Edit', 'MultiEdit']:
            content = ''
            if tool_name == 'Write':
                content = tool_input.get('content', '')
            elif tool_name == 'Edit':
                content = tool_input.get('new_string', '')
            elif tool_name == 'MultiEdit':
                edits = tool_input.get('edits', [])
                content = ' '.join([e.get('new_string', '') for e in edits])
            
            # Check for library imports without recent context7
            import re
            lib_patterns = [
                (r'import\s+(?!os|sys|json|re|time|pathlib|subprocess|shutil|tempfile|collections|itertools|functools|datetime|random|math|string|copy|hashlib|base64|urllib|socket|struct|pickle|csv|io|contextlib|warnings|traceback|inspect|types|typing|dataclasses|enum|abc|argparse|logging|configparser|platform|glob|fnmatch|stat|pwd|grp|termios|tty|pty|fcntl|select|signal|threading|multiprocessing|concurrent|queue|heapq|bisect|array|weakref|gc|atexit|builtins|__future__|__main__|unittest|doctest|pdb|profile|cProfile|timeit|trace|sys|os)\b', 'Python non-stdlib import'),
                (r'require\s*\(["\'](?!fs|path|os|util|crypto|http|https|url|querystring|stream|buffer|child_process|cluster|process|events|domain|readline|repl|vm|assert|dns|net|dgram|tls)\w+', 'Node.js package require'),
            ]
            
            for pattern, desc in lib_patterns:
                if re.search(pattern, content, re.MULTILINE):
                    # Check time since last context7
                    last_context7_file = f"/tmp/claude_last_context7_{ppid}"
                    time_since_context7 = float('inf')
                    
                    if os.path.exists(last_context7_file):
                        import time
                        with open(last_context7_file) as f:
                            last_time = float(f.read().strip())
                            time_since_context7 = time.time() - last_time
                    
                    if time_since_context7 > 300:  # 5 minutes
                        tracker.track_fabrication("library_without_context7", 
                                                f"{desc} in {tool_input.get('file_path', 'unknown')}")
                        response["message"] = (
                            f"⚠️ LIBRARY USAGE WITHOUT CONTEXT7 DETECTED\n"
                            f"   {desc}\n"
                            f"   Last context7: {int(time_since_context7/60)}m ago\n"
                            f"   💰 This will cost 5000+ tokens to debug!\n"
                            f"   REQUIRED: Use mcp__context7__ first!"
                        )
                    break
        
        # Check for edit without read pattern
        if tool_name in ['Edit', 'MultiEdit']:
            file_path = tool_input.get('file_path', '')
            last_read_file = f"/tmp/claude_last_read_{ppid}"
            
            # Check if we've read recently
            read_recently = False
            if os.path.exists(last_read_file):
                import time
                with open(last_read_file) as f:
                    last_time = float(f.read().strip())
                    if time.time() - last_time < 300:  # 5 minutes
                        read_recently = True
            
            # Check session for recent reads of this file
            session_file = f"/tmp/claude_session_{ppid}.json"
            if os.path.exists(session_file):
                with open(session_file) as f:
                    session = json.load(f)
                    recent_tools = session.get('tools', [])[-10:]  # Last 10 tools
                    for tool in recent_tools:
                        if tool.get('tool') == 'Read':
                            read_recently = True
                            break
            
            if not read_recently:
                tracker.track_fabrication("edit_without_read", f"Editing {file_path} without reading")
                if not response.get("allow", True):
                    # Already blocked, keep that message
                    pass
                else:
                    response["message"] = (
                        f"⚠️ EDITING WITHOUT READING DETECTED\n"
                        f"   File: {file_path}\n"
                        f"   This WILL cause errors!\n"
                        f"   REQUIRED: Read the file first!"
                    )
        
        # If blocked, exit with error
        if not response["allow"]:
            print(response["message"], file=sys.stderr)
            sys.exit(1)
        
        # If warning message, print it
        if response["message"]:
            print(response["message"], file=sys.stderr)
        
        # Generate and display current status
        status = tracker.get_injection_text()
        print(f"\n{status}\n", file=sys.stderr)
        
        # Let the tool proceed
        sys.exit(0)
        
    except Exception as e:
        # Log error but don't block tool execution
        print(f"Behavioral tracker error (non-blocking): {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()
