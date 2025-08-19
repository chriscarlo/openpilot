#!/usr/bin/env python3
"""
Pattern detection and tracking hook for PreToolUse
Detects library usage without context7, tracks tool usage patterns
"""

import sys
import json
import os
import time
import re
from pathlib import Path

def main():
    try:
        data = json.load(sys.stdin)
        tool = data.get('tool_name', '')
        tool_input = data.get('tool_input', {})
        
        # Get session ID
        ppid = os.getppid()
        session_file = f"/tmp/claude_session_{ppid}.json"
        violations_file = f"/tmp/claude_violations_{ppid}.txt"
        
        # Load session data
        session = {}
        if os.path.exists(session_file):
            try:
                with open(session_file) as f:
                    session = json.load(f)
            except:
                pass
        
        # Initialize session structure
        session.setdefault('tools', [])
        session.setdefault('last_context7', 0)
        session.setdefault('library_uses', [])
        
        # Track this tool usage
        session['tools'].append({
            'tool': tool,
            'time': time.time(),
            'timestamp': time.strftime('%H:%M:%S')
        })
        
        # Check for library usage patterns in Write/Edit operations
        if tool in ['Write', 'Edit', 'MultiEdit']:
            content = ''
            if tool == 'Write':
                content = tool_input.get('content', '')
            elif tool == 'Edit':
                content = tool_input.get('new_string', '')
            elif tool == 'MultiEdit':
                edits = tool_input.get('edits', [])
                content = ' '.join([e.get('new_string', '') for e in edits])
            
            # Library detection patterns
            lib_patterns = [
                (r'import\s+(?!os|sys|json|re|time|pathlib|subprocess|shutil|tempfile|collections|itertools|functools|datetime|random|math|string|copy|hashlib|base64|urllib|socket|struct|pickle|csv|io|contextlib|warnings|traceback|inspect|types|typing|dataclasses|enum|abc|argparse|logging|configparser|platform|glob|fnmatch|stat|pwd|grp|termios|tty|pty|fcntl|select|signal|threading|multiprocessing|concurrent|queue|heapq|bisect|array|weakref|gc|atexit|builtins|__future__|__main__|unittest|doctest|pdb|profile|cProfile|timeit|trace|sys|os)\b', 'Python non-stdlib import'),
                (r'from\s+(?!os|sys|json|re|time|pathlib|subprocess|shutil|tempfile|collections|itertools|functools|datetime|random|math|string|copy|hashlib|base64|urllib|socket|struct|pickle|csv|io|contextlib|warnings|traceback|inspect|types|typing|dataclasses|enum|abc|argparse|logging|configparser|platform|glob|fnmatch|stat|pwd|grp|termios|tty|pty|fcntl|select|signal|threading|multiprocessing|concurrent|queue|heapq|bisect|array|weakref|gc|atexit|builtins|__future__|__main__|unittest|doctest|pdb|profile|cProfile|timeit|trace)\s+\w+\s+import', 'Python non-stdlib from import'),
                (r'require\s*\(["\'](?!fs|path|os|util|crypto|http|https|url|querystring|stream|buffer|child_process|cluster|process|events|domain|readline|repl|vm|assert|dns|net|dgram|tls)\w+', 'Node.js package require'),
                (r'import\s+[{*].*from\s+["\'](?!\.|\@)', 'JavaScript ES6 import'),
                (r'use\s+(?!strict|warnings|vars|utf8|feature|constant|base|lib|Exporter|Data|File|IO|Time|POSIX|Carp|Scalar|List|Hash|Cwd|FindBin|Getopt|Pod|Test)\w+[;:]', 'Perl/Rust use statement'),
                (r'#include\s+[<"](?!stdio|stdlib|string|math|time|ctype|limits|float|stddef|stdint|stdbool|stdarg|assert|errno|signal|setjmp|locale|wchar|wctype|unistd|fcntl|sys/)', 'C/C++ non-standard include'),
                (r'using\s+(?!System|std)\w+;', 'C# using statement'),
                (r'import\s+(?!java\.lang|java\.util|java\.io)\w+', 'Java import'),
            ]
            
            # Check for library usage
            for pattern, desc in lib_patterns:
                if re.search(pattern, content, re.MULTILINE):
                    # Check time since last context7
                    time_since_ctx7 = time.time() - session.get('last_context7', 0)
                    
                    if time_since_ctx7 > 300:  # 5 minutes
                        violation_msg = f"{time.strftime('%H:%M:%S')}: {desc} without context7 (last: {int(time_since_ctx7/60)}m ago)"
                        
                        # Log violation
                        with open(violations_file, 'a') as f:
                            f.write(f"{violation_msg}\n")
                        
                        # Track in session
                        session['library_uses'].append({
                            'time': time.time(),
                            'pattern': desc,
                            'file': tool_input.get('file_path', 'unknown'),
                            'without_context7': True
                        })
                        
                        # Print warning with token cost
                        print(f"⚠️ LIBRARY USAGE WITHOUT CONTEXT7: {desc}", file=sys.stderr)
                        print(f"   File: {tool_input.get('file_path', 'unknown')}", file=sys.stderr)
                        print(f"   Last context7: {int(time_since_ctx7/60)} minutes ago", file=sys.stderr)
                        print(f"   💰 Guessing API = 100x debug tokens. Context7 lookup = 20 tokens.", file=sys.stderr)
                        print(f"   REQUIRED: Use mcp__context7__ to verify APIs first!", file=sys.stderr)
                    
                    break  # Only report first match
        
        # Update last context7 time if this is a context7 call
        if 'context7' in tool.lower():
            session['last_context7'] = time.time()
            print("✓ Context7 usage recorded", file=sys.stderr)
        
        # Check for TodoWrite usage
        if tool == 'TodoWrite':
            session['last_todo'] = time.time()
            Path(f"/tmp/claude_session_plan_{ppid}.txt").touch()
        
        # Warn if no TodoWrite in a while for complex operations
        if tool in ['Write', 'Edit', 'MultiEdit'] and session.get('last_todo', 0) == 0:
            print("⚠️ No TodoWrite used yet - consider creating a plan!", file=sys.stderr)
        
        # Save session
        with open(session_file, 'w') as f:
            json.dump(session, f, indent=2)
        
        # Let the tool proceed
        sys.exit(0)
        
    except Exception as e:
        # Log error but don't block tool execution
        print(f"Hook error (non-blocking): {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()
