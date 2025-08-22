#!/usr/bin/env python3
"""
Track all tool usage to detect patterns and potential fabrication
This hook runs on EVERY tool call to build behavioral patterns
"""

import json
import os
import sys
import re
import time
from datetime import datetime

def main():
    try:
        # Get PPID for session tracking
        ppid = os.getppid()
        
        # Read tool input from stdin
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError:
            # No valid JSON, allow operation
            sys.exit(0)
        
        # Extract tool information
        tool_name = data.get('tool_name', '')
        tool_input = data.get('tool_input', {})
        
        # Track tool usage patterns
        pattern_file = f"/tmp/claude_patterns_{ppid}.json"
        
        # Load existing patterns
        if os.path.exists(pattern_file):
            with open(pattern_file, 'r') as f:
                patterns = json.load(f)
        else:
            patterns = {
                "session_start": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_frequency": {},
                "suspicious_patterns": [],
                "verification_rate": 0.0,
                "rush_indicators": 0
            }
        
        # Record this tool call
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "has_params": bool(tool_input)
        }
        patterns["tool_calls"].append(call_record)
        
        # Update frequency counter
        if tool_name not in patterns["tool_frequency"]:
            patterns["tool_frequency"][tool_name] = 0
        patterns["tool_frequency"][tool_name] += 1
        
        # Detect suspicious patterns
        
        # Pattern 1: Multiple edits without reads
        recent_tools = [c["tool"] for c in patterns["tool_calls"][-10:]]
        edit_tools = ["Edit", "Write", "MultiEdit"]
        read_tools = ["Read", "Grep", "LS"]
        
        edit_count = sum(1 for t in recent_tools if t in edit_tools)
        read_count = sum(1 for t in recent_tools if t in read_tools)
        
        if edit_count > 3 and read_count == 0:
            patterns["suspicious_patterns"].append({
                "type": "edits_without_reads",
                "timestamp": datetime.now().isoformat(),
                "details": f"{edit_count} edits, 0 reads in last 10 operations"
            })
            print("WARNING: Multiple edits without reading files!", file=sys.stderr)
        
        # Pattern 2: Library usage without documentation lookup
        if tool_name in edit_tools:
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            
            # Common library imports
            library_patterns = [
                (r'import\s+tensorflow', 'TensorFlow usage without docs'),
                (r'import\s+torch', 'PyTorch usage without docs'),
                (r'import\s+pandas', 'Pandas usage without docs'),
                (r'import\s+numpy', 'NumPy usage without docs'),
                (r'from\s+sklearn', 'Scikit-learn usage without docs'),
                (r'import\s+requests', 'Requests library without docs'),
                (r'import\s+flask', 'Flask usage without docs'),
                (r'import\s+django', 'Django usage without docs'),
                (r'require\(["\']express', 'Express.js usage without docs'),
                (r'require\(["\']react', 'React usage without docs'),
                (r'require\(["\']vue', 'Vue.js usage without docs'),
            ]
            
            for pattern, description in library_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Check if context7 was used recently
                    recent_context7 = any("context7" in str(c) for c in patterns["tool_calls"][-20:])
                    if not recent_context7:
                        patterns["suspicious_patterns"].append({
                            "type": "library_without_docs",
                            "timestamp": datetime.now().isoformat(),
                            "details": description
                        })
                        print(f"WARNING: {description}", file=sys.stderr)
                    break
        
        # Pattern 3: Rush indicators (rapid tool calls)
        if len(patterns["tool_calls"]) >= 2:
            last_time = datetime.fromisoformat(patterns["tool_calls"][-2]["timestamp"])
            current_time = datetime.fromisoformat(call_record["timestamp"])
            time_diff = (current_time - last_time).total_seconds()
            
            if time_diff < 2:  # Less than 2 seconds between calls
                patterns["rush_indicators"] += 1
                if patterns["rush_indicators"] > 5:
                    print("WARNING: Slow down! Rapid tool usage detected.", file=sys.stderr)
        
        # Pattern 4: Claims without verification
        if tool_name in ["Write", "Edit"]:
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            
            # Patterns indicating claims
            claim_patterns = [
                r'#.*[Tt]his\s+(will|should|must|does)\s+',
                r'#.*[Aa]lways\s+',
                r'#.*[Nn]ever\s+',
                r'#.*[Gg]uaranteed\s+',
                r'#.*[Dd]efinitely\s+',
            ]
            
            for pattern in claim_patterns:
                if re.search(pattern, content):
                    # Check if there was recent verification
                    recent_verification = any(
                        c["tool"] in ["Read", "Grep", "WebSearch", "WebFetch"] 
                        for c in patterns["tool_calls"][-5:]
                    )
                    if not recent_verification:
                        patterns["suspicious_patterns"].append({
                            "type": "unverified_claim",
                            "timestamp": datetime.now().isoformat(),
                            "details": "Making claims without recent verification"
                        })
                        print("WARNING: Making claims without verification!", file=sys.stderr)
                    break
        
        # Calculate verification rate
        total_calls = len(patterns["tool_calls"])
        if total_calls > 0:
            verification_tools = ["Read", "Grep", "LS", "WebSearch", "WebFetch"]
            verification_count = sum(
                1 for c in patterns["tool_calls"] 
                if c["tool"] in verification_tools
            )
            patterns["verification_rate"] = (verification_count / total_calls) * 100
        
        # Save patterns
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        # Keep only last 100 tool calls to prevent file growth
        if len(patterns["tool_calls"]) > 100:
            patterns["tool_calls"] = patterns["tool_calls"][-100:]
            with open(pattern_file, 'w') as f:
                json.dump(patterns, f, indent=2)
        
        # Report if verification rate is too low
        if total_calls > 20 and patterns["verification_rate"] < 30:
            print(f"LOW VERIFICATION RATE: {patterns['verification_rate']:.1f}%", file=sys.stderr)
            print("Remember: Read and verify before writing!", file=sys.stderr)
        
        # Allow the operation to proceed
        sys.exit(0)
        
    except Exception as e:
        # Don't block on errors
        print(f"Tool tracking error: {e}", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()