#!/usr/bin/env bash
set -e
rm -f /tmp/claude_*_${PPID}* 2>/dev/null || true
echo "$(date): Session ${PPID} started with ENFORCEMENT ACTIVE" >> ~/.claude/enforcement.log
cat <<'MSG'
🔴 ANTI-FABRICATION ENFORCEMENT INITIALIZED
• READ before claiming
• Context7 for APIs
• WebSearch for facts
• TodoWrite for planning

💰 CRITICAL: Every shortcut = 10-100x more tokens to fix.
📊 Real math: Guess (5 tokens) → Debug (500+ tokens) → User frustration (5000+ tokens)
MSG
