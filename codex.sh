#!/bin/bash
# OpenAI Codex CLI Launch Script for TICI
# Modeled after claude.sh for consistent environment setup

# Remount filesystems for write access
sudo mount -o remount,rw /
sudo mount -o remount,rw,exec /persist

# Set up SSH for git operations (if needed by Codex)
export GIT_SSH_COMMAND="ssh -i /persist/ssh/git_keys/claude_github_key -o StrictHostKeyChecking=no"

# Source the Claude environment setup (contains NVM/Node setup)
if [ -f "/persist/comma/claude/setup_claude_env.sh" ]; then
    source /persist/comma/claude/setup_claude_env.sh
fi

# Ensure Node/npm are in PATH
export NVM_DIR="/data/persist/comma/nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

# Add Node to PATH if NVM isn't loaded but Node exists
if ! command -v node >/dev/null 2>&1; then
    NODE_BIN=$(find "$NVM_DIR/versions/node" -name bin -type d 2>/dev/null | head -1)
    if [ -n "$NODE_BIN" ]; then
        export PATH="$NODE_BIN:$PATH"
    fi
fi

# Verify Codex is accessible
if command -v codex >/dev/null 2>&1; then
    echo "OpenAI Codex CLI environment loaded successfully"
    echo "Launching Codex CLI..."
    codex "$@"
else
    echo "Error: Codex CLI not found in PATH"
    echo "Please ensure Codex is installed by running:"
    echo "  sudo -E env PATH=\"/data/persist/comma/nvm/versions/node/v22.18.0/bin:\$PATH\" npm install -g @openai/codex"
    exit 1
fi