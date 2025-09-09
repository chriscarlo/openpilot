#!/bin/bash
# Codex CLI Launch Script for AGNOS (+ tmux persistent session helper)
# Designed for comma three (C3/C3X) hardware running AGNOS

set -e  # Exit on error

# Configuration
CODEX_AUTH_PORT=1455
NODE_VERSION="v22.18.0"
NVM_DIR="/data/persist/comma/nvm"
PERSIST_CODEX_DIR="/data/.codex"  # Persistent storage across reboots
HOME_CODEX_DIR="$HOME/.codex"
CODEX_TMUX_SESSION_NAME="${CODEX_TMUX_SESSION_NAME:-codex}"
CODEX_TMUX_SENTINEL="${CODEX_TMUX_SENTINEL:-}"
CODEX_TMUX_CONF_LINK="$HOME/.tmux.conf"
CODEX_TMUX_CONF_PERSIST="$PERSIST_CODEX_DIR/tmux.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_auth() { echo -e "${CYAN}[AUTH]${NC} $1"; }

# Banner
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "          Codex CLI for AGNOS - Credential Auth Only"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# --- tmux helpers (persistent user session across SSH disconnects) ---

ensure_tmux_available() {
    if command -v tmux >/dev/null 2>&1; then
        return 0
    fi
    print_warn "tmux is not installed; persistence on disconnect won't work"
    echo ""
    echo "To install tmux on Ubuntu/AGNOS-based systems:"
    echo "  sudo apt-get update && sudo apt-get install -y tmux"
    echo ""
    return 1
}

setup_tmux_config() {
    # Ensure persistent dir and home symlink for codex (reused below)
    [ -d "$PERSIST_CODEX_DIR" ] || mkdir -p "$PERSIST_CODEX_DIR"

    # Create a sane default tmux config in persistent storage if missing
    if [ ! -f "$CODEX_TMUX_CONF_PERSIST" ]; then
        cat > "$CODEX_TMUX_CONF_PERSIST" <<'TMUXCONF'
# Minimal, readable defaults for device work
# Defaults favor standard tmux behavior; iOS overrides appended below
set -g mouse on
set -g history-limit 500000
set -g assume-paste-time 10
set -g focus-events on
setw -g remain-on-exit on
# Keep terminal colors predictable
set -g default-terminal "screen-256color"
set -as terminal-overrides ',xterm-256color:RGB'
TMUXCONF
    fi

    # Ensure smooth scroll/copy-mode helpers exist (idempotent upsert)
    local begin_marker="# BEGIN CODEX_SCROLL_HELPERS"
    local end_marker="# END CODEX_SCROLL_HELPERS"
    if grep -qF "$begin_marker" "$CODEX_TMUX_CONF_PERSIST" 2>/dev/null; then
        # Strip old block
        awk -v bgn="$begin_marker" -v end="$end_marker" '
          BEGIN{skip=0}
          $0==bgn{skip=1; next}
          $0==end{skip=0; next}
          skip==0{print $0}
        ' "$CODEX_TMUX_CONF_PERSIST" > "$CODEX_TMUX_CONF_PERSIST.tmp" && mv "$CODEX_TMUX_CONF_PERSIST.tmp" "$CODEX_TMUX_CONF_PERSIST"
    fi
    cat >> "$CODEX_TMUX_CONF_PERSIST" <<'TMUXHELP'
# BEGIN CODEX_SCROLL_HELPERS
# Scrollwheel auto-enters copy-mode; PageUp/PageDown work intuitively
setw -g mode-keys vi
set -g mouse on
bind -T root WheelUpPane if -F '#{pane_in_mode}' 'send-keys -M' 'copy-mode -e; send-keys -M'
bind -T root WheelDownPane if -F '#{pane_in_mode}' 'send-keys -M' 'send-keys -M'
bind -T copy-mode-vi WheelUpPane send -N 1 -X scroll-up
bind -T copy-mode-vi WheelDownPane send -N 1 -X scroll-down
bind -T copy-mode WheelUpPane send -N 1 -X scroll-up
bind -T copy-mode WheelDownPane send -N 1 -X scroll-down
bind -n S-PageUp copy-mode -e
bind -T copy-mode-vi S-PageUp send -X page-up
bind -T copy-mode-vi S-PageDown send -X page-down
bind -T copy-mode S-PageUp send -X page-up
bind -T copy-mode S-PageDown send -X page-down
# Keep a large scrollback
set -g history-limit 500000

# Note: Alternate screen is disabled by default elsewhere in this config so
# your terminal's own scrollback works naturally inside tmux.
# END CODEX_SCROLL_HELPERS
TMUXHELP

    # Minimal block (idempotent upsert): disable mouse, status bar, and alt screen
    local ios_begin="# BEGIN CODEX_IOS_MINIMAL"
    local ios_end="# END CODEX_IOS_MINIMAL"
    if grep -qF "$ios_begin" "$CODEX_TMUX_CONF_PERSIST" 2>/dev/null; then
        awk -v bgn="$ios_begin" -v end="$ios_end" '
          BEGIN{skip=0}
          $0==bgn{skip=1; next}
          $0==end{skip=0; next}
          skip==0{print $0}
        ' "$CODEX_TMUX_CONF_PERSIST" > "$CODEX_TMUX_CONF_PERSIST.tmp" && mv "$CODEX_TMUX_CONF_PERSIST.tmp" "$CODEX_TMUX_CONF_PERSIST"
    fi
    cat >> "$CODEX_TMUX_CONF_PERSIST" <<'TMUXIOS'
# BEGIN CODEX_IOS_MINIMAL
# Make tmux behave like a normal shell in iOS SSH apps
set -g status on
set -g mouse off
set -g history-limit 100000
setw -g alternate-screen off
set -ga terminal-overrides ",xterm*:smcup@:rmcup@"
# END CODEX_IOS_MINIMAL
TMUXIOS

    # Home is ephemeral; point ~/.tmux.conf at persistent copy every run
    if [ -L "$CODEX_TMUX_CONF_LINK" ] || [ -e "$CODEX_TMUX_CONF_LINK" ]; then
        rm -f "$CODEX_TMUX_CONF_LINK" || true
    fi
    ln -s "$CODEX_TMUX_CONF_PERSIST" "$CODEX_TMUX_CONF_LINK" 2>/dev/null || true
}

tmux_start_or_attach() {
    # Skip if already inside tmux or sentinel active
    if [ -n "$TMUX" ] || [ -n "$CODEX_TMUX_SENTINEL" ]; then
        return 0
    fi

    # Only attach/create when running interactively (TTY present)
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        return 0
    fi

    # Ensure tmux exists; if not, just warn and continue without tmux
    if ! ensure_tmux_available; then
        return 0
    fi

    # Create persistent config/symlink quietly
    setup_tmux_config || true

    # Helper to enforce runtime tmux options (applies even on existing servers)
    apply_tmux_runtime_options() {
        local hist_limit="${CODEX_TMUX_HISTORY_LIMIT:-500000}"
        # Base settings
        tmux set -g focus-events on 2>/dev/null || true
        tmux setw -g remain-on-exit on 2>/dev/null || true
        tmux set -g history-limit "$hist_limit" 2>/dev/null || true
        tmux set -g default-terminal "screen-256color" 2>/dev/null || true
        tmux set -as terminal-overrides ',xterm-256color:RGB' 2>/dev/null || true

        # Always-on minimal behavior for better mobile SSH scrolling
        tmux set -g mouse off 2>/dev/null || true
        tmux set -g status on 2>/dev/null || true
        tmux setw -g alternate-screen off 2>/dev/null || true
        tmux set -ga terminal-overrides ',*:smcup@:rmcup@' 2>/dev/null || true
        # Make sure subshells know alternate screen is disabled
        tmux set-environment -g CODEX_TMUX_DISABLE_ALTERNATE_SCREEN 1 2>/dev/null || true

        # Also (re)source user config in case it changed
        tmux source-file "$CODEX_TMUX_CONF_LINK" 2>/dev/null || true
    }

    # Attach if session exists; create otherwise and run this script inside it
    if tmux has-session -t "$CODEX_TMUX_SESSION_NAME" 2>/dev/null; then
        print_info "Attaching to tmux session '$CODEX_TMUX_SESSION_NAME'"
        apply_tmux_runtime_options || true
        exec tmux attach -t "$CODEX_TMUX_SESSION_NAME"
    else
        print_info "Creating tmux session '$CODEX_TMUX_SESSION_NAME' and attaching"
        # Run this script inside the new session, marked with sentinel to avoid recursion
        # Pass through original arguments.
        tmux -f "$CODEX_TMUX_CONF_LINK" new-session -d -s "$CODEX_TMUX_SESSION_NAME" \
            env CODEX_TMUX_SENTINEL=1 "$0" "$@"
        # Ensure the brand-new server has desired options as well
        apply_tmux_runtime_options || true
        exec tmux attach -t "$CODEX_TMUX_SESSION_NAME"
    fi
}

# Check SSH port forwarding for OAuth callback
check_ssh_forwarding() {
    # Skip port forwarding check if already authenticated
    if codex login status >/dev/null 2>&1; then
        return 0
    fi
    
    if [ -n "$SSH_CONNECTION" ]; then
        # Check if port is forwarded without using grep
        PORT_FORWARDED=false
        if command -v ss >/dev/null 2>&1; then
            if ss -tln 2>/dev/null | awk '{print $4}' | cut -d: -f2 | awk -v port="$CODEX_AUTH_PORT" '$1==port {exit 0} END {exit 1}'; then
                PORT_FORWARDED=true
            fi
        elif command -v netstat >/dev/null 2>&1; then
            if netstat -tln 2>/dev/null | awk '{print $4}' | cut -d: -f2 | awk -v port="$CODEX_AUTH_PORT" '$1==port {exit 0} END {exit 1}'; then
                PORT_FORWARDED=true
            fi
        fi
        
        if [ "$PORT_FORWARDED" = "true" ]; then
            print_info "Port $CODEX_AUTH_PORT is forwarded for OAuth authentication ✓"
        else
            print_warn "SSH detected but port $CODEX_AUTH_PORT not forwarded!"
            echo ""
            echo "  ⚠️  REQUIRED FOR AUTHENTICATION:"
            echo ""
            echo "  Reconnect with port forwarding:"
            echo "    ssh -L $CODEX_AUTH_PORT:localhost:$CODEX_AUTH_PORT comma@<device-ip>"
            echo ""
            echo "  Or use Termius iOS with Local Port Forward:"
            echo "    Local Port: $CODEX_AUTH_PORT"
            echo "    Intermediate Host: <your-comma3x-ip>"
            echo "    Destination: localhost:$CODEX_AUTH_PORT"
            echo ""
            read -p "Press Enter to continue anyway, or Ctrl+C to exit and reconnect... "
        fi
    else
        print_info "Local session detected (not SSH)"
    fi
}

# Setup Node.js environment
setup_node_env() {
    print_step "Setting up Node.js environment..."
    
    # Load NVM if available
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        export NVM_DIR="$NVM_DIR"
        . "$NVM_DIR/nvm.sh"
        print_info "NVM environment loaded"
    fi
    
    # Ensure node is in PATH
    if ! command -v node >/dev/null 2>&1; then
        NODE_BIN="$NVM_DIR/versions/node/$NODE_VERSION/bin"
        if [ -d "$NODE_BIN" ]; then
            export PATH="$NODE_BIN:$PATH"
            print_info "Added Node.js $NODE_VERSION to PATH"
        else
            print_error "Node.js not found at expected location: $NODE_BIN"
            print_error "Please ensure Node.js is installed via NVM"
            exit 1
        fi
    fi
    
    # Verify Node.js
    NODE_ACTUAL=$(node --version 2>/dev/null || echo "unknown")
    print_info "Node.js version: $NODE_ACTUAL"
}

# Install auto-attach-on-SSH snippet into ~/.bashrc (ephemeral per boot)
install_ssh_auto_attach() {
    local bashrc="$HOME/.bashrc"
    local begin_marker="# BEGIN CODEX_TMUX_AUTO_ATTACH"
    local end_marker="# END CODEX_TMUX_AUTO_ATTACH"

    # Make sure file exists
    if [ ! -f "$bashrc" ]; then
        touch "$bashrc"
    fi

    # Build snippet
    local snippet
    read -r -d '' snippet <<'SNIP'
# BEGIN CODEX_TMUX_AUTO_ATTACH
# Auto-attach/create a tmux session on SSH interactive logins
if [ -n "$SSH_TTY" ] && [ -z "$TMUX" ] && [ -t 0 ] && [ -t 1 ]; then
  # Allow disabling with a persistent flag file
  if [ -f /persist/ssh/disable_tmux_auto_attach ] || [ -f /persist/disable_tmux_auto_attach ]; then
    : # disabled by flag
  elif command -v tmux >/dev/null 2>&1; then
    sess="${CODEX_TMUX_SESSION_NAME:-codex}"
    # Use our config file when starting the server the first time
    if tmux has-session -t "$sess" 2>/dev/null; then
      exec tmux attach -t "$sess"
    else
      # On first creation, immediately run Codex launcher inside tmux
      # so sessions start in YOLO mode automatically.
      exec tmux -f "$HOME/.tmux.conf" new -As "$sess" \
        /data/openpilot/codex.sh
    fi
  fi
fi
# END CODEX_TMUX_AUTO_ATTACH
SNIP

    # Remove any existing block, then append fresh block
    if grep -q -F "$begin_marker" "$bashrc" >/dev/null 2>&1; then
        # Use awk to strip old block safely
        awk -v bgn="$begin_marker" -v end="$end_marker" '
          BEGIN{skip=0}
          $0==bgn{skip=1; next}
          $0==end{skip=0; next}
          skip==0{print $0}
        ' "$bashrc" > "$bashrc.tmp" && mv "$bashrc.tmp" "$bashrc"
    fi
    {
      echo "";
      echo "$snippet";
    } >> "$bashrc"
}

# Setup persistent credential storage (handles ephemeral home directory)
setup_credential_storage() {
    print_step "Setting up persistent credential storage..."
    
    # CRITICAL: Home directory is ephemeral (tmpfs) on AGNOS
    # We MUST recreate the symlink every time the script runs
    
    # Ensure persistent directory exists in /data (survives reboots)
    if [ ! -d "$PERSIST_CODEX_DIR" ]; then
        mkdir -p "$PERSIST_CODEX_DIR"
        chmod 700 "$PERSIST_CODEX_DIR"
        print_info "Created persistent credential storage at $PERSIST_CODEX_DIR"
    else
        print_info "Using existing credential storage at $PERSIST_CODEX_DIR"
    fi
    
    # ALWAYS recreate symlink (home is ephemeral, gets wiped on reboot)
    if [ -e "$HOME_CODEX_DIR" ] || [ -L "$HOME_CODEX_DIR" ]; then
        # Remove whatever is there (could be stale symlink or directory)
        rm -rf "$HOME_CODEX_DIR"
    fi
    
    # Create fresh symlink to persistent storage
    ln -s "$PERSIST_CODEX_DIR" "$HOME_CODEX_DIR"
    print_info "Created symlink: $HOME_CODEX_DIR → $PERSIST_CODEX_DIR (persistent)"
    
    # Verify the symlink was created correctly
    if [ -L "$HOME_CODEX_DIR" ] && [ "$(readlink "$HOME_CODEX_DIR")" = "$PERSIST_CODEX_DIR" ]; then
        print_info "✓ Credential persistence configured correctly"
    else
        print_error "Failed to create symlink for credential persistence!"
        exit 1
    fi
}

# Verify Codex binary
verify_codex_binary() {
    print_step "Verifying Codex installation..."
    
    if ! command -v codex >/dev/null 2>&1; then
        print_error "Codex CLI not found in PATH"
        echo ""
        echo "  To install Codex:"
        echo "    npm install -g @openai/codex"
        echo ""
        echo "  Or with full path:"
        echo "    $NVM_DIR/versions/node/$NODE_VERSION/bin/npm install -g @openai/codex"
        echo ""
        exit 1
    fi
    
    # Get version
    CODEX_VERSION=$(codex --version 2>&1 || echo "unknown")
    print_info "Codex version: $CODEX_VERSION"
    
    # Verify it's the correct binary for our architecture
    ARCH=$(arch)
    print_info "System architecture: $ARCH"
}

# Check authentication status
check_auth_status() {
    print_step "Checking authentication status..."
    
    if codex login status >/dev/null 2>&1; then
        print_auth "✓ Already authenticated with credentials"
        echo ""
        CODEX_USER=$(codex login status 2>&1 | awk '/Logged in as|User:/ {gsub(/.*: /, ""); print}' || echo "unknown")
        if [ "$CODEX_USER" != "unknown" ]; then
            print_info "Authenticated as: $CODEX_USER"
        fi
    else
        print_auth "⚠️  Not authenticated - credential login required"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "                   CREDENTIAL AUTHENTICATION"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  To authenticate with credentials:"
        echo ""
        echo "  1. Ensure SSH port forwarding is active (see above)"
        echo "  2. Run: codex login"
        echo "  3. Open the URL in your LOCAL browser (not on device)"
        echo "  4. Complete OAuth flow with your OpenAI account"
        echo "  5. The callback will return to localhost:$CODEX_AUTH_PORT"
        echo ""
        echo "  The auth server will listen on http://localhost:$CODEX_AUTH_PORT"
        echo "  for the OAuth callback from auth.openai.com"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Offer to start login
        echo ""
        read -p "Start credential login now? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_auth "Starting OAuth credential flow..."
            echo ""
            codex login
            echo ""
            
            # Check if login succeeded
            if codex login status >/dev/null 2>&1; then
                print_auth "✓ Authentication successful!"
            else
                print_error "Authentication may have failed. Please try again."
            fi
        fi
    fi
}

# Main execution
main() {
    # If not already in tmux, create/attach persistent session first.
    # This will exec tmux and not return in normal usage; when a new
    # session is created, this script is re-run inside it with sentinel set.
    tmux_start_or_attach "$@"

    # Setup environment
    setup_node_env
    setup_credential_storage
    verify_codex_binary
    # Install auto-attach on SSH for future logins this boot
    install_ssh_auto_attach || true
    
    # Check SSH forwarding
    check_ssh_forwarding
    
    # Check auth
    check_auth_status
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    print_info "Codex environment ready"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    
    # Launch Codex with provided arguments
    if [ $# -eq 0 ]; then
        print_info "Starting interactive Codex session in YOLO mode..."
        print_warn "Running with --dangerously-bypass-approvals-and-sandbox"
        print_info "Using GPT-5 with high reasoning effort"
        echo ""
        echo "  Tip: Use 'codex --help' for all options"
        echo "       Use 'codex logout' to remove stored credentials"
        echo ""
        if [ -n "$TMUX" ] || [ -n "$CODEX_TMUX_SENTINEL" ]; then
            # Inside tmux: run Codex, then drop to interactive shell so the window stays useful
            codex --dangerously-bypass-approvals-and-sandbox \
                  --model gpt-5 \
                  -c 'model_reasoning_effort="high"'
            echo ""
            print_info "Codex session ended. Dropping to interactive shell inside tmux."
            exec bash -l
        else
            # Outside tmux: replace process
            exec codex --dangerously-bypass-approvals-and-sandbox \
                       --model gpt-5 \
                       -c 'model_reasoning_effort="high"'
        fi
    else
        # If user provided arguments, pass them through with dangerous flag + GPT-5 high
        if [ -n "$TMUX" ] || [ -n "$CODEX_TMUX_SENTINEL" ]; then
            codex --dangerously-bypass-approvals-and-sandbox \
                  --model gpt-5 \
                  -c 'model_reasoning_effort="high"' \
                  "$@"
            echo ""
            print_info "Command finished. Dropping to interactive shell inside tmux."
            exec bash -l
        else
            exec codex --dangerously-bypass-approvals-and-sandbox \
                       --model gpt-5 \
                       -c 'model_reasoning_effort="high"' \
                       "$@"
        fi
    fi
}

# Handle script arguments
case "${1:-}" in
    auth|login)
        # Direct auth command
        setup_node_env
        setup_credential_storage
        check_ssh_forwarding
        echo ""
        print_auth "Starting credential authentication flow..."
        echo ""
        exec codex login "${@:2}"
        ;;
    logout)
        # Logout command
        setup_node_env
        print_auth "Removing stored credentials..."
        exec codex logout "${@:2}"
        ;;
    status)
        # Check status
        setup_node_env
        exec codex login status
        ;;
    *)
        # Normal execution
        main "$@"
        ;;
esac
