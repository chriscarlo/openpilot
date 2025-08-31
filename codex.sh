#!/bin/bash
# Codex CLI Launch Script for AGNOS - Credential Authorization Only
# Designed for comma three (C3/C3X) hardware running AGNOS

set -e  # Exit on error

# Configuration
CODEX_AUTH_PORT=1455
NODE_VERSION="v22.18.0"
NVM_DIR="/data/persist/comma/nvm"
PERSIST_CODEX_DIR="/data/.codex"  # Persistent storage across reboots
HOME_CODEX_DIR="$HOME/.codex"

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
    # Setup environment
    setup_node_env
    setup_credential_storage
    verify_codex_binary
    
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
        # Launch with dangerous bypass flag for YOLO mode + GPT-5 high reasoning
        exec codex --dangerously-bypass-approvals-and-sandbox \
                   --model gpt-5 \
                   -c 'model_reasoning_effort="high"'
    else
        # If user provided arguments, pass them through with dangerous flag + GPT-5 high
        exec codex --dangerously-bypass-approvals-and-sandbox \
                   --model gpt-5 \
                   -c 'model_reasoning_effort="high"' \
                   "$@"
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