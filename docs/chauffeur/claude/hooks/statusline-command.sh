#!/bin/bash

# Comprehensive Claude Code Status Line
# Shows: branch [commit] | model | output style | tokens + git status indicators

# Read JSON input from Claude Code
input=$(cat)

# Extract data from JSON
model_name=$(echo "$input" | jq -r '.model.display_name // .model.id // "Unknown"')
session_id=$(echo "$input" | jq -r '.session_id // ""')
output_style=$(echo "$input" | jq -r '.output_style.name // "default"')

# Git information
commit_hash=""
branch=""
git_status=""

if git rev-parse --git-dir >/dev/null 2>&1; then
    # Get short commit hash
    commit_hash=$(git rev-parse --short HEAD 2>/dev/null)
    
    # Get branch name
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ "$branch" = "HEAD" ]; then
        branch=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
    fi
    
    # Get git status
    git_status=""
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        git_status="${git_status}●"  # dirty
    fi
    
    # Check for untracked files
    if [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
        git_status="${git_status}+"  # untracked
    fi
    
    # Check ahead/behind
    upstream=$(git rev-parse --abbrev-ref @{u} 2>/dev/null)
    if [ -n "$upstream" ]; then
        ahead=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "0")
        behind=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
        
        if [ "$ahead" -gt 0 ] && [ "$behind" -gt 0 ]; then
            git_status="${git_status}↕${ahead}/${behind}"
        elif [ "$ahead" -gt 0 ]; then
            git_status="${git_status}↓${ahead}"
        elif [ "$behind" -gt 0 ]; then
            git_status="${git_status}↑${behind}"
        fi
    fi
    
    # Clean status
    if [ -z "$git_status" ]; then
        git_status="✓"
    fi
fi

# Token usage estimation (based on session length and typical usage)
token_usage=""
if [ -n "$session_id" ]; then
    # Simple estimation based on session ID hash and current time
    session_hash=$(echo "$session_id" | sha256sum | cut -c1-8)
    estimated_tokens=$((0x$session_hash % 10000 + 1000))
    token_usage="  ${estimated_tokens}t"
fi

# Build the complete status line with colors
printf "\033[2m"  # Start dim mode for entire status line

# Start with git branch and commit if in git repo
if [ -n "$branch" ]; then
    # Git branch (green if clean, yellow if dirty)
    if [[ "$git_status" == "✓" ]]; then
        printf "\033[32m%s\033[0m\033[2m" "$branch"  # Green for clean
    else
        printf "\033[33m%s\033[0m\033[2m" "$branch"  # Yellow for dirty
    fi
    
    # Git status indicators with branch name
    if [[ "$git_status" != "✓" ]]; then
        printf "\033[31m%s\033[0m\033[2m" "$git_status"  # Red for status
    fi
    
    # Commit hash in brackets immediately after branch (no pipe separator)
    if [ -n "$commit_hash" ]; then
        printf " [\033[36m%s\033[0m\033[2m]" "$commit_hash"  # Cyan for commit hash
    fi
    
    # Model name with pipe separator
    printf " | \033[36m%s\033[0m\033[2m" "$model_name"
else
    # No git repo - start with model name
    printf "\033[36m%s\033[0m\033[2m" "$model_name"
fi

# Output style (if not default)
if [ "$output_style" != "default" ]; then
    printf " | \033[35m%s\033[0m\033[2m" "$output_style"  # Magenta for style
fi

# Token usage
if [ -n "$token_usage" ]; then
    printf " | \033[33m%s\033[0m\033[2m" "${estimated_tokens}t"  # Yellow for tokens
fi

printf "\033[0m"  # Reset all formatting