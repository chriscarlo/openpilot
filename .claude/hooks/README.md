# Openpilot Claude Code Hooks

## Overview

This directory contains Claude Code hooks specifically tailored for openpilot development. These hooks help ensure code quality and prevent common mistakes by:

- Tracking which files have been read before editing
- Reminding to run tests and linters after making changes
- Running automatic linting on edited Python files
- Maintaining session state for better context awareness

## Hooks Included

### 1. **session_start.sh** (SessionStart)
- Initializes development session
- Shows current git branch
- Provides quick reminders about common commands

### 2. **track_read.py** (PreToolUse - Read)
- Tracks all files that have been read
- Maintains a session log of read operations
- Enables verification for edit operations

### 3. **check_edit_read.py** (PreToolUse - Edit/Write)
- Verifies files have been read before editing
- Blocks edits on unread files to prevent blind modifications
- Allows creation of new files without reading

### 4. **run_linters.py** (PostToolUse - Edit/Write)
- Automatically runs ruff and mypy on edited Python files
- Provides immediate feedback on code quality issues
- Updates session state to track linting status

### 5. **prompt_inject.sh** (UserPromptSubmit)
- Injects verification reminders into prompts
- Shows session statistics (files read/edited)
- Reminds to run tests and linters when needed

## Installation

### Quick Install (Local to Project)

The hooks are already configured in `.claude/hooks/`. To use them:

1. These hooks are project-specific and stored in the repository
2. They will be automatically available when working on this project
3. No global installation needed

### Global Install (For All Projects)

To use these hooks globally in Claude Code:

```bash
# Copy settings to Claude config directory
cp .claude/hooks/settings.json ~/.claude/settings.json

# Copy hook scripts
cp .claude/hooks/*.sh ~/.claude/hooks/
cp .claude/hooks/*.py ~/.claude/hooks/

# Make scripts executable
chmod +x ~/.claude/hooks/*.sh
chmod +x ~/.claude/hooks/*.py

# Restart Claude Code to activate
```

## Testing

Run the test suite to verify hooks are working:

```bash
python3 .claude/hooks/test_hooks.py
```

Expected output:
- ✅ All hooks should execute successfully
- ✅ Session initialization confirmed
- ✅ Read tracking functional
- ✅ Edit blocking working
- ✅ Linters running

## How It Works

### Session Tracking

Each Claude Code session maintains a JSON file at `/tmp/openpilot_session_$$.json` containing:
- List of files read
- List of files edited
- Test run status
- Linter run status
- Current git branch

### Read-Before-Edit Verification

The system enforces that files must be read before editing:
1. `track_read.py` logs each file read operation
2. `check_edit_read.py` verifies the file was read before allowing edits
3. New files can be created without reading

### Automatic Linting

When Python files are edited:
1. `run_linters.py` automatically runs ruff for style checks
2. Optionally runs mypy for type checking (if available)
3. Results are displayed immediately
4. Session tracks that linting has occurred

### Smart Reminders

The `prompt_inject.sh` hook monitors your session and injects reminders when:
- Files have been edited but tests haven't run
- Files have been edited but linter hasn't run
- Provides session statistics for awareness

## Customization

### Adjusting Linter Settings

Edit `run_linters.py` to:
- Add more linters (e.g., pylint, black)
- Change timeout values
- Modify output formatting

### Changing Reminder Thresholds

Edit `prompt_inject.sh` to:
- Adjust when reminders appear
- Customize reminder messages
- Add project-specific checks

### Adding New Checks

Create new hooks by:
1. Adding new script to `.claude/hooks/`
2. Registering in `settings.json`
3. Following the pattern of existing hooks

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Hooks not firing | Check that scripts are executable: `chmod +x .claude/hooks/*.sh` |
| Session data missing | Session file may have been cleaned up, restart Claude Code |
| False edit blocks | Check `/tmp/openpilot_session_*.json` for read history |
| Linter errors | Ensure ruff is installed: `pip install ruff` |

## Best Practices

1. **Always read before editing** - This ensures you understand the code context
2. **Run tests frequently** - Use `pytest` after making changes
3. **Lint before committing** - Run `./scripts/lint/lint.sh` for full linting
4. **Build to verify** - Use `scons -u -j$(nproc)` to ensure code compiles

## Integration with Openpilot Workflow

These hooks complement the openpilot development workflow:

- **RTI Development**: Automatically checks Python formatting in RTI modules
- **Safety**: Prevents blind edits that could affect safety-critical code
- **Quality**: Maintains code standards through automatic linting
- **Testing**: Reminds to run tests for modified components

## Contributing

To improve these hooks:
1. Test changes with `test_hooks.py`
2. Document new features in this README
3. Ensure backward compatibility
4. Follow existing code patterns

## License

These hooks are part of the openpilot project and follow the same licensing terms.