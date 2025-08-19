# Architecture Mismatch Fix - UI Crash on tici

## Problem
The UI process was crashing with SIGSEGV on the tici device (ARM aarch64) after recent changes.

## Root Cause
Binary files compiled for x86-64 architecture in WSL development environment were being used on ARM device:
- `third_party/libjson11.a` was compiled for x86-64
- Various `.o` object files in `selfdrive/ui/` were x86-64
- `moc_*.cc` generated files and their `.o` files were x86-64

## Solution
1. Removed `third_party/libjson11.a` (already in .gitignore)
2. Removed all `.o` object files from `selfdrive/ui/`
3. Removed all `moc_*.cc` generated files from `selfdrive/ui/`

These files will be automatically rebuilt with the correct architecture when compiling on the target device.

## Prevention
- Never commit binary files (`.o`, `.a`, `.so`) unless they're in architecture-specific directories
- Always clean build artifacts before committing: `scons -c`
- Keep `.gitignore` updated with all build artifacts
- When developing in WSL/x86, be aware that binaries are not portable to ARM devices

## Verification
Check architecture of any binary file:
```bash
file <filename>
```
Should show `aarch64` for tici, not `x86-64`