# Submodule Flattening Plan for chriscarlo/openpilot

## Overview
This plan converts all git submodules to regular directories while preserving the EXACT directory structure and symlinks. No paths or symlinks will change - only the git submodule machinery will be removed.

## Key Principle
**NOTHING MOVES** - All directories stay exactly where they are, all symlinks remain unchanged. We only convert submodule directories into regular directories with actual files.

## Current Structure (WILL NOT CHANGE)
```
/data/openpilot/
├── msgq -> msgq_repo/msgq (symlink stays)
├── msgq_repo/ (converts from submodule to regular directory)
├── opendbc -> opendbc_repo/opendbc (symlink stays)
├── opendbc_repo/ (converts from submodule to regular directory)
├── panda/ (converts from submodule to regular directory)
├── rednose -> rednose_repo/rednose (symlink stays)
├── rednose_repo/ (converts from submodule to regular directory)
├── teleoprtc -> teleoprtc_repo/teleoprtc (symlink stays)
├── teleoprtc_repo/ (converts from submodule to regular directory)
├── tinygrad -> tinygrad_repo/tinygrad (symlink stays)
├── tinygrad_repo/ (converts from submodule to regular directory)
└── sunnypilot/neural_network_data/ (converts from submodule to regular directory)
```

## Step-by-Step Process

### Phase 1: Preparation
```bash
# Create new branch
git checkout -b chauffeur-dev2-flattened

# Ensure all submodules are fully initialized
git submodule update --init --recursive

# Create a safety backup tag
git tag backup-before-flattening
```

### Phase 2: Handle LFS Files
```bash
# Download ALL LFS files to ensure we have actual file content
git lfs fetch --all
git lfs checkout

# For each LFS file, we'll convert it to a regular file
# This script preserves exact file locations
for file in $(git lfs ls-files | cut -d' ' -f3); do
    if [ -f "$file" ]; then
        # Get actual file content
        actual_file=$(mktemp)
        cp "$file" "$actual_file"
        
        # Remove from git index only
        git rm --cached "$file"
        
        # Put actual content back in exact same location
        mv "$actual_file" "$file"
        
        # Re-add as regular file
        git add "$file"
    fi
done

# Remove LFS tracking configuration
git rm .gitattributes
```

### Phase 3: Convert Each Submodule In-Place

#### For msgq_repo:
```bash
# Save the submodule content
mv msgq_repo msgq_repo_backup

# Remove submodule git tracking
git rm --cached msgq_repo
rm -rf .git/modules/msgq_repo

# Move content back to exact same location
mv msgq_repo_backup msgq_repo

# Remove .git file that makes it a submodule
rm msgq_repo/.git

# Add as regular directory
git add msgq_repo/
```

#### For opendbc_repo:
```bash
# Save the submodule content
mv opendbc_repo opendbc_repo_backup

# Remove submodule git tracking
git rm --cached opendbc_repo
rm -rf .git/modules/opendbc_repo

# Move content back to exact same location
mv opendbc_repo_backup opendbc_repo

# Remove .git file that makes it a submodule
rm opendbc_repo/.git

# Add as regular directory
git add opendbc_repo/
```

#### For panda:
```bash
# Save the submodule content
mv panda panda_backup

# Remove submodule git tracking
git rm --cached panda
rm -rf .git/modules/panda

# Move content back to exact same location
mv panda_backup panda

# Remove .git file that makes it a submodule
rm panda/.git

# Add as regular directory
git add panda/
```

#### For rednose_repo:
```bash
# Save the submodule content
mv rednose_repo rednose_repo_backup

# Remove submodule git tracking
git rm --cached rednose_repo
rm -rf .git/modules/rednose_repo

# Move content back to exact same location
mv rednose_repo_backup rednose_repo

# Remove .git file that makes it a submodule
rm rednose_repo/.git

# Add as regular directory
git add rednose_repo/
```

#### For teleoprtc_repo:
```bash
# Save the submodule content
mv teleoprtc_repo teleoprtc_repo_backup

# Remove submodule git tracking
git rm --cached teleoprtc_repo
rm -rf .git/modules/teleoprtc_repo

# Move content back to exact same location
mv teleoprtc_repo_backup teleoprtc_repo

# Remove .git file that makes it a submodule
rm teleoprtc_repo/.git

# Add as regular directory
git add teleoprtc_repo/
```

#### For tinygrad_repo:
```bash
# Save the submodule content
mv tinygrad_repo tinygrad_repo_backup

# Remove submodule git tracking
git rm --cached tinygrad_repo
rm -rf .git/modules/tinygrad_repo

# Move content back to exact same location
mv tinygrad_repo_backup tinygrad_repo

# Remove .git file that makes it a submodule
rm tinygrad_repo/.git

# Add as regular directory
git add tinygrad_repo/
```

#### For sunnypilot/neural_network_data:
```bash
# Save the submodule content
mv sunnypilot/neural_network_data sunnypilot/neural_network_data_backup

# Remove submodule git tracking
git rm --cached sunnypilot/neural_network_data
rm -rf .git/modules/sunnypilot/neural_network_data

# Move content back to exact same location
mv sunnypilot/neural_network_data_backup sunnypilot/neural_network_data

# Remove .git file that makes it a submodule
rm sunnypilot/neural_network_data/.git

# Add as regular directory
git add sunnypilot/neural_network_data/
```

### Phase 4: Remove Submodule Configuration
```bash
# Remove .gitmodules file
git rm .gitmodules

# Clean git config (may fail if sections don't exist, that's ok)
git config --remove-section submodule.panda 2>/dev/null || true
git config --remove-section submodule.opendbc 2>/dev/null || true
git config --remove-section submodule.msgq 2>/dev/null || true
git config --remove-section submodule.rednose_repo 2>/dev/null || true
git config --remove-section submodule.teleoprtc_repo 2>/dev/null || true
git config --remove-section submodule.tinygrad 2>/dev/null || true
git config --remove-section "submodule.sunnypilot/neural_network_data" 2>/dev/null || true
```

### Phase 5: Commit the Flattened Structure
```bash
git commit -m "Convert all submodules to regular directories, preserving exact structure

- All directories remain in exact same locations
- All symlinks remain unchanged
- Only submodule git machinery removed
- LFS files converted to regular files"
```

### Phase 6: Cherry-pick the Target Commit
```bash
# The commit modifies files in opendbc_repo/opendbc/
# Since we kept the structure, paths remain the same
git cherry-pick 1f53b7cbefb0b92ce963e4087a69523088cba74a
```

### Phase 7: Verification
```bash
# Verify symlinks still work
ls -la msgq  # Should show: msgq -> msgq_repo/msgq
ls -la opendbc  # Should show: opendbc -> opendbc_repo/opendbc
ls -la rednose  # Should show: rednose -> rednose_repo/rednose
ls -la teleoprtc  # Should show: teleoprtc -> teleoprtc_repo/teleoprtc
ls -la tinygrad  # Should show: tinygrad -> tinygrad_repo/tinygrad

# Verify no submodules remain
git submodule status  # Should show nothing

# Verify directories exist with content
ls msgq_repo/msgq/
ls opendbc_repo/opendbc/
ls panda/
ls rednose_repo/rednose/
ls teleoprtc_repo/teleoprtc/
ls tinygrad_repo/tinygrad/
ls sunnypilot/neural_network_data/

# Test build
scons -u -j$(nproc)
```

## Alternative Approach: Using Subtree Merge

If the manual approach encounters issues, we can use git's subtree merge strategy:

```bash
# For each submodule, convert to subtree
# Example for msgq_repo:
git rm --cached msgq_repo
git commit -m "Remove msgq_repo submodule"

# Add the submodule content as a subtree at the EXACT same path
git subtree add --prefix=msgq_repo https://github.com/sunnypilot/msgq.git master --squash

# Remove the .git file
rm msgq_repo/.git
```

## Critical Success Factors

1. **No Path Changes**: Every file remains at its exact current path
2. **Symlinks Intact**: All symlinks continue to work exactly as before
3. **Build Works**: The codebase builds successfully after conversion
4. **Cherry-pick Clean**: The target commit applies without path adjustments

## What This Accomplishes

- ✅ Removes dependency on external git repositories
- ✅ Removes dependency on Git LFS
- ✅ Removes dependency on GitLab
- ✅ All files become actual files in your repository
- ✅ No structural changes - everything stays exactly where it is
- ✅ Symlinks continue to work exactly as before

## What This Does NOT Do

- ❌ Does not move any files
- ❌ Does not rename any directories  
- ❌ Does not change any symlinks
- ❌ Does not change the codebase structure

## Troubleshooting

### If a submodule won't convert:
```bash
# Force deinit
git submodule deinit -f <path>
git rm --cached <path>
rm -rf .git/modules/<path>
```

### If symlinks break:
They shouldn't! The whole point is nothing moves. But if they do:
```bash
# Recreate symlink (example for msgq)
ln -sf msgq_repo/msgq msgq
```

### If LFS files show as text pointers:
```bash
# Force download before converting
git lfs pull
git lfs fetch --all
```