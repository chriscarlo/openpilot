sudo mount -o remount,rw /
sudo mount -o remount,rw,exec /persist
export GIT_SSH_COMMAND="ssh -i /persist/ssh/git_keys/claude_github_key -o StrictHostKeyChecking=no"
/persist/scripts/launchClaude.sh
