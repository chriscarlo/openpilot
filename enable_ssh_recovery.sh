#!/usr/bin/env bash
# Emergency SSH Recovery Script
# Run this to forcibly enable SSH access when locked out

echo "Emergency SSH Recovery - Enabling SSH access..."

# Set the params directly using Python
python3 << 'EOF'
import sys
sys.path.insert(0, '/data/openpilot')

from common.params import Params
import requests

params = Params()

# Force enable SSH
params.putBool("SshEnabled", True)
print("✓ SSH enabled")

# Set GitHub username
params.put("GithubUsername", "chriscarlo")
print("✓ GitHub username set to: chriscarlo")

# Try to fetch SSH keys
try:
    response = requests.get("https://github.com/chriscarlo.keys", timeout=10)
    if response.status_code == 200 and response.text.strip():
        params.put("GithubSshKeys", response.text)
        print("✓ SSH keys fetched from GitHub")
    else:
        print("⚠ Could not fetch SSH keys (HTTP %d)" % response.status_code)
except Exception as e:
    print("⚠ Could not fetch SSH keys:", str(e))

print("\nSSH recovery complete. You should now be able to SSH into the device.")
EOF

echo "SSH recovery script finished."