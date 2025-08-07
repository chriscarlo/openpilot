#!/usr/bin/env python3
"""
SSH Recovery Script - Ensures SSH is enabled with hardcoded credentials at boot.
This is a critical recovery tool for when the UI crashes and prevents SSH configuration.
"""

import requests
import time
from openpilot.common.params import Params

def setup_ssh_recovery():
    """Force enable SSH with chriscarlo's GitHub keys for recovery access."""
    params = Params()

    # Always enable SSH on boot for recovery
    params.put_bool("SshEnabled", True)
    print("SSH enabled at boot for recovery")

    # Check if we already have keys
    existing_username = params.get("GithubUsername", encoding='utf8')
    existing_keys = params.get("GithubSshKeys", encoding='utf8')

    # If no keys or different user, set up chriscarlo's keys
    if not existing_keys or existing_username != "chriscarlo":
        try:
            print("Fetching SSH keys for chriscarlo...")
            keys = requests.get("https://github.com/chriscarlo.keys", timeout=10)

            if keys.status_code == 200 and keys.text.strip():
                params.put("GithubUsername", "chriscarlo")
                params.put("GithubSshKeys", keys.text)
                print("SSH keys for chriscarlo successfully configured")
            else:
                print(f"Failed to fetch keys: HTTP {keys.status_code}")
                # Even if we can't get keys, at least enable SSH
        except Exception as e:
            print(f"Error fetching SSH keys: {e}")
            # Continue anyway - at least SSH will be enabled
    else:
        print(f"SSH keys already configured for {existing_username}")

    return True

if __name__ == "__main__":
    # Retry a few times in case network isn't ready
    for attempt in range(3):
        try:
            if setup_ssh_recovery():
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
