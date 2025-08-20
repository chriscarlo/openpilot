#!/usr/bin/env python3
"""
Fetch and install GitHub SSH keys for a user.
This is separated from the manager to avoid blocking boot with network requests.
"""

import sys
import time
import requests
from openpilot.common.params import Params

def fetch_github_keys(username="chriscarlo", max_retries=3):
  """Fetch SSH keys from GitHub and store them in params."""
  params = Params()
  
  # Check if keys already exist
  existing_keys = params.get("GithubSshKeys", encoding='utf8')
  if existing_keys:
    print(f"SSH keys already present for {username}")
    return True
  
  # Try to fetch keys with retries
  for attempt in range(max_retries):
    try:
      print(f"Fetching SSH keys for {username} (attempt {attempt + 1}/{max_retries})")
      response = requests.get(f"https://github.com/{username}.keys", timeout=10)
      
      if response.status_code == 200 and response.text.strip():
        params.put("GithubSshKeys", response.text)
        print(f"Successfully fetched and stored SSH keys for {username}")
        return True
      else:
        print(f"Failed to fetch keys: HTTP {response.status_code}")
        
    except requests.exceptions.RequestException as e:
      print(f"Network error: {e}")
    
    if attempt < max_retries - 1:
      time.sleep(5)  # Wait before retry
  
  print(f"Failed to fetch SSH keys after {max_retries} attempts")
  return False

if __name__ == "__main__":
  username = sys.argv[1] if len(sys.argv) > 1 else "chriscarlo"
  success = fetch_github_keys(username)
  sys.exit(0 if success else 1)