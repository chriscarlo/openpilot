#!/usr/bin/env python3
"""
API Key Manager for RTI System

Helper utilities for managing RapidAPI key persistence across the RTI system.
"""

import os

# Priority order for API key locations
API_KEY_PATHS = [
    '/data/persist/rapidapi_key',           # Standard openpilot persist location
    '/persist/rapidapi_key',                # Alternative persist location
    '/data/openpilot/persist/rapidapi_key', # Local project persist
    '/data/openpilot/rapidapi_key'          # Project root fallback
]

# Environment variable names to check
ENV_VAR_NAMES = ['RAPIDAPI_KEY', 'WAZE_API_KEY', 'RTI_API_KEY']


def get_api_key() -> str | None:
    """
    Get the RapidAPI key from environment variables or persistent files.
    
    Returns:
        API key string if found, None otherwise
    """
    # First check environment variables
    for env_var in ENV_VAR_NAMES:
        api_key = os.getenv(env_var)
        if api_key and api_key.strip():
            return api_key.strip()

    # Then check persistent file locations
    for path in API_KEY_PATHS:
        try:
            with open(path) as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
        except (OSError, FileNotFoundError, PermissionError):
            continue

    return None


def save_api_key(api_key: str, path: str | None = None) -> bool:
    """
    Save API key to persistent storage.
    
    Args:
        api_key: The API key to save
        path: Optional specific path to save to (defaults to first writable path)
        
    Returns:
        True if successful, False otherwise
    """
    if not api_key or not api_key.strip():
        return False

    paths_to_try = [path] if path else API_KEY_PATHS

    for save_path in paths_to_try:
        if not save_path:
            continue

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Write the API key
            with open(save_path, 'w') as f:
                f.write(api_key.strip())

            # Verify it was written correctly
            with open(save_path) as f:
                if f.read().strip() == api_key.strip():
                    return True

        except (PermissionError, OSError):
            continue

    return False


def validate_api_key(api_key: str | None) -> bool:
    """
    Basic validation of API key format.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False

    key = api_key.strip()

    # RapidAPI keys are typically 50 characters with mixed alphanumeric
    return (
        len(key) >= 40 and                    # Minimum reasonable length
        len(key) <= 60 and                    # Maximum reasonable length
        any(c.isalpha() for c in key) and     # Contains letters
        any(c.isdigit() for c in key)        # Contains digits
    )


if __name__ == '__main__':
    """Command line interface for API key management."""
    import sys

    if len(sys.argv) < 2:
        print("RTI API Key Manager")
        print("Usage:")
        print("  python3 api_key_manager.py get     - Get current API key")
        print("  python3 api_key_manager.py set KEY - Save new API key")
        print("  python3 api_key_manager.py check   - Check if key is valid")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'get':
        api_key = get_api_key()
        if api_key:
            print(f"API key found: {api_key[:8]}...")
            print("Key is ready for use")
        else:
            print("No API key found")
            print("Available locations:", API_KEY_PATHS)
            sys.exit(1)

    elif command == 'set':
        if len(sys.argv) < 3:
            print("Error: Please provide API key")
            sys.exit(1)

        new_key = sys.argv[2]
        if not validate_api_key(new_key):
            print("Warning: API key format looks invalid")

        if save_api_key(new_key):
            print("API key saved successfully")
        else:
            print("Failed to save API key")
            sys.exit(1)

    elif command == 'check':
        api_key = get_api_key()
        if api_key:
            is_valid = validate_api_key(api_key)
            print(f"API key found: {api_key[:8]}...")
            print(f"Format validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        else:
            print("No API key found")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
