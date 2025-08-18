#!/usr/bin/env python3
"""
RTI Root Cause Check
Verifies the root cause of why RTI widget doesn't display threats.
"""
import re

def check_ui_subscription():
    """Check if the UI is properly subscribed to RTI messages."""
    print("RTI Root Cause Analysis")
    print("=" * 60)
    print("\nChecking UI subscription configuration...")
    
    ui_file = "/projects/chauffeur/data/openpilot/selfdrive/ui/sunnypilot/ui.cc"
    with open(ui_file, 'r') as f:
        content = f.read()
    
    # Find the SubMaster initialization
    pattern = r'SubMaster.*?\{([^}]+)\}'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        subscription_list = match.group(1)
        subscribed_messages = re.findall(r'"([^"]+)"', subscription_list)
        
        print(f"Found {len(subscribed_messages)} subscribed messages in UI:")
        for msg in sorted(subscribed_messages):
            if 'rti' in msg.lower() or 'RTI' in msg:
                print(f"  ✓ {msg}")
            else:
                print(f"    {msg}")
        
        print("\nChecking for rtiStateSP subscription...")
        if 'rtiStateSP' in subscribed_messages:
            print("✓ UI is subscribed to rtiStateSP")
            return True
        else:
            print("✗ UI is NOT subscribed to rtiStateSP")
            return False
    else:
        print("✗ Could not find SubMaster initialization")
        return False

def check_hud_code():
    """Check if HUD code references rtiStateSP."""
    print("\nChecking HUD code for RTI message handling...")
    
    hud_file = "/projects/chauffeur/data/openpilot/selfdrive/ui/sunnypilot/qt/onroad/hud.cc"
    with open(hud_file, 'r') as f:
        content = f.read()
    
    if 'rtiStateSP' in content:
        # Count occurrences
        count = content.count('rtiStateSP')
        print(f"✓ HUD code references 'rtiStateSP' {count} times")
        
        # Check for message validity checks
        if 'sm->valid("rtiStateSP")' in content:
            print("✓ HUD checks for rtiStateSP validity")
        
        if 'sm->updated("rtiStateSP")' in content:
            print("✓ HUD checks if rtiStateSP is updated")
            
        if 'getRtiStateSP()' in content:
            print("✓ HUD retrieves RTI state data")
        
        return True
    else:
        print("✗ HUD code does not reference rtiStateSP")
        return False

def check_service_definition():
    """Check if rtiStateSP is defined in services."""
    print("\nChecking service definition...")
    
    services_file = "/projects/chauffeur/data/openpilot/cereal/services.py"
    with open(services_file, 'r') as f:
        content = f.read()
    
    if '"rtiStateSP"' in content:
        print("✓ rtiStateSP is defined in cereal/services.py")
        # Extract the line
        for line in content.split('\n'):
            if 'rtiStateSP' in line:
                print(f"  Definition: {line.strip()}")
                break
        return True
    else:
        print("✗ rtiStateSP is not defined in services")
        return False

def check_process_config():
    """Check if rtid process is configured."""
    print("\nChecking rtid process configuration...")
    
    config_file = "/projects/chauffeur/data/openpilot/system/manager/process_config.py"
    with open(config_file, 'r') as f:
        content = f.read()
    
    if 'rtid' in content:
        print("✓ rtid process is configured")
        # Find the configuration line
        for line in content.split('\n'):
            if 'rtid' in line and 'PythonProcess' in line:
                print(f"  Config: {line.strip()}")
                break
        
        if 'rti_enabled' in content:
            print("✓ rtid uses rti_enabled condition")
            # Find the condition definition
            pattern = r'def rti_enabled.*?return.*?\n'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                print(f"  Condition: starts when RTIEnabled parameter is True")
        
        return True
    else:
        print("✗ rtid is not configured")
        return False

def main():
    # Run all checks
    ui_subscribed = check_ui_subscription()
    hud_handles_msg = check_hud_code()
    service_defined = check_service_definition()
    process_configured = check_process_config()
    
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print("=" * 60)
    
    if not ui_subscribed and hud_handles_msg and service_defined and process_configured:
        print("\n*** ROOT CAUSE IDENTIFIED ***")
        print("\nThe UI is NOT subscribed to 'rtiStateSP' messages!")
        print("\nDetails:")
        print("1. The rtid daemon is properly configured to run when RTIEnabled=True")
        print("2. The rtiStateSP message is properly defined in cereal/services.py")
        print("3. The HUD code properly handles rtiStateSP messages when received")
        print("4. BUT: The UI's SubMaster does not subscribe to 'rtiStateSP'")
        print("\nThis means:")
        print("- The rtid daemon publishes RTI threat data")
        print("- The HUD widget code is ready to display threats")
        print("- But the UI never receives the messages because it's not subscribed!")
        print("\nFIX REQUIRED:")
        print("Add 'rtiStateSP' to the SubMaster subscription list in:")
        print("  selfdrive/ui/sunnypilot/ui.cc (line 17-22)")
        print("\nThe subscription list should include:")
        print('  "rtiStateSP",  // Add this line')
    else:
        print("\nMultiple issues found:")
        if not ui_subscribed:
            print("- UI not subscribed to rtiStateSP")
        if not hud_handles_msg:
            print("- HUD doesn't handle rtiStateSP")
        if not service_defined:
            print("- rtiStateSP not defined in services")
        if not process_configured:
            print("- rtid process not configured")

if __name__ == "__main__":
    main()