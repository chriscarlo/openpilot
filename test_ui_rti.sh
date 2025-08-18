#!/bin/bash
# UI RTI Test Launcher
echo "Launching UI with RTI test environment..."

# Set environment for onroad UI
export FORCE_ONROAD_UI=1

# Ensure RTI is enabled
echo "true" > /tmp/params/d/RTIEnabled
echo "true" > /tmp/params/d/RTIHUDEnabled

# Launch UI in background
./selfdrive/ui/ui &
UI_PID=$!

echo "UI launched with PID: $UI_PID"
echo "Press Enter to stop UI..."
read

kill $UI_PID 2>/dev/null
echo "UI stopped"
