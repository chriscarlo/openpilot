# UI Testing Guide

## Launching the Onroad HUD

To launch the UI in onroad mode for testing without cameras:

```bash
FORCE_ONROAD_UI=1 ./selfdrive/ui/ui &
```

The `FORCE_ONROAD_UI=1` environment variable bypasses the camera requirements and forces the UI directly into onroad mode, displaying the HUD elements (speed, MAX box, lateral acceleration meter, etc.) without needing actual driving data.

## Prerequisites

Ensure you have a virtual display running if testing in a headless environment:
```bash
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

## Taking Screenshots

To capture screenshots of the running UI:
```bash
DISPLAY=:99 import -window root /tmp/screenshot.png
```