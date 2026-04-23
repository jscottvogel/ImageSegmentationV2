#!/bin/bash
# Attempt to set the power limit of the AMD GPU to 80% to prevent thermal shutdowns
echo "Checking current GPU power limits..."
cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap
echo "Attempting to reduce power limit..."
