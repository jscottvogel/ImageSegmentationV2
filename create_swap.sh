#!/bin/bash
set -e
if [ -f /swapfile ]; then
    echo "Swapfile already exists at /swapfile."
    exit 0
fi

echo "Creating 16GB swap file..."
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "Swap file created and enabled successfully!"
free -h
