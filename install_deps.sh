#!/bin/bash
set -e

echo "Updating system..."
sudo apt update

echo "Installing dependencies..."
sudo apt install -y python3-pip python3-picamera2 \
    libatlas-base-dev libjpeg-dev libqtgui4 \
    libqt4-test libilmbase-dev libopenexr-dev \
    libgstreamer1.0-dev

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Install complete!"

