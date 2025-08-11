#!/bin/bash
set -e

echo "Updating system..."
sudo apt update
sudo apt full-upgrade -y

echo "Installing dependencies (Qt4 removed; using current libs)..."
sudo apt install -y \
  python3-pip \
  python3-picamera2 \
  libatlas-base-dev \
  libjpeg-dev \
  libopenexr-dev \
  libgstreamer1.0-dev \
  libgtk-3-0 \
  libgl1 \
  python3-pyqt5

echo "Upgrading pip and installing Python packages..."
python3 -m pip install --upgrade pip
# OpenCV (GUI via GTK); MediaPipe; NumPy
python3 -m pip install opencv-python==4.8.1.78 mediapipe==0.10.11 numpy

echo "All set."

