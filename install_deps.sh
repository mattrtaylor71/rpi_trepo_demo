#!/bin/bash
set -e

echo "==> Updating system"
sudo apt update
sudo apt full-upgrade -y

echo "==> Installing OS-level deps"
sudo apt install -y \
  python3-full python3-venv python3-pip \
  python3-picamera2 \
  libatlas-base-dev libjpeg-dev libopenexr-dev \
  libgstreamer1.0-dev libgtk-3-0 libgl1 \
  python3-opencv  # optional fallback; we still install a newer wheel in venv

# Project venv lives alongside the code
VENV=".venv"

echo "==> Creating virtual environment at ${VENV}"
python3 -m venv "${VENV}"

echo "==> Upgrading pip in venv"
"${VENV}/bin/pip" install --upgrade pip

echo "==> Installing Python packages in venv"
# OpenCV wheel, MediaPipe, NumPy
"${VENV}/bin/pip" install \
  opencv-python==4.8.1.78 \
  "mediapipe==0.10.18" \
  "numpy<2.0"


echo "==> Done. To run your script:"
echo "source ${VENV}/bin/activate && python gestures.py"

