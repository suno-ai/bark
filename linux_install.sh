#!/bin/bash

# Ensure the script is run as root this was tested on Debain 12
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# Update and install essential packages
apt-get update && apt-get upgrade -y
apt-get install -y python3-pip python3-venv git

# Create a directory for your setup
mkdir -p /opt/my_project
cd /opt/my_project

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Clone the transformers and Bark repository and install them
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
cd ..

git clone https://github.com/suno-ai/bark.git
cd bark
pip install .

echo "Setup complete. All packages are installed."
