#!/bin/bash

# Install dependencies for the project
sudo apt install -y libjpeg-dev zlib1g-dev wget

# Install requirements
pip install -r requirements.txt

# Check if git submodule was recursively cloned
git submodule update --init --recursive

# Navigate to RepViT submodule and install package
sudo chmod -R 777 RepViT
cd RepViT/sam
pip install --user -e . 

# Dowload the sam weights using wget
mkdir weights
cd weights
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt


echo "Dependencies installed successfully"


