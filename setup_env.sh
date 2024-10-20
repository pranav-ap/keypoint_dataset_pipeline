#!/bin/bash

# Set up Miniconda installation directory
MINICONDA_DIR="$HOME/miniconda3"

# Create Miniconda directory
mkdir -p "$MINICONDA_DIR"

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_DIR"/miniconda.sh
bash "$MINICONDA_DIR"/miniconda.sh -b -u -p "$MINICONDA_DIR"

# Remove the installer
rm "$MINICONDA_DIR"/miniconda.sh

# Initialize Conda (adds it to the path)
source "$MINICONDA_DIR"/bin/conda init

conda init --all

# Create and activate the conda environment
conda create -n kd_pipeline -y python=3.13
conda activate kd_pipeline

# Install requirements from requirements.txt
pip install -r requirements.txt

# Display Python version and installed packages
python --version
pip list
