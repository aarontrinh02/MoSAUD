#!/bin/bash

CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

conda create -n supe python=3.10
conda activate supe

# Install Python dependencies
pip install --upgrade setuptools wheel
pip install -r requirements.txt
pip install numpy==1.23.5 # Some compatibility issues with gym 0.23.1 (required by D4RL) and newer numpy versions

# Clone necessary code to run the HILP baseline (from a fork of original HILP implementation), and move it to the correct directory
git clone https://github.com/wilcoxsonm21/HILP
mv HILP/hilp .
mv HILP/jaxrl_m .
rm -rf HILP

# Download Visual AntMaze Dataset
gdown --id 1EQCYsEDi3qUlq4NuffLJ28iDpXL4I9_n
mkdir -p data/antmaze_topview_6_60
mv antmaze-large-diverse-v2.npz data/antmaze_topview_6_60

# Download and unzip opal and hilp pretrained checkpoints
pip install gdown
gdown --id 1IbwkUG2notEu-fGQgYNj8KReEmkOuxre
unzip hilp_checkpoints.zip
rm -rf hilp_checkpoints.zip

gdown --id 1coWpxslJCIDkXYuR-i9H4_lbhc_9yS32
unzip opal_checkpoints.zip
rm -rf opal_checkpoints.zip