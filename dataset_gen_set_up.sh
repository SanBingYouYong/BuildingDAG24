#!/bin/bash

# Step 1: Git clone the repository and change to the directory
# git clone https://github.com/SanBingYouYong/BuildingDAG24.git
# cd BuildingDAG24

# Step 2: Download Blender and prepare it
cd ./blender
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz

# Archive Blender's own Python directory
tar -czf blender_python_backup.tar.gz -C blender-3.2.2-linux-x64/3.2 python/

# Remove the Python directory after archiving
rm -rf blender-3.2.2-linux-x64/3.2/python/

cd ../  # Go back to the project root directory

# Step 3: Install dependencies using conda and pip
conda create -n dag_distort python=3.10 -y
# conda activate dag_distort  # need conda init bash
# pip install opencv-python PyYAML tqdm numpy

# Optional: Check if conda installed its own `freestyle` package
FREESTYLE_PATH=$(conda info --base)/envs/dag_distort/lib/python3.10/site-packages/freestyle

# Safety check to ensure FREESTYLE_PATH contains the expected path structure
EXPECTED_PATH_PART="envs/dag_distort/lib/python3.10/site-packages/freestyle"

if [[ "$FREESTYLE_PATH" != *"$EXPECTED_PATH_PART"* ]]; then
    echo "Error: FREESTYLE_PATH ($FREESTYLE_PATH) does not contain the expected path structure. Aborting."
    exit 1
fi

# Safety check before removing anything
if [ "$FREESTYLE_PATH" == "$HOME" ] || [ "$FREESTYLE_PATH" == "/" ]; then
    echo "Error: The FREESTYLE_PATH points to a critical directory ($FREESTYLE_PATH). Aborting to prevent data loss."
    exit 1
fi

if [ -d "$FREESTYLE_PATH" ]; then
    echo "Found freestyle package in conda environment. Replacing it with Blender's freestyle..."
    rm -rf "$FREESTYLE_PATH"/*
    cp -r ./blender/blender-3.2.2-linux-x64/3.2/scripts/freestyle/* "$FREESTYLE_PATH"
    echo "Freestyle package has been replaced."
else
    echo "Freestyle package not found in conda environment. Skipping replacement."
fi

echo "Setup completed successfully!"
