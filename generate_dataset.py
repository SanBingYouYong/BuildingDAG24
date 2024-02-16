import os
import numpy as np

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))
sys.path.append(str(file.parents[0]))

from tqdm import tqdm


# use commandline to call blender in background and execute script
# $BLENDER32 dataset.blend -b -P dataset_gen.py
import subprocess
import time
import re

# Path to Blender executable
BLENDER32 = os.environ["BLENDER32"]

# Path to dataset.blend and dataset_gen.py
dataset_blend = "./dataset.blend"
dataset_gen_py = "./dataset_gen.py"

# Log file path
log_file = "./datasets/dataset_gen_log.txt"

# Command to execute
command = [BLENDER32, dataset_blend, "-b", "-P", dataset_gen_py]

# Start timer
start_time = time.time()

# Initialize progress bar
pbar = tqdm(desc="Rendering", unit="image")

# Regular expression pattern to extract time information
time_pattern = re.compile(r"Time: (\d+):(\d+\.\d+) \(Saving: (\d+):(\d+\.\d+)\)")

# Execute the command and capture output
with open(log_file, "w") as f:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        # Update progress bar based on output
        match = time_pattern.search(line)
        if match:
            render_time = float(match.group(2))
            pbar.update(1)
            pbar.set_postfix(render_time=f"{render_time:.2f}s")
        f.write(line)  # Write the output to the log file

# Close progress bar
pbar.close()

# Calculate elapsed time
elapsed_time = time.time() - start_time

# Show timer
print(f"Total execution time: {elapsed_time:.2f} seconds")

