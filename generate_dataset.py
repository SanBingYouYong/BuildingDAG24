import os
import sys
import subprocess
import time
import re
from tqdm import tqdm
import signal
import datetime

# Path to Blender executable
BLENDER32 = os.environ.get("BLENDER32")

# Path to dataset.blend and dataset_gen.py
dataset_blend = "./dataset.blend"
dataset_gen_py = "./dataset_gen.py"

# Args
args = sys.argv[1:]
if len(args) == 4:
    num_batches, batch_size, num_varying_params, device = map(int, args)
else:
    print("Usage: blender -b -P dataset_gen.py <num_batches> <batch_size> <num_varying_params> <device>")
    sys.exit(1)

# Log file path
log_file = f"./datasets/dataset_gen_log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

# Calculate total number of images
total_images = num_batches * batch_size

# Initialize progress bar
pbar = tqdm(total=total_images, desc="Rendering", unit="image")

# Regular expression pattern to match sentences starting with "Saved: './datasets/test_dataset/images/batch0_sample0.png'"
# saved_pattern = re.compile(r"Saved: '\./datasets/test_dataset/images/batch\d+_sample\d+\.png'")

# Handler for termination signals
def signal_handler(sig, frame):
    print("Terminating process...")
    process.terminate()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Command to execute
command = [BLENDER32, dataset_blend, "-b", "-P", dataset_gen_py, str(num_batches), str(batch_size), str(num_varying_params), str(device)]

# Start timer
start_time = time.time()

# Execute the command and capture output
with open(log_file, "w") as f:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        # Check if the line matches the saved pattern
        if line.startswith("Saved:"):
            # Update progress bar
            pbar.update(1)
        if not line.startswith("Fra"):
            # Write the output to the log file
            f.write(line)
        # Check if the process has terminated
        if process.poll() is not None:
            break

# Close progress bar
pbar.close()

# Calculate elapsed time
elapsed_time = time.time() - start_time

# Show timer
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Log file written to: {log_file}")
