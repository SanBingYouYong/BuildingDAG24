# import typing
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[0]  # 0 for background mode, 1 for gui mode... 
print(f"parent: {parent}")
sys.path.append(str(parent))

from ui_external_inference import inference

inference()

# run $BLENDER d_render.blend -b -P d_load_and_render.py
# BLENDER = "blender/"
command = "./blender/blender-3.2.2-linux-x64/blender -b .\d_render.blend -P d_load_and_render.py"

import subprocess
subprocess.run(command, shell=True)

