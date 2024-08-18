# import typing
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[0]  # 0 for background mode, 1 for gui mode... 
print(f"parent: {parent}")
sys.path.append(str(parent))

from ui_external_inference import inference

import subprocess


# run $BLENDER d_render.blend -b -P d_load_and_render.py


# determine win or linux
if sys.platform == "win32":
    BLENDER = "./blender/blender-3.2.2-windows-x64/blender.exe"
elif sys.platform == "linux":
    BLENDER = "./blender/blender-3.2.2-linux-x64/blender"
else:
    raise Exception("Unsupported platform")
RENDER_FILE = "./d_render.blend"
RENDER_SCRIPT = "./d_load_and_render.py"
command = [BLENDER, "-b", RENDER_FILE, "-P", RENDER_SCRIPT]


def _check_file_exists(blender, render_file, render_script):
    if not os.path.exists(blender):
        raise FileNotFoundError(f"Blender executable not found at {blender}")
    if not os.path.exists(render_file):
        raise FileNotFoundError(f"Render file not found at {render_file}")
    if not os.path.exists(render_script):
        raise FileNotFoundError(f"Render script not found at {render_script}")


def infer_load_render(_blender: str=BLENDER,
                      _render_file: str=RENDER_FILE,
                      _render_script: str=RENDER_SCRIPT):
    _check_file_exists(_blender, _render_file, _render_script)
    inference()
    command = [_blender, "-b", _render_file, "-P", _render_script]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if process.returncode is not None:
        print(f"Process returned with code {process.returncode}")
    if process.stderr is not None:
        print(f"Error: {process.stderr.readlines()}")
    for line in process.stdout:
        if line.startswith("Saved:"):
            print(line)


