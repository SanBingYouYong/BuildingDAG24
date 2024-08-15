import bpy
from bpy.types import Context

# import typing
import os
import sys
from pathlib import Path
import shutil
import numpy as np
# import yaml
import cv2

# import local modules
file = Path(__file__).resolve()
parent = file.parents[0]  # 0 for background mode, 1 for gui mode... 
print(f"parent: {parent}")
sys.path.append(str(parent))

from paramload import DAGParamLoader
from render import DAGRenderer
from params import DAGParams

'''
expected usage: 
$BLENDER d_render.blend -b -P d_load_and_render.py
    reads the inference output and writes rendered images to the output folder
'''

distort = True
path = "./inference/output.yml"
img_path = path[:-4] + ".png"

path = os.getcwd() + path
img_path = os.getcwd() + img_path

loader = DAGParamLoader()
params = DAGParams(path)
loader.load_dag_params_with_return_part(params, 0)

renderer = DAGRenderer()
# renderer.render(file_path=img_path, distortion=distort)  # with dataset generation processings  # background + distort = crash
bpy.context.scene.render.filepath = img_path
bpy.ops.render.render(write_still=True)  # plain rendering with freestyle lines


