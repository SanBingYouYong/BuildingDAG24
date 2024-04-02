import bpy

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))


from distortion import DRUtils

building = bpy.data.objects["Building"]
# hide
building.hide_viewport = True
DRUtils.get_visibles(building, mode="EDGE")
