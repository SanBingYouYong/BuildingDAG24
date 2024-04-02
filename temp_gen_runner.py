import bpy

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[0]  # or 1
sys.path.append(str(parent))


# building = bpy.data.objects["Building"]
# node = None
# for mod in building.modifiers:
#     if mod.type == "NODES" and mod.node_group.name == "Building For Distortion Render":
#         node = mod
#         break
# # iterate through inputs and load values
# inputs = node.node_group.inputs
# for input_key, input_val in inputs.items():
#     print(input_key, input_val)
#     id = input_val.identifier
#     input_type = input_val.type
#     print(id, input_type)

# import bmesh

# # bigger_cube = bpy.data.objects["Cube"]
# smaller_cube = bpy.data.objects["Cube.001"]
# # select both and enter editmode
# # bigger_cube.select_set(True)
# smaller_cube.select_set(True)
# obj = smaller_cube
# # edit mode
# bpy.ops.object.editmode_toggle()
# me = obj.data
# bm = bmesh.from_edit_mesh(me)
# bpy.ops.mesh.select_mode(type='EDGE')
# bpy.ops.mesh.select_all(action='DESELECT')
# # select visible verts
# DRUtils._select_border(bpy.context)
# obj.update_from_editmode()
# selected_verts = [v.index for v in bm.verts if v.select]
# selected_edges = [e.index for e in bm.edges if e.select]
# selected_faces = [f.index for f in bm.faces if f.select]
# bpy.ops.object.editmode_toggle()
# print(selected_edges)

# obj = bpy.context.active_object
# # scale it
# obj.scale = (2, 2, 2)

from dataset_gen import DAGDatasetGenerator

generator = DAGDatasetGenerator(dataset_name="distortiontest")
generator.populate_dataset_wrt_batches(num_batches=2, batch_size=2, num_varying_params=5, render=True, distortion=True)
