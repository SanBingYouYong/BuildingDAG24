import bpy
import geometry_script as gs

# import local modules
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from building import building

# import importlib
# importlib.reload(building)


params = [
    2, [1, 1, 1], 0, [1, 1, 1]
]


bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object

treename = "Building"
nodetree = bpy.data.node_groups[treename]
if not nodetree:
    raise Exception("Node tree not found")
nodegroup = nodetree.copy()

obj_modifier = obj.modifiers.new(name=treename, type="NODES")
obj_modifier.node_group = nodegroup
inputs = [
            input_key for (input_key, input_val) in obj.modifiers[treename].items() 
                if input_key[:6] == "Input_" and input_key[-1] != 'e'
]
params_index = 0
for input_key in inputs:
    input_group = obj.modifiers[treename][input_key]
    # print(type(input_group))
    if input_group == 0:  # 0 for single value
        obj.modifiers[treename][input_key] = params[params_index]
        params_index += 1
    else:  # now there's only IDPropertyArray, in future there might be more
        for i in range(len(input_group)):
            obj.modifiers[treename][input_key][i] = params[params_index][i]
        params_index += 1




# building(
#     bm_type=2, bm_size=gs.Vector(x=1, y=1, z=1),
#     roof_type=0, roof_size=gs.Vector(x=1, y=1, z=1)
# )