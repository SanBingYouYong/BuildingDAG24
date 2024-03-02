import geometry_script as gs
import bpy

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))


dag_node_name = "BuildingDAG"
building_name = "Building"

# get the building object
building = bpy.data.objects[building_name]
# get the building's dag node
for mod in building.modifiers:
    if mod.type == "NODES" and mod.node_group.name == dag_node_name:
        dag_mod = mod
        break

# example input value update
# identifier = dag_mod.node_group.inputs["Bm Base Shape"].identifier
# dag_mod[identifier] = 0

inputs = dag_mod.node_group.inputs
for input_key, input_val in inputs.items():
    print(input_key)
    id = input_val.identifier
    input_type = input_val.type  # INT, VALUE, BOOLEAN, VECTOR are expected
    # if input_type == "INT":
    #     dag_mod[id] = 1
    # elif input_type == "VALUE":
    #     dag_mod[id] = 1.0
    # elif input_type == "BOOLEAN":
    #     dag_mod[id] = True
    # elif input_type == "VECTOR":
    #     for i in range(len(dag_mod[id])):
    #         dag_mod[id][i] = 1.0
    # else:
    #     raise ValueError(f"Unexpected input type: {input_type}")

    



# update obj
building.data.update()
