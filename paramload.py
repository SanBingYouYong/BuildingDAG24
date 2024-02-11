import bpy

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from params import DAGParams


class DAGParamLoader():
    def __init__(self) -> None:
        self.single_val_inputs = [
            "INT", "VALUE", "BOOLEAN"
        ]
        self.multi_val_inputs = [
            "VECTOR"
        ]

    def _load_single_input(self, dag_mod, id, val):
        dag_mod[id] = val

    def _load_vector_input(self, dag_mod, id, val):
        for i in range(len(dag_mod[id])):
            dag_mod[id][i] = val[i]

    def load_dag_params(self, params_file_path: str, 
                        dag_node_name: str="BuildingDAG",
                        building_name: str="Building",
                        ):
        # load params
        params = DAGParams(params_file_path)
        # get the building object
        building = bpy.data.objects[building_name]
        # get the building's dag node modifier reference
        for mod in building.modifiers:
            if mod.type == "NODES" and mod.node_group.name == dag_node_name:
                dag_mod = mod
                break
        # iterate through inputs and load values
        inputs = dag_mod.node_group.inputs
        for input_key, input_val in inputs.items():
            id = input_val.identifier
            val = params.params[input_key]
            input_type = input_val.type
            if input_type in self.single_val_inputs:
                self._load_single_input(dag_mod, id, val)
            elif input_type in self.multi_val_inputs:
                self._load_vector_input(dag_mod, id, val)
            else:
                raise ValueError(f"Unexpected input type: {input_type}")
        # update obj
        building.data.update()


if __name__ == "__main__":
    loader = DAGParamLoader()
    loader.load_dag_params("te.yaml")
