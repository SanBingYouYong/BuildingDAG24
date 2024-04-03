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
    
    @staticmethod
    def load_single_input(dag_mod, id, val):
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
        inputs = dag_mod.node_group.inputs  # if dag_mod unassigned, check which node you are using! change dag_node_name as appropriate
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
    
    def load_dag_params_with_return_part(self, params: DAGParams, 
                                         return_part: int,
                                         return_part_label: str="Return Part",
                                         dag_node_name: str="Building For Distortion Render",
                                         building_name: str="Building",
                                         ):
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
            if input_key == return_part_label:
                self._load_single_input(dag_mod, input_val.identifier, return_part)
                continue
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
    
    def change_return_part(self, return_part: int,
                           return_part_label: str="Return Part",
                           dag_node_name: str="Building For Distortion Render",
                           building_name: str="Building",
                           ):
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
            if input_key == return_part_label:
                self._load_single_input(dag_mod, input_val.identifier, return_part)
                break
        # update obj
        building.data.update()
    
    @staticmethod
    def change_return_part_static(return_part: int,
                                  return_part_label: str="Return Part",
                                  dag_node_name: str="Building For Distortion Render",
                                  building_name: str="Building",
                                  ):
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
            if input_key == return_part_label:
                DAGParamLoader.load_single_input(dag_mod, input_val.identifier, return_part)
                break
        # update obj
        building.data.update()
    
    @staticmethod
    def get_param_vals(param_names: list, dag_node_name: str="Building For Distortion Render",
                      building_name: str="Building",
                      ):
        # get the building object
        building = bpy.data.objects[building_name]
        # get the building's dag node modifier reference
        for mod in building.modifiers:
            if mod.type == "NODES" and mod.node_group.name == dag_node_name:
                dag_mod = mod
                break
        param_values = {}
        # iterate through inputs and get values
        inputs = dag_mod.node_group.inputs
        for input_key, input_val in inputs.items():
            if input_key in param_names:
                val = dag_mod[input_val.identifier]
                val_type = input_val.type
                if val_type == "VECTOR":
                    val = list(val)
                elif val_type == "BOOLEAN":
                    val = bool(val)
                elif val_type == "INT":
                    val = int(val)
                elif val_type == "VALUE":
                    val = float(val)
                else:
                    raise ValueError(f"Unexpected input type: {val_type}")
                param_values[input_key] = val
        return param_values


if __name__ == "__main__":
    params = DAGParamLoader.get_param_vals(["Return Part", "Bm Size", "Has Window Ledge"])
    print(params)
    # loader = DAGParamLoader()
    # # loader.load_dag_params("./inference/output.yml")
    # loader.load_dag_params_with_return_part(DAGParams("./datasets/sample.yml"), 3)
    # loader.change_return_part(4)
