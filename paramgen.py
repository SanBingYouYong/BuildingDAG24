import numpy as np
from pprint import pprint


class DParamBool():
    def __init__(self, name: str, possible_values: list) -> None:
        self.name = name
        self.values = possible_values

    def get_bin_indices(self):
        return list(range(len(self.values)))

    def sample_within_bin(self, bin_index: int):
        return self.values[bin_index]  # for booleans, bin_index is the same as the value
    
    def sample_any_bins(self):
        return bool(np.random.choice(self.values))
    
    def save_ranges(self):
        return {
            "name": self.name,
            "type": "bool",
            "values": self.values,
        }

        

class DParamStates():
    def __init__(self, name: str, possible_values: list) -> None:
        self.name = name
        self.values = possible_values

    def get_bin_indices(self):
        return list(range(len(self.values)))
    
    def sample_within_bin(self, bin_index: int):
        return self.values[bin_index]  # similar to bools
    
    def sample_any_bins(self):
        return int(np.random.choice(self.values))
    
    def save_ranges(self):
        return {
            "name": self.name,
            "type": "states",
            "values": self.values,
        }

class DParamInt():
    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max
        self.num_bins = self.max - self.min + 1
        # set up bins
        self.bins = np.arange(self.min, self.max + 1)

    def get_bin_indices(self):
        return list(range(self.num_bins))
    
    def sample_within_bin(self, bin_index: int):
        return int(self.bins[bin_index])
    
    def sample_any_bins(self):
        # decide bin and then sample within bin
        bin_index = np.random.randint(0, self.num_bins)
        return self.sample_within_bin(bin_index)
    
    def save_ranges(self):
        return {
            "name": self.name,
            "type": "int",
            "min": self.min,
            "max": self.max
        }

class DParamFloat():
    def __init__(self, name: str, min: float, max: float, num_bins: int=10):
        self.name = name
        self.min = min
        self.max = max
        self.num_bins = num_bins
        # set up bins
        self.bins = np.linspace(self.min, self.max, self.num_bins + 1)

    def get_bin_indices(self):
        return list(range(self.num_bins))
    
    def sample_within_bin(self, bin_index: int):
        return float(np.random.uniform(self.bins[bin_index], self.bins[bin_index + 1]))
    
    def sample_any_bins(self):
        # decide bin and then sample within bin
        bin_index = np.random.randint(0, self.num_bins)
        return self.sample_within_bin(bin_index)
    
    def save_ranges(self):
        return {
            "name": self.name,
            "type": "float",
            "min": self.min,
            "max": self.max
        }



class DParamVectorF():
    def __init__(
        self,
        name: str,
        xmin: float, xmax: float, 
        ymin: float, ymax: float, 
        zmin: float, zmax: float, 
        num_xbins: int=10, num_ybins: int=10, num_zbins: int=10
    ) -> None:
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        # note that we use different bin choices for each dimension to avoid too cubic shapes
        self.num_xbins = num_xbins
        self.num_ybins = num_ybins
        self.num_zbins = num_zbins
        # set up bins
        self.xbins = np.linspace(self.xmin, self.xmax, self.num_xbins + 1)
        self.ybins = np.linspace(self.ymin, self.ymax, self.num_ybins + 1)
        self.zbins = np.linspace(self.zmin, self.zmax, self.num_zbins + 1)

    def get_bin_indices(self):
        # return 3 ranges
        return [
            list(range(self.num_xbins)),
            list(range(self.num_ybins)),
            list(range(self.num_zbins)),
        ]
    
    def sample_within_bin(self, index_xbin: int, index_ybin: int, index_zbin: int):
        return [
            np.random.uniform(self.xbins[index_xbin], self.xbins[index_xbin + 1]),
            np.random.uniform(self.ybins[index_ybin], self.ybins[index_ybin + 1]),
            np.random.uniform(self.zbins[index_zbin], self.zbins[index_zbin + 1]),
        ]
        
    def sample_any_bins(self):
        # decide bin and then sample within bin
        index_xbin = np.random.randint(0, self.num_xbins)
        index_ybin = np.random.randint(0, self.num_ybins)
        index_zbin = np.random.randint(0, self.num_zbins)
        return self.sample_within_bin(index_xbin, index_ybin, index_zbin)
    
    def save_ranges(self):
        return {
            "name": self.name,
            "type": "vector",
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "zmin": self.zmin,
            "zmax": self.zmax,
        }


class DAGParamGenerator():
    def __init__(self):
        self.bm_base_shape = DParamStates("Bm Base Shape", [0, 1])
        self.bm_size = DParamVectorF("Bm Size", 
                                     0.5, 1.0, 
                                     0.5, 1.0, 
                                     0.5, 1.0)  # bm size seems to need more refinement
        self.num_floors = DParamInt("Num Floors", 1, 5)
        self.rf_base_shape = DParamStates("Rf Base Shape", [0, 1, 2])
        self.rf_size = DParamVectorF("Rf Size", 
                                     0.5, 1.0, 
                                     0.5, 1.0, 
                                     0.5, 1.0)
        self.num_windows_each_side = DParamInt("Num Windows Each Side", 1, 5)  
        self.windows_left_right_offset = DParamFloat("Windows Left Right Offset", -1.0, 1.0)
        self.windows_height_offset = DParamFloat("Windows Height Offset", -1.0, 1.0)
        self.window_shape_size = DParamVectorF("Window Shape Size", 
                                               0.5, 1.0, 
                                               0.5, 1.0, 
                                               0.5, 1.0)
        self.window_panel_area = DParamVectorF("Window Panel Area",
                                                0.5, 1.0, 
                                                0.5, 1.0, 
                                                0.5, 1.0)
        self.window_divided_horizontal = DParamBool("Window Divided Horizontal", [True, False])
        self.window_divided_vertical = DParamBool("Window Divided Vertical", [True, False])
        self.window_interpanel_offset_percentage_y = DParamFloat("Window Interpanel Offset Percentage Y", 0.0, 1.0)
        self.window_interpanel_offset_percentage_z = DParamFloat("Window Interpanel Offset Percentage Z", 0.0, 1.0)
        self.has_window_ledge = DParamBool("Has Window Ledge", [True, False])
        self.window_ledge_shape_size = DParamVectorF("Window Ledge Shape Size", 
                                              0.5, 1.0, 
                                              0.5, 1.0, 
                                              0.5, 1.0)
        self.window_ledge_extrusion_x = DParamFloat("Window Ledge Extrusion X", 0.0, 1.0)
        self.window_ledge_extrusion_z = DParamFloat("Window Ledge Extrusion Z", 0.0, 1.0)
        self.window_ledges_height_offset = DParamFloat("Window Ledges Height Offset", 0.0, 1.0)
        self.has_floor_ledge = DParamBool("Has Floor Ledge", [True, False])
        self.floor_ledge_size_x = DParamFloat("Floor Ledge Size X", 0.1, 1.0)
        self.floor_ledge_size_z = DParamFloat("Floor Ledge Size Z", 0.1, 1.0)
        self.floor_ledge_extrusion_x = DParamFloat("Floor Ledge Extrusion X", 0.0, 1.0)
        self.floor_ledge_extrusion_z = DParamFloat("Floor Ledge Extrusion Z", 0.0, 1.0)

        self.params = [
            self.bm_base_shape,
            self.bm_size,
            self.num_floors,
            self.rf_base_shape,
            self.rf_size,
            self.num_windows_each_side,
            self.windows_left_right_offset,
            self.windows_height_offset,
            self.window_shape_size,
            self.window_panel_area,
            self.window_divided_horizontal,
            self.window_divided_vertical,
            self.window_interpanel_offset_percentage_y,
            self.window_interpanel_offset_percentage_z,
            self.has_window_ledge,
            self.window_ledge_shape_size,
            self.window_ledge_extrusion_x,
            self.window_ledge_extrusion_z,
            self.window_ledges_height_offset,
            self.has_floor_ledge,
            self.floor_ledge_size_x,
            self.floor_ledge_size_z,
            self.floor_ledge_extrusion_x,
            self.floor_ledge_extrusion_z,
        ]

        self.four_parts_decoders = {
            "Building Mass Decoder": [
                "Bm Base Shape",
                "Bm Size",
                "Num Floors",
            ],
            "Roof Decoder": [
                "Rf Base Shape",
                "Rf Size",
            ],
            "Window Decoder": [
                "Num Windows Each Side",
                "Windows Left Right Offset",
                "Windows Height Offset",
                "Window Shape Size",
                "Window Panel Area",
                "Window Divided Horizontal",
                "Window Divided Vertical",
                "Window Interpanel Offset Percentage Y",
                "Window Interpanel Offset Percentage Z",
                "Has Window Ledge",
                "Window Ledge Shape Size",
                "Window Ledge Extrusion X",
                "Window Ledge Extrusion Z",
                "Window Ledges Height Offset",
            ],
            "Floor Ledge Decoder": [
                "Has Floor Ledge",
                "Floor Ledge Size X",
                "Floor Ledge Size Z",
                "Floor Ledge Extrusion X",
                "Floor Ledge Extrusion Z",
            ],
        }
        self.divided_decoders = {
            "Building Mass Decoder": [
                "Bm Base Shape",
                "Bm Size",
            ],
            "Facade Decoder": [
                "Num Floors",
                "Num Windows Each Side",
            ],
            "Roof Decoder": [
                "Rf Base Shape",
                "Rf Size",
            ],
            "Window Decoder": [
                "Windows Left Right Offset",
                "Windows Height Offset",
                "Window Shape Size",
                "Window Panel Area",
                "Window Divided Horizontal",
                "Window Divided Vertical",
                "Window Interpanel Offset Percentage Y",
                "Window Interpanel Offset Percentage Z",
            ],
            "Window Ledge Decoder": [
                "Has Window Ledge",
                "Window Ledge Shape Size",
                "Window Ledge Extrusion X",
                "Window Ledge Extrusion Z",
                "Window Ledges Height Offset",
            ],
            "Floor Ledge Decoder": [
                "Has Floor Ledge",
                "Floor Ledge Size X",
                "Floor Ledge Size Z",
                "Floor Ledge Extrusion X",
                "Floor Ledge Extrusion Z",
            ],
        }
        self.detailed_decoders = {
            "Building Mass Decoder": [
                "Bm Base Shape",
                "Bm Size",
            ],
            "Facade Decoder": [
                "Num Floors",
                "Num Windows Each Side",
            ],
            "Roof Decoder": [
                "Rf Base Shape",
                "Rf Size",
            ],
            "Window Main Decoder": [
                "Windows Left Right Offset",
                "Windows Height Offset",
                "Window Shape Size",
            ],
            "Window Panel Decoder": [
                "Window Panel Area",
                "Window Divided Horizontal",
                "Window Divided Vertical",
                "Window Interpanel Offset Percentage Y",
                "Window Interpanel Offset Percentage Z",
            ],
            "Window Ledge Decoder": [
                "Has Window Ledge",
                "Window Ledge Shape Size",
                "Window Ledge Extrusion X",
                "Window Ledge Extrusion Z",
                "Window Ledges Height Offset",
            ],
            "Floor Ledge Decoder": [
                "Has Floor Ledge",
                "Floor Ledge Size X",
                "Floor Ledge Size Z",
                "Floor Ledge Extrusion X",
                "Floor Ledge Extrusion Z",
            ],
        }
        self.optimized_decoders = {
            "Bm Base Shape Classifier": [
                "Bm Base Shape",
            ],
            "Rf Base Shape Classifier": [
                "Rf Base Shape",
            ],
            "Window Divided Horizontal Classifier": [
                "Window Divided Horizontal",
            ],
            "Window Divided Vertical Classifier": [
                "Window Divided Vertical",
            ],
            "Has Window Ledge Classifier": [
                "Has Window Ledge",
            ],
            "Has Floor Ledge Classifier": [
                "Has Floor Ledge",
            ],
            "Bm Rf Size Regressor": [
                "Bm Size",
                "Rf Size",
                "Num Floors",
                "Num Windows Each Side",
            ],
            "Window Main Regressor": [
                "Windows Left Right Offset",
                "Windows Height Offset",
                "Window Shape Size",
            ],
            "Window Panel Regressor": [
                "Window Panel Area",
                "Window Interpanel Offset Percentage Y",
                "Window Interpanel Offset Percentage Z",
            ],
            "Window Ledge Regressor": [
                "Window Ledge Shape Size",
                "Window Ledge Extrusion X",
                "Window Ledge Extrusion Z",
                "Window Ledges Height Offset",
            ],
            "Floor Ledge Regressor": [
                "Floor Ledge Size X",
                "Floor Ledge Size Z",
                "Floor Ledge Extrusion X",
                "Floor Ledge Extrusion Z",
            ],
        }
        self.switches = {
            "Has Window Ledge": [
                "Window Ledge Shape Size",
                "Window Ledge Extrusion X",
                "Window Ledge Extrusion Z",
                "Window Ledges Height Offset",
            ],
            "Has Floor Ledge": [
                "Floor Ledge Size X",
                "Floor Ledge Size Z",
                "Floor Ledge Extrusion X",
                "Floor Ledge Extrusion Z",
            ],
            "Window Divided Horizontal": [
                "Window Interpanel Offset Percentage Y",
            ],
            "Window Divided Vertical": [
                "Window Interpanel Offset Percentage Z",
            ],
            "Reversed Mapping": {
                "Window Ledge Shape Size": "Has Window Ledge",
                "Window Ledge Extrusion X": "Has Window Ledge",
                "Window Ledge Extrusion Z": "Has Window Ledge",
                "Window Ledges Height Offset": "Has Window Ledge",
                "Floor Ledge Size X": "Has Floor Ledge",
                "Floor Ledge Size Z": "Has Floor Ledge",
                "Floor Ledge Extrusion X": "Has Floor Ledge",
                "Floor Ledge Extrusion Z": "Has Floor Ledge",
                "Window Interpanel Offset Percentage Y": "Window Divided Horizontal",
                "Window Interpanel Offset Percentage Z": "Window Divided Vertical",
            }
        }
        self.cam_angles = [30, 35, 40, 45, 50, 55, 60]

    
    def generate_param(self):
        raise DeprecationWarning("Use generate_batch_params instead")
        # np.random.seed(42)
        dag_params = {}
        for param in self.params:
            dag_params[param.name] = param.sample()
        return dag_params
    
    '''
    For a batch, everyone shares same set of varying/fixed params and bins for fixed params
    '''
    def generate_batch_params(self, num_varying_params: int=5, count: int=10):
        # choose varying params
        varying_params_indices = np.random.choice(len(self.params), num_varying_params, replace=False)
        fixed_params_indices = [i for i in range(len(self.params)) if i not in varying_params_indices]
        # choose bin for fixed params
        fixed_params_bins = {}
        for i in fixed_params_indices:
            # if param is vector, choose bin for each dimension
            if isinstance(self.params[i], DParamVectorF):
                bin_indices = self.params[i].get_bin_indices()
                fixed_params_bins[self.params[i].name] = [
                    np.random.choice(bin_indices[0]),
                    np.random.choice(bin_indices[1]),
                    np.random.choice(bin_indices[2]),
                ]
            else:
                fixed_params_bins[self.params[i].name] = np.random.choice(self.params[i].get_bin_indices())
        # sample params
        batch_dag_params = []
        for i in range(count):
            batch_dag_params.append(
                self.generate_param_wrt_bins(varying_params_indices, fixed_params_bins)
            )
        return batch_dag_params
    
    def generate_param_wrt_bins(self, varying_params_indices: list, fixed_params_bins: dict):
        # np.random.seed(42)
        dag_params = {}
        # sample params
        for i, param in enumerate(self.params):
            if i in varying_params_indices:
                dag_params[param.name] = param.sample_any_bins()
            else:
                if isinstance(param, DParamVectorF):
                    dag_params[param.name] = param.sample_within_bin(
                        fixed_params_bins[param.name][0],
                        fixed_params_bins[param.name][1],
                        fixed_params_bins[param.name][2],
                    )
                else:
                    dag_params[param.name] = param.sample_within_bin(fixed_params_bins[param.name])
        return dag_params
    
    def save_ranges(self):
        ranges = {}
        for param in self.params:
            ranges[param.name] = param.save_ranges()
        return ranges
    
    def save_decoders(self, lod: int=2):
        if lod == 0:
            decoders = self.four_parts_decoders
        elif lod == 1:
            decoders = self.divided_decoders
        elif lod == 2:
            decoders = self.detailed_decoders
        elif lod == 3:
            decoders = self.optimized_decoders
        else:
            raise ValueError("lod can only be 0, 1, 2, 3")
        return decoders
    
    def save_switches(self):
        return self.switches


if __name__ == "__main__":
    dag_param_ranges_gen = DAGParamGenerator()
    dag_params = dag_param_ranges_gen.generate_param()
    pprint(dag_params)
        
