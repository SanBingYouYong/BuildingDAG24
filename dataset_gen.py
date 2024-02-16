import os
import bpy
import yaml
import numpy as np

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))
sys.path.append(str(file.parents[0]))

from params import DAGParams
from paramgen import DAGParamGenerator
from paramload import DAGParamLoader
from render import DAGRenderer
from tqdm import tqdm


class DAGDatasetGenerator():
    def __init__(self, 
                 dataset_name: str, 
                 dataset_root_path: str="./datasets", 
                 mkdir: bool=True) -> None:
        self.dataset_name = dataset_name
        self.dataset_root_path = dataset_root_path
        self.dataset_path = os.path.join(self.dataset_root_path, self.dataset_name)
        self.dataset_images_folder = os.path.join(self.dataset_path, "images")
        self.dataset_params_folder = os.path.join(self.dataset_path, "params")
        if mkdir:
            self._mkdirs()
        self.param_generator = DAGParamGenerator()
        self.param_loader = DAGParamLoader()
        self.param_renderer = DAGRenderer()
    
    def _mkdirs(self):
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.dataset_images_folder, exist_ok=True)
        os.makedirs(self.dataset_params_folder, exist_ok=True)

    def populate_dataset_wrt_batches(self, num_batches: int, batch_size: int=10, num_varying_params: int=5, render: bool=True):
        # save ranges
        ranges_path = os.path.join(self.dataset_path, "ranges.yml")
        ranges = self.param_generator.save_ranges()
        with open(ranges_path, "w") as f:
            yaml.dump(ranges, f)
        batch_cam_angles = {}
        for i in tqdm(range(num_batches)):
            batch = self.param_generator.generate_batch_params(
                num_varying_params=num_varying_params, count=batch_size
            )
            # choose a cam angle
            cam_angle = int(np.random.choice(self.param_generator.cam_angles))
            batch_cam_angles[f"batch{i}"] = cam_angle
            self.param_renderer.update_lr_angle(cam_angle)
            for j, param in enumerate(batch):
                sample_id = "batch{}_sample{}".format(i, j)
                sample_params_path = os.path.join(self.dataset_params_folder, f"{sample_id}.yml")
                sample_param = DAGParams()
                sample_param.set_params(param)
                sample_param.save_params(sample_params_path)
                if render:
                    # load shape params into blender
                    self.param_loader.load_dag_params(sample_params_path)
                    # render image
                    sample_image_path = os.path.join(self.dataset_images_folder, f"{sample_id}.png")
                    self.param_renderer.render(sample_image_path)
        self.write_batch_cam_angles(batch_cam_angles)
    
    def write_decoders(self):
        '''
        Output one extra file along with ranges.yml 

            decoders.yml:  

                decoder_name: [param_names]
        '''
        decoders = self.param_generator.save_decoders()
        decoders_path = os.path.join(self.dataset_path, "decoders.yml")
        with open(decoders_path, "w") as f:
            yaml.dump(decoders, f)
# TODO: refactor these to one yml file maybe
    def write_switches(self):
        switches = self.param_generator.save_switches()
        switches_path = os.path.join(self.dataset_path, "switches.yml")
        with open(switches_path, "w") as f:
            yaml.dump(switches, f)

    def write_batch_cam_angles(self, batch_cam_angles):
        batch_cam_angles_path = os.path.join(self.dataset_path, "batch_cam_angles.yml")
        with open(batch_cam_angles_path, "w") as f:
            yaml.dump(batch_cam_angles, f)

    def use_device(self, device: int):
        '''
        -1: CPU
        0 or int index: GPU
        '''
        self.param_renderer.use_device(device)


if __name__ == "__main__":
    generator = DAGDatasetGenerator("test_dataset")
    devices = generator.use_device(0)
    print(devices)
    # # check for args
    # args = sys.argv[1:]
    # if len(args) > 0:
    #     num_batches = int(args[0])
    #     batch_size = int(args[1])
    #     num_varying_params = int(args[2])
    #     generator.populate_dataset_wrt_batches(num_batches, batch_size, num_varying_params)
    # else:
    #     generator.populate_dataset_wrt_batches(10, 10, 5)

    # generator.populate_dataset_wrt_batches(10, 10, 5)
    # generator.write_decoders()
    # generator.write_switches()
