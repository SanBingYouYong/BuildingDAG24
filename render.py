import bpy
import bmesh
import cv2
import numpy as np
import random

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from distortion import DRStraight, DRCylindrical, DRSphered, DRUtils, Dedicated_Renderer
# from building4distortion import building_for_distortion_render
from paramload import DAGParamLoader

COMPONENTS = ["Building Mass", "Roof", "Windows", "Ledges"]
TYPE_PARAMS = ["Bm Base Shape", "Rf Base Shape"]



class DAGRenderer():
    def __init__(self, 
                 temp_file_path: str="./temp.png", 
                 down_angle: float=30., 
                 lr_angle: float=35.,
                 view_angle_disturb_range: float=20.,
                 camera_track_radius: float=2.,
                 building_name: str="Building",
                 node_name: str="Building For Distortion Render") -> None:
        self.temp_file_path = temp_file_path
        self.camera = bpy.data.objects["Camera"]
        self.camera_track = bpy.data.objects["CameraTrack"]
        self.down_angle = down_angle
        self.lr_angle = lr_angle
        self.view_angle_disturb_range = view_angle_disturb_range
        self.camera_track_radius = camera_track_radius
        self.building_name = building_name
        self.node_name = node_name
    
    def _post_process(self, file_path: str=None):
        path = file_path if file_path else self.temp_file_path
        # Load the image
        image = cv2.imread(path)
        # Convert the image to greyscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding to make background black and lines white
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # save image
        cv2.imwrite(path, thresholded_image)

    def _update_camera(self):
        new_down_angle = self.down_angle + np.random.uniform(-self.view_angle_disturb_range, self.view_angle_disturb_range)
        new_lr_angle = self.lr_angle + np.random.uniform(-self.view_angle_disturb_range, self.view_angle_disturb_range)
        # sin(h/2) = angle
        height = 2 * np.arcsin(np.deg2rad(new_down_angle))
        # cos(h/2) = radius
        radius = 2 * np.arccos(np.deg2rad(new_down_angle))
        self.camera_track.rotation_euler[2] = np.deg2rad(new_lr_angle)
        self.camera_track.location[2] = height
        track_radius_scaler = radius / self.camera_track_radius
        self.camera_track.scale = (track_radius_scaler, track_radius_scaler, track_radius_scaler)

    def render(self, file_path: str=None, distortion: bool=False):
        # update camera position
        self._update_camera()
        # render image
        path = file_path if file_path else self.temp_file_path
        bpy.context.scene.render.filepath = path
        if not distortion:
            bpy.ops.render.render(write_still=True)
        else:
            # render with distortion
            self.render_with_distortion()
        # post process
        self._post_process(path)

    def render_with_distortion(self):
        # check if the distortion renderer node is present
        building = bpy.data.objects[self.building_name]
        # set building as active
        bpy.context.view_layer.objects.active = building
        building.select_set(True)
        node = None
        for mod in building.modifiers:
            if mod.type == "NODES" and mod.node_group.name == self.node_name:
                node = mod
                break
        if not node:
            raise ValueError(f"Distortion renderer node '{self.node_name}' not found")
        # get component shape types
        shape_types = DAGParamLoader.get_param_vals(TYPE_PARAMS)
        # choose de-reg level for this shot
        de_reg = random.choice(list(DRUtils.DeRegLevels)[1:])  # perfect and light too similar
        # manual: 
        # rf
        DAGParamLoader.change_return_part_static(2)
        renderer = self._get_renderer("Roof", shape_types)
        rf_dup = self._get_actual_mesh(building, reset_active=False)  # leave the active mesh "selected"
        rf_curves = renderer.obj_to_curves_only(rf_dup, de_reg)
        # bm
        DAGParamLoader.change_return_part_static(1)
        renderer = self._get_renderer("Building Mass", shape_types)
        bm_dup = self._get_actual_mesh(building, reset_active=False)
        # rf_dup.select_set(True)  # breaks for cylindrical shapes
        bm_curves = renderer.obj_to_curves_only(bm_dup, de_reg)
        # windows
        DAGParamLoader.change_return_part_static(3)
        renderer = self._get_renderer("Windows", shape_types)
        windows_dup = self._get_actual_mesh(building, reset_active=False)
        bm_dup.select_set(True)  # block invisible windows
        windows_curves = renderer.obj_to_curves_only(windows_dup, de_reg)
        bm_dup.select_set(False)
        # ledges
        DAGParamLoader.change_return_part_static(4)
        renderer = self._get_renderer("Ledges", shape_types)
        ledges_dup = self._get_actual_mesh(building, reset_active=False)
        ledges_curves = renderer.obj_to_curves_only(ledges_dup, de_reg)
        # # reset building as active
        # self._set_as_active(building)
        # combine and mark
        curves = rf_curves + bm_curves + windows_curves + ledges_curves
        extruded_curves = DRUtils.mark_curves_as_freestyle(curves)
        # render
        bpy.data.objects.remove(windows_dup)  # rf and bm mesh can block some invisible curves
        bpy.data.objects.remove(ledges_dup)
        # shrink bm and rf by scaling
        bm_dup.scale = (0.99, 0.99, 0.99)
        rf_dup.scale = (0.99, 0.99, 0.99)
        # update: no shink. select visible objects and exclude bm and rf, delete invisible curves
        
        building.hide_render = True
        bpy.ops.render.render(write_still=True)
        # clean up
        bpy.data.objects.remove(rf_dup)
        bpy.data.objects.remove(bm_dup)
        for curve in extruded_curves:
            bpy.data.objects.remove(curve)
        building.hide_render = False
        # set back to full model display
        DAGParamLoader.change_return_part_static(0)

    def _get_renderer(self, component_name, shape_types):
        if component_name == "Building Mass":
            if shape_types["Bm Base Shape"] == 0:
                return DRStraight()
            elif shape_types["Bm Base Shape"] == 1:
                return DRCylindrical()
        elif component_name == "Roof":
            if shape_types["Rf Base Shape"] == 0:
                return DRStraight()
            elif shape_types["Rf Base Shape"] == 1:
                return DRCylindrical()
            elif shape_types["Rf Base Shape"] == 2:
                return DRSphered()
        elif component_name == "Windows":
            return DRStraight()
        elif component_name == "Ledges":
            # if shape_types["Bm Base Shape"] == 0:
            #     return DRStraight()
            # elif shape_types["Bm Base Shape"] == 1:
            #     return DRCylindrical()
            return DRStraight()  # cylinders mess up everything... 
        else:
            raise ValueError(f"Unexpected component name: {component_name}")

    
    def _get_visibles(self, obj, mode="EDGE", hide_mesh=True):
        '''
        Get visible elements from actual mesh duplicate of obj, and by default hide the generated dup mesh. 
        '''
        dup = self._get_actual_mesh(obj)
        self._set_as_active(dup)
        obj.hide_viewport = True
        verts, edges, faces = DRUtils.get_visibles(dup, mode)
        obj.hide_viewport = False
        if hide_mesh:
            dup.hide_viewport = True
        self._set_as_active(obj)
        return verts, edges, faces

    def _set_as_active(self, obj):
        '''
        De-select and de-activate the current obj, then select and activate the new obj.
        '''
        bpy.context.object.select_set(False)
        bpy.context.view_layer.objects.active = obj
        bpy.context.object.select_set(True)

    def _get_actual_mesh(self, building, reset_active=True):
        '''
        Copy paste a building from dag node and convert the duplicate to actual mesh, then return the mesh.
        '''
        self._set_as_active(building)
        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        bpy.ops.object.convert(target='MESH')
        if reset_active:
            self._set_as_active(building)
        return dup
    
    def update_lr_angle(self, lr_angle: float):
        self.lr_angle = lr_angle

    def use_device(self, device: int):
        '''
        -1: CPU
        0 or other int: CUDA, and also GPU index
        '''
        use_device = "CPU" if device == -1 else "CUDA"
        
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()
        devices = cycles_preferences.devices
        # print(devices.items())

        if not devices:
            raise RuntimeError("Unsupported device type")

        for available_device in devices:
            available_device.use = False  # disable all first
        if use_device == "CPU":
            bpy.context.scene.cycles.device = "CPU"
            for available_device in devices:
                if available_device.type == "CPU":
                    print('activated cpu', available_device.name)
                    available_device.use = True
        else:
            bpy.context.scene.cycles.device = "GPU"
            gpu_counter = 0
            for available_device in devices:
                if available_device.type == "CUDA" and gpu_counter == device:
                    print('activated gpu', available_device.name)
                    available_device.use = True
                gpu_counter += 1
        cycles_preferences.compute_device_type = "NONE" if device == -1 else "CUDA"
        # cycles_preferences.refresh_devices()
        # return the activated device
        # for available_device in devices:
        #     print(available_device.name, available_device.type, available_device.use)
        return [device.name for device in devices if device.use]
    
    def check_devices(self):
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.refresh_devices()
        devices = cycles_preferences.devices
        for available_device in devices:
            print(available_device.name, available_device.type, available_device.use)

    # def use_device(self, device: int):
    #     preferences = bpy.context.preferences
    #     cycles_preferences = preferences.addons["cycles"].preferences
    #     cycles_preferences.refresh_devices()
    #     devices = cycles_preferences.devices

    #     if not devices:
    #         raise RuntimeError("Unsupported device type")
        
    #     use_cpu = device == -1
    #     device_type = "CPU" if use_cpu else "CUDA"

    #     activated_gpus = []
    #     for device in devices:
    #         if device.type == "CPU":
    #             device.use = use_cpu
    #         else:
    #             device.use = not use_cpu
    #             activated_gpus.append(device.name)
    #             print('activated gpu', device.name)

    #     cycles_preferences.compute_device_type = device_type
    #     bpy.context.scene.cycles.device = "GPU"

    #     return activated_gpus


if __name__ == "__main__":
    renderer = DAGRenderer()
    # renderer.render("datasets/test_dataset/images/1.png")
    renderer.render_with_distortion()
