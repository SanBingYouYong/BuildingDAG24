import bpy
import cv2
import numpy as np

from distortion import DRStraight, DRCylindrical, DRSphered, DRUtils, Dedicated_Renderer
from building4distortion import building_for_distortion_render
import bmesh


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
        node = None
        for mod in building.modifiers:
            if mod.type == "NODES" and mod.node_group.name == self.node_name:
                node = mod
                break
        if not node:
            raise ValueError(f"Distortion renderer node '{self.node_name}' not found")
        # get visible edges on full building
        fb_verts, fb_edges, fb_faces = self._get_visibles(building)
        print(f"Visible edges on full building: {len(fb_edges)}")
        

    
    def _get_visibles(self, obj, mode="EDGE", hide_mesh=True):
        '''
        Get visible elements from actual mesh duplicate of obj, and by default hide the generated dup mesh. 
        '''
        dup = self._get_actual_mesh(obj)
        self._set_as_active(dup)
        verts, edges, faces = DRUtils.get_visibles(dup, mode)
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

    def _get_actual_mesh(self, building):
        '''
        Copy paste a building from dag node and convert the duplicate to actual mesh, then return the mesh.
        '''
        self._set_as_active(building)
        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        bpy.ops.object.convert(target='MESH')
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
