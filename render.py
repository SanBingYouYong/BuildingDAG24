import bpy
import cv2
import numpy as np


class DAGRenderer():
    def __init__(self, 
                 temp_file_path: str="./temp.png", 
                 down_angle: float=30., 
                 lr_angle: float=35.,
                 view_angle_disturb_range: float=20.,
                 camera_track_radius: float=2.) -> None:
        self.temp_file_path = temp_file_path
        self.camera = bpy.data.objects["Camera"]
        self.camera_track = bpy.data.objects["CameraTrack"]
        self.down_angle = down_angle
        self.lr_angle = lr_angle
        self.view_angle_disturb_range = view_angle_disturb_range
        self.camera_track_radius = camera_track_radius
    
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

    def render(self, file_path: str=None):
        # update camera position
        self._update_camera()
        # render image
        path = file_path if file_path else self.temp_file_path
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        # post process
        self._post_process(path)
    
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
    renderer.render("datasets/test_dataset/images/1.png")
