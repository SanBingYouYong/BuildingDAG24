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


if __name__ == "__main__":
    renderer = DAGRenderer()
    renderer.render("datasets/test_dataset/images/1.png")
