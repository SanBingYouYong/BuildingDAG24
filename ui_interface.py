import bpy
from bpy.types import Context

import typing
import os
import sys
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
import logging
import numpy as np
import yaml
from PIL import Image

import torch
from torch.utils.data import DataLoader

# sys.path.append("/media/Work/itx_ubuntu_work/BuildingDAG24")

# import local modules
file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from params import DAGParams
from paramgen import DAGParamGenerator
from paramload import DAGParamLoader
from render import DAGRenderer
from tqdm import tqdm

from nn_models import EncoderDecoderModel
from ui_external_inference import inference, batch_inference


def resize_and_convert(img_path: str, invert=True) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((512, 512))
    # convert to grayscale and save as RGB
    binarized_image = resized_image.convert("L")
    if invert:
        # invert the image
        binarized_image = Image.eval(binarized_image, lambda x: 255 - x)
    # convert back to RGB
    binarized_image = binarized_image.convert("RGB")
    binarized_image.save(img_path)
    print(f"PIL saved img to {img_path}")


def load_param_to_shape(yml_path: str=None):
    loader = DAGParamLoader()
    path = yml_path if yml_path else "./inference/output.yml"
    loader.load_dag_params(path)

class CaptureAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.capture_annotation_operator"
    bl_label = "Capture Annotation & GeoCode"

    def execute(self, context: Context):
        print("You've called Capture Annotation & GeoCode Inference.")

        # get domain
        scene = context.scene
        img_path = "./inference/sketch.png"  # TODO: extract this to a global variable
        obj = bpy.data.objects["Building"]

        # hide "procedural shape"
        obj.hide_viewport = True
        # hide camera (and its background image)
        bpy.data.objects["Camera"].hide_viewport = True

        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        resize_and_convert(img_path)

        # gc_single_image_inference_entrypoint(domain)
        inference()
        # param2obj_entrypoint(domain, obj)
        load_param_to_shape()

        # bring it back in
        obj.hide_viewport = False
        bpy.data.objects["Camera"].hide_viewport = False
        return {"FINISHED"}


class ClearAllAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.clear_all_annotation_operator"
    bl_label = "Clear All Annotation"

    def execute(self, context: Context):
        print("You've called Clear All Annotation.")
        scene = context.scene
        for annotation_layer in scene.grease_pencil.layers:
            annotation_layer.clear()
        return {"FINISHED"}

class ClearBackgroundImageOperator(bpy.types.Operator):
    bl_idname = "object.clear_background_image_operator"
    bl_label = "Clear Background Image"

    def execute(self, context: Context):
        print("You've called Clear Background Image.")
        context.scene.background_image_path = ""  # Clear the background image path
        update_camera_background_image(context)  # Update the camera background image
        return {"FINISHED"}

class ToggleCameraViewOperator(bpy.types.Operator):
    bl_idname = "object.toggle_camera_view_operator"
    bl_label = "Toggle Camera View"

    def execute(self, context: Context):
        print("You've called Toggle Camera View.")
        bpy.ops.view3d.view_camera()
        return {"FINISHED"}

class ImageInferenceOperator(bpy.types.Operator):
    bl_idname = "object.image_inference_operator"
    bl_label = "Image Inference"

    def execute(self, context: Context):
        print("You've called Image Inference.")
        # copy paste background image to corresponding dataset of domain
        scene = context.scene
        img_path = "./inference/sketch.png"
        # translate background image path to path that shutil recognise
        img_path = bpy.path.abspath(img_path)
        inf_img_path = bpy.path.abspath(scene.background_image_path)
        shutil.copyfile(inf_img_path, img_path)
        resize_and_convert(img_path, invert=not scene.proper_background_image)
        # gc_single_image_inference_entrypoint(domain)
        inference()
        # param2obj_entrypoint(domain, obj)
        load_param_to_shape()
        return {"FINISHED"}

class BatchRenderOperator(bpy.types.Operator):
    bl_idname = "object.batch_render_operator"
    bl_label = "Batch Render"

    def execute(self, context: Context):
        print("You've called Batch Render.")
        # get domain
        scene = context.scene
        obj = bpy.data.objects["Building"]
        old_alpha = scene.background_image_opacity
        old_background = scene.background_image_path
        scene.background_image_opacity = 0.1
        # get images folder
        images_folder = bpy.path.abspath(scene.batch_render_images_folder)
        if not os.path.exists(images_folder):
            print(f"Folder {images_folder} does not exist")
            return {"CANCELLED"}
        # attempt to make the "pred" dir inside the folder
        pred_folder = os.path.join(images_folder, "pred")
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        # set inference image path
        inference_image_path = "./inference/sketch.png"

        print(f"Batch rendering images in {images_folder}")
        
        imgs = os.listdir(images_folder)

        params = batch_inference(imgs, images_folder, pred_folder)

        for (img_path, param_path) in params:
            # param2obj_entrypoint(domain, obj)
            load_param_to_shape(param_path)
            scene.background_image_path = img_path
            # render viewport
            img_name = param_path.replace(".yml", ".png")
            bpy.context.scene.render.filepath = os.path.join(pred_folder, img_name)
            bpy.ops.render.opengl(write_still=True)
            print(f"Rendered {img_name}")
        
        # clean up
        scene.background_image_opacity = old_alpha
        scene.background_image_path = old_background
        load_param_to_shape()  # load back original shape params
        return {"FINISHED"}


class GeoCodeInterfacePanel(bpy.types.Panel):
    bl_label = "GeoCode Interface Panel"
    bl_idname = "SIUI_PT_GeoCodeInterfacePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="GeoCode")
        # box.prop(scene, "geocode_domain_options", text="GeoCode Domain")
        box.operator("object.capture_annotation_operator")

        box = layout.box()
        box.label(text="View")
        # Toggle to show/hide the current shape
        box.prop(scene, "show_current_shape", text="Show Current Shape")
        box.prop(scene, "slider_value", text="Horizontal View Angle (left/right)")
        box.prop(scene, "slider_value_updown", text="Vertical View Angle (up/down)")
        # switch to camera view
        box.operator("object.toggle_camera_view_operator", text="Toggle Camera View")
        box.operator("object.clear_all_annotation_operator")

        box = layout.box()
        box.label(text="Background Image")
        box.prop(scene, "show_background_image", text="Show Background Image")
        # Background image path input
        box.prop(scene, "background_image_path", text="Background Image")
        # Background image opacity slider
        box.prop(scene, "background_image_opacity", text="Opacity")
        # Clear background image button
        box.operator("object.clear_background_image_operator", text="Clear Background Image")
        # use a pre-saved image to inference
        box.label(text="Use This Image For Inference")
        box.prop(scene, "proper_background_image", text="Image is Processed Already")
        # Inference button
        box.operator("object.image_inference_operator", text="Image Inference")
        # Batch render button
        box.prop(scene, "batch_render_images_folder", text="Batch Render Images Folder")
        box.operator("object.batch_render_operator", text="Batch Render")


def update_slider_value(self, context):
    print("slider value updated")
    scene = context.scene
    print(scene.slider_value)
    # update z rotation of "CameraTrack" object accordingly
    bpy.data.objects["CameraTrack"].rotation_euler[2] = (
        scene.slider_value * 3.1415926 / 180.0
    )
    print("updated camera rotation")

def update_slider_value_combined(self, context):
    print("Combined update on camera rotation")
    scene = context.scene
    new_lr_angle = scene.slider_value
    new_down_angle = scene.slider_value_updown
    # sin(h/2) = angle
    height = 2 * np.arcsin(np.deg2rad(new_down_angle))
    # cos(h/2) = radius
    radius = 2 * np.arccos(np.deg2rad(new_down_angle))
    camera_track = bpy.data.objects["CameraTrack"]
    camera_track.rotation_euler[2] = np.deg2rad(new_lr_angle)
    camera_track.location[2] = height
    track_radius_scaler = radius / 2
    camera_track.scale = (track_radius_scaler, track_radius_scaler, track_radius_scaler)


def update_camera_background_image(context):
    camera = context.space_data.camera
    if camera is not None:
        # Set the background image for the camera
        camera.data.background_images.clear()
        if context.scene.background_image_path != "":
            camera.data.show_background_images = True
            bg_img = camera.data.background_images.new()
            bg_img.image = bpy.data.images.load(context.scene.background_image_path)
            bg_img.alpha = context.scene.background_image_opacity
        else:
            camera.data.show_background_images = False

def update_camera_background_opacity(context):
    camera = context.space_data.camera
    if camera is not None:
        # Update the background image opacity for the camera
        for bg_img in camera.data.background_images:
            bg_img.alpha = context.scene.background_image_opacity


def update_background_image_path(self, context):
    # Update the camera background image when the path is changed
    update_camera_background_image(context)

def update_background_image_opacity(self, context):
    # Update the camera background image opacity when the opacity is changed
    update_camera_background_opacity(context)

def update_show_current_shape(self, context):
    # Update the visibility of the current shape
    scene = context.scene
    object_name = "Building"
    if object_name in bpy.data.objects:
        bpy.data.objects[object_name].hide_viewport = not scene.show_current_shape

def update_show_background_image(self, context):
    # Update whether camera background image is shown
    camera = context.space_data.camera
    if camera is not None:
        camera.data.show_background_images = context.scene.show_background_image




def register():
    # bpy.types.Scene.annotation_image_path: bpy.types.StringProperty = bpy.props.StringProperty(
    #     name="Annotation Image Path Property",
    #     subtype='FILE_PATH',
    #     default="//datasets//SingleImg//test//sketches//single_img_-30.0_15.0.png"
    # )

    bpy.types.Scene.slider_value = bpy.props.FloatProperty(
        name="View Angle (horizontal)", default=0.0, min=0.0, max=90.0, update=update_slider_value_combined
    )
    bpy.types.Scene.slider_value_updown = bpy.props.FloatProperty(
        name="View Angle (vertical)", default=30.0, min=0.0, max=50.0, update=update_slider_value_combined
    )
    bpy.types.Scene.background_image_path = bpy.props.StringProperty(
        name="Background Image Path",
        subtype='FILE_PATH',
        default="",
        description="Path to the background image for reference in camera view",
        update=update_background_image_path
    )
    bpy.types.Scene.background_image_opacity = bpy.props.FloatProperty(
        name="Background Image Opacity",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Opacity of the background image in camera view",
        update=update_background_image_opacity
    )
    bpy.types.Scene.show_current_shape = bpy.props.BoolProperty(
        name="Show Current Shape",
        default=True,
        description="Toggle to show/hide the current shape in the viewport",
        update=update_show_current_shape
    )
    bpy.types.Scene.show_background_image = bpy.props.BoolProperty(
        name="Show Background Image",
        default=True,
        description="Toggle to show/hide the background image in the viewport",
        update=update_show_background_image
    )
    bpy.types.Scene.proper_background_image = bpy.props.BoolProperty(
        name="Proper Background Image",
        default=True,
        description="Whether the background image is a properly processed image or a sketch straight from datasets",
    )
    bpy.types.Scene.batch_render_images_folder = bpy.props.StringProperty(
        name="Batch Render Images Folder",
        subtype='DIR_PATH',
        default="",
        description="Folder containing images to batch render",
    )




    bpy.utils.register_class(CaptureAnnotationOperator)
    bpy.utils.register_class(ClearAllAnnotationOperator)
    bpy.utils.register_class(GeoCodeInterfacePanel)
    bpy.utils.register_class(ClearBackgroundImageOperator)
    bpy.utils.register_class(ToggleCameraViewOperator)
    bpy.utils.register_class(ImageInferenceOperator)
    bpy.utils.register_class(BatchRenderOperator)


def unregister():
    # del bpy.types.Scene.annotation_image_path

    bpy.utils.unregister_class(CaptureAnnotationOperator)
    bpy.utils.unregister_class(ClearAllAnnotationOperator)
    bpy.utils.unregister_class(GeoCodeInterfacePanel)
    bpy.utils.unregister_class(ClearBackgroundImageOperator)
    bpy.utils.unregister_class(ToggleCameraViewOperator)
    bpy.utils.unregister_class(ImageInferenceOperator)
    bpy.utils.unregister_class(BatchRenderOperator)


if __name__ == "__main__":
    register()
