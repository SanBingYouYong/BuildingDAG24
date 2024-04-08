import bpy
from bpy.types import Context

# import typing
import os
import sys
from pathlib import Path
import shutil
# from subprocess import Popen, PIPE
# import logging
import numpy as np
# import yaml
from PIL import Image

# import local modules
file = Path(__file__).resolve()
parent = file.parents[1]
print(f"parent: {parent}")
sys.path.append(str(parent))

# from params import DAGParams
# from paramgen import DAGParamGenerator
from paramload import DAGParamLoader


class DistortionOperator(bpy.types.Operator):
    bl_idname = "object.distortion_operator"
    bl_label = "Distort Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        return {'FINISHED'}

class RenderAsImage(bpy.types.Operator):
    bl_idname = "object.render_as_image"
    bl_label = "Render as Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        return {'FINISHED'}
    
class DistortAndRender(bpy.types.Operator):
    bl_idname = "object.distort_and_render"
    bl_label = "Distort and Render"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        return {'FINISHED'}
    
class DeleteCurrentCurves(bpy.types.Operator):
    bl_idname = "object.delete_current_curves"
    bl_label = "Delete Current Curves"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        return {'FINISHED'}

class DeleteAllCurves(bpy.types.Operator):
    bl_idname = "object.delete_all_curves"
    bl_label = "Delete All Curves"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        return {'FINISHED'}


class DistortionPanel(bpy.types.Panel):
    bl_label = "Distortion Panel"
    bl_idname = "SIUI_PT_DistortionPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # View
        box = layout.box()
        box.label(text="View", icon='CAMERA_STEREO')

        box.prop(scene, "show_current_shape", text="Show Current Shape")
        box.prop(scene, "show_spawned_curves", text="Show Spawned Curves")
        
        box.prop(scene, "cam_angle_hori", text="Camera Horizontal Angle")
        box.prop(scene, "cam_angle_vert", text="Camera Vertical Angle")
        box.prop(scene, "toggle_cam", text="Toggle Camera")

        # Manipulate
        box = layout.box()
        box.label(text="Manipulate", icon='MODIFIER_DATA')
        box.operator("object.delete_current_curves", text="Delete Current Curves")
        box.operator("object.delete_all_curves", text="Delete All Curves")
        
        # Distort Into Curves
        box = layout.box()
        box.label(text="Distort Into Curves", icon='OUTLINER_DATA_CURVES')
        box.prop(scene, "remove_obj_after_spawn", text="Remove Object After Spawn")
        box.operator("object.distortion_operator", text="Distort Into Curves")

        # Render as Image
        box = layout.box()
        box.label(text="Render as Image", icon='CAMERA_DATA')
        box.prop(scene, "image_path", text="Image Path")
        box.operator("object.render_as_image", text="Render as Image")

        # Combined
        box = layout.box()
        box.label(text="Combined", icon='RNA_ADD')
        box.operator("object.distort_and_render", text="Distort and Render")

    
def register():
    bpy.types.Scene.dr_type = bpy.props.EnumProperty(
        items=[('dr_straight', 'Straight Line', "Dedicated Distortion Renderer for objects containing only straight edges", 'META_CUBE', 0),
               ('dr_cylinder', 'Cylindrical', "Dedicated Distortion Renderer for cylinders", 'MESH_CYLINDER', 1),
               ('dr_sphere', 'Half-spherical', "Dedicated Distortion Renderer for half-spheres", 'MATSPHERE', 2)],
        name="DR Type",
        description="Select the type of DR to use"
    )
    # TODO: show/hide both viewport and render? 
    bpy.types.Scene.show_current_shape = bpy.props.BoolProperty(
        name="Show Current Shape",
        description="Show the current shape",
        default=False
    )
    bpy.types.Scene.show_spawned_curves = bpy.props.BoolProperty(
        name="Show Spawned Curves",
        description="Show the spawned curves",
        default=False
    )
    bpy.types.Scene.cam_angle_hori = bpy.props.FloatProperty(
        name="Camera Horizontal Angle",
        description="Set the camera horizontal angle",
        default=45.0,
        min=0.0,
        max=90.0
    )
    bpy.types.Scene.cam_angle_vert = bpy.props.FloatProperty(
        name="Camera Vertical Angle",
        description="Set the camera vertical angle",
        default=30.0, 
        min=0.0,
        max=90.0
    )
    bpy.types.Scene.toggle_cam = bpy.props.BoolProperty(
        name="Toggle Camera",
        description="Toggle the camera",
        default=False
    )
    bpy.types.Scene.remove_obj_after_spawn = bpy.props.BoolProperty(
        name="Remove Object After Spawn",
        description="Remove the object after spawning distorted curves",
        default=False
    )
    bpy.types.Scene.image_path = bpy.props.StringProperty(
        name="Image Path",
        description="Set the image path",
        default=""
    )


    bpy.utils.register_class(DistortionPanel)
    bpy.utils.register_class(DistortionOperator)
    bpy.utils.register_class(RenderAsImage)
    bpy.utils.register_class(DistortAndRender)
    bpy.utils.register_class(DeleteCurrentCurves)
    bpy.utils.register_class(DeleteAllCurves)

def unregister():
    bpy.utils.unregister_class(DistortionPanel)
    bpy.utils.unregister_class(DistortionOperator)
    bpy.utils.unregister_class(RenderAsImage)
    bpy.utils.unregister_class(DistortAndRender)
    bpy.utils.unregister_class(DeleteCurrentCurves)
    bpy.utils.unregister_class(DeleteAllCurves)


if __name__ == "__main__":
    register()
