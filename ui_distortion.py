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
# from paramload import DAGParamLoader
from distortion import Dedicated_Renderer, DRStraight, DRCylindrical, DRSphered, DRUtils


def find_curves(obj_name: str):
    '''
    Find all curves related to the object with the given name
    '''
    curves = []
    for obj in bpy.data.objects:
        if obj.type == 'CURVE' and obj_name in obj.name and obj.name != obj_name and "BezierCurve_" in obj.name:
            curves.append(obj)
    return curves


class DistortionOperator(bpy.types.Operator):
    bl_idname = "object.distortion_operator"
    bl_label = "Distort Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        dr_type = context.scene.dr_type
        dl = context.scene.distort_level
        obj = bpy.context.active_object  # no need to hide, since editmode will take care of visibility on single obj
        bpy.ops.object.duplicate()  # create duplicate to avoid affecting original object
        dup = bpy.context.active_object
        renderer = None
        if dr_type == 'dr_straight':
            renderer = DRStraight()
        elif dr_type == 'dr_cylinder':
            renderer = DRCylindrical()
        elif dr_type == 'dr_sphere':
            renderer = DRSphered()
        else:
            raise ValueError(f"Invalid DR type: {dr_type}")
        de_reg = None
        if dl == 'dl_perfect':
            de_reg = DRUtils.DeRegLevels.PERFECT
        elif dl == 'dl_light':
            de_reg = DRUtils.DeRegLevels.LIGHT
        elif dl == 'dl_medium':
            de_reg = DRUtils.DeRegLevels.MEDIUM
        elif dl == 'dl_heavy':
            de_reg = DRUtils.DeRegLevels.HEAVY
        else:
            raise ValueError(f"Invalid distortion level: {dl}")
        spawned_curves = renderer.obj_to_curves_only(dup, de_reg=de_reg, obj_name=obj.name)
        bpy.data.objects.remove(dup)
        return {'FINISHED'}

class RenderAsImage(bpy.types.Operator):
    bl_idname = "object.render_as_image"
    bl_label = "Render as Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        img_path = context.scene.image_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = img_path

        freestyle = context.scene.use_freestyle
        if freestyle:
            bpy.context.scene.render.use_freestyle = True
        else:
            bpy.context.scene.render.use_freestyle = False
        
        viewport = context.scene.viewport_render
        if viewport and freestyle:
            self.report({'WARNING'}, "Freestyle is not supported in viewport render")
        if viewport:
            original_engine = bpy.context.scene.render.engine
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.ops.render.opengl(write_still=True)
            bpy.context.scene.render.engine = original_engine
        else:
            original_engine = bpy.context.scene.render.engine
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.ops.render.render(write_still=True)
            bpy.context.scene.render.engine = original_engine
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
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        curves = find_curves(obj.name)
        for curve in curves:
            bpy.data.objects.remove(curve)
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
        box.prop(scene, "dr_type", text="Object Type")
        box.prop(scene, "distort_level", text="Distortion Level")
        box.operator("object.distortion_operator", text="Distort Into Curves")

        # Render as Image
        box = layout.box()
        box.label(text="Render as Image", icon='CAMERA_DATA')
        box.prop(scene, "use_freestyle", text="Use Freestyle")
        box.prop(scene, "viewport_render", text="Viewport Render")
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
        name="Object Type",
        description="Select the type of DR to use"
    )
    bpy.types.Scene.distort_level = bpy.props.EnumProperty(
        items=[ ('dl_perfect', 'Perfect', "No distortion", 'SEQUENCE_COLOR_04', 0),
                ('dl_light', 'Light', "Low level of distortion", 'SEQUENCE_COLOR_03', 0),
                ('dl_medium', 'Medium', "Medium level of distortion", 'SEQUENCE_COLOR_02', 1),
                ('dl_heavy', 'Heavy', "High level of distortion", 'SEQUENCE_COLOR_01', 2)],
        name="Distortion Level",
        description="Select the level of distortion"
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
        subtype='FILE_PATH',
        description="Set the image path",
        default=""
    )
    bpy.types.Scene.use_freestyle = bpy.props.BoolProperty(
        name="Use Freestyle",
        description="Use Freestyle",
        default=True
    )
    bpy.types.Scene.viewport_render = bpy.props.BoolProperty(
        name="Viewport Render",
        description="Capture the viewport as an image",
        default=False
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
