import bpy

obj = bpy.context.active_object
print(f"{obj.visible_get()}")
