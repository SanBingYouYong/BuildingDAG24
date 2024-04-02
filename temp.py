import bpy


# render
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = 'temp.png'
bpy.ops.render.render(write_still=True)
