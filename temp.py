import bpy

# save path
output_path = "./temp.png"
bpy.context.scene.render.filepath = output_path

# Render the viewport
bpy.ops.render.opengl(write_still=True)
