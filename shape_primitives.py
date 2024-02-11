import geometry_script as gs

# SP: Shape Primitives

@gs.tree("SP Cube")
def gs_cube(size: gs.Vector) -> gs.Geometry:
    return gs.cube(size=size)

@gs.tree("SP Cylinder")
def gs_cylinder(radius: gs.FloatDistance, depth: gs.FloatDistance) -> gs.Geometry:
    return gs.cylinder(radius=radius, depth=depth).mesh  # note that top, side and bottom are discarded

@gs.tree("SP Cylinder Scaled")
def gs_cylinder_scaled(size: gs.Vector) -> gs.Geometry:
    '''
    Use size/scale instead of radius and depth to represent shape.
    '''
    # return gs.cylinder(radius=0.5, depth=1).mesh.set_shade_smooth().transform(scale=size)
    return gs.cylinder(radius=0.5, depth=1).mesh.transform(scale=size)

@gs.tree("SP Sphere")
def gs_sphere(radius: gs.FloatDistance) -> gs.Geometry:
    return gs.uv_sphere(radius=radius)

@gs.tree("SP Sphere Scaled")
def gs_sphere_scaled(size: gs.Vector) -> gs.Geometry:
    '''
    Use size/scale instead of radius to represent shape.
    '''
    return gs.uv_sphere().transform(scale=size / 2)

@gs.tree("SP Half Sphere")
def gs_half_sphere(size: gs.Vector) -> gs.Geometry:
    sphere = gs.uv_sphere().transform(scale=size / 2)
    cutter_size = gs.combine_xyz(x=size.x * 2, y=size.y * 2, z=size.z)
    translation_vector = gs.combine_xyz(x=0, y=0, z=-size.z / 2)
    cutter_cube = gs.cube(size=cutter_size).transform(translation=translation_vector)
    shape = gs.mesh_boolean(operation=gs.MeshBoolean.Operation.DIFFERENCE, mesh_1=sphere, mesh_2=[cutter_cube])
    return shape.transform(translation=translation_vector)


