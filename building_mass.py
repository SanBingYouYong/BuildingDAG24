import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from shape_primitives import gs_cube, gs_cylinder, gs_cylinder_scaled, gs_sphere, gs_sphere_scaled


# TODO: add operations for creating more complex shapes


@gs.tree("Building Mass")
def building_mass(bm_type: gs.Int, bm_size: gs.Vector):
    """
    Generate a building mass based on the given shape and size.
    Note: sphere (2) is not implemented now.

    Parameters:
    - bm_type (gs.Int): The shape of the building mass. 0 for cube, 1 for cylinder.
    - bm_size (gs.Vector): The size of the building mass. Primitives shapes use a unified xyz scale to represent size.

    Returns:
    - The generated building mass.
    - Four curve line Geometries representing the four sides of the building mass. position at the center of the bottom floor.
    """
    # use compare to determine shape type: currently switch nodes can have T/F outputs only, so 3 types need 2 compares (cube?cylinder?sphere)
    is_cube = gs.compare(
        operation=gs.Compare.Operation.EQUAL,
        data_type=gs.Compare.DataType.INT,
        a=bm_type,
        b=0,
    )
    is_cylinder = gs.compare(
        operation=gs.Compare.Operation.EQUAL,
        data_type=gs.Compare.DataType.INT,
        a=bm_type,
        b=1,
    )
    cube = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs_cube(size=bm_size),
    )
    # edges
    cube_line_positive_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs.curve_line(
            start=gs.combine_xyz(x=bm_size.x / 2, y=-bm_size.y / 2, z=-bm_size.z / 2),
            end=gs.combine_xyz(x=bm_size.x / 2, y=bm_size.y / 2, z=-bm_size.z / 2),
        ),
    )
    cube_line_positive_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs.curve_line(
            start=gs.combine_xyz(x=bm_size.x / 2, y=bm_size.y / 2, z=-bm_size.z / 2),
            end=gs.combine_xyz(x=-bm_size.x / 2, y=bm_size.y / 2, z=-bm_size.z / 2),
        ),
    )
    cube_line_negative_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs.curve_line(
            start=gs.combine_xyz(x=-bm_size.x / 2, y=bm_size.y / 2, z=-bm_size.z / 2),
            end=gs.combine_xyz(x=-bm_size.x / 2, y=-bm_size.y / 2, z=-bm_size.z / 2),
        ),
    )
    cube_line_negative_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs.curve_line(
            start=gs.combine_xyz(x=-bm_size.x / 2, y=-bm_size.y / 2, z=-bm_size.z / 2),
            end=gs.combine_xyz(x=bm_size.x / 2, y=-bm_size.y / 2, z=-bm_size.z / 2),
        ),
    )
    depth = bm_size.z
    cylinder = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs_cylinder_scaled(size=bm_size),
    )
    nighty_degree = gs.math(operation=gs.Math.Operation.RADIANS, value=90)
    deg_45 = gs.math(operation=gs.Math.Operation.RADIANS, value=45)
    deg_135 = gs.math(operation=gs.Math.Operation.RADIANS, value=135)
    deg_225 = gs.math(operation=gs.Math.Operation.RADIANS, value=225)
    cylinder_line_positive_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.arc(
            radius=0.5,
            start_angle=-deg_45, 
            sweep_angle=nighty_degree
        ).transform(
            translation=gs.combine_xyz(x=0, y=0, z=-depth / 2),
            scale=bm_size)
    )
    cylinder_line_positive_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.arc(
            radius=0.5,
            start_angle=deg_45, 
            sweep_angle=nighty_degree
        ).transform(
            translation=gs.combine_xyz(x=0, y=0, z=-depth / 2),
            scale=bm_size)
    )
    cylinder_line_negative_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.arc(
            radius=0.5,
            start_angle=deg_135, 
            sweep_angle=nighty_degree
        ).transform(
            translation=gs.combine_xyz(x=0, y=0, z=-depth / 2),
            scale=bm_size)
    )
    cylinder_line_negative_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.arc(
            radius=0.5,
            start_angle=deg_225, 
            sweep_angle=nighty_degree
        ).transform(
            translation=gs.combine_xyz(x=0, y=0, z=-depth / 2),
            scale=bm_size)
    )
    bm_shape = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION, mesh_2=[cube, cylinder]
    )
    line_positive_x = gs.join_geometry(
        geometry=[cube_line_positive_x, cylinder_line_positive_x]
    )
    line_positive_y = gs.join_geometry(
        geometry=[cube_line_positive_y, cylinder_line_positive_y]
    )
    line_negative_x = gs.join_geometry(
        geometry=[cube_line_negative_x, cylinder_line_negative_x]
    )
    line_negative_y = gs.join_geometry(
        geometry=[cube_line_negative_y, cylinder_line_negative_y]
    )
    return bm_shape, line_positive_x, line_positive_y, line_negative_x, line_negative_y
