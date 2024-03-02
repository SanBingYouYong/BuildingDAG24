import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from shape_primitives import gs_cube, gs_sphere, gs_cylinder_scaled, gs_half_sphere


@gs.tree("Fixed Window")
def fixed_window(
    size: gs.Vector,
    window_panel_area: gs.Float,
    divided_horizontal: gs.Bool,
    divided_vertical: gs.Bool,  # TODO: custom divisions (3x3 etc.)
    panel_offset: gs.Float,
    extrusion: gs.Float,
):
    """
    Default window type, supports a whole panel, two panels divided horizontally or vertically, or four panels divded both direction.

    Parameters:
    - size: the size of the window.
    - window_panel_area: the percentage of the window panel to the total window size.
    - divided_horizontal: whether the window panel is divided horizontally.
    - divided_vertical: whether the window panel is divided vertically.
    - panel_offset: the offset of the window panels from each other if divided. percentage of the window panel size (on that axis).
    - extrusion: extrusion of window panel into the window frame.
    """
    # initialize window frame with cube
    # initialize window panel of corresponding amount to divided booleans
    # move panel to correct position according to extrusion
    # subtract panel from frame
    # TODO: for simplicity, offset from center is not included
    frame = gs_cube(size=(1, 0.1, 1)).transform(scale=size)
    window_panel_multiplier = gs.combine_xyz(
        x=window_panel_area, y=window_panel_area, z=window_panel_area
    )
    extrusion_multiplier = gs.combine_xyz(x=extrusion, y=extrusion, z=extrusion)
    panel_size_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY, value=(size.x, window_panel_multiplier)
    )
    panel_size_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY, value=(size.z, window_panel_multiplier)
    )
    panel_size_y = gs.math(
        operation=gs.Math.Operation.MULTIPLY, value=(size.y, extrusion_multiplier)
    )
    panel_size = gs.combine_xyz(x=panel_size_x, y=panel_size_y, z=panel_size_z)

    origin_anchor = gs.combine_xyz(x=0, y=0, z=0)
    both_anchor_up = gs.combine_xyz(x=-panel_size.x / 4, y=0, z=-panel_size.z / 4)  # TODO: update up/low split to use mesh line too, for supporting more divisions
    both_anchor_low = gs.combine_xyz(x=-panel_size.x / 4, y=0, z=+panel_size.z / 4)
    horizontal_anchor = gs.combine_xyz(x=-panel_size.x / 4, y=0, z=0)
    horizontal_offset = gs.combine_xyz(x=panel_size.x / 2, y=0, z=0)
    vertical_anchor = gs.combine_xyz(x=0, y=0, z=-panel_size.z / 4)
    vertical_offset = gs.combine_xyz(x=0, y=0, z=panel_size.z / 2)

    is_whole = gs.boolean_math(
        operation=gs.BooleanMath.Operation.NOR,
        boolean=(divided_horizontal, divided_vertical),
    )
    is_both = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(divided_horizontal, divided_vertical),
    )
    is_horizontal = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(
            divided_horizontal,
            gs.boolean_math(
                operation=gs.BooleanMath.Operation.NOT, boolean=divided_vertical
            ),
        ),
    )
    is_vertical = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(
            gs.boolean_math(
                operation=gs.BooleanMath.Operation.NOT, boolean=divided_horizontal
            ),
            divided_vertical,
        ),
    )

    # go through checks one by one. 
    single_panel_size = gs.switch(input_type=gs.Switch.InputType.VECTOR, switch=is_whole, false=None, true=panel_size)
    single_panel_size = gs.switch(input_type=gs.Switch.InputType.VECTOR, 
        switch=is_both, false=single_panel_size, 
        true=gs.combine_xyz(
            x=gs.math(
                operation=gs.Math.Operation.SUBTRACT,
                value=(
                    panel_size.x / 2,
                    gs.math(
                        operation=gs.Math.Operation.DIVIDE, value=(
                            gs.math(
                                operation=gs.Math.Operation.MULTIPLY, value=(panel_size.x, panel_offset)
                            ), 2)
                    ),
                ),
            ),
            y=panel_size.y,
            z=gs.math(
                operation=gs.Math.Operation.SUBTRACT,
                value=(
                    panel_size.z / 2,
                    gs.math(
                        operation=gs.Math.Operation.DIVIDE, value=(
                            gs.math(
                                operation=gs.Math.Operation.MULTIPLY, value=(panel_size.z, panel_offset)
                            ), 2)
                    ),
                ),
            ),
        ),
    )
    single_panel_size = gs.switch(input_type=gs.Switch.InputType.VECTOR, 
        switch=is_horizontal, false=single_panel_size, 
        true=gs.combine_xyz(
            x=gs.math(
                operation=gs.Math.Operation.SUBTRACT,
                value=(
                    panel_size.x / 2,
                    gs.math(
                        operation=gs.Math.Operation.DIVIDE, value=(
                            gs.math(
                                operation=gs.Math.Operation.MULTIPLY, value=(panel_size.x, panel_offset)
                            ), 2)
                    ),
                ),
            ),
            y=panel_size.y,
            z=panel_size.z,
        ),
    )
    single_panel_size = gs.switch(input_type=gs.Switch.InputType.VECTOR, 
        switch=is_vertical,
        false=single_panel_size,
        true=gs.combine_xyz(
            x=panel_size.x,
            y=panel_size.y,
            z=gs.math(
                operation=gs.Math.Operation.SUBTRACT,
                value=(
                    panel_size.z / 2,
                    gs.math(
                        operation=gs.Math.Operation.DIVIDE, value=(
                            gs.math(
                                operation=gs.Math.Operation.MULTIPLY, value=(panel_size.z, panel_offset)
                            ), 2)
                    ),
                ),
            ),
        ),
    )
    # apply size
    test_panel_shape = gs_cube(size=single_panel_size)  # TODO: replace with actual panels

    meshline_whole = gs.switch(  # default is geometry
        switch=is_whole,
        false=None,
        true=gs.mesh_line(
            mode=gs.MeshLine.Mode.OFFSET, count=1, start_location=origin_anchor
        ),
    )
    panels_whole = gs.instance_on_points(points=meshline_whole, instance=test_panel_shape)
    meshline_both = gs.switch(
        switch=is_both,
        false=None,
        true=gs.join_geometry(
            geometry=[
                gs.mesh_line(mode=gs.MeshLine.Mode.OFFSET, count=2, start_location=both_anchor_up, offset=horizontal_offset),
                gs.mesh_line(mode=gs.MeshLine.Mode.OFFSET, count=2, start_location=both_anchor_low, offset=horizontal_offset),
            ]
        )
    )
    panels_both = gs.instance_on_points(points=meshline_both, instance=test_panel_shape)
    meshline_horizontal = gs.switch(
        switch=is_horizontal,
        false=None,
        true=gs.mesh_line(
            mode=gs.MeshLine.Mode.OFFSET,
            count=2,
            start_location=horizontal_anchor,
            offset=horizontal_offset,
        ),
    )
    panels_horizontal = gs.instance_on_points(
        points=meshline_horizontal, instance=test_panel_shape
    )
    meshline_vertical = gs.switch(
        switch=is_vertical,
        false=None,
        true=gs.mesh_line(
            mode=gs.MeshLine.Mode.OFFSET,
            count=2,
            start_location=vertical_anchor,
            offset=vertical_offset,
        ),
    )
    panels_vertical = gs.instance_on_points(
        points=meshline_vertical, instance=test_panel_shape
    )

    return gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION,
        mesh_2=[panels_whole, panels_both, panels_horizontal, panels_vertical, frame],
    )



@gs.tree("Window")
def window(window_type: gs.Int, window_size: gs.Vector):
    """
    Placeholder, use Fixed Window for now instead. 
    Generate a window based on the given shape and size.

    Parameters:
    - window_type (gs.Int): The shape of the window. 0 for cube, 1 for cylinder, 2 for half sphere.
    - window_size (gs.Vector): The size of the window. Primitives shapes use a unified xyz scale to represent size.

    Returns:
    - The generated window.

    Raises:
    - ValueError: If the shape is not a valid value (0, 1 or 2).
    """
    # use compare to determine shape type: currently switch nodes can have T/F outputs only, so 3 types need 2 compares (flat?gabled?hipped)
    is_cube = gs.compare(
        operation=gs.Compare.Operation.EQUAL,
        data_type=gs.Compare.DataType.INT,
        a=window_type,
        b=0,
    )
    is_cylinder = gs.compare(
        operation=gs.Compare.Operation.EQUAL,
        data_type=gs.Compare.DataType.INT,
        a=window_type,
        b=1,
    )
    is_sphere = gs.compare(
        operation=gs.Compare.Operation.EQUAL,
        data_type=gs.Compare.DataType.INT,
        a=window_type,
        b=2,
    )
    cube = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cube,
        false=None,
        true=gs_cube(size=window_size),
    )
    cylinder = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs_cylinder_scaled(size=window_size),
    )
    sphere = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_sphere,
        false=None,
        true=gs_half_sphere(size=window_size),
    )
    window_shape = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION, mesh_2=[cube, cylinder, sphere]
    )
    return window_shape
