import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from shape_primitives import gs_cube, gs_sphere, gs_cylinder_scaled, gs_half_sphere


@gs.tree("Window Panel MeshLine Anchors Generator")
def window_panel_meshline_anchors_generator(
    panel_size: gs.Vector,
    interpanel_offset_percentage_y: gs.Float,
    interpanel_offset_percentage_z: gs.Float,
):
    """
    Computes all four meshlines for different window division patterns.
    """
    _panel_offset_y_half = panel_size.y * interpanel_offset_percentage_y / 2
    _panel_offset_z_half = panel_size.z * interpanel_offset_percentage_z / 2
    _panel_size_y_half = panel_size.y / 2
    _panel_size_z_half = panel_size.z / 2
    # anchors
    anchor_whole = (0, 0, 0)
    ## when divided both, have two anchors vertically and offset them horizontally
    # anchor_both_up = (0, -_panel_offset_y_half - _panel_size_y_half, _panel_offset_z_half + _panel_size_z_half)
    anchor_both_up = gs.combine_xyz(
        x=0,
        y=-_panel_offset_y_half - _panel_size_y_half,
        z=_panel_offset_z_half + _panel_size_z_half,
    )
    # anchor_both_down = (0, -_panel_offset_y_half - _panel_size_y_half, -_panel_offset_z_half - _panel_size_z_half)
    anchor_both_down = gs.combine_xyz(
        x=0,
        y=-_panel_offset_y_half - _panel_size_y_half,
        z=-_panel_offset_z_half - _panel_size_z_half,
    )
    # anchor_horizontal = (0, -_panel_offset_y_half - _panel_size_y_half, 0)
    anchor_horizontal = gs.combine_xyz(
        x=0, y=-_panel_offset_y_half - _panel_size_y_half, z=0
    )
    # anchor_vertical = (0, 0, _panel_offset_z_half + _panel_size_z_half)
    anchor_vertical = gs.combine_xyz(
        x=0, y=0, z=_panel_offset_z_half + _panel_size_z_half
    )
    # directions
    direction_horizontal = gs.combine_xyz(
        x=0, y=_panel_size_y_half * 2 + _panel_offset_y_half * 2, z=0
    )
    direction_vertical = gs.combine_xyz(
        x=0, y=0, z=-_panel_size_z_half * 2 - _panel_offset_z_half * 2
    )
    # meshlines
    meshline_whole = gs.mesh_line(
        mode=gs.MeshLine.Mode.OFFSET,
        count=1,
        start_location=anchor_whole,
    )
    meshline_both = gs.join_geometry(
        geometry=[
            gs.mesh_line(
                mode=gs.MeshLine.Mode.OFFSET,
                count=2,
                start_location=anchor_both_up,
                offset=direction_horizontal,
            ),
            gs.mesh_line(
                mode=gs.MeshLine.Mode.OFFSET,
                count=2,
                start_location=anchor_both_down,
                offset=direction_horizontal,
            ),
        ]
    )
    meshline_horizontal = gs.mesh_line(
        mode=gs.MeshLine.Mode.OFFSET,
        count=2,
        start_location=anchor_horizontal,
        offset=direction_horizontal,
    )
    meshline_vertical = gs.mesh_line(
        mode=gs.MeshLine.Mode.OFFSET,
        count=2,
        start_location=anchor_vertical,
        offset=direction_vertical,
    )
    # return meshlines
    return meshline_whole, meshline_both, meshline_horizontal, meshline_vertical


@gs.tree("Window Panels Generator")
def window_panels_generator(
    window_size: gs.Vector,
    window_panel_percentage: gs.Vector,
    window_divided_horizontal: gs.Bool,
    window_divided_vertical: gs.Bool,
    window_interpanel_offset_percentage_y: gs.Float,  # TODO: figure out how to properly make model ignore one input from vector (GeoCode)
    window_interpanel_offset_percentage_z: gs.Float,
):
    """
    Calculates the window panel size and instantiate on corresponding meshline.
    """
    # get actual sizes
    window_panel_size = gs.combine_xyz(
        x=window_size.x * window_panel_percentage.x,
        y=window_size.y * window_panel_percentage.y,
        z=window_size.z * window_panel_percentage.z,
    )
    window_interpanel_offset = gs.combine_xyz(
        x=0,
        y=window_panel_size.y * window_interpanel_offset_percentage_y,
        z=window_panel_size.z * window_interpanel_offset_percentage_z,
    )
    # check panel division
    _panel_whole = gs.boolean_math(
        operation=gs.BooleanMath.Operation.NOR,
        boolean=(window_divided_horizontal, window_divided_vertical),
    )
    _panel_both = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(window_divided_horizontal, window_divided_vertical),
    )
    _panel_horizontal = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(
            window_divided_horizontal,
            gs.boolean_math(
                operation=gs.BooleanMath.Operation.NOT, boolean=window_divided_vertical
            ),
        ),
    )
    _panel_vertical = gs.boolean_math(
        operation=gs.BooleanMath.Operation.AND,
        boolean=(
            window_divided_vertical,
            gs.boolean_math(
                operation=gs.BooleanMath.Operation.NOT,
                boolean=window_divided_horizontal,
            ),
        ),
    )
    # pre-calculate panel sizes
    single_panel_size = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=_panel_whole,
        false=None,
        true=window_panel_size,
    )
    single_panel_size = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=_panel_both,
        false=single_panel_size,
        true=gs.combine_xyz(
            x=window_panel_size.x,
            y=window_panel_size.y / 2 - window_interpanel_offset.y / 2,
            z=window_panel_size.z / 2 - window_interpanel_offset.z / 2,
        ),
    )
    single_panel_size = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=_panel_horizontal,
        false=single_panel_size,
        true=gs.combine_xyz(
            x=window_panel_size.x,
            y=window_panel_size.y / 2 - window_interpanel_offset.y / 2,
            z=window_panel_size.z,
        ),
    )
    single_panel_size = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=_panel_vertical,
        false=single_panel_size,
        true=gs.combine_xyz(
            x=window_panel_size.x,
            y=window_panel_size.y,
            z=window_panel_size.z / 2 - window_interpanel_offset.z / 2,
        ),
    )
    # instantiate shape with proper size
    single_panel_shape = gs.cube(size=single_panel_size)
    # generate meshlines
    (
        meshline_whole,
        meshline_both,
        meshline_horizontal,
        meshline_vertical,
    ) = window_panel_meshline_anchors_generator(
        panel_size=single_panel_size,
        interpanel_offset_percentage_y=window_interpanel_offset_percentage_y,
        interpanel_offset_percentage_z=window_interpanel_offset_percentage_z,
    )
    # finalize meshline
    meshline = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=_panel_whole,
        false=None,
        true=meshline_whole,
    )
    meshline = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=_panel_both,
        false=meshline,
        true=meshline_both,
    )
    meshline = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=_panel_horizontal,
        false=meshline,
        true=meshline_horizontal,
    )
    meshline = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=_panel_vertical,
        false=meshline,
        true=meshline_vertical,
    )
    # instantiate shape on meshline
    panels = gs.instance_on_points(points=meshline, instance=single_panel_shape)
    # return panels
    return panels


@gs.tree("Window Panel Into Frame")
def window_panel_into_frame(
    panel_body_offset: gs.Float, panel_mesh: gs.Geometry, frame_mesh: gs.Geometry
):
    """
    Offset the panel and cut into the frame.
    """
    panel_cutter = panel_mesh.transform(
        translation=gs.combine_xyz(x=panel_body_offset, y=0, z=0)
    )
    return gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.DIFFERENCE,
        mesh_1=frame_mesh,
        mesh_2=[panel_cutter],
    )

@gs.tree("Window Ledge Shape Generator")
def window_ledge_shape_generator(
    ledge_size: gs.Vector,
    ledge_extrusion_x: gs.Float,
    ledge_extrusion_z: gs.Float,
):
    '''
    Generates the shape of the ledge.
    '''
    ledge = gs_cube(size=ledge_size)
    cutter = gs_cube(size=ledge_size)
    extrusion_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.x,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, ledge_extrusion_x)),
        ),
    )
    extrusion_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.z,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, ledge_extrusion_z)),
        ),
    )
    cutter_translation = gs.combine_xyz(x=extrusion_x, y=0, z=-extrusion_z)
    cutter = cutter.transform(translation=cutter_translation)
    ledge = ledge.mesh_boolean(
        operation=gs.MeshBoolean.Operation.DIFFERENCE, mesh_2=[cutter]
    )
    return ledge



@gs.tree("Window With Ledge")
def window_with_ledge(
    window_size: gs.Vector,
    window_panel_percentage: gs.Vector,
    window_divided_horizontal: gs.Bool,
    window_divided_vertical: gs.Bool,
    window_interpanel_offset_percentage_y: gs.Float,  # TODO: figure out how to properly make model ignore one input from vector (GeoCode)
    window_interpanel_offset_percentage_z: gs.Float,
    has_ledge: gs.Bool,
    ledge_size: gs.Vector,
    ledge_elevation_from_window: gs.Float,
    ledge_extrusion_x: gs.Float,
    ledge_extrusion_z: gs.Float,
):
    """
    Combines window and ledge as one component.

    Parameters:
    - window_size: the size of the window.
    - window_panel_percentage: the size of the window panel. Percentage of whole window size. 0~1. x axis define the former extrusion of panel into the frame.
    - window_divided_horizontal: whether the window panel is divided horizontally.
    - window_divided_vertical: whether the window panel is divided vertically.
    - window_interpanel_offset_percentage_y: the offset between the window panel on the y axis. Percentage of window panel size y. 0~1: 0 for no offset, 1 for full offset.
    - window_interpanel_offset_percentage_z: the offset between the window panel on the z axis. Percentage of window panel size z. 0~1: 0 for no offset, 1 for full offset.
    - has_ledge: whether the window has a ledge.
    - ledge_size: the size of the ledge.
    - ledge_elevation_from_window: the elevation of the ledge from the window. 0~1 as percentage of ledge size z. # TODO: consider floor height constraint outside of this component.
    - ledge_extrusion_x: the extrusion of the ledge from outmost face back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - ledge_extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    """
    scaled_window_size = gs.combine_xyz(
        x=window_size.x,
        y=window_size.y,
        z=window_size.z,
    )
    window_body = gs_cube(size=scaled_window_size)
    window_panels = window_panels_generator(
        window_size=scaled_window_size,
        window_panel_percentage=window_panel_percentage,
        window_divided_horizontal=window_divided_horizontal,
        window_divided_vertical=window_divided_vertical,
        window_interpanel_offset_percentage_y=window_interpanel_offset_percentage_y,
        window_interpanel_offset_percentage_z=window_interpanel_offset_percentage_z,
    )
    _panel_x_offset = (
        scaled_window_size.x / 2 - scaled_window_size.x * window_panel_percentage.x / 2
    )
    final_window_shape = window_panel_into_frame(
        panel_body_offset=_panel_x_offset,
        panel_mesh=window_panels,
        frame_mesh=window_body,
    )
    ledge_shape = window_ledge_shape_generator(
        ledge_size=ledge_size, 
        ledge_extrusion_x=ledge_extrusion_x,
        ledge_extrusion_z=ledge_extrusion_z,
    )
    ledge_window_height_offset = ledge_elevation_from_window * ledge_size.z
    ledge_translation = gs.combine_xyz(
        x=0,
        y=0,
        z=ledge_window_height_offset + scaled_window_size.z / 2 + ledge_size.z / 2,
    )
    ledge_shape = ledge_shape.transform(translation=ledge_translation)
    ledge_shape = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=has_ledge,
        false=None,
        true=ledge_shape,
    )
    final_window_shape = gs.join_geometry(geometry=[final_window_shape, ledge_shape])
    return final_window_shape
