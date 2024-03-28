import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from building_mass import building_mass
from roof import roof
from window_with_ledge import window_with_ledge
from ledge import cubed_floor_ledge_for_extrusion


@gs.tree("Building Window Instantiator")
def building_window_instantiator(
    edge_line_pos_x: gs.Geometry,
    edge_line_pos_y: gs.Geometry,
    edge_line_neg_x: gs.Geometry,
    edge_line_neg_y: gs.Geometry,
    floor_height: gs.Float,
    num_windows_each_side: gs.Int,
    windows_left_right_offset: gs.Float,
    windows_height_offset: gs.Float,
    window_shape_size: gs.Vector,
    window_panel_area: gs.Vector,
    window_divided_horizontal: gs.Bool,
    window_divided_vertical: gs.Bool,
    window_interpanel_offset_percentage_y: gs.Float,
    window_interpanel_offset_percentage_z: gs.Float,
    has_window_ledge: gs.Bool,
    ledge_shape_size: gs.Vector,
    ledge_extrusion_x: gs.Float,
    ledge_extrusion_z: gs.Float,
    ledges_height_offset: gs.Float,
):
    """
    Instantiates windows on single floor.

    Params:
    - edge_line_pos_x: the edge line on the positive x axis.
    - edge_line_pos_y: the edge line on the positive y axis.
    - edge_line_neg_x: the edge line on the negative x axis.
    - edge_line_neg_y: the edge line on the negative y axis.
    - floor_height: the height of each floor.
    - num_windows_each_side: the number of windows on each side of each floor of building mass.
    - windows_left_right_offset: the left-/right+ offset of windows from center of each floor of building mass. -1~1. percentage of half window size x to prevent overflowing windows on each end.
    - windows_height_offset: the z offset of windows from center line of each floor of building mass. exact percentage of window size z.
    - window_shape_size: the exact size of each window.
    - window_panel_area: the percentage of the window panel to the total window size.
    - window_divided_horizontal: whether the window panel is divided horizontally.
    - window_divided_vertical: whether the window panel is divided vertically.
    - window_interpanel_offset_percentage_y: the offset between the window panel on the y axis. Percentage of window panel size y. 0~1: 0 for no offset, 1 for full offset.
    - window_interpanel_offset_percentage_z: the offset between the window panel on the z axis. Percentage of window panel size z. 0~1: 0 for no offset, 1 for full offset.
    - ledge_shape_size: the size of each ledge.
    - ledge_extrusion_x: the extrusion of the ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - ledge_extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    - ledges_height_offset: the z offset of ledges from the top of each window. the elevation of the ledge from the window. 0~1 as percentage of ledge size z.
    """
    window_with_ledge_shape = window_with_ledge(
        window_size=window_shape_size,
        window_panel_percentage=window_panel_area,
        window_divided_horizontal=window_divided_horizontal,
        window_divided_vertical=window_divided_vertical,
        window_interpanel_offset_percentage_y=window_interpanel_offset_percentage_y,
        window_interpanel_offset_percentage_z=window_interpanel_offset_percentage_z,
        has_ledge=has_window_ledge,
        ledge_size=ledge_shape_size,
        ledge_elevation_from_window=ledges_height_offset,
        ledge_extrusion_x=ledge_extrusion_x,
        ledge_extrusion_z=ledge_extrusion_z,
    )
    # apply height offset
    edge_line_pos_x = edge_line_pos_x.transform(
        translation=gs.combine_xyz(
            x=0, y=0, z=floor_height / 2 + windows_height_offset * window_shape_size.z
        )
    )
    edge_line_pos_y = edge_line_pos_y.transform(
        translation=gs.combine_xyz(
            x=0, y=0, z=floor_height / 2 + windows_height_offset * window_shape_size.z
        )
    )
    edge_line_neg_x = edge_line_neg_x.transform(
        translation=gs.combine_xyz(
            x=0, y=0, z=floor_height / 2 + windows_height_offset * window_shape_size.z
        )
    )
    edge_line_neg_y = edge_line_neg_y.transform(
        translation=gs.combine_xyz(
            x=0, y=0, z=floor_height / 2 + windows_height_offset * window_shape_size.z
        )
    )
    # trim curve from edges to the center to not have overflowing windows on each end
    trim_factor = window_shape_size.y
    edge_window_separation_factor = 1.5
    # consider left and right offset here
    window_lr_offset_factor = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(trim_factor / 2, windows_left_right_offset),
    )  # /2 again to make +-0.5 +-1.
    window_line_pos_x = gs.trim_curve(
        mode=gs.TrimCurve.Mode.FACTOR,
        curve=edge_line_pos_x,
        start=(trim_factor * edge_window_separation_factor)
        + window_lr_offset_factor,  # TODO: tweak magic number or make it work with some other factors
        end=(1 - trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
    )
    window_line_pos_y = gs.trim_curve(
        mode=gs.TrimCurve.Mode.FACTOR,
        curve=edge_line_pos_y,
        start=(trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
        end=(1 - trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
    )
    window_line_neg_x = gs.trim_curve(
        mode=gs.TrimCurve.Mode.FACTOR,
        curve=edge_line_neg_x,
        start=(trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
        end=(1 - trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
    )
    window_line_neg_y = gs.trim_curve(
        mode=gs.TrimCurve.Mode.FACTOR,
        curve=edge_line_neg_y,
        start=(trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
        end=(1 - trim_factor * edge_window_separation_factor) + window_lr_offset_factor,
    )

    # orientation
    # orient on positive x axis
    nighty_degree = gs.math(operation=gs.Math.Operation.RADIANS, value=90)
    ctp_res_pos_x = (
        gs.curve_to_points(  # points_pos_x, tangent_pos_x, normal_pos_x, rotation_pos_x
            mode=gs.CurveToPoints.Mode.COUNT,
            curve=window_line_pos_x,
            count=num_windows_each_side,
        )
    )
    points_pos_x = ctp_res_pos_x.points
    rotation_pos_x = gs.rotate_euler(
        space=gs.RotateEuler.Space.LOCAL,
        rotation=ctp_res_pos_x.rotation,
        rotate_by=gs.combine_xyz(x=nighty_degree, y=0, z=0),
    )
    # orient on positive y axis
    ctp_res_pos_y = (
        gs.curve_to_points(  # points_pos_y, tangent_pos_y, normal_pos_y, rotation_pos_y
            mode=gs.CurveToPoints.Mode.COUNT,
            curve=window_line_pos_y,
            count=num_windows_each_side,
        )
    )
    points_pos_y = ctp_res_pos_y.points
    rotation_pos_y = gs.rotate_euler(
        space=gs.RotateEuler.Space.LOCAL,
        rotation=ctp_res_pos_y.rotation,
        rotate_by=gs.combine_xyz(x=nighty_degree, y=0, z=0),
    )
    # orient on negative x axis
    ctp_res_neg_x = (
        gs.curve_to_points(  # points_neg_x, tangent_neg_x, normal_neg_x, rotation_neg_x
            mode=gs.CurveToPoints.Mode.COUNT,
            curve=window_line_neg_x,
            count=num_windows_each_side,
        )
    )
    points_neg_x = ctp_res_neg_x.points
    rotation_neg_x = gs.rotate_euler(
        space=gs.RotateEuler.Space.LOCAL,
        rotation=ctp_res_neg_x.rotation,
        rotate_by=gs.combine_xyz(x=nighty_degree, y=0, z=0),
    )
    # orient on negative y axis
    ctp_res_neg_y = (
        gs.curve_to_points(  # points_neg_y, tangent_neg_y, normal_neg_y, rotation_neg_y
            mode=gs.CurveToPoints.Mode.COUNT,
            curve=window_line_neg_y,
            count=num_windows_each_side,
        )
    )
    points_neg_y = ctp_res_neg_y.points
    rotation_neg_y = gs.rotate_euler(
        space=gs.RotateEuler.Space.LOCAL,
        rotation=ctp_res_neg_y.rotation,
        rotate_by=gs.combine_xyz(x=nighty_degree, y=0, z=0),
    )

    # instantiation of windows
    windows_pos_x = gs.instance_on_points(
        points=points_pos_x,
        instance=window_with_ledge_shape,
        rotation=rotation_pos_x,
    )
    windows_pos_y = gs.instance_on_points(
        points=points_pos_y,
        instance=window_with_ledge_shape,
        rotation=rotation_pos_y,
    )
    windows_neg_x = gs.instance_on_points(
        points=points_neg_x,
        instance=window_with_ledge_shape,
        rotation=rotation_neg_x,
    )
    windows_neg_y = gs.instance_on_points(
        points=points_neg_y,
        instance=window_with_ledge_shape,
        rotation=rotation_neg_y,
    )
    return gs.join_geometry(
        geometry=[windows_pos_x, windows_pos_y, windows_neg_x, windows_neg_y]
    )


@gs.tree("Building Floor Ledges Instantiator")
def building_floor_ledges_instantiator(
    ledge_shape_size_x: gs.Float,
    ledge_shape_size_z: gs.Float,
    ledge_extrusion_x: gs.Float,
    ledge_extrusion_z: gs.Float,
    floor_height: gs.Float,
    bm_line_pos_x: gs.Geometry,
    bm_line_pos_y: gs.Geometry,
    bm_line_neg_x: gs.Geometry,
    bm_line_neg_y: gs.Geometry,
):
    fledge_shape_curve = cubed_floor_ledge_for_extrusion(
        size_x=ledge_shape_size_x,
        size_z=ledge_shape_size_z,
        extrusion_x=ledge_extrusion_x,
        extrusion_z=ledge_extrusion_z,
    )
    # get edge curves, add floor height offset, curve to mesh with profile curve being the shape curve
    fledge_line_pos_x = bm_line_pos_x.transform(
        translation=gs.combine_xyz(x=0, y=0, z=floor_height)
    )
    fledge_shape_pos_x = gs.curve_to_mesh(
        curve=fledge_line_pos_x, profile_curve=fledge_shape_curve, fill_caps=True
    ).set_shade_smooth(shade_smooth=False)
    fledge_line_pos_y = bm_line_pos_y.transform(
        translation=gs.combine_xyz(x=0, y=0, z=floor_height)
    )
    fledge_shape_pos_y = gs.curve_to_mesh(
        curve=fledge_line_pos_y, profile_curve=fledge_shape_curve, fill_caps=True
    ).set_shade_smooth(shade_smooth=False)
    fledge_line_neg_x = bm_line_neg_x.transform(
        translation=gs.combine_xyz(x=0, y=0, z=floor_height)
    )
    fledge_shape_neg_x = gs.curve_to_mesh(
        curve=fledge_line_neg_x, profile_curve=fledge_shape_curve, fill_caps=True
    ).set_shade_smooth(shade_smooth=False)
    fledge_line_neg_y = bm_line_neg_y.transform(
        translation=gs.combine_xyz(x=0, y=0, z=floor_height)
    )
    fledge_shape_neg_y = gs.curve_to_mesh(
        curve=fledge_line_neg_y, profile_curve=fledge_shape_curve, fill_caps=True
    ).set_shade_smooth(shade_smooth=False)
    # return gs.join_geometry(geometry=[fledge_shape_pos_x, fledge_shape_pos_y, fledge_shape_neg_x, fledge_shape_neg_y])
    # repeat num_floors times
    single_floor_fledges = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION,
        mesh_2=[
            fledge_shape_pos_x,
            fledge_shape_pos_y,
            fledge_shape_neg_x,
            fledge_shape_neg_y,
        ],
    )
    return single_floor_fledges


@gs.tree("XYZ Scale Clamper")
def xyz_scale_clamper(scale: gs.Vector) -> gs.Vector:
    _max_xy = gs.math(
        operation=gs.Math.Operation.MAXIMUM, value=(scale.x, scale.y)
    )
    _max = gs.math(operation=gs.Math.Operation.MAXIMUM, value=(_max_xy, scale.z))
    _min_xy = gs.math(
        operation=gs.Math.Operation.MINIMUM, value=(scale.x, scale.y)
    )
    _min = gs.math(operation=gs.Math.Operation.MINIMUM, value=(_min_xy, scale.z))
    clamped_scale_x = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=scale.x,
        from_min=_min,
        from_max=_max,
        to_min=0.5,
        to_max=1,
    )
    clamped_scale_y = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=scale.y,
        from_min=_min,
        from_max=_max,
        to_min=0.5,
        to_max=1,
    )
    clamped_scale_z = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=scale.z,
        from_min=_min,
        from_max=_max,
        to_min=0.5,
        to_max=1,
    )
    clamped_scale = gs.combine_xyz(x=clamped_scale_x, y=clamped_scale_y, z=clamped_scale_z)
    return clamped_scale



# TODO: change shape int type to float interpolation values? 0~1
@gs.tree("Basic Building")
def basic_building(
    bm_base_shape: gs.Int,
    bm_size: gs.Vector,
    num_floors: gs.Int,
    rf_base_shape: gs.Int,
    rf_size: gs.Vector,
    num_windows_each_side: gs.Int,
    windows_left_right_offset: gs.Float,
    windows_height_offset: gs.Float,
    window_shape_size: gs.Vector,
    window_panel_area: gs.Vector,
    window_divided_horizontal: gs.Bool,
    window_divided_vertical: gs.Bool,
    window_interpanel_offset_percentage_y: gs.Float,
    window_interpanel_offset_percentage_z: gs.Float,
    has_window_ledge: gs.Bool,
    window_ledge_shape_size: gs.Vector,
    window_ledge_extrusion_x: gs.Float,
    window_ledge_extrusion_z: gs.Float,
    window_ledges_height_offset: gs.Float,
    has_floor_ledge: gs.Bool,
    floor_ledge_size_x: gs.Float,
    floor_ledge_size_z: gs.Float,
    floor_ledge_extrusion_x: gs.Float,
    floor_ledge_extrusion_z: gs.Float,
):
    """
    BM size now is not clamped but normalized by diagonal length 1. May need to restrict input range in Blender before handing in to user. 

    Parameters:
    - bm_base_shape: the primitive shape used by each floor. to be combined together to form the main body
    - num_floors: the number of floors (base_shapes) to stay on top of each other
    - bm_size: the size of the base_shape, relative to each other, will be clamped to 0~1 as 0.5 to 1. 
    - rf_base_shape: the primitive shape used by the roof.
    - rf_size: the size of the roof, 0~1 percentage of bm_size 50%~100%.
    - num_windows_each_side: the number of windows on each side of each floor of building mass.
    - windows_left_right_offset: the left-/right+ offset of windows from center of each floor of building mass. -1~1. percentage of half window size x to prevent overflowing windows on each end.
    - windows_height_offset: the z offset of windows from center line of each floor of building mass. -1~1. percentage of half window size z.
    - window_shape_size: the size of each window.
    - window_panel_area: the percentage of the window panel to the total window size.
    - window_divided_horizontal: whether the window panel is divided horizontally.
    - window_divided_vertical: whether the window panel is divided vertically.
    - window_interpanel_offset_percentage_y: the offset between the window panel on the y axis. Percentage of left out window area size y from panels. 0~1: 0 for no offset, 1 for full offset.
    - window_interpanel_offset_percentage_z: the offset between the window panel on the z axis. Percentage of left out window area size z from panels. 0~1: 0 for no offset, 1 for full offset.
    - has_window_ledge: whether to have ledges on top of each window.
    - window_ledge_shape_size: the size of each ledge.
    - window_ledge_extrusion_x: the extrusion of the ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - window_ledge_extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    - window_ledges_height_offset: the z offset of ledges from the top of each window. the elevation of the ledge from the window. 0~1 as percentage of ledge size z.
    - has_floor_ledge: whether to have floor ledges separating each floor.
    - floor_ledge_size_x: the size of each floor ledge on the x axis.
    - floor_ledge_size_z: the size of each floor ledge on the z axis.
    - floor_ledge_extrusion_x: the extrusion of the floor ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - floor_ledge_extrusion_z: the extrusion of the floor ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    """
    ### Building Mass
    # calculate initial floor height
    init_floor_height = gs.math(
        operation=gs.Math.Operation.DIVIDE, value=(bm_size.z, num_floors)
    )
    # add one floor height (for roof) to bm_size for diag calculation
    building_size = gs.combine_xyz(
        x=bm_size.x, y=bm_size.y, z=gs.math(operation=gs.Math.Operation.ADD, value=(bm_size.z, init_floor_height))
    )
    sq_building_size = gs.combine_xyz(
        x=gs.math(operation=gs.Math.Operation.POWER, value=(building_size.x, 2)),
        y=gs.math(operation=gs.Math.Operation.POWER, value=(building_size.y, 2)),
        z=gs.math(operation=gs.Math.Operation.POWER, value=(building_size.z, 2)),
    )
    # calculate diagonal of xyz
    diag = gs.math(
        operation=gs.Math.Operation.SQRT,
        value=gs.math(
            operation=gs.Math.Operation.ADD,
            value=(gs.math(operation=gs.Math.Operation.ADD, value=(sq_building_size.x, sq_building_size.y)), sq_building_size.z),
        ),
    )
    # get factor to resize diagonal to 1
    diag_factor = gs.math(
        operation=gs.Math.Operation.DIVIDE, value=(1, diag)
    )
    # resize bm_size with diag_factor  # note: drop building_size now and continue to use bm_size
    building_size = gs.vector_math(
        operation=gs.VectorMath.Operation.SCALE,
        vector=building_size, scale=diag_factor,
    )
    clamped_bm_size = gs.vector_math(
        operation=gs.VectorMath.Operation.SCALE,
        vector=bm_size, scale=diag_factor,
    )
    # calculate floor size (xyz) using bm_size (xyz) divided by num_floors
    floor_height = gs.math(
        operation=gs.Math.Operation.DIVIDE, value=(clamped_bm_size.z, num_floors)
    )
    floor_size = gs.combine_xyz(x=clamped_bm_size.x, y=clamped_bm_size.y, z=floor_height)
    # instantiate floor body using bm_base_shape
    (
        bm_shape,
        bm_line_pos_x,
        bm_line_pos_y,
        bm_line_neg_x,
        bm_line_neg_y,
    ) = building_mass(bm_type=bm_base_shape, bm_size=floor_size)
    # return gs.join_geometry(geometry=[bm_line_pos_x, bm_line_pos_y, bm_line_neg_x, bm_line_neg_y])
    # repeat num_floors times
    floor_grow_vector = gs.combine_xyz(x=0, y=0, z=floor_height)
    floor_anchor_points = gs.mesh_line(
        mode=gs.MeshLine.Mode.OFFSET, count=num_floors, offset=floor_grow_vector
    )
    bm_instances = gs.instance_on_points(points=floor_anchor_points, instance=bm_shape)
    bm_final = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION, mesh_2=[bm_instances]
    )

    ### Roof
    # clamp roof size proportional to bm size
    clamped_rf_size = gs.combine_xyz(  # now roof size is a percentage of (clamped) bm size
        x=clamped_bm_size.x * rf_size.x,
        y=clamped_bm_size.y * rf_size.y,
        z=floor_height * rf_size.z  # z to be percent floor height
    )
    # calculate elevation for roof
    res = gs.bounding_box(bm_final)
    r_pos_z = res.max.z + clamped_rf_size.z / 2  # clamped_bm_size is not exactly match here, so use bounding box
    r_pos = gs.combine_xyz(x=0, y=0, z=r_pos_z)
    rf_final = roof(roof_type=rf_base_shape, roof_size=clamped_rf_size).transform(
        translation=r_pos
    )

    ### Windows
    # calculate window size (xyz) using bm_size (xyz) divided by num_floors and num windows each side
    # use the smaller scale among x and y size
    x_smaller_than_y = gs.compare(
        operation=gs.Compare.Operation.LESS_THAN,
        data_type=gs.Compare.DataType.FLOAT,
        a=clamped_bm_size.x,
        b=clamped_bm_size.y,
    )
    smaller_horizontal_span = gs.switch(
        input_type=gs.Switch.InputType.FLOAT,
        switch=x_smaller_than_y,
        false=clamped_bm_size.y,
        true=clamped_bm_size.x,
    )
    max_window_size = gs.combine_xyz(
        x=0.01, 
        y=smaller_horizontal_span / (num_windows_each_side + 1),  # +1 to prevent overflowing windows on each end; also leave room for l/r offsets
        z=clamped_bm_size.z / (num_floors + 2),  # same, to leave room for height offset and ledges. 
    )
    # clamped_window_size = xyz_scale_clamper(scale=window_shape_size)
    clamped_window_size = window_shape_size
    true_window_size = gs.combine_xyz(
        x=clamped_window_size.x * max_window_size.x,
        y=clamped_window_size.y * max_window_size.y,
        z=clamped_window_size.z * max_window_size.z, 
    )
    # window height offset max is 1 - window size z
    true_window_height_offset = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=windows_height_offset,
        from_min=-1,
        from_max=1,
        to_min=-0.3,
        to_max=0.1,
    )
    true_window_height_offset = true_window_height_offset / 2 - 0.1
    true_ledge_size = gs.combine_xyz(  # transfered 0.1 scaling on x and z from ledge shape generator to here
        x=window_ledge_shape_size.x * true_window_size.x * 2,
        y=window_ledge_shape_size.y * true_window_size.y,
        z=window_ledge_shape_size.z * true_window_size.z * 0.1,
    )
    window_interpanel_max_offset_y = 1 - window_panel_area.y
    window_interpanel_max_offset_z = 1 - window_panel_area.z
    true_interpanel_offset_y = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(window_interpanel_offset_percentage_y, window_interpanel_max_offset_y),
    )
    true_interpanel_offset_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(window_interpanel_offset_percentage_z, window_interpanel_max_offset_z),
    )
    scaled_window_ledge_extrusion_x = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=window_ledge_extrusion_x,
        from_min=0,
        from_max=1,
        to_min=0,
        to_max=0.5,
    )
    single_floor_windows = building_window_instantiator(
        edge_line_pos_x=bm_line_pos_x,
        edge_line_pos_y=bm_line_pos_y,
        edge_line_neg_x=bm_line_neg_x,
        edge_line_neg_y=bm_line_neg_y,
        floor_height=floor_height,
        num_windows_each_side=num_windows_each_side,
        windows_left_right_offset=windows_left_right_offset,
        windows_height_offset=true_window_height_offset,
        window_shape_size=true_window_size,
        window_panel_area=window_panel_area,
        window_divided_horizontal=window_divided_horizontal,
        window_divided_vertical=window_divided_vertical,
        window_interpanel_offset_percentage_y=true_interpanel_offset_y,
        window_interpanel_offset_percentage_z=true_interpanel_offset_z,
        has_window_ledge=has_window_ledge,
        ledge_shape_size=true_ledge_size,
        ledge_extrusion_x=scaled_window_ledge_extrusion_x,
        ledge_extrusion_z=window_ledge_extrusion_z,
        ledges_height_offset=window_ledges_height_offset,
    )
    # return windows
    # all floors
    all_floor_windows = gs.instance_on_points(
        points=floor_anchor_points, instance=single_floor_windows
    )
    all_floor_windows = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION,
        mesh_2=[all_floor_windows],
    )
    # return all_floor_windows

    ### Ledges - Floor (floor ledge - fledge for short)
    true_fledge_size_x = floor_ledge_size_x * 0.01 * 2  # window size x is 0.01'd as well
    true_fledge_size_z = floor_ledge_size_z * floor_height * 0.1
    scaled_fledge_extrusion_x = gs.map_range(
        clamp=True,
        interpolation_type=gs.MapRange.InterpolationType.LINEAR,
        data_type=gs.MapRange.DataType.FLOAT,
        value=floor_ledge_extrusion_x,
        from_min=0,
        from_max=1,
        to_min=0,
        to_max=0.5,
    )
    single_floor_fledges = building_floor_ledges_instantiator(
        ledge_shape_size_x=true_fledge_size_x,
        ledge_shape_size_z=true_fledge_size_z,
        ledge_extrusion_x = scaled_fledge_extrusion_x,
        ledge_extrusion_z = floor_ledge_extrusion_z,
        floor_height = floor_height,
        bm_line_pos_x = bm_line_pos_x,
        bm_line_pos_y = bm_line_pos_y,
        bm_line_neg_x = bm_line_neg_x,
        bm_line_neg_y = bm_line_neg_y,
    )
    all_floor_fledges = gs.instance_on_points(
        points=floor_anchor_points, instance=single_floor_fledges
    )
    all_floor_fledges = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=has_floor_ledge,
        false=None,
        true=all_floor_fledges,
    )
    all_floor_fledges = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION,
        mesh_2=[all_floor_fledges],
    )

    ### Combine
    # final = gs.mesh_boolean(
    #     operation=gs.MeshBoolean.Operation.UNION,
    #     mesh_2=[bm_final, rf_final, all_floor_windows, all_floor_fledges],
    # )
    final = gs.join_geometry(
        geometry=[bm_final, rf_final, all_floor_windows, all_floor_fledges]
    )

    ### center on z axis
    res = gs.bounding_box(final)
    zmin = res.min.z
    zmax = res.max.z
    z_center = (zmax + zmin) / 2
    final = final.transform(
        translation=gs.combine_xyz(x=0, y=0, z=-z_center)
    )
    bm_final = bm_final.transform(
        translation=gs.combine_xyz(x=0, y=0, z=-z_center)
    )
    rf_final = rf_final.transform(
        translation=gs.combine_xyz(x=0, y=0, z=-z_center)
    )
    all_floor_windows = all_floor_windows.transform(
        translation=gs.combine_xyz(x=0, y=0, z=-z_center)
    )
    all_floor_fledges = all_floor_fledges.transform(
        translation=gs.combine_xyz(x=0, y=0, z=-z_center)
    )
    return final, bm_final, rf_final, all_floor_windows, all_floor_fledges
    # return {  # causes wrapper method to not work
    #     "Final Building": final,
    #     "Building Mass": bm_final,
    #     "Roof": rf_final,
    #     "Windows": all_floor_windows,
    #     "Floor Ledges": all_floor_fledges,
    # }
