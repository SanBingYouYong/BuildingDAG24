import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from basic_building import basic_building


@gs.tree("Building For Distortion Render")
def building_for_distortion_render(
    return_part: gs.Int,
    # original basic building parameters
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
    '''
    Basically a wrapper for basic_building, but also controls what to choose as mesh for distortion pipeline. 
    
    Parameters:
    - return_part (gs.Int): 0 for final whole building, 1 for building mass, 2 for roof, 3 for windows, 4 for floor ledges.
    - others: basic_building node tree params.
    '''
    final_building, building_mass, roof, windows, floor_ledges = basic_building(
        bm_base_shape=bm_base_shape,
        bm_size=bm_size,
        num_floors=num_floors,
        rf_base_shape=rf_base_shape,
        rf_size=rf_size,
        num_windows_each_side=num_windows_each_side,
        windows_left_right_offset=windows_left_right_offset,
        windows_height_offset=windows_height_offset,
        window_shape_size=window_shape_size,
        window_panel_area=window_panel_area,
        window_divided_horizontal=window_divided_horizontal,
        window_divided_vertical=window_divided_vertical,
        window_interpanel_offset_percentage_y=window_interpanel_offset_percentage_y,
        window_interpanel_offset_percentage_z=window_interpanel_offset_percentage_z,
        has_window_ledge=has_window_ledge,
        window_ledge_shape_size=window_ledge_shape_size,
        window_ledge_extrusion_x=window_ledge_extrusion_x,
        window_ledge_extrusion_z=window_ledge_extrusion_z,
        window_ledges_height_offset=window_ledges_height_offset,
        has_floor_ledge=has_floor_ledge,
        floor_ledge_size_x=floor_ledge_size_x,
        floor_ledge_size_z=floor_ledge_size_z,
        floor_ledge_extrusion_x=floor_ledge_extrusion_x,
        floor_ledge_extrusion_z=floor_ledge_extrusion_z,
    )
    # switch the chosen output
    return_whole = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=return_part, b=0)
    return_building_mass = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=return_part, b=1)
    return_roof = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=return_part, b=2)
    return_windows = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=return_part, b=3)
    return_floor_ledges = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=return_part, b=4)
    # return the chosen output
    final_building = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, switch=return_whole, false=None, true=final_building)
    building_mass = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, switch=return_building_mass, false=None, true=building_mass)
    roof = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, switch=return_roof, false=None, true=roof)
    windows = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, switch=return_windows, false=None, true=windows)
    floor_ledges = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, switch=return_floor_ledges, false=None, true=floor_ledges)
    # combine and return one piece
    res = gs.join_geometry(geometry=[final_building, building_mass, roof, windows, floor_ledges])
    return res
