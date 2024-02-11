import geometry_script as gs

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from shape_primitives import gs_cube


@gs.tree("Cubed Floor Ledge")
def cubed_floor_ledge(
    size_x: gs.Float, size_z: gs.Float, extrusion_x: gs.Float, extrusion_z: gs.Float
):
    """
    Cube shaped ledges with custom sizes and extrusions. Default size y size 1. made for each floor.

    Parameters:
    - size_x: the size of the ledge on the x axis. i.e. forward/away from building surface.
    - size_z: the size of the ledge on the z axis. i.e. height of ledge.
    - extrusion_x: the extrusion of the ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    """
    # initialize ledge with cube
    # duplicate cube for booleans
    # calculate cutter cube using extrusion params
    # subtract cutter cube from ledge
    shape_factor = 0.1
    ledge_size = gs.combine_xyz(x=shape_factor * size_x, y=1, z=shape_factor * size_z)
    ledge = gs_cube(size=ledge_size)
    cutter = gs_cube(size=ledge_size)
    extrusion_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.x,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_x)),
        ),
    )
    extrusion_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.z,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_z)),
        ),
    )
    cutter_translation = gs.combine_xyz(x=extrusion_x, y=0, z=-extrusion_z)
    cutter = cutter.transform(translation=cutter_translation)
    ledge = ledge.mesh_boolean(
        operation=gs.MeshBoolean.Operation.DIFFERENCE, mesh_2=[cutter]
    )
    return ledge

@gs.tree("Cubed Floor Ledge For Extrusion")
def cubed_floor_ledge_for_extrusion(
    size_x: gs.Float, size_z: gs.Float, extrusion_x: gs.Float, extrusion_z: gs.Float
):
    """
    Cube shaped ledges with custom sizes and extrusions. Returns a curve as plane to be extruded along a curve (edge line of floor). 

    Parameters:
    - size_x: the size of the ledge on the x axis. i.e. forward/away from building surface.
    - size_z: the size of the ledge on the z axis. i.e. height of ledge.
    - extrusion_x: the extrusion of the ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    """
    # initialize grid as plane, use cube (transformed due to extrusion) to cut it. return the cutted plane directly for later extrusions. 
    ledge_size = gs.combine_xyz(x=size_x, y=size_z, z=1)  # note how y and z are flipped
    grid = gs.grid(size_x=ledge_size.x, size_y=ledge_size.y, vertices_x=2, vertices_y=2)
    cutter = gs_cube(size=ledge_size)
    extrusion_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.x,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_x)),
        ),
    )
    extrusion_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.y,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_z)),
        ),
    )
    cutter_translation = gs.combine_xyz(x=extrusion_x, y=-extrusion_z, z=0)
    cutter = cutter.transform(translation=cutter_translation)
    deg_180 = gs.math(operation=gs.Math.Operation.RADIANS, value=180)
    ledge = grid.mesh_boolean(
        operation=gs.MeshBoolean.Operation.DIFFERENCE, mesh_2=[cutter]
    ).mesh_to_curve().transform(rotation=gs.combine_xyz(x=deg_180, y=0, z=0))
    return ledge

    


@gs.tree("Cubed Window Ledge")
def cubed_window_ledge(size: gs.Vector, extrusion_x: gs.Float, extrusion_z: gs.Float):
    """
    Cube shaped ledges with custom sizes and extrusions. made for each window.

    Parameters:
    - size: the size of the ledge.
    - extrusion_x: the extrusion of the ledge from end back to building surface on the x axis. Percentage of ledge size x. 0~1: 0 for no extrusion, 1 for full extrusion.
    - extrusion_z: the extrusion of the ledge from bottom back to ledge top on the z axis. Percentage of ledge size z. 0~1: 0 for no extrusion, 1 for full extrusion.
    """
    # initialize ledge with cube
    # duplicate cube for booleans
    # calculate cutter cube using extrusion params
    # subtract cutter cube from ledge
    shape_factor = 0.1
    ledge_size = gs.combine_xyz(
        x=shape_factor * size.x, y=size.y, z=shape_factor * size.z
    )
    ledge = gs_cube(size=ledge_size)
    cutter = gs_cube(size=ledge_size)
    extrusion_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.x,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_x)),
        ),
    )
    extrusion_z = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(
            ledge_size.z,
            gs.math(operation=gs.Math.Operation.SUBTRACT, value=(1, extrusion_z)),
        ),
    )
    cutter_translation = gs.combine_xyz(x=extrusion_x, y=0, z=-extrusion_z)
    cutter = cutter.transform(translation=cutter_translation)
    ledge = ledge.mesh_boolean(
        operation=gs.MeshBoolean.Operation.DIFFERENCE, mesh_2=[cutter]
    )
    return ledge
