import geometry_script as gs

# import local modules
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from shape_primitives import gs_cube, gs_sphere, gs_cylinder_scaled, gs_half_sphere


# TODO: add proper roof shapes
# TODO: add operations for creating more complex shapes

@gs.tree("Roof")
def roof(roof_type: gs.Int, roof_size: gs.Vector):
    """
    Generate a roof based on the given shape and size.

    Parameters:
    - roof_type (gs.Int): The shape of the roof. 0 for cube, 1 for cylinder, 2 for half sphere.
    - roof_size (gs.Vector): The size of the roof. Primitives shapes use a unified xyz scale to represent size.

    Returns:
    - The generated roof.

    Raises:
    - ValueError: If the shape is not a valid value (0, 1 or 2).
    """
    # use compare to determine shape type: currently switch nodes can have T/F outputs only, so 3 types need 2 compares (flat?gabled?hipped)
    is_cube = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=roof_type, b=0)
    is_cylinder = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=roof_type, b=1)
    is_sphere = gs.compare(operation=gs.Compare.Operation.EQUAL, data_type=gs.Compare.DataType.INT, a=roof_type, b=2)
    # half size for non-half-spheres; to unify scale for half-sphere and other shapes. 
    half_height_rf_size = gs.combine_xyz(
        x=roof_size.x,
        y=roof_size.y,
        z=roof_size.z / 2
    )
    cube = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, 
                     switch=is_cube, 
                     false=None, 
                     true=gs_cube(size=half_height_rf_size).transform(translation=gs.combine_xyz(x=0, y=0, z=-roof_size.z / 4))
                     )
    cylinder = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, 
                       switch=is_cylinder, 
                       false=None, 
                       true=gs_cylinder_scaled(size=half_height_rf_size).transform(translation=gs.combine_xyz(x=0, y=0, z=-roof_size.z / 4))
                       )
    sphere = gs.switch(input_type=gs.Switch.InputType.GEOMETRY, 
                       switch=is_sphere, 
                       false=None, 
                       true=gs_half_sphere(size=roof_size)
                       )
    roof_shape = gs.mesh_boolean(operation=gs.MeshBoolean.Operation.UNION, mesh_2=[cube, cylinder, sphere])
    return roof_shape