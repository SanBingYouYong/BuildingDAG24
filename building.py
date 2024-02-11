import geometry_script as gs

# import local modules
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from building_mass import building_mass
from roof import roof


@gs.tree("Building Test")
def building_test(bm_type: gs.Int, bm_size: gs.Vector, roof_type: gs.Int, roof_size: gs.Vector):
    """
    Generate a building based on the given building mass and roof.
    """
    bm = building_mass(bm_type=bm_type, bm_size=bm_size)
    # calculate elevation for roof
    res = gs.bounding_box(bm)
    r_pos_z = res.max.z + roof_size.z
    r_pos = gs.combine_xyz(x=0, y=0, z=r_pos_z)  # add xy position? GeoCode fixed at 0,0
    r = roof(roof_type=roof_type, roof_size=roof_size).transform(translation=r_pos)
    # r = building_mass(bm_type=roof_type, bm_size=roof_size).transform(translation=r_pos)
    return gs.mesh_boolean(operation=gs.MeshBoolean.Operation.UNION, mesh_2=[bm, r])
