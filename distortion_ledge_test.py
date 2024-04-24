import bpy
from bpy.types import Object
from typing import List, Tuple
import bmesh

# import local modules
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parents[1]
sys.path.append(str(parent))

from distortion import DRUtils

def get_sharps(obj: Object, threshold: float=0.523599) -> List[List[Tuple[float, float, float]]]:
    bpy.ops.object.editmode_toggle()
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    # select sharp edges
    bpy.ops.mesh.edges_select_sharp(sharpness=threshold)
    obj.update_from_editmode()
    
    sharp_edges = []
    for e in bm.edges:
        if e.select:
            # check for vertical edges to exclude
            if e.verts[0].co.x == e.verts[1].co.x and e.verts[0].co.y == e.verts[1].co.y:
                print('Vertical edge found, skipping')
                continue
            sharp_edges.append((e.verts[0].index, e.verts[1].index))
    
    # find connected edges
    connected_edges = find_connected_edges(sharp_edges)
    print(connected_edges)
    # raise

    ce_as_vert_coords = []
    bm.verts.ensure_lookup_table()
    for edge_seq in connected_edges:
        if len(edge_seq) == 1:  # single edge (not on a curve), skip
            continue
        vert_coords = []
        for edge in edge_seq:
            vert_coords.append(bm.verts[edge].co)
        ce_as_vert_coords.append(vert_coords)
    
    bpy.ops.object.editmode_toggle()
    return ce_as_vert_coords

def find_connected_edges(edges: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    def dfs(node, visited, connected_edges):
        visited[node] = True
        connected_edges.append(node)
        for edge in edges:
            if node in edge:
                neighbor = edge[0] if edge[1] == node else edge[1]
                if neighbor < len(visited) and not visited[neighbor]:
                    dfs(neighbor, visited, connected_edges)
    
    visited = [False] * len(edges)  # Keep track of visited vertices
    connected_edge_sequences = []
    for i in range(len(edges)):
        if not visited[i]:
            connected_edges = []
            dfs(i, visited, connected_edges)
            connected_edge_sequences.append(connected_edges)
    
    return connected_edge_sequences


if __name__ == "__main__":
    obj = bpy.context.active_object
    connected_sharp_edges = get_sharps(obj)
    # print(connected_sharp_edges)
    print(len(connected_sharp_edges))
    count = 0
    for edge_seq in connected_sharp_edges:
        if len(edge_seq) > 1:
            print(edge_seq)
            count += 1
    print(count)
    disturbed_curves = DRUtils.disturb_curves(connected_sharp_edges, obj)
    spawned_curves = DRUtils.spawn_curves(connected_sharp_edges)
