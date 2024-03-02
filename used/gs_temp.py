import geometry_script as gs

# test with a simple cube building mass node
@gs.tree("Test Test")
def testtest(radius: gs.FloatDistance, height: gs.FloatDistance):
    # convert radius to float
    meshline_horizontal = gs.switch(
        switch=True,
        false=None,
        true=gs.mesh_line(mode=gs.MeshLine.Mode.OFFSET, count=2)
    )
    return meshline_horizontal


@gs.tree("Separation Circle Test")
def separation_circle_test(bm_size: gs.Vector):
    is_cylinder = True
    cyl_offset_x = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(bm_size.x / 2, 
               gs.math(
                   operation=gs.Math.Operation.COSINE,
                   value=0.78
               ))
    )
    cyl_offset_y = gs.math(
        operation=gs.Math.Operation.MULTIPLY,
        value=(bm_size.y / 2, 
               gs.math(
                   operation=gs.Math.Operation.SINE,
                   value=0.78
               ))
    )
    cylinder_line_positive_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.quadratic_bezier(
            start=gs.combine_xyz(x=cyl_offset_x, y=-cyl_offset_y, z=0), # TODO: add height offsets
            middle=gs.combine_xyz(x=bm_size.x / 2, y=0, z=0),
            end=gs.combine_xyz(x=cyl_offset_x, y=cyl_offset_y, z=0),
        ),
    )
    cylinder_line_positive_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.quadratic_bezier(
            start=gs.combine_xyz(x=cyl_offset_x, y=cyl_offset_y, z=0), # TODO: add height offsets
            middle=gs.combine_xyz(x=0, y=bm_size.y / 2, z=0),
            end=gs.combine_xyz(x=-cyl_offset_x, y=cyl_offset_y, z=0),
        ),
    )
    cylinder_line_negative_x = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.quadratic_bezier(
            start=gs.combine_xyz(x=-cyl_offset_x, y=cyl_offset_y, z=0), # TODO: add height offsets
            middle=gs.combine_xyz(x=-bm_size.x / 2, y=0, z=0),
            end=gs.combine_xyz(x=-cyl_offset_x, y=-cyl_offset_y, z=0),
        ),
    )
    cylinder_line_negative_y = gs.switch(
        input_type=gs.Switch.InputType.GEOMETRY,
        switch=is_cylinder,
        false=None,
        true=gs.quadratic_bezier(
            start=gs.combine_xyz(x=-cyl_offset_x, y=-cyl_offset_y, z=0), # TODO: add height offsets
            middle=gs.combine_xyz(x=0, y=-bm_size.y / 2, z=0),
            end=gs.combine_xyz(x=cyl_offset_x, y=-cyl_offset_y, z=0),
        )
    )
    return gs.join_geometry(
        geometry=[
            gs.curve_circle(radius=0.5).transform(translation=gs.combine_xyz(x=0, y=0, z=0.5)),
            cylinder_line_positive_x, cylinder_line_positive_y, cylinder_line_negative_x, cylinder_line_negative_y]
    )


@gs.tree("Vector Test")
def vector_test(
    window_size: gs.Vector,
    window_panel_percentage: gs.Vector,
    window_divided_horizontal: gs.Bool,
    window_divided_vertical: gs.Bool,
    window_interpanel_offset: gs.Float, 
    window_panel_extrusion: gs.Float, 
):
    # get panel size
    window_panel_size = gs.combine_xyz(
        x=window_size.x * window_panel_percentage.x,
        y=window_size.y * window_panel_percentage.y,
        z=window_size.z * window_panel_percentage.z,
    )
    return gs.cube(), window_panel_size

@gs.tree("Math Abrv Test")
def math_abrv_test():
    cube = gs.cube()
    size = gs.combine_xyz(x=1, y=2, z=3)
    math = -size.x - size.y
    tp = gs.combine_xyz(
        x=-size.x, 
        y=-size.x - size.y,
        z=size.x + size.z
    )
    return cube, math, tp

@gs.tree("Switch Test")
def switch_test(
    a: gs.Bool, b: gs.Bool, c: gs.Bool, d: gs.Bool
):
    result = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=a,
        false=None,
        true=gs.combine_xyz(x=1, y=2, z=3),
    )
    result = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=b,
        false=result,
        true=gs.combine_xyz(x=4, y=5, z=6),
    )
    result = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=c,
        false=result,
        true=gs.combine_xyz(x=7, y=8, z=9),
    )
    result = gs.switch(
        input_type=gs.Switch.InputType.VECTOR,
        switch=d,
        false=result,
        true=gs.combine_xyz(x=10, y=11, z=12),
    )
    return result

@gs.tree("Geometry IO Test")
def geometry_io_test(
    geom_a: gs.Geometry, 
    geom_b: gs.Geometry,
) -> gs.Geometry:
    cut = gs.mesh_boolean(
        operation=gs.MeshBoolean.Operation.UNION,
        mesh_2=[geom_a, geom_b],
    )
    return cut, geom_a, geom_b

@gs.tree("Diag Test")
def diag_test(bm_size: gs.Vector):
    # bm_size = gs.combine_xyz(x=1, y=2, z=3)
    sq_bm_size = gs.combine_xyz(
        x=gs.math(operation=gs.Math.Operation.POWER, value=(bm_size.x, 2)),
        y=gs.math(operation=gs.Math.Operation.POWER, value=(bm_size.y, 2)),
        z=gs.math(operation=gs.Math.Operation.POWER, value=(bm_size.z, 2)),
    )
    diag = gs.math(
        operation=gs.Math.Operation.SQRT,
        value=gs.math(
            operation=gs.Math.Operation.ADD,
            value=(gs.math(operation=gs.Math.Operation.ADD, value=(sq_bm_size.x, sq_bm_size.y)), sq_bm_size.z),
        ),
    )
    # get factor to resize diagonal to 1
    diag_factor = gs.math(
        operation=gs.Math.Operation.DIVIDE, value=(1, diag)
    )
    # resize bm_size with diag_factor
    clamped_bm_size = gs.vector_math(
        operation=gs.VectorMath.Operation.SCALE,
        vector=bm_size, scale=diag_factor,
    )
    return clamped_bm_size
