
// Auto-generated benchmark case.
// Units are millimetres.

pipe_outer_radius = 21.0;
pipe_inner_radius = 15.0;
bend_centerline_radius = 51.0;
straight_len_x = 24.0;
straight_len_y = 70.0;

bolt_station_x = 51.0;
bolt_station_y = 36.0;

side_hole_radius = 3.2;

bolt_shaft_radius = 2.0;
bolt_head_radius = 5.4;
bolt_head_thickness = 4.0;
shaft_start_z = -25.5;
shaft_end_z = 31.0;
shaft_length = 56.5;
shaft_center_z = 2.75;
head_center_z = -27.0;

nut_inner_radius = 2.15;
nut_outer_radius = 5.4;
nut_thickness = 4.0;
nut_center_z = 27.0;

lid_outer_radius = 21.0;
lid_thickness = 6.0;
lid_center_x = -27.0;
lid_center_y = -51.0;

eps = 0.05;

module quarter_torus(radius_major, radius_minor, fn_major=96, fn_minor=64) {
    rotate_extrude(angle=90, convexity=10, $fn=fn_major)
        translate([radius_major, 0, 0])
            circle(r=radius_minor, $fn=fn_minor);
}

module pipe_outer_solid() {
    union() {
        // Bend centerline: from (0, -R) to (R, 0).
        // This makes the +X and +Y straight segments tangent to the elbow.
        rotate([0, 0, -90])
            quarter_torus(bend_centerline_radius, pipe_outer_radius);

        // -X straight pipe segment.
        translate([-straight_len_x, -bend_centerline_radius, 0])
            rotate([0, 90, 0])
                cylinder(
                    h=straight_len_x,
                    r=pipe_outer_radius,
                    center=false,
                    $fn=96
                );

        // +Y straight pipe segment.
        translate([bend_centerline_radius, 0, 0])
            rotate([-90, 0, 0])
                cylinder(
                    h=straight_len_y,
                    r=pipe_outer_radius,
                    center=false,
                    $fn=96
                );
    }
}

module pipe_inner_void() {
    union() {
        rotate([0, 0, -90])
            quarter_torus(bend_centerline_radius, pipe_inner_radius);

        // Inner void for -X straight pipe.
        translate([-straight_len_x - eps, -bend_centerline_radius, 0])
            rotate([0, 90, 0])
                cylinder(
                    h=straight_len_x + 2 * eps,
                    r=pipe_inner_radius,
                    center=false,
                    $fn=96
                );

        // Inner void for +Y straight pipe.
        translate([bend_centerline_radius, -eps, 0])
            rotate([-90, 0, 0])
                cylinder(
                    h=straight_len_y + 2 * eps,
                    r=pipe_inner_radius,
                    center=false,
                    $fn=96
                );
    }
}

module bolt_side_holes() {
    translate([bolt_station_x, bolt_station_y, 0])
        cylinder(h=4 * pipe_outer_radius, r=side_hole_radius, center=true, $fn=64);
}

module part_j_elbow_pipe() {
    difference() {
        pipe_outer_solid();
        pipe_inner_void();
        bolt_side_holes();
    }
}

module bolt_thread_marker() {
    for (i = [0:4]) {
        translate([
            bolt_station_x,
            bolt_station_y,
            nut_center_z - nut_thickness * 0.45 + i * nut_thickness * 0.18
        ])
            cylinder(
                h=0.22,
                r=bolt_shaft_radius + 0.18,
                center=true,
                $fn=64
            );
    }
}

module part_i_bolt() {
    union() {
        translate([bolt_station_x, bolt_station_y, shaft_center_z])
            cylinder(
                h=shaft_length,
                r=bolt_shaft_radius,
                center=true,
                $fn=64
            );

        translate([bolt_station_x, bolt_station_y, head_center_z])
            cylinder(
                h=bolt_head_thickness,
                r=bolt_head_radius,
                center=true,
                $fn=64
            );

        bolt_thread_marker();
    }
}

module part_k_nut() {
    translate([bolt_station_x, bolt_station_y, nut_center_z])
        difference() {
            cylinder(
                h=nut_thickness,
                r=nut_outer_radius,
                center=true,
                $fn=6
            );

            cylinder(
                h=nut_thickness + 2 * eps,
                r=nut_inner_radius,
                center=true,
                $fn=64
            );
        }
}

module part_l_cover_lid() {
    translate([lid_center_x, lid_center_y, 0])
        rotate([0, 90, 0])
            cylinder(
                h=lid_thickness,
                r=lid_outer_radius,
                center=true,
                $fn=96
            );
}

module assembly_colored() {
    color([0.00, 0.52, 0.56, 0.42])
        part_j_elbow_pipe();

    color([0.93, 0.72, 0.05, 1.00])
        part_i_bolt();

    color([0.70, 0.22, 0.17, 1.00])
        part_k_nut();

    color([0.08, 0.27, 0.40, 0.55])
        part_l_cover_lid();
}
