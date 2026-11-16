include <common_params.scad>;

translate([0, 0, ring_z])
difference() {
    cylinder(h = ring_thickness, r = ring_outer_radius);
    translate([0, 0, -0.1])
        cylinder(h = ring_thickness + 0.2, r = ring_inner_radius);
}