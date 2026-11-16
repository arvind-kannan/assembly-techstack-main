include <common_params.scad>;

hole_radius = 3.100000;

translate([0, 0, cylinder_height])
difference() {
    cylinder(h = cap_thickness, r = cylinder_radius);
    translate([0, 0, -0.1])
        cylinder(h = cap_thickness + 0.2, r = hole_radius);
}