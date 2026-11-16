include <common_params.scad>;

difference() {
    cylinder(h = cylinder_height, r = cylinder_radius);
    translate([0, 0, -0.1])
        cylinder(h = cylinder_height + 0.2, r = cylinder_radius - wall_thickness);
}