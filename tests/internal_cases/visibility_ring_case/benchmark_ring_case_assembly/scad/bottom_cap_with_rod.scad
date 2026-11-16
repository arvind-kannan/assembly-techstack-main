include <common_params.scad>;

union() {
    // bottom cap
    cylinder(h = cap_thickness, r = cylinder_radius);

    // rod attached to bottom cap
    translate([0, 0, cap_thickness])
        cylinder(h = rod_height, r = rod_radius);
}