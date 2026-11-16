"""
Build files for the battery replacement product (ticket "Build the battery replacement product").

Reads the Yonghang model (lid outline, 41 screw holes) and writes:
  lid_cnc.dxf        lid outline (R20 corners) + 41x D4.8 holes + D8 pin hole + 8x12 pin slot, mm, z ignored
  nut_positions.csv  per profile: which screw/pin sits at which distance from the profile's start end
Frame (top ring, 40x40 profiles, slot centrelines under the screw lines), all in model coordinates.
It follows the lid's stepped outline, so the stepped end is the outer edge:
  long_minus  y=-417, x -701..499     long_plus  y=+417, x -701..499   (run through)
  left        x=-681, y -397..397     (between the long ones)
  right       x=+681, y -287..273     (runs through; the step_h pieces butt into it)
  step_h_minus y=-267, x 459..661     step_h_plus y=+253, x 459..661
  step_v_minus x=+479, y -397..-287   step_v_plus x=+479, y 273..397  (between long and step_h)

Run with any venv that has trimesh (e.g. the pomdp_disassembly_gym one); battery_yonghang/ is not in this repo yet:
    python make_build_files.py <path to battery_yonghang> .
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

HOLE_D = 4.8  # ISO 273 medium-coarse clearance for M4
PIN_D = 8.0
PIN_SLOT_LEN = 12.0  # slot length along the pin-to-pin line (8 wide)
PIN_A = (
    -681.0,
    -292.0,
)  # round hole; on the left profile, between screws at y=-237 and -347
PIN_B = (681.0, 105.0)  # slot; on the right profile, between screws at y=56.9 and 152.9

# name: (axis, fixed coordinate, start, end); axis = coordinate that varies along the profile
PROFILES = {
    "long_minus": ("x", -417.0, -701.0, 499.0),
    "long_plus": ("x", 417.0, -701.0, 499.0),
    "left": ("y", -681.0, -397.0, 397.0),
    "right": ("y", 681.0, -287.0, 273.0),
    "step_h_minus": ("x", -267.0, 459.0, 661.0),
    "step_h_plus": ("x", 253.0, 459.0, 661.0),
    "step_v_minus": ("y", 479.0, -397.0, -287.0),
    "step_v_plus": ("y", 479.0, 273.0, 397.0),
}
MIN_END_DIST = 12.0  # a slot nut needs this much profile beyond its centre

# lid outline, counter-clockwise: (x, y) corner points and (cx, cy, r) fillets between them
OUTLINE = [
    ("L", (-691, -407), (-691, 407)),
    ("A", (-671, 407), (-691, 407), (-671, 427)),
    ("L", (-671, 427), (469, 427)),
    ("A", (469, 407), (469, 427), (489, 407)),
    ("L", (489, 407), (489, 283)),
    ("A", (509, 283), (489, 283), (509, 263)),
    ("L", (509, 263), (671, 263)),
    ("A", (671, 243), (671, 263), (691, 243)),
    ("L", (691, 243), (691, -257)),
    ("A", (671, -257), (691, -257), (671, -277)),
    ("L", (671, -277), (509, -277)),
    ("A", (509, -297), (509, -277), (489, -297)),
    ("L", (489, -297), (489, -407)),
    ("A", (469, -407), (489, -407), (469, -427)),
    ("L", (469, -427), (-671, -427)),
    ("A", (-671, -407), (-671, -427), (-691, -407)),
]


def screw_holes(model_dir):
    """Hole centres (x, y) of the 41 lid screws, from the screw meshes, in model order."""
    holes = []
    for f in sorted(Path(model_dir, "Battery_Yonghang").glob("*_lid_screw_*.stl")):
        lo, hi = trimesh.load(f).bounds
        holes.append(
            (f.stem, round((lo[0] + hi[0]) / 2, 1), round((lo[1] + hi[1]) / 2, 1))
        )
    assert len(holes) == 41, len(holes)
    return holes


def profile_of(x, y):
    for name, (axis, fixed, start, end) in PROFILES.items():
        along, across = (x, y) if axis == "x" else (y, x)
        if (
            abs(across - fixed) < 0.5
            and start + MIN_END_DIST <= along <= end - MIN_END_DIST
        ):
            return name, along - start
    raise ValueError(f"no profile slot under ({x}, {y})")


def dxf(entities):
    out = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$INSUNITS",
        "70",
        "4",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    for e in entities:
        out += e
    return "\n".join(out + ["0", "ENDSEC", "0", "EOF", ""])


def line(layer, p, q):
    return [
        "0",
        "LINE",
        "8",
        layer,
        "10",
        f"{p[0]:.3f}",
        "20",
        f"{p[1]:.3f}",
        "11",
        f"{q[0]:.3f}",
        "21",
        f"{q[1]:.3f}",
    ]


def arc(layer, c, r, a0, a1):
    return [
        "0",
        "ARC",
        "8",
        layer,
        "10",
        f"{c[0]:.3f}",
        "20",
        f"{c[1]:.3f}",
        "40",
        f"{r:.3f}",
        "50",
        f"{a0:.4f}",
        "51",
        f"{a1:.4f}",
    ]


def circle(layer, c, d):
    return [
        "0",
        "CIRCLE",
        "8",
        layer,
        "10",
        f"{c[0]:.3f}",
        "20",
        f"{c[1]:.3f}",
        "40",
        f"{d / 2:.3f}",
    ]


def outline_entities():
    ents = []
    for seg in OUTLINE:
        if seg[0] == "L":
            ents.append(line("OUTLINE", seg[1], seg[2]))
            continue
        c, p, q = seg[1:]
        r = math.dist(c, p)
        assert abs(r - 20) < 1e-6 and abs(math.dist(c, q) - 20) < 1e-6
        ap, aq = (
            math.degrees(math.atan2(v[1] - c[1], v[0] - c[0])) % 360 for v in (p, q)
        )
        # DXF arcs run counter-clockwise; a 90 deg fillet runs p->q (convex) or q->p (concave)
        ents.append(
            arc("OUTLINE", c, r, ap, aq)
            if (aq - ap) % 360 == 90
            else arc("OUTLINE", c, r, aq, ap)
        )
    return ents


def slot_entities(c, toward, width, length):
    u = np.subtract(toward, c) / math.dist(c, toward)
    n = np.array([-u[1], u[0]])
    h, r = (length - width) / 2, width / 2
    e0, e1 = np.array(c) - u * h, np.array(c) + u * h
    ang = math.degrees(math.atan2(u[1], u[0]))
    return [
        line("PIN", e0 + n * r, e1 + n * r),
        line("PIN", e0 - n * r, e1 - n * r),
        arc("PIN", e1, r, ang - 90, ang + 90),
        arc("PIN", e0, r, ang + 90, ang + 270),
    ]


def main(model_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    holes = screw_holes(model_dir)
    ents = outline_entities()
    ents += [circle("HOLES_D4.8", (x, y), HOLE_D) for _, x, y in holes]
    ents.append(circle("PIN", PIN_A, PIN_D))
    ents += slot_entities(PIN_B, PIN_A, PIN_D, PIN_SLOT_LEN)
    (out / "lid_cnc.dxf").write_text(dxf(ents))

    rows = [(*profile_of(x, y), name, x, y) for name, x, y in holes]
    rows += [
        (*profile_of(*PIN_A), "pin_A_round", *PIN_A),
        (*profile_of(*PIN_B), "pin_B_slot", *PIN_B),
    ]
    rows.sort(key=lambda r: (list(PROFILES).index(r[0]), r[1]))
    with open(out / "nut_positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "profile",
                "profile_length_mm",
                "dist_from_start_end_mm",
                "item",
                "x_model",
                "y_model",
            ]
        )
        for prof, dist, item, x, y in rows:
            _, _, start, end = PROFILES[prof]
            w.writerow([prof, round(end - start, 1), round(dist, 1), item, x, y])

    # sanity: pins must not repeat under a 180 deg turn of the lid (one-way fit)
    assert math.dist((-PIN_A[0], -PIN_A[1]), PIN_B) > 20
    counts = {p: sum(r[0] == p for r in rows) for p in PROFILES}
    print(f"{len(holes)} holes, nuts per profile: {counts}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
