"""
Technical drawings (A4 landscape, scale 1:10, mm) of the lid and the base plate of the battery replacement product.

Datum = lower-left corner of each part seen from the top (for the lid: the intersection of its two straight edges,
the corner itself is R20). X to the right, Y up. Outline dimensions are drawn on the part, every hole is listed
in a coordinate table.

Writes drawing_lid.pdf / .png and drawing_base_plate.pdf / .png. Run from this folder with a venv that has
trimesh and matplotlib:
    python make_drawings.py
"""

import csv
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_build_files import HOLE_D, OUTLINE, PIN_A, PIN_B, PIN_D, PIN_SLOT_LEN
from matplotlib.patches import Arc, Circle, Rectangle
from render_frame import MODULES, POSTS, PRESSURE_BLOCKS

HERE = Path(__file__).parent
SCALE = 10  # 1:10
PAPER = (297, 210)  # A4 landscape
ORIGIN = (30, 100)  # datum on paper
TABLE_X, TABLE_TOP = 205, 200
LID_DATUM = (-691.0, -427.0)
BASE_DATUM = (-701.0, -437.0)
BASE_OUTLINE = [
    (-701, -437),
    (499, -437),
    (499, -287),
    (701, -287),
    (701, 273),
    (499, 273),
    (499, 437),
    (-701, 437),
]
POST_HOLE_D = 9.0  # M8 clearance, countersunk from below (90 deg, D16.5)
LW, THIN = 0.5, 0.25
FS = 6


class Sheet:
    def __init__(self, origin):
        self.fig = plt.figure(figsize=(PAPER[0] / 25.4, PAPER[1] / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAPER[0])
        self.ax.set_ylim(0, PAPER[1])
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.o = origin
        self.ax.add_patch(
            Rectangle((8, 8), PAPER[0] - 16, PAPER[1] - 16, fill=False, lw=0.7)
        )

    def p(self, X, Y):
        """part coordinates (mm, from the datum) -> paper mm"""
        return self.o[0] + X / SCALE, self.o[1] + Y / SCALE

    def line(self, a, b, lw=LW, color="k", **kw):
        (x0, y0), (x1, y1) = self.p(*a), self.p(*b)
        self.ax.plot(
            [x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", **kw
        )

    def cross(self, X, Y, size=1.2):
        x, y = self.p(X, Y)
        self.ax.plot([x - size, x + size], [y, y], color="k", lw=0.12)
        self.ax.plot([x, x], [y - size, y + size], color="k", lw=0.12)

    def text(self, X, Y, s, size=FS, paper_offset=(0, 0), **kw):
        x, y = self.p(X, Y)
        self.ax.text(x + paper_offset[0], y + paper_offset[1], s, fontsize=size, **kw)

    def dim(self, a, b, offset, label=None):
        """Linear dimension between part points a and b, drawn `offset` paper-mm to the side (+ = left of a->b)."""
        (x0, y0), (x1, y1) = self.p(*a), self.p(*b)
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = -uy, ux
        d0 = (x0 + nx * offset, y0 + ny * offset)
        d1 = (x1 + nx * offset, y1 + ny * offset)
        s = 1 if offset > 0 else -1
        for (px, py), (qx, qy) in (((x0, y0), d0), ((x1, y1), d1)):
            self.ax.plot(
                [px + nx * s * 1, qx + nx * s * 1.5],
                [py + ny * s * 1, qy + ny * s * 1.5],
                color="k",
                lw=THIN,
            )
        self.ax.annotate(
            "",
            d1,
            d0,
            arrowprops=dict(
                arrowstyle="<|-|>",
                lw=THIN,
                color="k",
                mutation_scale=5,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        if label is None:
            label = f"{math.dist(a, b):g}"
        ang = math.degrees(math.atan2(uy, ux))
        if ang > 90.1 or ang < -89.9:
            ang -= 180
        mx, my = (d0[0] + d1[0]) / 2, (d0[1] + d1[1]) / 2
        self.ax.text(
            mx + nx * 1.2 * (1 if ang == math.degrees(math.atan2(uy, ux)) else -1),
            my + ny * 1.2 * (1 if ang == math.degrees(math.atan2(uy, ux)) else -1),
            label,
            fontsize=FS,
            rotation=ang,
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", pad=0.3),
        )

    def leader(self, at, text_at_paper, label):
        x, y = self.p(*at)
        self.ax.annotate(
            label,
            (x, y),
            (x + text_at_paper[0], y + text_at_paper[1]),
            fontsize=FS,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=THIN,
                color="k",
                mutation_scale=4,
                shrinkA=0,
                shrinkB=0,
            ),
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", pad=0.3),
        )

    def datum(self):
        x, y = self.p(0, 0)
        self.ax.annotate(
            "",
            (x + 14, y),
            (x, y),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="tab:red"),
        )
        self.ax.annotate(
            "",
            (x, y + 14),
            (x, y),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="tab:red"),
        )
        self.ax.text(x + 15, y - 1.5, "X", color="tab:red", fontsize=FS + 1, va="top")
        self.ax.text(x - 1.5, y + 15, "Y", color="tab:red", fontsize=FS + 1, ha="right")
        self.ax.add_patch(Circle((x, y), 1.2, fc="tab:red", ec="none"))
        self.ax.text(
            x - 2, y - 2, "0,0", color="tab:red", fontsize=FS, ha="right", va="top"
        )

    def table(self, x, y_top, header, rows, widths, row_h=3.6, size=5.5):
        cols = [x]
        for w in widths:
            cols.append(cols[-1] + w)
        for i, row in enumerate([header] + rows):
            yy = y_top - (i + 1) * row_h
            for c, cell in enumerate(row):
                self.ax.text(
                    cols[c] + 1,
                    yy + row_h / 2,
                    cell,
                    fontsize=size,
                    va="center",
                    weight="bold" if i == 0 else "normal",
                )
            self.ax.plot([x, cols[-1]], [yy, yy], color="k", lw=0.5 if i == 0 else 0.15)
        y_bot = y_top - (len(rows) + 1) * row_h
        for cx in cols:
            self.ax.plot([cx, cx], [y_top, y_bot], color="k", lw=0.15)
        self.ax.plot([x, cols[-1]], [y_top, y_top], color="k", lw=0.5)
        return y_bot

    def title_block(self, name, material, notes):
        x0, y0, w = 177, 9, 110
        rows = [
            ("Part", name),
            ("Material", material),
            ("Scale", "1:10 on A4 (print at 100 %)"),
            ("Units", "mm, view from top"),
            ("Tolerances", "ISO 2768-m unless noted"),
            ("Product", "Battery replacement product"),
            ("Date", date.today().isoformat()),
        ]
        h = 5.2
        for i, (k, v) in enumerate(reversed(rows)):
            yy = y0 + i * h
            self.ax.plot([x0, x0 + w], [yy + h, yy + h], color="k", lw=0.3)
            self.ax.text(
                x0 + 1.5, yy + h / 2, k, fontsize=FS, va="center", color="0.35"
            )
            self.ax.text(
                x0 + 22,
                yy + h / 2,
                v,
                fontsize=FS + (1 if k == "Part" else 0),
                va="center",
                weight="bold" if k == "Part" else "normal",
            )
        self.ax.add_patch(Rectangle((x0, y0), w, len(rows) * h, fill=False, lw=0.7))
        self.ax.plot([x0 + 20, x0 + 20], [y0, y0 + len(rows) * h], color="k", lw=0.3)
        for i, n in enumerate(notes):
            self.ax.text(
                x0,
                y0 + len(rows) * h + 3 + (len(notes) - 1 - i) * 3.4,
                n,
                fontsize=FS - 0.5,
            )

    def save(self, stem):
        for ext in ("pdf", "png"):
            self.fig.savefig(HERE / f"{stem}.{ext}", dpi=200 if ext == "png" else None)
        plt.close(self.fig)


def rel(pt, datum):
    return round(pt[0] - datum[0], 1), round(pt[1] - datum[1], 1)


def fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def screw_holes():
    with open(HERE / "nut_positions.csv") as f:
        rows = [r for r in csv.DictReader(f) if "screw" in r["item"]]
    holes = [
        (int(r["item"].split("_")[-1]), float(r["x_model"]), float(r["y_model"]))
        for r in rows
    ]
    return sorted(holes)


def lid():
    s = Sheet(origin=ORIGIN)
    d = LID_DATUM
    # outline: lines + R20 arcs, as in lid_cnc.dxf
    for seg in OUTLINE:
        if seg[0] == "L":
            s.line(rel(seg[1], d), rel(seg[2], d))
        else:
            c, p, q = (rel(v, d) for v in seg[1:])
            ap, aq = (
                math.degrees(math.atan2(v[1] - c[1], v[0] - c[0])) for v in (p, q)
            )
            t0, t1 = (ap, aq) if (aq - ap) % 360 == 90 else (aq, ap)
            cx, cy = s.p(*c)
            s.ax.add_patch(
                Arc((cx, cy), 40 / SCALE, 40 / SCALE, theta1=t0, theta2=t1, lw=LW)
            )
    s.datum()
    # screw holes
    for n, x, y in screw_holes():
        X, Y = rel((x, y), d)
        s.ax.add_patch(Circle(s.p(X, Y), HOLE_D / 2 / SCALE, fill=False, lw=THIN))
        s.cross(X, Y)
        vx, vy = 691 - X, (427 - Y) * 1.6  # label pushed towards the part centre
        L = math.hypot(vx, vy)
        inward = (4.2 * vx / L, 3.6 * vy / L)
        s.text(
            X,
            Y,
            f"{n}",
            size=4.5,
            paper_offset=inward,
            ha="center",
            va="center",
            color="0.25",
        )
    # pin hole + slot
    A, B = rel(PIN_A, d), rel(PIN_B, d)
    s.ax.add_patch(
        Circle(s.p(*A), PIN_D / 2 / SCALE, fill=False, lw=LW, color="tab:red")
    )
    ang = math.atan2(B[1] - A[1], B[0] - A[0])
    u = (math.cos(ang), math.sin(ang))
    h, r = (PIN_SLOT_LEN - PIN_D) / 2, PIN_D / 2
    e0, e1 = (B[0] - u[0] * h, B[1] - u[1] * h), (B[0] + u[0] * h, B[1] + u[1] * h)
    n_ = (-u[1], u[0])
    for sgn in (1, -1):
        s.line(
            (e0[0] + n_[0] * r * sgn, e0[1] + n_[1] * r * sgn),
            (e1[0] + n_[0] * r * sgn, e1[1] + n_[1] * r * sgn),
            lw=THIN,
            color="tab:red",
        )
    for e, a0 in ((e1, -90), (e0, 90)):
        s.ax.add_patch(
            Arc(
                s.p(*e),
                PIN_D / SCALE,
                PIN_D / SCALE,
                theta1=math.degrees(ang) + a0,
                theta2=math.degrees(ang) + a0 + 180,
                lw=THIN,
                color="tab:red",
            )
        )
    s.leader(A, (14, 10), f"P1  Ø{PIN_D:g} H8\n(round pin)")
    s.leader(
        B,
        (-30, 6),
        f"P2  slot {PIN_D:g} H8 × {PIN_SLOT_LEN:g}\naxis {math.degrees(ang):.2f}° to X\n(on line P1–P2)",
    )
    s.line(A, B, lw=0.15, linestyle=(0, (8, 3, 1, 3)), color="tab:red")
    # pressure blocks (bonded on top, not machined)
    for i, (xa, xb, ya, yb) in enumerate(PRESSURE_BLOCKS, 1):
        X0, Y0 = rel((xa, ya), d)
        X1, Y1 = rel((xb, yb), d)
        x0, y0 = s.p(X0, Y0)
        s.ax.add_patch(
            Rectangle(
                (x0, y0),
                (X1 - X0) / SCALE,
                (Y1 - Y0) / SCALE,
                fill=False,
                lw=THIN,
                ls="--",
            )
        )
        s.text((X0 + X1) / 2, Y1, f"B{i}", paper_offset=(0, 2), ha="center", size=5)

    # outline dimensions (from the datum)
    W, H = 1382, 854
    s.dim((0, 0), (W, 0), -26)  # overall length
    s.dim((0, 0), (1180, 0), -18)  # main body length
    s.dim((1180, 0), (W, 0), -18)  # step length 202
    s.dim((0, 0), (0, H), 14)  # overall width
    s.dim((W, 150), (W, 690), -14)  # step width 540
    s.dim((W, 0), (W, 150), -14)  # step offset from Y=0
    s.dim((W, 690), (W, H), -14)  # 164
    s.leader((1362 + 14.14, 170 - 14.14), (9, -8), "R20")  # convex step corner
    s.leader(
        (1200 - 14.14, 130 + 14.14), (-4, -14), "R20 (concave)"
    )  # concave step corner
    s.text(
        0,
        H,
        "all 8 outside corners R20, both inside corners R20;\nall edges straight, all angles 90°",
        paper_offset=(2, 6),
        size=FS,
    )

    # tables
    rows = [
        [f"{n}", fmt(rel((x, y), d)[0]), fmt(rel((x, y), d)[1]), f"Ø{HOLE_D:g} thru"]
        for n, x, y in screw_holes()
    ]
    rows += [
        ["P1", fmt(A[0]), fmt(A[1]), f"Ø{PIN_D:g} H8 thru"],
        [
            "P2",
            fmt(B[0]),
            fmt(B[1]),
            f"slot {PIN_D:g}H8×{PIN_SLOT_LEN:g}, {math.degrees(ang):.2f}°",
        ],
    ]
    s.table(
        TABLE_X, TABLE_TOP, ["Hole", "X", "Y", "Size"], rows, [9, 13, 13, 40], row_h=3.1
    )
    brows = []
    for i, (xa, xb, ya, yb) in enumerate(PRESSURE_BLOCKS, 1):
        X0, Y0 = rel((xa, ya), d)
        X1, Y1 = rel((xb, yb), d)
        brows.append(
            [
                f"B{i}",
                f"{fmt(X0)}–{fmt(X1)}",
                f"{fmt(Y0)}–{fmt(Y1)}",
                f"{fmt(X1 - X0)} × {fmt(Y1 - Y0)} × 5",
            ]
        )
    s.ax.text(
        20,
        64,
        "Pressure blocks B1–B5: printed, 5 mm high, bonded on the top face (not machined)",
        fontsize=FS,
        weight="bold",
    )
    s.table(
        20,
        62,
        ["Block", "X from–to", "Y from–to", "Size"],
        brows,
        [12, 28, 28, 28],
        row_h=3.6,
    )
    s.title_block(
        "Lid",
        "Hard PVC 5 mm, opaque matte — CNC-routed, never laser-cut",
        [
            "Datum 0,0 = intersection of the left and bottom edges (corner itself R20).",
            f"Hole numbers = lid_screw_NN in the model; 41× Ø{HOLE_D:g} for M4 (ISO 273 medium).",
            "CNC file: lid_cnc.dxf (origin at part centre: X = x + 691, Y = y + 427).",
            "Deburr all holes. Dashed: bonded pressure blocks B1–B5.",
        ],
    )
    s.save("drawing_lid")


def base_plate():
    s = Sheet(origin=ORIGIN)
    d = BASE_DATUM
    pts = [rel(p, d) for p in BASE_OUTLINE]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        s.line(a, b)
    s.datum()
    holes = [rel(p, d) for p in POSTS]
    order = sorted(range(len(holes)), key=lambda i: (holes[i][0], holes[i][1]))
    rows = []
    for k, i in enumerate(order, 1):
        X, Y = holes[i]
        s.ax.add_patch(Circle(s.p(X, Y), POST_HOLE_D / 2 / SCALE, fill=False, lw=THIN))
        s.ax.add_patch(Circle(s.p(X, Y), 16.5 / 2 / SCALE, fill=False, lw=0.15, ls=":"))
        s.cross(X, Y)
        s.text(
            X, Y, f"H{k}", size=5, paper_offset=(4, 3 if Y < 437 else -3), va="center"
        )
        rows.append([f"H{k}", fmt(X), fmt(Y), f"Ø{POST_HOLE_D:g}, csk 90° Ø16.5 below"])
    mrows = []
    for k, (x, y) in enumerate(sorted(MODULES), 1):
        X, Y = rel((x - 125, y - 125), d)
        x0, y0 = s.p(X, Y)
        s.ax.add_patch(
            Rectangle(
                (x0, y0),
                250 / SCALE,
                250 / SCALE,
                fill=False,
                lw=0.15,
                ls="--",
                color="0.4",
            )
        )
        s.text(X + 125, Y + 125, f"M{k}", size=5, ha="center", va="center", color="0.4")
        mrows.append(
            [
                f"M{k}",
                f"{fmt(X)}–{fmt(X + 250)}",
                f"{fmt(Y)}–{fmt(Y + 250)}",
                "250 × 250",
            ]
        )

    W, H = 1402, 874
    s.dim((0, 0), (W, 0), -26)
    s.dim((0, 0), (1200, 0), -18)
    s.dim((1200, 0), (W, 0), -18)
    s.dim((0, 0), (0, H), 14)
    s.dim((W, 150), (W, 710), -14)
    s.dim((W, 0), (W, 150), -14)
    s.dim((W, 710), (W, H), -14)
    s.text(
        0,
        H,
        "all corners sharp, all angles 90°;\noutline = outer outline of the profile frame",
        paper_offset=(2, 6),
        size=FS,
    )

    s.table(
        TABLE_X, TABLE_TOP, ["Hole", "X", "Y", "Size"], rows, [9, 13, 13, 40], row_h=3.6
    )
    s.ax.text(
        20,
        64,
        "Module footprints M1–M8 (dashed; for placing the locating corners, not machined)",
        fontsize=FS,
        weight="bold",
    )
    s.table(
        20,
        62,
        ["Module", "X from–to", "Y from–to", "Size"],
        mrows,
        [12, 28, 28, 20],
        row_h=3.4,
    )
    s.title_block(
        "Base plate",
        "MDF or aluminium composite, 10 mm",
        [
            "Datum 0,0 = bottom-left corner, view from top (frame side).",
            "H1–H8: M8 countersunk screws from below into the post core bores.",
            "The frame's posts stand centred on H1–H8 (40 × 40).",
        ],
    )
    s.save("drawing_base_plate")


if __name__ == "__main__":
    lid()
    base_plate()
    print("written: drawing_lid.pdf/.png, drawing_base_plate.pdf/.png")
