# PROTOTYPE (#78): generates mockup.svg, the half/half layout (viewer left, graph + next steps right).
W, H = 1400, 820
C = dict(
    bg="#f4f5fa",
    card="#ffffff",
    text="#1c1d26",
    muted="#6b6f80",
    border="#d9dbe6",
    accent="#5b6bf0",
    soft="#eef0ff",
    goal="#1fae66",
    canvas="#eceef5",
    warn="#d98a00",
    rec="#d9534f",
)
o = []
a = o.append
a(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Arial, sans-serif" font-size="13">'
)
a(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>')


def btn(x, y, w, label, primary=False, caret=False, h=30):
    f = C["accent"] if primary else C["card"]
    t = "#fff" if primary else C["text"]
    s = C["accent"] if primary else C["border"]
    a(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" stroke="{s}"/>'
    )
    a(
        f'<text x="{x + w / 2 - (6 if caret else 0)}" y="{y + h / 2 + 4.5}" text-anchor="middle" fill="{t}" font-weight="600">{label}</text>'
    )
    if caret:
        a(
            f'<path d="M{x + w - 17} {y + h / 2 - 2} l5 5 l5 -5" fill="none" stroke="{t}" stroke-width="1.6"/>'
        )


def badge(x, y, w, label, color=None):
    a(
        f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="11" fill="{C["card"]}" stroke="{C["border"]}"/>'
    )
    a(
        f'<text x="{x + w / 2}" y="{y + 15}" text-anchor="middle" fill="{color or C["muted"]}" font-size="12" font-weight="600">{label}</text>'
    )


def small_caps(x, y, text, anchor="start"):
    a(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{C["muted"]}" font-size="11" font-weight="700" letter-spacing=".05em">{text}</text>'
    )


# ---------------- left: product viewer (stylised placeholder for the 3D view)
LX, LW, TOP, HH = 12, 676, 12, 796
a(
    f'<rect x="{LX}" y="{TOP}" width="{LW}" height="{HH}" rx="14" fill="{C["canvas"]}" stroke="{C["border"]}"/>'
)
cx, cy = 350, 470
a(
    f'<path d="M{cx - 175} {cy - 40} L{cx - 175} {cy + 190} Q{cx} {cy + 290} {cx + 175} {cy + 190} L{cx + 175} {cy - 40} Z" fill="#3f6fb5"/>'
)
a(
    f'<path d="M{cx - 175} {cy + 150} Q{cx} {cy + 250} {cx + 175} {cy + 150} L{cx + 175} {cy + 190} Q{cx} {cy + 290} {cx - 175} {cy + 190} Z" fill="#7a7a7a"/>'
)
a(f'<ellipse cx="{cx}" cy="{cy - 40}" rx="175" ry="82" fill="#d9534f"/>')
a(f'<ellipse cx="{cx}" cy="{cy - 40}" rx="125" ry="56" fill="#e8e9f0"/>')
a(f'<ellipse cx="{cx}" cy="{cy - 30}" rx="100" ry="42" fill="#5cb85c" opacity=".55"/>')
a(
    f'<rect x="{cx - 48}" y="{cy - 120}" width="96" height="86" fill="#3f9d4a"/>'
    f'<ellipse cx="{cx}" cy="{cy - 120}" rx="48" ry="19" fill="#5cc26a"/><ellipse cx="{cx}" cy="{cy - 120}" rx="32" ry="12" fill="#2e7a38"/>'
)
for sx, sy, col in [
    (cx - 165, cy - 70, "#e377b0"),
    (cx + 165, cy - 70, "#9a6fd0"),
    (cx - 30, cy + 20, "#35b7b0"),
    (cx + 10, cy - 135, "#e8b030"),
]:
    a(f'<circle cx="{sx}" cy="{sy}" r="9" fill="{col}"/>')
a(
    f'<text x="{LX + LW / 2}" y="{TOP + HH - 14}" text-anchor="middle" fill="{C["muted"]}" font-size="11.5">drag to orbit · scroll to zoom · camera always fitted to the whole product</text>'
)

# toolbar over the viewer
btn(28, 28, 76, "Load")
btn(112, 28, 76, "Store")
btn(196, 28, 140, "Set targets", caret=True)
btn(LX + LW - 16 - 80 - 8 - 70, 28, 70, "Frame")
btn(LX + LW - 16 - 80, 28, 80, "X-ray")

# open sub-menu of "Set targets"
mx, my, mw, mh = 196, 64, 300, 262
a(
    f'<rect x="{mx}" y="{my + 2}" width="{mw}" height="{mh}" rx="10" fill="#000" opacity=".06"/>'
)
a(
    f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" fill="{C["card"]}" stroke="{C["border"]}"/>'
)
small_caps(mx + 14, my + 24, "PART ROLES")
small_caps(mx + mw - 125, my + 24, "BASE")
small_caps(mx + mw - 58, my + 24, "TARGET")
parts = [
    ("01_upper_lid", "#d9534f", 0, 0),
    ("02_stator", "#3f6fb5", 1, 0),
    ("03_rotor_short", "#5cb85c", 0, 1),
    ("04_screw_1", "#e8b030", 0, 0),
    ("05_screw_2", "#9a6fd0", 0, 0),
    ("06_screw_3", "#35b7b0", 0, 0),
    ("07_screw_4", "#e377b0", 0, 0),
    ("08_lower_lid", "#7a7a7a", 1, 0),
]
for i, (n, col, b, t) in enumerate(parts):
    y = my + 38 + i * 23
    if t:
        a(
            f'<rect x="{mx + 6}" y="{y - 3}" width="{mw - 12}" height="22" rx="6" fill="{C["soft"]}"/>'
        )
    a(
        f'<rect x="{mx + 14}" y="{y + 3}" width="10" height="10" rx="3" fill="{col}"/><text x="{mx + 32}" y="{y + 13}" fill="{C["text"]}">{n}</text>'
    )
    for bx, on in ((mx + mw - 125, b), (mx + mw - 52, t)):
        a(
            f'<rect x="{bx}" y="{y + 1}" width="14" height="14" rx="4" fill="{C["accent"] if on else C["card"]}" stroke="{C["accent"] if on else C["border"]}"/>'
        )
        if on:
            a(
                f'<path d="M{bx + 3} {y + 8} l3 3 l5 -6" fill="none" stroke="#fff" stroke-width="1.8"/>'
            )
a(
    f'<line x1="{mx}" y1="{my + mh - 38}" x2="{mx + mw}" y2="{my + mh - 38}" stroke="{C["border"]}"/>'
)
a(
    f'<text x="{mx + 14}" y="{my + mh - 14}" fill="{C["warn"]}" font-size="11.5">edits make the graph stale</text>'
)
btn(mx + mw - 78, my + mh - 33, 66, "Apply", primary=True, h=26)

badge(28, TOP + HH - 52, 92, "8 parts left")
badge(128, TOP + HH - 52, 70, "depth 0")

# ---------------- right: graph + next steps
RX, RW = 700, 688
a(
    f'<rect x="{RX}" y="{TOP}" width="{RW}" height="{HH}" rx="14" fill="{C["card"]}" stroke="{C["border"]}"/>'
)
btn(RX + 16, 28, 130, "Generate graph", primary=True)
btn(RX + 154, 28, 100, "Export", caret=True)
badge(RX + RW - 16 - 220, 32, 220, "18 states · 34 actions · up to date")
a(f'<line x1="{RX}" y1="72" x2="{RX + RW}" y2="72" stroke="{C["border"]}"/>')

layers = [1, 4, 5, 4, 2, 1, 1]
gx0, gy0, gdx, gdy = RX + 50, 112, 93, 40
pos = []
for d, n in enumerate(layers):
    off = (max(layers) - n) * gdy / 2
    pos.append([(gx0 + d * gdx, gy0 + off + i * gdy) for i in range(n)])
path = [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (5, 0), (6, 0)]
for d in range(len(layers) - 1):
    for i, p in enumerate(pos[d]):
        for j, q in enumerate(pos[d + 1]):
            near = abs(i / max(1, layers[d] - 1) - j / max(1, layers[d + 1] - 1)) < 0.6
            if near or layers[d] == 1 or layers[d + 1] == 1:
                hot = (d, i) in path and (d + 1, j) in path
                a(
                    f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" stroke="{C["accent"] if hot else C["border"]}" stroke-width="{3 if hot else 1}"/>'
                )
cur = (3, 1)
for d in range(len(layers)):
    for i, (x, y) in enumerate(pos[d]):
        goal = d == len(layers) - 1
        on = (d, i) in path
        c = (d, i) == cur
        fill = (
            C["goal"] if goal else C["accent"] if c else C["soft"] if on else C["card"]
        )
        a(
            f'<circle cx="{x}" cy="{y}" r="{12 if c else 9}" fill="{fill}" stroke="{C["accent"] if on or c else C["muted"]}" stroke-width="1.5"/>'
        )
a(
    f'<text x="{pos[3][1][0]}" y="{pos[3][1][1] - 20}" text-anchor="middle" font-size="11" fill="{C["accent"]}" font-weight="700">you are here</text>'
)
a(
    f'<text x="{pos[0][0][0]}" y="{pos[0][0][1] - 18}" text-anchor="middle" font-size="11" fill="{C["muted"]}">start</text>'
)
a(
    f'<text x="{pos[6][0][0]}" y="{pos[6][0][1] - 18}" text-anchor="middle" font-size="11" fill="{C["goal"]}">goal</text>'
)
a(
    f'<text x="{RX + 16}" y="350" fill="{C["muted"]}" font-size="11.5">click any state to jump there · blue = path taken</text>'
)

# next steps (compact) + record
ny = 372
a(f'<line x1="{RX}" y1="{ny}" x2="{RX + RW}" y2="{ny}" stroke="{C["border"]}"/>')
small_caps(RX + 16, ny + 26, "NEXT STEPS")
btn(RX + RW - 16 - 70 - 8 - 84 - 8 - 80, ny + 8, 80, "← Back", h=26)
btn(RX + RW - 16 - 70 - 8 - 84, ny + 8, 84, "↻ Restart", h=26)
rx0 = RX + RW - 16 - 70
a(
    f'<rect x="{rx0}" y="{ny + 8}" width="70" height="26" rx="8" fill="#fdeceb" stroke="{C["rec"]}"/>'
    f'<circle cx="{rx0 + 16}" cy="{ny + 21}" r="5" fill="{C["rec"]}"/><text x="{rx0 + 28}" y="{ny + 26}" fill="{C["rec"]}" font-weight="700">Rec</text>'
)
steps = [
    ("1", "07_screw_4", "#e377b0", "Screwing · H5", "30.0 s"),
    ("2", "06_screw_3", "#35b7b0", "Screwing · H5", "30.1 s"),
    ("3", "05_screw_2", "#9a6fd0", "Screwing · H5", "31.3 s"),
    ("4", "04_screw_1", "#e8b030", "Screwing · H5", "33.1 s"),
]
for i, (n, name, col, sub, t) in enumerate(steps):
    y = ny + 44 + i * 36
    a(
        f'<rect x="{RX + 16}" y="{y}" width="{RW - 32}" height="30" rx="8" fill="{C["card"]}" stroke="{C["accent"] if i == 0 else C["border"]}"/>'
    )
    a(
        f'<circle cx="{RX + 34}" cy="{y + 15}" r="9" fill="{C["bg"]}" stroke="{C["border"]}"/><text x="{RX + 34}" y="{y + 19}" text-anchor="middle" font-size="11" fill="{C["muted"]}">{n}</text>'
    )
    a(
        f'<rect x="{RX + 52}" y="{y + 10}" width="10" height="10" rx="3" fill="{col}"/><text x="{RX + 70}" y="{y + 19}" font-weight="700" fill="{C["text"]}">{name}</text>'
    )
    a(
        f'<text x="{RX + 220}" y="{y + 19}" fill="{C["muted"]}" font-size="12">{sub}</text>'
    )
    badge(RX + RW - 16 - 64, y + 4, 54, t)

ry = ny + 44 + 4 * 36 + 14
small_caps(RX + 16, ry + 4, "RECORDED  ·  3 steps")
a(
    f'<text x="{RX + RW - 16}" y="{ry + 4}" text-anchor="end" fill="{C["accent"]}" font-weight="600">Clear  ·  Save as sequence</text>'
)
x = RX + 16
for i, (name, col) in enumerate(
    [("08_lower_lid", "#7a7a7a"), ("07_screw_4", "#e377b0"), ("05_screw_2", "#9a6fd0")]
):
    a(
        f'<rect x="{x}" y="{ry + 14}" width="130" height="26" rx="8" fill="{C["soft"]}" stroke="{C["border"]}"/>'
        f'<rect x="{x + 9}" y="{ry + 22}" width="10" height="10" rx="3" fill="{col}"/><text x="{x + 25}" y="{ry + 31}" font-size="12">{i + 1}. {name}</text>'
    )
    x += 138
a("</svg>")
open("mockup.svg", "w", encoding="utf-8").write("\n".join(o))
