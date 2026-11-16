# examples/build_disassembly_ui.py
"""
Build a self-contained, interactive HTML page for exploring a disassembly
graph with a *live* WebGL 3D viewer: free orbit/pan/zoom camera, and each
action animates the target part physically leaving the assembly along its
real removal direction (a spin-and-back-out for screws, a straight pull
for gripped parts) rather than a static picture.

Why live 3D instead of baked screenshots (see navigate_disassembly_graph.py
and the earlier PNG-based build_disassembly_ui.py): a fixed camera angle
flattens small parts (screws) into indistinguishable dots, and a static
color-highlight doesn't convey what an action actually *does*. Exporting
the real (decimated) part meshes into the page and rendering them with a
small hand-written WebGL renderer fixes both, and -- because geometry is
stored once per *part* rather than once per rendered state/action image --
the resulting file is smaller than the screenshot-based version despite
being fully interactive.

Usage:
    python examples/build_disassembly_ui.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np
from build_env import build_env

from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent
from disassembly_pomdp.mesh_export import export_part_mesh
from disassembly_pomdp.rendering import _PALETTE, HIGHLIGHT_COLOR


def _find_root(graph):
    for n, d in graph.nodes(data=True):
        if d.get("depth") == 0:
            return n
    return next(iter(graph.nodes()))


def _scene_bounds(assembly) -> tuple[list, float]:
    """Union bounding box (center, radius) of every part -- for initial camera fit."""
    mins, maxs = [], []
    for part in assembly.parts.values():
        bbox = part.mesh.get_axis_aligned_bounding_box()
        mins.append(np.asarray(bbox.get_min_bound()))
        maxs.append(np.asarray(bbox.get_max_bound()))
    lo = np.min(np.stack(mins), axis=0)
    hi = np.max(np.stack(maxs), axis=0)
    center = ((lo + hi) / 2.0).tolist()
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    return center, radius


def build_page_data(graph, base_assembly) -> dict:
    part_names = sorted(p.name for p in base_assembly.parts.values())
    color_by_part = {
        name: list(_PALETTE[i % len(_PALETTE)]) for i, name in enumerate(part_names)
    }

    print(f"Exporting {len(part_names)} part meshes (decimating heavy ones)...")
    parts_geo = {}
    for name in part_names:
        part = base_assembly.get_part_by_name(name)
        geo = export_part_mesh(part)
        geo["color"] = color_by_part[name]
        parts_geo[name] = geo
        print(f"  {name}: {geo['index_count'] // 3} triangles")

    scene_center, scene_radius = _scene_bounds(base_assembly)

    node_ids = {key: f"s{i}" for i, key in enumerate(graph.nodes())}

    nodes = {}
    for key, node_id in node_ids.items():
        d = graph.nodes[key]
        nodes[node_id] = {
            "parts_present": d["parts_present"],
            "parts_removed": d["parts_removed"],
            "n_parts_present": d["n_parts_present"],
            "depth": d.get("depth", 0),
            "is_goal": bool(d.get("is_goal", False)),
            "inspectable_parts": d.get("inspectable_parts", []),
            "non_inspectable_parts": d.get("non_inspectable_parts", []),
        }

    edges_by_node: dict[str, list] = {node_id: [] for node_id in node_ids.values()}
    for u, v, _k, ed in graph.edges(keys=True, data=True):
        edges_by_node[node_ids[u]].append(
            {
                "target": node_ids[v],
                "label": ed["label"],
                "capability": ed["capability"],
                "tool": ed["tool"],
                "target_part": ed["target_part"],
                "action_time": ed["action_time"],
                "direction": ed.get("direction"),
            }
        )
    for node_id in edges_by_node:
        edges_by_node[node_id].sort(key=lambda e: e["action_time"])

    return {
        "root": node_ids[_find_root(graph)],
        "parts": parts_geo,
        "nodes": nodes,
        "edgesByNode": edges_by_node,
        "sceneCenter": scene_center,
        "sceneRadius": scene_radius,
        "highlightColor": list(HIGHLIGHT_COLOR),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Disassembly Explorer</title>
<style>
  :root {
    --bg: #f4f5fa;
    --card-bg: #ffffff;
    --text: #1c1d26;
    --muted: #6b6f80;
    --border: #e6e7ee;
    --accent: #5b6bf0;
    --accent-soft: #eef0ff;
    --goal: #1fae66;
    --goal-soft: #e6f7ee;
    --danger-soft: #fdeceb;
    --shadow: 0 1px 2px rgba(20,21,40,0.04), 0 8px 24px rgba(20,21,40,0.06);
    --shadow-hover: 0 4px 10px rgba(20,21,40,0.08), 0 16px 32px rgba(20,21,40,0.10);
    --radius: 18px;
    --canvas-bg: #eef0f6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14151c;
      --card-bg: #1c1e28;
      --text: #eef0f6;
      --muted: #9aa0b4;
      --border: #2b2d3a;
      --accent: #7b8aff;
      --accent-soft: #262a4a;
      --goal: #35d488;
      --goal-soft: #163629;
      --danger-soft: #3a2323;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
      --shadow-hover: 0 4px 10px rgba(0,0,0,0.35), 0 16px 32px rgba(0,0,0,0.45);
      --canvas-bg: #10111a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 28px; position: sticky; top: 0; z-index: 10;
    background: color-mix(in srgb, var(--bg) 85%, transparent);
    backdrop-filter: blur(10px); border-bottom: 1px solid var(--border);
  }
  .topbar-title { font-size: 17px; font-weight: 650; letter-spacing: -0.01em; }
  .topbar-actions { display: flex; gap: 8px; }
  .btn {
    border: 1px solid var(--border); background: var(--card-bg); color: var(--text);
    padding: 8px 14px; border-radius: 10px; font-size: 13.5px; font-weight: 550;
    cursor: pointer; transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
  }
  .btn:hover:not(:disabled) { box-shadow: var(--shadow); transform: translateY(-1px); }
  .btn:disabled { opacity: 0.35; cursor: default; }
  .btn-small { padding: 4px 10px; font-size: 11.5px; border-radius: 8px; }
  .btn-small.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  .breadcrumb { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; padding: 12px 28px; font-size: 13px; color: var(--muted); }
  .crumb { border: none; background: transparent; color: var(--muted); cursor: pointer; font-size: 13px; padding: 4px 8px; border-radius: 8px; font-family: inherit; }
  .crumb:hover:not(:disabled) { background: var(--accent-soft); color: var(--accent); }
  .crumb-current { color: var(--text); font-weight: 650; cursor: default; }
  .crumb-sep { color: var(--border); }

  .main { max-width: 1080px; margin: 0 auto; padding: 8px 28px 60px; display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; }
  @media (max-width: 860px) { .main { grid-template-columns: 1fr; } }

  .viewer-card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); overflow: hidden; display: flex; flex-direction: column;
  }
  .canvas-wrap { position: relative; aspect-ratio: 4 / 3; background: var(--canvas-bg); }
  #viewer-canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  #viewer-canvas:active { cursor: grabbing; }
  .canvas-hint {
    position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
    font-size: 11px; color: var(--muted); background: color-mix(in srgb, var(--card-bg) 75%, transparent);
    padding: 3px 10px; border-radius: 999px; pointer-events: none; white-space: nowrap;
  }
  .webgl-fallback {
    display: none; position: absolute; inset: 0; align-items: center; justify-content: center;
    text-align: center; padding: 20px; color: var(--muted); font-size: 13px;
  }
  .viewer-info { padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
  .badge-row, .action-badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: var(--accent-soft); color: var(--accent); }
  .badge-goal { background: var(--goal-soft); color: var(--goal); }
  .badge-neutral { background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .badge-tool { background: var(--accent-soft); color: var(--accent); }
  .badge-time { background: var(--bg); color: var(--muted); border: 1px solid var(--border); }

  .removed-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 650; }
  .info-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .chip-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
  .chip { font-size: 12.5px; padding: 4px 10px; border-radius: 8px; background: var(--danger-soft); color: var(--text); border: 1px solid var(--border); }
  .chip-inspectable { background: var(--accent-soft); color: var(--accent); border: 1px solid transparent; }
  .muted { color: var(--muted); font-size: 13px; }

  .actions-panel { display: flex; flex-direction: column; gap: 12px; }
  .section-title { font-size: 15px; font-weight: 650; margin: 4px 0 0; }

  .action-row {
    display: flex; align-items: center; gap: 12px; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: 14px; padding: 10px 14px;
    cursor: pointer; box-shadow: var(--shadow); transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  }
  .action-row:hover, .action-row.hovered { transform: translateX(3px); box-shadow: var(--shadow-hover); border-color: var(--accent); }
  .action-row.disabled { opacity: 0.45; pointer-events: none; }
  .action-num {
    flex: 0 0 auto; width: 24px; height: 24px; border-radius: 999px; background: var(--bg);
    border: 1px solid var(--border); font-size: 12px; font-weight: 700; color: var(--muted);
    display: flex; align-items: center; justify-content: center;
  }
  .action-swatch { flex: 0 0 auto; width: 14px; height: 14px; border-radius: 4px; }
  .action-main { flex: 1; min-width: 0; }
  .action-title { font-size: 13.5px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .action-sub { font-size: 11.5px; color: var(--muted); }
  .action-badges { flex: 0 0 auto; }

  .empty-state {
    display: none; flex-direction: column; align-items: center; justify-content: center;
    gap: 10px; padding: 50px 20px; color: var(--muted); background: var(--card-bg);
    border: 1px dashed var(--border); border-radius: var(--radius);
  }
  .empty-emoji { font-size: 34px; }

  .footer { text-align: center; padding: 4px 18px 24px; color: var(--muted); font-size: 12.5px; grid-column: 1 / -1; }
  kbd { background: var(--card-bg); border: 1px solid var(--border); border-bottom-width: 2px; border-radius: 5px; padding: 1px 6px; font-family: inherit; font-size: 11.5px; }
</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-title">Disassembly Explorer</div>
    <div class="topbar-actions">
      <button id="back-btn" class="btn">&larr; Back</button>
      <button id="restart-btn" class="btn">&#8635; Restart</button>
    </div>
  </header>

  <nav id="breadcrumb" class="breadcrumb"></nav>

  <main class="main">
    <section class="viewer-card">
      <div class="canvas-wrap">
        <canvas id="viewer-canvas"></canvas>
        <div class="canvas-hint">drag to orbit &middot; scroll to zoom</div>
        <div id="webgl-fallback" class="webgl-fallback">WebGL isn't available in this browser, so the live 3D view can't be shown.</div>
      </div>
      <div class="viewer-info">
        <div id="hero-badges" class="badge-row"></div>
        <div>
          <div class="info-row">
            <div class="removed-label">Inspectable now</div>
            <button id="xray-btn" class="btn btn-small">&#128065; X-ray view</button>
          </div>
          <div id="inspectable-chips" class="chip-row"></div>
        </div>
        <div>
          <div class="removed-label">Removed so far</div>
          <div id="removed-chips" class="chip-row"></div>
        </div>
      </div>
    </section>

    <section class="actions-panel">
      <h2 class="section-title">Available actions</h2>
      <div id="actions-list"></div>
      <div id="empty-state" class="empty-state">
        <div class="empty-emoji">&#9989;</div>
        <div>No further actions from this state.</div>
      </div>
    </section>

    <div class="footer">
      Click an action (or press <kbd>1</kbd>&ndash;<kbd>9</kbd>) to watch it happen &middot;
      <kbd>Backspace</kbd> to undo &middot; drag/scroll the 3D view any time
    </div>
  </main>

<script>
const DATA = __DATA_JSON__;

// ============================================================
// mat4 / mat3 helpers (column-major, matches WebGL convention)
// ============================================================
const mat4 = {
  identity() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); },
  multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
      }
    }
    return o;
  },
  translation(x, y, z) {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]);
  },
  rotationAxisAngle(axis, angle) {
    let [x,y,z] = axis;
    const len = Math.hypot(x,y,z) || 1;
    x/=len; y/=len; z/=len;
    const c = Math.cos(angle), s = Math.sin(angle), t = 1-c;
    return new Float32Array([
      t*x*x+c,   t*x*y+s*z, t*x*z-s*y, 0,
      t*x*y-s*z, t*y*y+c,   t*y*z+s*x, 0,
      t*x*z+s*y, t*y*z-s*x, t*z*z+c,   0,
      0,0,0,1
    ]);
  },
  perspective(fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy/2);
    const nf = 1/(near-far);
    return new Float32Array([
      f/aspect,0,0,0,
      0,f,0,0,
      0,0,(far+near)*nf,-1,
      0,0,2*far*near*nf,0
    ]);
  },
  lookAt(eye, target, up) {
    let zx = eye[0]-target[0], zy = eye[1]-target[1], zz = eye[2]-target[2];
    let zl = Math.hypot(zx,zy,zz) || 1; zx/=zl; zy/=zl; zz/=zl;
    let xx = up[1]*zz - up[2]*zy, xy = up[2]*zx - up[0]*zz, xz = up[0]*zy - up[1]*zx;
    let xl = Math.hypot(xx,xy,xz) || 1; xx/=xl; xy/=xl; xz/=xl;
    const yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
      -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
      -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
      1
    ]);
  },
  toMat3(m) {
    return new Float32Array([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]]);
  },
};

function easeInOutCubic(t) { return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2, 3)/2; }

// ============================================================
// WebGL setup
// ============================================================
const canvas = document.getElementById('viewer-canvas');
let gl = canvas.getContext('webgl2');
let isWebGL2 = !!gl;
if (!gl) gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
if (!gl) {
  document.getElementById('webgl-fallback').style.display = 'flex';
}
let indexExt = null;
if (gl && !isWebGL2) {
  indexExt = gl.getExtension('OES_element_index_uint');
}

const VS_SRC = `
  attribute vec3 aPosition;
  attribute vec3 aNormal;
  uniform mat4 uModel;
  uniform mat4 uViewProj;
  uniform mat3 uNormalMatrix;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uViewProj * worldPos;
  }
`;
const FS_SRC = `
  precision mediump float;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  uniform vec3 uColor;
  uniform vec3 uLightDir;
  uniform vec3 uCameraPos;
  uniform float uAlpha;
  void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    float diff = max(dot(N, L), 0.0);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    vec3 halfDir = normalize(L + viewDir);
    float spec = pow(max(dot(N, halfDir), 0.0), 28.0) * 0.35 * uAlpha;
    vec3 ambient = uColor * 0.5;
    vec3 diffuse = uColor * diff * 0.65;
    gl_FragColor = vec4(ambient + diffuse + vec3(spec), uAlpha);
  }
`;

let program = null, uModelLoc, uViewProjLoc, uNormalMatLoc, uColorLoc, uLightDirLoc, uCameraPosLoc, uAlphaLoc, aPositionLoc, aNormalLoc;
const partBuffers = {};

if (gl) {
  function compileShader(src, type) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(sh));
    }
    return sh;
  }
  const vs = compileShader(VS_SRC, gl.VERTEX_SHADER);
  const fs = compileShader(FS_SRC, gl.FRAGMENT_SHADER);
  program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
  }
  uModelLoc = gl.getUniformLocation(program, 'uModel');
  uViewProjLoc = gl.getUniformLocation(program, 'uViewProj');
  uNormalMatLoc = gl.getUniformLocation(program, 'uNormalMatrix');
  uColorLoc = gl.getUniformLocation(program, 'uColor');
  uLightDirLoc = gl.getUniformLocation(program, 'uLightDir');
  uCameraPosLoc = gl.getUniformLocation(program, 'uCameraPos');
  uAlphaLoc = gl.getUniformLocation(program, 'uAlpha');
  aPositionLoc = gl.getAttribLocation(program, 'aPosition');
  aNormalLoc = gl.getAttribLocation(program, 'aNormal');

  function b64ToFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function b64ToUint32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Uint32Array(bytes.buffer);
  }

  for (const [name, geo] of Object.entries(DATA.parts)) {
    const positions = b64ToFloat32(geo.positions_b64);
    const normals = b64ToFloat32(geo.normals_b64);
    const indices = b64ToUint32(geo.indices_b64);

    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const normBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);

    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    partBuffers[name] = { posBuf, normBuf, idxBuf, indexCount: geo.index_count, color: geo.color, bboxCenter: geo.bbox_center, bboxDiag: geo.bbox_diag };
  }
}

// ============================================================
// Orbit camera (Z is the model's "pole" axis -- these parts are flat
// discs with Z as the thin/through axis, so orbiting around Z with a
// moderate elevation gives a 3/4 view instead of a flat top-down one).
// ============================================================
let camAzimuth = 0.7, camElevation = 0.55;
let camRadius = DATA.sceneRadius * 2.4;
const camTarget = DATA.sceneCenter;
const MIN_ELEV = -1.45, MAX_ELEV = 1.45;

function eyePosition() {
  const ce = Math.cos(camElevation), se = Math.sin(camElevation);
  return [
    camTarget[0] + camRadius * ce * Math.sin(camAzimuth),
    camTarget[1] + camRadius * ce * Math.cos(camAzimuth),
    camTarget[2] + camRadius * se,
  ];
}

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointercancel', () => { dragging = false; });
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  camAzimuth -= dx * 0.008;
  camElevation = Math.max(MIN_ELEV, Math.min(MAX_ELEV, camElevation + dy * 0.008));
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const scale = Math.exp(e.deltaY * 0.0015);
  camRadius = Math.max(DATA.sceneRadius * 0.25, Math.min(DATA.sceneRadius * 8, camRadius * scale));
}, { passive: false });

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
}
window.addEventListener('resize', resizeCanvas);

// ============================================================
// Navigation state
// ============================================================
let historyStack = [];
let current = DATA.root;
let hoveredEdge = null;
let hoverStartTime = 0;
let xrayMode = false;

const anim = { active: false, part: null, kind: null, dir: [0,0,1], travel: 0, center: [0,0,0], startTime: 0, duration: 1000, onComplete: null };

function playRemoval(edge) {
  if (anim.active) return;
  const part = partBuffers[edge.target_part];
  const dir = edge.direction || [0, 0, 1];
  const isScrew = edge.capability.toLowerCase().includes('screw');
  anim.active = true;
  anim.part = edge.target_part;
  anim.kind = isScrew ? 'helical' : 'linear';
  anim.dir = dir;
  anim.center = part ? part.bboxCenter : [0,0,0];
  anim.travel = (part ? part.bboxDiag : 50) * 1.4;
  anim.startTime = performance.now();
  anim.duration = isScrew ? 1400 : 950;
  anim.onComplete = () => { goTo(edge.target); };
  renderActionsList(); // reflect "disassembling..." disabled state
}

function currentAnimModel() {
  const t = Math.min(1, (performance.now() - anim.startTime) / anim.duration);
  const e = easeInOutCubic(t);
  const d = anim.dir, c = anim.center;
  const travelMat = mat4.translation(d[0]*e*anim.travel, d[1]*e*anim.travel, d[2]*e*anim.travel);
  let model;
  if (anim.kind === 'helical') {
    const angle = e * Math.PI * 4;
    const toOrigin = mat4.translation(-c[0], -c[1], -c[2]);
    const rot = mat4.rotationAxisAngle(d, angle);
    const back = mat4.translation(c[0], c[1], c[2]);
    model = mat4.multiply(travelMat, mat4.multiply(back, mat4.multiply(rot, toOrigin)));
  } else {
    model = travelMat;
  }
  if (t >= 1 && anim.active) {
    const cb = anim.onComplete;
    anim.active = false; anim.part = null; anim.onComplete = null;
    if (cb) cb();
  }
  return model;
}

// Small continuous back-and-forth nudge along the action's real removal
// direction (plus a slight spin for screws) while its action row is
// hovered -- makes it obvious which part an action refers to without
// committing to the full removal.
function currentHoverModel(part) {
  const d = hoveredEdge.direction || [0, 0, 1];
  const t = (performance.now() - hoverStartTime) / 1000;
  const wiggle = Math.sin(t * 3.2);
  const amp = (part.bboxDiag || 50) * 0.10;
  const isScrew = hoveredEdge.capability.toLowerCase().includes('screw');
  const travel = mat4.translation(d[0]*wiggle*amp, d[1]*wiggle*amp, d[2]*wiggle*amp);
  if (!isScrew) return travel;
  const c = part.bboxCenter;
  const angle = wiggle * 0.35;
  const toOrigin = mat4.translation(-c[0], -c[1], -c[2]);
  const rot = mat4.rotationAxisAngle(d, angle);
  const back = mat4.translation(c[0], c[1], c[2]);
  return mat4.multiply(travel, mat4.multiply(back, mat4.multiply(rot, toOrigin)));
}

// ============================================================
// Render loop
// ============================================================
function frame() {
  requestAnimationFrame(frame);
  if (!gl || !program) return;
  resizeCanvas();

  const node = DATA.nodes[current];
  const eye = eyePosition();
  const aspect = canvas.width / canvas.height;
  const view = mat4.lookAt(eye, camTarget, [0, 0, 1]);
  const proj = mat4.perspective(Math.PI / 4, aspect, Math.max(DATA.sceneRadius * 0.01, 0.01), DATA.sceneRadius * 30);
  const viewProj = mat4.multiply(proj, view);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  gl.uniformMatrix4fv(uViewProjLoc, false, viewProj);
  gl.uniform3fv(uLightDirLoc, [0.4, -0.5, 0.9]);
  gl.uniform3fv(uCameraPosLoc, eye);

  const animModel = anim.active ? currentAnimModel() : null;
  const inspectableSet = xrayMode ? new Set(node.inspectable_parts) : null;

  function isSpecial(name) {
    return (anim.active && name === anim.part) || (!anim.active && hoveredEdge && name === hoveredEdge.target_part);
  }
  function alphaFor(name) {
    if (!xrayMode || isSpecial(name)) return 1.0;
    return inspectableSet.has(name) ? 0.15 : 1.0;
  }

  function drawPart(name, alpha) {
    const part = partBuffers[name];
    if (!part) return;
    let model = mat4.identity();
    let color = part.color;
    if (anim.active && name === anim.part) {
      model = animModel;
    } else if (!anim.active && hoveredEdge && name === hoveredEdge.target_part) {
      model = currentHoverModel(part);
      color = DATA.highlightColor;
    }

    gl.uniformMatrix4fv(uModelLoc, false, model);
    gl.uniformMatrix3fv(uNormalMatLoc, false, mat4.toMat3(model));
    gl.uniform3fv(uColorLoc, color);
    gl.uniform1f(uAlphaLoc, alpha);

    gl.bindBuffer(gl.ARRAY_BUFFER, part.posBuf);
    gl.enableVertexAttribArray(aPositionLoc);
    gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, part.normBuf);
    gl.enableVertexAttribArray(aNormalLoc);
    gl.vertexAttribPointer(aNormalLoc, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, part.idxBuf);
    gl.drawElements(gl.TRIANGLES, part.indexCount, gl.UNSIGNED_INT, 0);
  }

  // Two-pass draw: opaque parts first (depth write on), then transparent
  // ("x-rayed") parts on top (depth write off, alpha blended) so hidden
  // parts show through without corrupting the opaque depth buffer.
  gl.depthMask(true);
  gl.disable(gl.BLEND);
  for (const name of node.parts_present) {
    if (alphaFor(name) >= 1.0) drawPart(name, 1.0);
  }

  gl.depthMask(false);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  for (const name of node.parts_present) {
    const a = alphaFor(name);
    if (a < 1.0) drawPart(name, a);
  }
  gl.depthMask(true);
  gl.disable(gl.BLEND);
}
requestAnimationFrame(frame);

// ============================================================
// UI (breadcrumb / badges / chips / action list)
// ============================================================
const els = {
  breadcrumb: document.getElementById('breadcrumb'),
  heroBadges: document.getElementById('hero-badges'),
  inspectableChips: document.getElementById('inspectable-chips'),
  removedChips: document.getElementById('removed-chips'),
  actionsList: document.getElementById('actions-list'),
  emptyState: document.getElementById('empty-state'),
  backBtn: document.getElementById('back-btn'),
  restartBtn: document.getElementById('restart-btn'),
  xrayBtn: document.getElementById('xray-btn'),
};

function currentPath() { return [...historyStack, current]; }

function goTo(id) {
  if (id === current || anim.active) return;
  historyStack.push(current);
  current = id;
  hoveredEdge = null;
  renderUI();
}
function goBack() {
  if (anim.active || !historyStack.length) return;
  current = historyStack.pop();
  hoveredEdge = null;
  renderUI();
}
function restart() {
  if (anim.active) return;
  if (current !== DATA.root || historyStack.length) {
    historyStack = [];
    current = DATA.root;
    hoveredEdge = null;
    renderUI();
  }
}
function goToBreadcrumb(idx) {
  if (anim.active) return;
  const p = currentPath();
  if (idx >= p.length - 1) return;
  historyStack = p.slice(0, idx);
  current = p[idx];
  hoveredEdge = null;
  renderUI();
}
function edgeLeadingTo(fromId, toId) {
  return (DATA.edgesByNode[fromId] || []).find(e => e.target === toId);
}
function addBadge(container, text, cls) {
  const b = document.createElement('span');
  b.className = 'badge ' + cls;
  b.textContent = text;
  container.appendChild(b);
}

function renderBreadcrumb() {
  const p = currentPath();
  els.breadcrumb.innerHTML = '';
  p.forEach((id, idx) => {
    if (idx > 0) {
      const sep = document.createElement('span');
      sep.className = 'crumb-sep';
      sep.textContent = '›';
      els.breadcrumb.appendChild(sep);
    }
    const btn = document.createElement('button');
    const isCurrent = idx === p.length - 1;
    btn.className = 'crumb' + (isCurrent ? ' crumb-current' : '');
    if (idx === 0) {
      btn.textContent = 'Start';
    } else {
      const e = edgeLeadingTo(p[idx - 1], p[idx]);
      btn.textContent = e ? e.target_part : (DATA.nodes[id].n_parts_present + ' parts');
    }
    btn.disabled = isCurrent;
    btn.onclick = () => goToBreadcrumb(idx);
    els.breadcrumb.appendChild(btn);
  });
}

function renderHero() {
  const node = DATA.nodes[current];
  els.heroBadges.innerHTML = '';
  addBadge(els.heroBadges, node.n_parts_present + ' parts remaining', 'badge-neutral');
  addBadge(els.heroBadges, 'depth ' + node.depth, 'badge-neutral');
  if (node.is_goal) addBadge(els.heroBadges, '🏁 Goal state', 'badge-goal');

  els.inspectableChips.innerHTML = '';
  if (node.inspectable_parts.length === 0) {
    const span = document.createElement('span');
    span.className = 'muted';
    span.textContent = 'Nothing currently visible/inspectable.';
    els.inspectableChips.appendChild(span);
  } else {
    node.inspectable_parts.forEach(name => {
      const chip = document.createElement('span');
      chip.className = 'chip chip-inspectable';
      chip.textContent = name;
      els.inspectableChips.appendChild(chip);
    });
  }

  els.removedChips.innerHTML = '';
  if (node.parts_removed.length === 0) {
    const span = document.createElement('span');
    span.className = 'muted';
    span.textContent = 'Nothing removed yet.';
    els.removedChips.appendChild(span);
  } else {
    node.parts_removed.forEach(name => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = name;
      els.removedChips.appendChild(chip);
    });
  }
}

function renderActionsList() {
  const actions = DATA.edgesByNode[current] || [];
  els.actionsList.innerHTML = '';
  els.emptyState.style.display = actions.length === 0 ? 'flex' : 'none';
  els.actionsList.style.display = actions.length === 0 ? 'none' : 'flex';
  els.actionsList.style.flexDirection = 'column';
  els.actionsList.style.gap = '8px';

  actions.forEach((edge, i) => {
    const row = document.createElement('div');
    row.className = 'action-row' + (anim.active ? ' disabled' : '');
    row.onclick = () => playRemoval(edge);
    row.onmouseenter = () => { hoveredEdge = edge; hoverStartTime = performance.now(); };
    row.onmouseleave = () => { hoveredEdge = null; };

    const num = document.createElement('div');
    num.className = 'action-num';
    num.textContent = (i < 9) ? String(i + 1) : '•';
    row.appendChild(num);

    const swatch = document.createElement('div');
    swatch.className = 'action-swatch';
    const c = (DATA.parts[edge.target_part] || {}).color || [0.6,0.6,0.6];
    swatch.style.background = `rgb(${Math.round(c[0]*255)}, ${Math.round(c[1]*255)}, ${Math.round(c[2]*255)})`;
    row.appendChild(swatch);

    const mainDiv = document.createElement('div');
    mainDiv.className = 'action-main';
    const title = document.createElement('div');
    title.className = 'action-title';
    title.textContent = edge.target_part;
    mainDiv.appendChild(title);
    const sub = document.createElement('div');
    sub.className = 'action-sub';
    sub.textContent = edge.capability.replace('Capability', '');
    mainDiv.appendChild(sub);
    row.appendChild(mainDiv);

    const badges = document.createElement('div');
    badges.className = 'action-badges';
    addBadge(badges, edge.tool, 'badge-tool');
    addBadge(badges, edge.action_time.toFixed(1) + 's', 'badge-time');
    row.appendChild(badges);

    els.actionsList.appendChild(row);
  });

  els.backBtn.disabled = historyStack.length === 0 || anim.active;
  els.restartBtn.disabled = anim.active;
}

function renderUI() {
  renderBreadcrumb();
  renderHero();
  renderActionsList();
}

els.backBtn.addEventListener('click', goBack);
els.restartBtn.addEventListener('click', restart);
els.xrayBtn.addEventListener('click', () => {
  xrayMode = !xrayMode;
  els.xrayBtn.classList.toggle('active', xrayMode);
  els.xrayBtn.innerHTML = xrayMode ? '&#128065; Exit x-ray' : '&#128065; X-ray view';
});

document.addEventListener('keydown', (e) => {
  if (anim.active) return;
  if (e.key === 'Backspace' || e.key === 'b') { goBack(); return; }
  if (e.key >= '1' && e.key <= '9') {
    const idx = parseInt(e.key, 10) - 1;
    const actions = DATA.edgesByNode[current] || [];
    if (idx < actions.length) playRemoval(actions[idx]);
  }
});

renderUI();
</script>
</body>
</html>
"""


def main() -> None:
    out_dir = ROOT / "examples" / "runs" / "disassembly_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    agent = DisassemblyGraphAgent(env)
    graph = agent.build_graph(verbose=True)
    agent.compute_inspectable_parts(verbose=True)

    data = build_page_data(graph, agent.base_assembly)

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    out_path = out_dir / "ui.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    n_actions = sum(len(v) for v in data["edgesByNode"].values())
    print(
        f"Wrote {out_path} ({size_mb:.2f} MB, {len(data['nodes'])} states, {n_actions} actions)"
    )
    print(f"Open it with: open {out_path}")


if __name__ == "__main__":
    main()
