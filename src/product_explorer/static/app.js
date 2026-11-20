// Product explorer client (#134). Renders the view the server sends and forwards commands; no logic of its own.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

let V = null;          // the server's view
let MESH_VERSION = null;   // mesh_version of the product the meshes were loaded for
let MESH = null;       // part meshes + framing of the loaded product
let VIEW = null;       // 3D viewport
const UI = { hovered: null, picked: null, xray: false, busy: false, graphView: 'all' };

const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function toast(msg) { const t = el('div', 'toast'); t.textContent = msg; document.body.append(t); setTimeout(() => t.remove(), 3500); }

async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || `${r.status} ${r.statusText}`); throw new Error(data.detail || r.statusText); }
  return data;
}
async function command(path, body) {
  V = await api('POST', path, body); render();
  const names = (V.product || { parts: [] }).parts;   // a rename or an undo of one: the meshes are keyed by part name
  const meshesOld = MESH && V.product && V.product.mesh_version !== MESH_VERSION;   // a mesh update or an undo of one
  if (MESH && names.length && (meshesOld || names.length !== Object.keys(MESH.parts).length || names.some(n => !(n in MESH.parts)))) await loadMeshes();
  return V;
}

// the extraction depths (AT 269): value -> label
const DEPTHS = { disassembled_states: 'down to the disassembled states', full_teardown: 'full teardown' };
const graph = () => V.graph;
const edge = (id) => graph().edges[id];
const node = (id) => graph().nodes[id];
const stale = () => !!(graph() && graph().stale);
const stepTargets = e => e.target_parts || (e.target_part ? [e.target_part] : []);
const stepLabel = e => stepTargets(e).join(' + ');
const stepDisplay = e => { const names = stepTargets(e); return names.slice(0, 3).join(' + ') + (names.length > 3 ? ` + ${names.length - 3} more` : ''); };
const stepSummary = e => e.screw_macro ? `Screw batch · ${stepTargets(e).length} screws · ${e.n_stages} stages` : e.parallel ? 'Parallel removal' : `${(e.capability || '').replace('Capability', '')} · ${e.tool || ''}`;
const partColor = (n) => { const c = (MESH && MESH.parts[n] || {}).color || [.6, .6, .6]; return `rgb(${c.map(v => Math.round(v * 255)).join(',')})`; };

// ============================================================ 3D viewport (three.js)
// Framing comes from the WHOLE product (every part, removed or not), never from the parts present in the state,
// so the model neither jumps nor rescales while parts leave. Distance fits the smaller field of view (#13).
const FOV = 45;
function createViewport(host) {
  const wrap = el('div', 'vp'), msg = el('div', 'msg', 'Load a product folder.'); host.append(wrap); wrap.append(msg);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); wrap.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 1, 1e6); camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); camera.add(sun); sun.position.set(.4, .8, 1); scene.add(camera);
  const controls = new OrbitControls(camera, renderer.domElement);
  const meshes = {}, ray = new THREE.Raycaster(), anim = { on: false };
  let center = new THREE.Vector3(), radius = 1;

  const decode = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
  function setProduct(data) {
    for (const m of Object.values(meshes)) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
    for (const k of Object.keys(meshes)) delete meshes[k];
    for (const [name, g] of Object.entries(data.parts)) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(decode(g.positions_b64, Float32Array), 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(decode(g.normals_b64, Float32Array), 3));
      geo.setIndex(new THREE.BufferAttribute(decode(g.indices_b64, Uint32Array), 1));
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...g.color), roughness: .6, metalness: .1 });
      const m = new THREE.Mesh(geo, mat); m.name = name; m.userData = { c: new THREE.Vector3(...g.bbox_center), d: g.bbox_diag, color: mat.color.clone() };
      meshes[name] = m; scene.add(m);
    }
    center = new THREE.Vector3(...data.scene_center); radius = data.scene_radius;
    camera.near = radius * .01; camera.far = radius * 50; msg.textContent = ''; frame();
  }
  function frame() {
    const half = THREE.MathUtils.degToRad(FOV / 2);
    const dist = radius / Math.sin(Math.atan(Math.tan(half) * Math.min(1, camera.aspect))) * 1.05;
    const az = .7, elv = .55;
    camera.position.set(center.x + dist * Math.cos(elv) * Math.sin(az), center.y - dist * Math.cos(elv) * Math.cos(az), center.z + dist * Math.sin(elv));
    controls.target.copy(center); controls.update();
  }
  const isScrew = (e) => /screw/i.test(e.capability);
  function place(m, e, t) {   // part m moved t (0..1) along the removal of step e
    const d = new THREE.Vector3(...(e.direction || [0, 0, 1])).normalize(), travel = m.userData.d * 1.4;
    m.position.copy(d.clone().multiplyScalar(t * travel));
    if (isScrew(e)) {   // spin about the screw axis through the part's centre
      const q = new THREE.Quaternion().setFromAxisAngle(d, t * Math.PI * 4), c = m.userData.c;
      m.position.add(c.clone().sub(c.clone().applyQuaternion(q))); m.quaternion.copy(q);
    } else m.quaternion.identity();
  }
  function play(e, done) { const m = meshes[e.target_part]; if (!m) return done(); Object.assign(anim, { on: true, e, m, t0: performance.now(), dur: isScrew(e) ? 1400 : 950, done }); }

  let down = null;
  renderer.domElement.addEventListener('pointerdown', (ev) => { down = [ev.clientX, ev.clientY]; });
  renderer.domElement.addEventListener('pointerup', (ev) => {   // a click without drag picks a part
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1), camera);
    const hit = ray.intersectObjects(Object.values(meshes).filter(m => m.visible))[0];
    pick(hit ? hit.object.name : null);
  });

  (function loop() {
    requestAnimationFrame(loop);
    const r = wrap.getBoundingClientRect(), w = Math.max(1, r.width), h = Math.max(1, r.height);
    if (renderer.domElement.width !== Math.round(w * renderer.getPixelRatio())) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
    const present = new Set(V ? V.parts_present : []);
    for (const [name, m] of Object.entries(meshes)) {
      const special = name === UI.picked || (UI.hovered && stepTargets(UI.hovered).includes(name)) || (anim.on && name === anim.m.name);
      m.visible = present.has(name);
      m.material.color.copy(name === UI.picked ? new THREE.Color(1, .65, .1) : UI.hovered && stepTargets(UI.hovered).includes(name) ? new THREE.Color(...MESH.highlight_color) : m.userData.color);
      m.material.transparent = UI.xray && !special; m.material.opacity = m.material.transparent ? .18 : 1; m.material.depthWrite = !m.material.transparent;
      if (!(anim.on && m === anim.m)) {
        if (UI.hovered && stepTargets(UI.hovered).includes(name)) place(m, UI.hovered, (Math.sin(performance.now() / 1000 * 3.2) + 1) * .04);
        else { m.position.set(0, 0, 0); m.quaternion.identity(); }
      }
    }
    if (anim.on) {
      const t = Math.min(1, (performance.now() - anim.t0) / anim.dur); place(anim.m, anim.e, t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
      if (t >= 1) { anim.on = false; anim.done(); }
    }
    controls.update(); renderer.render(scene, camera);
  })();
  return { setProduct, frame, play, busy: () => anim.on, message: (text) => { msg.textContent = text; } };
}

// ============================================================ menus (one open at a time)
let openMenu = null;
function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
function showMenu(anchor, host, content) {
  const same = openMenu && openMenu._for === anchor; closeMenu(); if (same) return;
  const m = el('div', 'menu'); m._for = anchor; m.append(content);
  m.style.left = anchor.offsetLeft + 'px'; m.style.top = (anchor.offsetTop + anchor.offsetHeight + 6) + 'px';
  host.append(m); openMenu = m; m.addEventListener('click', e => e.stopPropagation());
}
document.addEventListener('click', closeMenu);

// ============================================================ commands
async function load(folder) {
  await command('/api/load', { folder }); UI.picked = null;
  await loadMeshes();
}
async function loadMeshes() {
  VIEW.message('Loading part meshes…');
  try { MESH_VERSION = V.product && V.product.mesh_version; VIEW.setProduct(MESH = await api('GET', '/api/meshes')); } catch (e) { VIEW.message(`Part meshes failed: ${e.message}. Load again.`); }
  render();
}
function watchJob() {   // progress of the running job until it is done
  const ws = new WebSocket(`ws://${location.host}/ws/job`);
  ws.onmessage = (m) => { V.job = JSON.parse(m.data); V.job.kind !== 'graph' ? render() : renderStatus(); };
  ws.onclose = async () => {
    V = await api('GET', '/api/state');
    if (V.job.error) toast(V.job.error);
    else if (V.job.cancelled) toast('Import cancelled; the open product is unchanged');
    else if (V.job.kind === 'import' || V.job.kind === 'update') { UI.picked = null; await loadMeshes(); }
    render();
  };
}
async function generate() {
  if (!V.product) return;
  await command('/api/generate', {
    extraction_depth: document.getElementById('depth').value,
    parallel: document.getElementById('parallel').checked,
    n_robots: Number(document.getElementById('robots').value),
    collapse_parallel_screws: document.getElementById('collapse-parallel').checked,
    min_separation: Number(document.getElementById('min-separation').value)
  });
  watchJob();
}
async function updateMeshes(body) {
  await command('/api/update', body);
  watchJob();
}
async function importFolder(body) {
  await command('/api/import', body);
  watchJob();
}
function play(id) {
  if (VIEW.busy() || stale()) return;
  UI.hovered = null; UI.picked = null;
  // Grouped steps update the complete assembly state atomically. They are
  // not animated as one part or presented as robot motion trajectories.
  if (edge(id).parallel) return command('/api/step', { edge: id });
  VIEW.play(edge(id), () => command('/api/step', { edge: id }));
}
// Connection authoring (#242): "Add…" and "Re-detect…" in the Connections menu collect two picked parts, then run.
function startPicking(kind, hint) { UI.adding = { kind, parts: [] }; toast(hint); }
async function pickForConnection(part) {
  const a = UI.adding; if (a.parts.includes(part)) return;
  a.parts.push(part); if (a.parts.length < 2) return toast(`${part} picked: now pick the ${a.kind === 'screw' ? 'held part' : 'second part'}`);
  UI.adding = null; const [p1, p2] = a.parts;
  if (a.kind === 'screw') {   // the server saves the screw first, whatever order the two were picked in
    const head = prompt(`Screw head of ${p1} in ${p2} (${V.screw_heads.join(', ')}):`, V.screw_heads[0]); if (!head) return;
    await command('/api/connection', { kind: 'screw', part1: p1, part2: p2, screw_head: head.trim() });
  } else if (a.kind === 'redetect') await command('/api/contacts/redetect', { part1: p1, part2: p2 });
  else await command('/api/connection', { kind: a.kind, part1: p1, part2: p2 });
}
async function pick(part) {
  if (part && UI.adding) return pickForConnection(part);
  UI.picked = part; UI.pickText = null; render();
  if (!part) return;
  const r = await api('GET', `/api/rejection?part=${encodeURIComponent(part)}`);
  if (UI.picked === part) { UI.pickText = r; render(); }
}
const defaultOut = () => (V.default_out_dir ? `${V.default_out_dir}/${V.product.name}` : '');

// ============================================================ left pane: viewer + toolbar
function buildLeft() {
  const left = document.getElementById('left');
  VIEW = createViewport(left);
  const bar = el('div', 'overlay'); left.append(bar);
  const bImport = el('button', 'btn', 'Import…');
  const bLoad = el('button', 'btn', 'Load'), bSave = el('button', 'btn', 'Save'), bStore = el('button', 'btn', 'Store'), bSet = el('button', 'btn', 'Set targets ▾');
  const bUpdate = el('button', 'btn', 'Update meshes…'); bUpdate.id = 'update';
  const bUndo = el('button', 'btn', '↶ Undo'), bRedo = el('button', 'btn', '↷ Redo'), bPart = el('button', 'btn', 'Part edits ▾'), bRename = el('button', 'btn', 'Rename…'), bCell = el('button', 'btn', 'Cell…');
  bSave.id = 'save'; bUndo.id = 'undo'; bRedo.id = 'redo'; bRename.id = 'rename'; bPart.id = 'part'; bCell.id = 'cell';
  const bFrame = el('button', 'btn', 'Frame'), bX = el('button', 'btn', 'X-ray');
  bStore.id = 'store'; bSet.id = 'set';
  bar.append(bLoad, bImport, bUpdate, bSave, bStore, bSet, bUndo, bRedo, bRename, bPart, bCell, el('span', 'grow'), bFrame, bX);
  bSave.onclick = async () => {
    let folder;   // the first save of an imported product asks for the product folder to write
    if (!V.product.folder) { folder = prompt('Target folder for the new product (new or empty):', ''); if (!folder) return; }
    await command('/api/save', folder ? { folder: folder.trim() } : undefined); toast('Saved the product model');
  };
  bImport.onclick = (e) => {
    e.stopPropagation();
    const box = el('div'), field = (label, value) => { const i = el('input'); i.value = value; const l = el('label', null, `${esc(label)} `); l.append(i); box.append(l, el('br')); return i; };
    box.append(el('div', 'status', 'Meshes must be in millimetres and share one product frame; no unit conversion happens.'));
    const folder = field('Mesh folder (STL, OBJ)', ''), depth = field('Subfolder depth', '0'), name = field('Product name (default: folder name)', ''),
      tol = field('Contact tolerance', '1.0'), angle = field('Contact angle', '-0.99');
    const go = el('button', 'btn pri', 'Import'), foot = el('div', 'foot'); foot.append(go); box.append(foot);
    go.onclick = () => {
      if (!folder.value.trim()) return toast('Enter the mesh folder');
      closeMenu();
      importFolder({ folder: folder.value.trim(), depth: +depth.value, name: name.value.trim() || null, tolerance: +tol.value, angle: +angle.value });
    };
    showMenu(bImport, left, box);
  };
  // Mesh update (#244): check a new CAD export folder first, tick the new files to add, then run the update job.
  bUpdate.onclick = (e) => {
    e.stopPropagation(); if (!V.product) return;
    const box = el('div'), field = (label, value) => { const i = el('input'); i.value = value; const l = el('label', null, `${esc(label)} `); l.append(i); box.append(l, el('br')); return i; };
    box.append(el('div', 'status', 'Parts are matched to the new folder by mesh file name. The folder itself is never changed.'));
    const folder = field('New mesh folder (STL, OBJ)', ''), depth = field('Subfolder depth', '0');
    const plan = el('div'), check = el('button', 'btn', 'Check'), foot = el('div', 'foot'); foot.append(check); box.append(plan, foot);
    check.onclick = async () => {
      if (!folder.value.trim()) return toast('Enter the new mesh folder');
      const p = await api('POST', '/api/update/check', { folder: folder.value.trim(), depth: +depth.value });
      plan.innerHTML = ''; const list = (title, names) => { if (names.length) plan.append(el('div', 'status', `${title}: ${esc(names.join(', '))}`)); };
      list('Changed (new mesh, contacts detected again)', p.changed); list('Removed (no mesh file)', p.removed); list('Merged, skipped (re-merge by hand)', p.merged);
      plan.append(el('div', 'status', `${p.unchanged.length} parts unchanged`));
      const boxes = p.new_files.map(f => { const c = el('input'); c.type = 'checkbox'; c.disabled = p.clashing.includes(f);
        const l = el('label', null, ` add ${esc(f)}${c.disabled ? ' (a part has this name)' : ''}`); l.prepend(c); plan.append(l, el('br')); return [f, c]; });
      const go = el('button', 'btn pri', 'Update'); foot.innerHTML = ''; foot.append(go);
      go.onclick = () => { closeMenu(); updateMeshes({ folder: folder.value.trim(), depth: +depth.value, add: boxes.filter(([, c]) => c.checked).map(([f]) => f) }); };
    };
    showMenu(bUpdate, left, box);
  };
  bUndo.onclick = () => command('/api/undo');
  bRedo.onclick = () => command('/api/redo');
  bRename.onclick = async () => {
    if (!UI.picked) return toast('Click a part first, then rename it');
    const name = prompt(`New name for ${UI.picked}:`, UI.picked); if (!name || name.trim() === UI.picked) return;
    await command('/api/rename', { part: UI.picked, name: name.trim() }); UI.picked = name.trim(); UI.pickText = null; pick(UI.picked);
  };
  bPart.onclick = (e) => {
    e.stopPropagation(); if (!V.product) return; const box = el('div');
    const item = (label, fn) => { const b = el('button', 'item', label); b.onclick = () => { closeMenu(); fn(); }; box.append(b); };
    const picked = () => { if (!UI.picked) toast('Click a part first'); return UI.picked; };
    item('Rename by rules file…', async () => {
      const f = prompt('Rules file (one rule per line: old new):', ''); if (!f) return;
      await command('/api/rename-by-rules', { rules_file: f.trim() }); UI.picked = null; UI.pickText = null;
    });
    item('Material and density…', async () => {
      const part = picked(); if (!part) return; const now = V.product.properties[part];
      const material = prompt(`Material of ${part}:`, now.material); if (!material) return;
      const density = Number(prompt(`Density of ${part} (kg/m3):`, now.density)); if (!(density > 0)) return;
      await command('/api/part/properties', { part, material: material.trim(), density });
    });
    item('Delete part', async () => {
      const part = picked(); if (!part || !confirm(`Delete ${part} with its connections?`)) return;
      await command('/api/part/delete', { part }); UI.picked = null; UI.pickText = null;
    });
    item('Merge parts into a subassembly…', () => mergeDialog(bPart, left));
    showMenu(bPart, left, box);
  };
  function mergeDialog(anchor, host) {
    const chosen = new Set(UI.picked ? [UI.picked] : []), box = el('div'), table = el('table');
    const scroll = el('div'); scroll.style.cssText = 'max-height:60vh;overflow:auto'; scroll.append(table);
    for (const name of V.product.parts) {
      const tr = el('tr', null, `<td><span class="sw" style="background:${partColor(name)}"></span> ${esc(name)}</td><td></td>`);
      const cb = el('input'); cb.type = 'checkbox'; cb.checked = chosen.has(name);
      cb.onchange = () => { if (cb.checked) chosen.add(name); else chosen.delete(name); };
      tr.children[1].append(cb); table.append(tr);
    }
    const foot = el('div', 'foot'), nameIn = el('input'), go = el('button', 'btn pri', 'Merge');
    nameIn.placeholder = 'name of the subassembly'; nameIn.onclick = (e) => e.stopPropagation();
    go.onclick = async () => { closeMenu(); await command('/api/merge', { parts: [...chosen], name: nameIn.value.trim() }); UI.picked = null; UI.pickText = null; };
    foot.append(nameIn, go); box.append(scroll, foot); showMenu(anchor, host, box);
  }
  const bConn = el('button', 'btn', 'Connections ▾'); bConn.id = 'connections'; bar.insertBefore(bConn, bRename.nextSibling);
  bConn.onclick = (e) => {
    e.stopPropagation(); if (!V.product) return; const box = el('div'), scroll = el('div');
    scroll.style.cssText = 'max-height:50vh;overflow:auto'; box.append(scroll);
    V.screw_warnings.forEach(w => scroll.append(el('div', 'muted', '⚠ ' + esc(w))));
    V.connections.filter(c => c.type !== 'geometric').forEach(c => {
      const row = el('div', 'row'), label = c.type === 'screw' ? `screw ${c.part1} → ${c.part2} (${c.screw_head})` : `${c.type.replace('_', ' ')} ${c.part1} ↔ ${c.part2}`;
      const ed = el('button', 'btn', 'Edit'), rm = el('button', 'btn', 'Remove');
      ed.onclick = async () => {
        const p1 = prompt(c.type === 'screw' ? 'Screw:' : 'First part:', c.part1), p2 = p1 && prompt(c.type === 'screw' ? 'Held part:' : 'Second part:', c.part2);
        const head = p2 && c.type === 'screw' ? prompt(`Screw head (${V.screw_heads.join(', ')}):`, c.screw_head) : null;
        if (p2 && (c.type !== 'screw' || head)) { closeMenu(); await command('/api/connection/edit', { index: c.index, part1: p1.trim(), part2: p2.trim(), screw_head: head && head.trim() }); }
      };
      rm.onclick = () => { closeMenu(); command('/api/connection/remove', { index: c.index }); };
      row.append(el('span', 'grow', esc(label)), ed, rm); scroll.append(row);
    });
    const item = (label, fn) => { const b = el('button', 'item', label); b.onclick = () => { closeMenu(); fn(); }; box.append(b); };
    item('Add screw connection…', () => startPicking('screw', 'Click the screw, then the held part'));
    item('Add safety ring…', () => startPicking('safety_ring', 'Click the two parts of the safety ring'));
    item('Add fixed connection…', () => startPicking('fixed', 'Click the two parts that are fixed'));
    item('Delete a geometric connection…', () => {
      const geo = V.connections.filter(c => c.type === 'geometric'), typed = prompt(`Geometric connection to delete (${geo.map(c => `${c.index}: ${c.part1} ↔ ${c.part2}`).join(', ')}) — index:`);
      if (typed !== null && typed.trim() !== '') command('/api/connection/remove', { index: +typed });
    });
    item('Re-detect contacts of a part pair…', () => startPicking('redetect', 'Click the two parts whose contacts are detected again'));
    showMenu(bConn, left, box);
  };
  bCell.onclick = async () => {
    const f = prompt('Cell file (JSON; its screwdrivers and grippers drive the available actions):', V.cell.path || ''); if (!f) return;
    try { await command('/api/cell', { path: f.trim() }); } catch { /* api() toasted the error; the cell in use stays */ }
  };
  bFrame.onclick = () => VIEW.frame();
  bX.onclick = () => { UI.xray = !UI.xray; bX.classList.toggle('on', UI.xray); };
  bLoad.onclick = () => { const f = prompt('Product folder (assembly XML + product_setup.json):', V.product ? V.product.folder : ''); if (f) load(f.trim()); };
  bStore.onclick = async () => { await command('/api/store'); toast('Stored product_setup.json'); };
  bSet.onclick = (e) => {
    e.stopPropagation(); if (!V.product) return;
    const draft = {}; V.setup.base_parts.forEach(p => draft[p] = 'base'); V.setup.target_parts.forEach(p => draft[p] = 'target');
    const box = el('div'), table = el('table'); table.innerHTML = '<tr><th>Part</th><th>Base</th><th>Target</th></tr>';
    const scroll = el('div'); scroll.style.cssText = 'max-height:60vh;overflow:auto'; scroll.append(table);
    for (const name of V.product.parts) {
      const tr = el('tr', null, `<td><span class="sw" style="background:${partColor(name)}"></span> ${esc(name)}</td><td></td><td></td>`);
      ['base', 'target'].forEach((k, i) => {
        const cb = el('input'); cb.type = 'checkbox'; cb.checked = draft[name] === k;
        cb.onchange = () => { if (cb.checked) draft[name] = k; else delete draft[name]; tr.classList.toggle('t', draft[name] === 'target');
          tr.querySelectorAll('input').forEach((o, j) => { o.checked = draft[name] === (j ? 'target' : 'base'); }); };
        tr.children[i + 1].append(cb);
      });
      tr.classList.toggle('t', draft[name] === 'target'); table.append(tr);
    }
    const foot = el('div', 'foot', '<span class="grow">changes make the graph stale</span>'), ap = el('button', 'btn pri', 'Apply');
    ap.onclick = () => { closeMenu(); command('/api/setup', { base_parts: Object.keys(draft).filter(p => draft[p] === 'base'), target_parts: Object.keys(draft).filter(p => draft[p] === 'target') }); };
    foot.append(ap); box.append(scroll, foot); showMenu(bSet, left, box);
  };
  const badges = el('div', 'badges'); badges.id = 'badges'; left.append(badges);
  left.append(el('div', 'hint', 'drag to orbit · right-drag to pan · scroll to zoom · click a part to see why it can or cannot come off'));
}

// ============================================================ right pane: graph + next steps
function buildRight() {
  const right = document.getElementById('right');
  const bar = el('div', 'bar'); right.append(bar);
  const bGen = el('button', 'btn pri', 'Generate graph'), bExp = el('button', 'btn', 'Export ▾'), status = el('span', 'pill'); status.id = 'gstatus';
  // the extraction depth, chosen when the graph is generated (AT 269); the export records it
  const depth = el('select', 'btn', Object.entries(DEPTHS).map(([v, label]) => `<option value="${v}">${label}</option>`).join(''));
  depth.id = 'depth'; depth.title = 'Extraction depth: stop where every target part is removed, or remove parts until none comes out';
  bar.append(bGen, depth, bExp, el('span', 'grow'), status);
  bGen.onclick = generate;
  bExp.onclick = (e) => {
    e.stopPropagation(); if (!V.product) return; const box = el('div');
    const item = (label, fn) => { const b = el('button', 'item', label); b.onclick = () => { closeMenu(); fn(); }; box.append(b); };
    item('Analysis graphs and plots…', async () => {
      const out = prompt('Output folder (full graph, shortest path and rejected attempts):', defaultOut()); if (!out) return;
      const r = await api('POST', '/api/export-analysis', { out_dir: out.trim() }); toast(`Exported to ${r.out_dir}`);
    });
    if (!graph()?.parallel) item('Sequence graph + URDFs…', async () => {
      const out = prompt('Output folder (sequence_graph.json, urdf/, mesh/, manifest.json):', defaultOut()); if (!out) return;
      const r = await api('POST', '/api/export', { out_dir: out.trim() }); toast(`Exported to ${r.out_dir}`);
    });
    if (!graph()?.parallel) item('Recorded sequence…', async () => {
      const out = prompt('Sequence file (a sequence graph export of the recorded path):', `${defaultOut()}/recorded_sequence.json`); if (!out) return;
      const r = await api('POST', '/api/rec/save', { path: out.trim() }); toast(`Saved ${r.path}`);
    });
    item('Product root…', async () => {
      const r = V.product_root, now = r.path ? `${r.path} (${r.source})` : 'none';
      const msg = `Product root in effect (export paths are recorded relative to it): ${now}\n` + (r.error ? `Problem: ${r.error}\n` : '')
        + `Order: app setting, else environment variable ${r.env_var}, else the 3D files repository of the product.\n`
        + 'New app setting (absolute folder; empty clears it):';
      const f = prompt(msg, r.setting || ''); if (f === null) return;
      await command('/api/product-root', { path: f.trim() || null });
    });
    showMenu(bExp, bar, box);
  };
  const options = el('div', 'bar graph-options');
  options.innerHTML = '<label><input id="parallel" type="checkbox"> Parallel</label>'
    + '<label>Robots <input id="robots" type="number" min="1" step="1" value="2"></label>'
    + '<label>Minimum gap (mm) <input id="min-separation" type="number" min="0" step="1" value="50"></label>'
    + '<label><input id="collapse-parallel" type="checkbox" checked> Collapse compatible screws</label>';
  const views = el('select', 'btn', '<option value="all">All transitions</option><option value="shortest">Shortest path</option><option value="rejected">Rejected parallel attempts</option>');
  views.id = 'graph-view'; views.onchange = () => { UI.graphView = views.value; renderGraph(); };
  options.append(views); right.append(options);
  const g = el('div'); g.id = 'graph'; right.append(g);
  const steps = el('div', 'steps'); steps.id = 'steps'; right.append(steps);
  const log = el('div', 'steps'); log.id = 'log'; right.append(log);
}

function renderGraph() {
  const g = document.getElementById('graph'); g.classList.remove('stale');
  if (!graph()) { g.innerHTML = `<div class="stalemsg"><span class="pill">${V.product ? 'Generate graph to explore' : 'No product loaded'}</span></div>`; return; }
  if (UI.graphView === 'rejected') return renderRejectedGraph(g);
  const shortest = graph().shortest || {edges: [], error: 'Generate graph to compute a shortest path'};
  if (UI.graphView === 'shortest' && shortest.error) { g.textContent = shortest.error; return; }
  const visibleEdges = Object.entries(graph().edges).filter(([id]) => UI.graphView !== 'shortest' || shortest.edges.includes(id));
  const visibleNodes = UI.graphView === 'shortest' ? new Set([graph().root, ...visibleEdges.flatMap(([,e]) => [e.source, e.target])]) : new Set(Object.keys(graph().nodes));
  const layers = {};
  for (const id of visibleNodes) (layers[node(id).depth] ||= []).push(id);
  const GX = 80, GY = 38, nl = Object.keys(layers).length, maxRows = Math.max(...Object.values(layers).map(l => l.length));
  const W = 110 + (nl - 1) * GX, H = 50 + (maxRows - 1) * GY, pos = {};
  for (const [d, ids] of Object.entries(layers)) ids.forEach((id, i) => { pos[id] = [55 + d * GX, 28 + (maxRows - ids.length) * GY / 2 + i * GY]; });
  const onPath = new Set([graph().root, ...V.path.map(e => edge(e).target)]), hot = new Set(V.path);
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  for (const [id, e] of visibleEdges) { const a = pos[e.source], b = pos[e.target], h = hot.has(id) || UI.graphView === 'shortest';
    svg += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${h ? 'var(--accent)' : e.parallel ? '#9b3a9e' : 'var(--border)'}" stroke-width="${h ? 3 : e.parallel ? 2 : 1}"><title>${esc(stepLabel(e))} · ${esc(stepSummary(e))} · ${e.action_time.toFixed(2)} s</title></line>`; }
  for (const [id, [x, y]] of Object.entries(pos)) { const n = node(id), cur = id === V.current, ps = onPath.has(id);
    svg += `<g data-id="${id}" style="cursor:pointer"><circle cx="${x}" cy="${y}" r="${cur ? 12 : 9}" fill="${n.is_goal ? 'var(--goal)' : cur ? 'var(--accent)' : ps ? 'var(--accent-soft)' : 'var(--card)'}" stroke="${ps || cur ? 'var(--accent)' : 'var(--muted)'}" stroke-width="1.5"/><title>${n.parts_present.length} parts left · removed: ${esc(n.parts_removed.join(', ') || 'none')}</title></g>`; }
  const c = pos[V.current]; if (c) svg += `<text x="${c[0]}" y="${c[1] - 17}" text-anchor="middle" fill="var(--accent)" font-weight="700">you are here</text>`; svg += '</svg>';
  g.innerHTML = svg;
  if (UI.graphView === 'shortest') {
    const details = el('div', 'analysis-details');
    details.append(el('b', null, `Shortest synchronized-stage time: ${shortest.total_time.toFixed(2)} s`));
    shortest.edges.forEach((id, index) => {
      const e = edge(id), item = el('div', 'analysis-stage');
      item.textContent = `${index+1}. ${stepSummary(e)}: ${stepDisplay(e)} — ${e.action_time.toFixed(2)} s`;
      (e.stages || []).forEach((stage, i) => item.append(el('div', 'muted', `Stage ${i+1}: ${esc(stepLabel(stage))} — ${stage.action_time.toFixed(2)} s`)));
      details.append(item);
    });
    if (!stale()) { const jump = el('button', 'btn', 'Go to shortest-path goal'); jump.onclick = () => command('/api/jump', {node: shortest.edges.length ? edge(shortest.edges.at(-1)).target : graph().root}); details.append(jump); }
    g.append(details);
  }
  g.classList.toggle('stale', stale());
  if (stale()) g.append(el('div', 'stalemsg', '<span class="pill warn">setup or product model changed: Generate graph</span>'));
  else g.querySelectorAll('g[data-id]').forEach(n => { n.onclick = () => { if (!VIEW.busy()) command('/api/jump', { node: n.dataset.id }); }; });
}

function renderRejectedGraph(container) {
  const attempts = graph().rejected_attempts || [];
  container.innerHTML = '';
  container.append(el('p', 'status', `${attempts.length} failed attempts. Red dashed arrows are rejections, not valid transitions.`));
  if (stale()) container.append(el('p', 'pill warn', 'Graph is stale: rebuild to refresh rejections.'));
  attempts.forEach(attempt => {
    const row = el('div', 'rejected-row');
    const state = el('button', 'btn'); state.textContent = `${attempt.parts_present.length} parts remaining`;
    state.title = attempt.parts_present.join(', '); state.disabled = !attempt.source || stale();
    state.onclick = () => command('/api/jump', {node: attempt.source});
    const arrow = el('span', 'rejected-arrow', '⇢');
    const detail = el('div'); detail.textContent = `${attempt.target_parts.join(' + ')} — ${attempt.reason}`;
    detail.title = JSON.stringify(attempt.details || {}, null, 2);
    row.append(state, arrow, detail); container.append(row);
  });
}

function renderSteps() {
  const box = document.getElementById('steps'); box.innerHTML = '';
  const head = el('div', 'row'); head.append(el('span', 'cap grow', 'Next steps'));
  const bBack = el('button', 'btn', '← Back'), bRe = el('button', 'btn', '↻ Restart'), bRec = el('button', 'btn rec' + (V.recording ? ' live' : ''), 'Rec');
  bBack.disabled = !V.path.length || stale(); bRe.disabled = !graph() || stale(); bRec.disabled = !graph() || stale() || graph().parallel;
  bBack.onclick = () => { if (!VIEW.busy()) command('/api/back'); };
  bRe.onclick = () => { if (!VIEW.busy()) command('/api/restart'); };
  bRec.onclick = () => command('/api/rec', { on: !V.recording });
  head.append(bBack, bRe, bRec); box.append(head);
  const list = el('div', 'list');
  V.next_steps.forEach((id, i) => {
    const e = edge(id), r = el('div', 'action');
    r.append(el('span', 'num', i < 9 ? i + 1 : '•'), el('span', 'sw'), el('b', null, esc(stepDisplay(e))), el('span', 'muted grow', esc(stepSummary(e))), el('span', 'pill', e.action_time.toFixed(1) + ' s'));
    r.children[1].style.background = partColor(stepTargets(e)[0]);
    r.title = (e.stages || [e]).map((stage, i) => `Stage ${i+1}: ${stepLabel(stage)} (${stage.action_time.toFixed(2)} s)`).join('\n');
    r.onclick = () => play(id); r.onmouseenter = () => { UI.hovered = e; }; r.onmouseleave = () => { UI.hovered = null; };
    list.append(r);
  });
  if (V.recheck && (!graph() || stale())) {   // after an edit: the available actions of the current state on the edited model
    list.append(el('div', 'status', 'Available actions now, re-checked on the edited product (Rebuild for next steps):'));
    if (V.recheck.error) list.append(el('div', 'muted', esc(V.recheck.error)));
    (V.recheck.available || []).forEach(a => {
      const r = el('div', 'action');
      r.append(el('span', 'sw'), el('b', null, esc(a.target_part)), el('span', 'muted grow', esc(`${a.capability.replace('Capability', '')} · ${a.tool}`)), el('span', 'pill', a.action_time.toFixed(1) + ' s'));
      r.children[0].style.background = partColor(a.target_part); r.onclick = () => pick(a.target_part); list.append(r);
    });
  }
  const groups = graph() ? graph().collapsed_screw_groups : [];
  if (graph() && !stale() && !V.next_steps.length) list.append(el('div', 'muted', node(V.current).is_goal ? '🏁 Goal state reached.' : 'No further steps.'));
  if (groups.length) list.append(el('div', 'status', `Interchangeable screws are explored in one order (${groups.map(g => `${g.length} screws: ${esc(g[0])} …`).join('; ')}).`));
  box.append(list);
  if (UI.picked) {
    const r = UI.pickText, p = el('div', 'picked' + (r ? (r.available ? ' yes' : ' no') : ''));
    p.textContent = r ? r.text : `${UI.picked}: checking…`; box.append(p);
  }
  const rh = el('div', 'row'); rh.append(el('span', 'cap grow', `Recorded · ${V.recorded.length} step${V.recorded.length === 1 ? '' : 's'}`));
  box.append(rh);
  const chips = el('div', 'chips');
  if (!V.recorded.length) chips.append(el('span', 'muted', V.recording ? 'Recording: click next steps.' : 'Press Rec for sequential steps; use Export ▸ Analysis graphs for parallel results.'));
  V.recorded.forEach((id, i) => chips.append(el('span', 'chip', `<span class="sw" style="background:${partColor(stepTargets(edge(id))[0])}"></span>${i + 1}. ${esc(stepLabel(edge(id)))}`)));
  box.append(chips);
}

function renderStatus() {
  const st = document.getElementById('gstatus'), j = V.job, g = graph();
  st.textContent = j.running && j.kind === 'import' ? `detecting contacts… ${j.expanded} of ${j.discovered} part pairs`
    : j.running && j.kind === 'update' ? `updating meshes… ${j.expanded} of ${j.discovered} part pairs`
    : j.running ? `generating… ${j.expanded} states expanded, ${j.discovered} found` : !g ? 'no graph' : stale() ? 'graph is stale'
    : `${Object.keys(g.nodes).length} states · ${Object.keys(g.edges).length} steps · ${g.parallel ? `${g.n_robots} robots · gap > ${g.min_separation} mm · ` : ''}${DEPTHS[g.extraction_depth] || 'depth unknown'} · up to date`;
  st.className = 'pill ' + (j.running || stale() ? 'warn' : g ? 'ok' : '');
}

// The report of the last mesh update (#244) and the status log of edits, warnings and job results.
function renderLog() {
  const box = document.getElementById('log'); box.innerHTML = '';
  const r = V.update_report;
  if (r) {
    box.append(el('h4', null, 'Mesh update report'));
    const row = (title, names) => names.length && box.append(el('div', 'status', `${title}: ${esc(names.join(', '))}`));
    row('Changed', r.changed); row('Removed', r.removed); row('Added', r.added); row('New files not added', r.not_added); row('Re-merge by hand', r.merged);
    box.append(el('div', 'status', `${r.unchanged} parts unchanged · ${r.redetected.length} part pairs detected again`));
  }
  if (V.status_log.length) {
    box.append(el('h4', null, 'Status log'));
    V.status_log.slice(-20).forEach(l => box.append(el('div', 'status', esc(l))));
  }
}

function render() {
  renderStatus();
  const on = (id, yes) => { const b = document.getElementById(id); if (b) b.disabled = !yes; };
  on('update', V.product); on('undo', V.can_undo); on('redo', V.can_redo); on('save', V.product && V.unsaved); on('rename', V.product); on('part', V.product); on('connections', V.product); on('cell', V.product);
  const gen = document.querySelector('#right .bar .pri'); if (gen) gen.textContent = graph() ? 'Rebuild graph' : 'Generate graph';
  const b = document.getElementById('badges'); b.innerHTML = '';
  if (V.job.running && V.job.kind === 'update') b.append(el('span', 'pill warn', `updating meshes: ${V.job.expanded} of ${V.job.discovered} part pairs`));
  if (V.job.running && V.job.kind === 'import') {
    const cancel = el('button', 'btn', 'Cancel import'); cancel.onclick = () => command('/api/cancel');
    b.append(el('span', 'pill warn', `importing: ${V.job.expanded} of ${V.job.discovered} part pairs`), cancel);
  }
  if (V.product) {
    V.warnings.forEach(w => b.append(el('span', 'pill warn', esc(w))));
    V.notices.forEach(n => b.append(el('span', 'pill', esc(n))));
    b.append(el('span', 'pill', esc(V.product.name)), el('span', 'pill', `${V.parts_present.length} parts left`));
    b.append(el('span', 'pill', V.cell.default ? 'default cell' : `cell: ${esc(V.cell.name)}`));
    if (!V.setup_stored) b.append(el('span', 'pill warn', 'setup not stored'));
    if (V.screw_warnings.length) b.append(el('span', 'pill warn', `${V.screw_warnings.length} screw connection warning${V.screw_warnings.length === 1 ? '' : 's'}`));
    if (V.unsaved) b.append(el('span', 'pill warn', 'unsaved edits'));
    if (graph() && !stale() && node(V.current).is_goal) b.append(el('span', 'pill ok', '🏁 goal'));
  }
  renderGraph(); renderSteps(); renderLog();
}

document.addEventListener('keydown', (e) => {
  if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || !V) return;
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); if (V.can_undo) command('/api/undo'); }
  else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Z')) { e.preventDefault(); if (V.can_redo) command('/api/redo'); }
  else if (e.key === 'Backspace' && V.path.length && !VIEW.busy()) command('/api/back');
  else if (e.key >= '1' && e.key <= '9') { const id = V.next_steps[+e.key - 1]; if (id) play(id); }
});

buildLeft(); buildRight();
V = await api('GET', '/api/state'); render();
if (V.product) await loadMeshes();
