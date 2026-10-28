"""``csc maps edit`` 的前端页面（单文件，无外部依赖）。

Python 侧只注入两个占位符：``__PAYLOAD__``（JSON）和 ``__MAP__``（图名）。
像素↔世界的公式与 :class:`csc.radar.MapTransform` 完全一致，改一处要改两处——
所以参数只从 Python 注入，JS 里不出现任何硬编码的 pos_x/scale。
"""

TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>地图打点 · __MAP__</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e6e9ef; --dim:#8b93a7;
    --entry:#3ddc84; --cover:#4da3ff; --region:#ffb545; --danger:#ff5d5d; --ok:#3ddc84;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--text);font:13px/1.5 "Segoe UI",system-ui,-apple-system,"Microsoft YaHei",sans-serif;overflow:hidden}
  #app{display:flex;height:100%}
  #side{width:352px;min-width:352px;display:flex;flex-direction:column;background:var(--panel);border-right:1px solid var(--line)}
  #side header{padding:14px 14px 10px;border-bottom:1px solid var(--line)}
  #side h1{font-size:15px;margin:0 0 4px;font-weight:600}
  .sub{color:var(--dim);font-size:12px;word-break:break-all}
  #subInfo{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;word-break:normal}
  .toolbar{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
  .kinds{display:flex;gap:6px}
  .kinds button{flex:1;padding:7px 4px;border-radius:7px;border:1px solid var(--line);background:#1e222b;color:var(--text);cursor:pointer;font-size:12px}
  .kinds button.on{border-color:var(--entry);background:#12331f;color:#c9ffe2}
  .kinds button[data-kind=cover].on{border-color:var(--cover);background:#13263b;color:#d3e8ff}
  .kinds button[data-kind=region].on{border-color:var(--region);background:#332919;color:#ffe4b3}
  .row{display:flex;gap:6px;align-items:center}
  .row label{display:flex;gap:4px;align-items:center;color:var(--dim);font-size:12px}
  select,input{background:#1e222b;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 6px;font-size:12px;min-width:0}
  input[type=number]{width:88px}
  .row button{padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:#1e222b;color:var(--text);cursor:pointer;font-size:12px}
  .hint{color:var(--dim);font-size:11px;line-height:1.6}
  #gateBox{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px;max-height:150px;overflow:auto}
  .chip{display:inline-flex;gap:6px;align-items:center;padding:3px 8px;border-radius:999px;background:#1e222b;border:1px solid var(--line);font-size:11px;margin:0 4px 4px 0}
  .chip.ok{border-color:#245c3d;color:#a9f0c4}
  .chip.bad{border-color:#5c2424;color:#ffb3b3}
  #list{flex:1;overflow:auto;padding:8px 8px 40px}
  .grp{margin:6px 2px}
  .grp h3{margin:6px 0;font-size:12px;color:var(--dim);font-weight:500;display:flex;gap:8px;align-items:center}
  .grp h3 b{color:var(--text);font-size:12px}
  .pt{display:flex;gap:6px;align-items:center;padding:4px 6px;border-radius:6px;cursor:pointer;border:1px solid transparent}
  .pt:hover{background:#1e222b}
  .pt.sel{background:#1c2740;border-color:#2f4b7c}
  .pt .dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
  .pt.entry .dot{background:var(--entry)} .pt.cover .dot{background:var(--cover)} .pt.region .dot{background:var(--region)}
  .pt input.name{flex:1;background:transparent;border:1px solid transparent;padding:2px 4px}
  .pt input.name:focus{border-color:var(--line);background:#141821}
  .pt .xyz{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums}
  .pt .src{font-size:10px;color:#6b7280;border:1px solid var(--line);border-radius:4px;padding:0 4px}
  .pt button{border:0;background:transparent;color:var(--dim);cursor:pointer;font-size:13px;padding:0 3px}
  .pt button:hover{color:var(--danger)}
  #side footer{border-top:1px solid var(--line);padding:10px 14px;display:flex;gap:6px;flex-wrap:wrap}
  #side footer button{flex:1;padding:8px 6px;border-radius:7px;border:1px solid var(--line);background:#1e222b;color:var(--text);cursor:pointer;font-size:12px}
  #side footer button.primary{background:#1f6f45;border-color:#2f8f5c;color:#eafff3;flex:2}
  #side footer button:disabled{opacity:.45;cursor:not-allowed}
  #status{width:100%;color:var(--dim);font-size:11px;min-height:16px;margin-top:6px;word-break:break-all}
  #status.err{color:var(--danger)}
  #status.ok{color:var(--ok)}
  #view{position:relative;flex:1;overflow:hidden;background:
     radial-gradient(circle at 40% 35%, #1a2030 0%, #0d1016 70%);cursor:crosshair}
  #stage{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}
  #radar{display:block;width:100%;height:100%;user-select:none;-webkit-user-drag:none;pointer-events:none;opacity:.92}
  #grid{position:absolute;inset:0;background-image:
      linear-gradient(to right,#ffffff12 1px,transparent 1px),
      linear-gradient(to bottom,#ffffff12 1px,transparent 1px);
    background-size:100px 100px}
  #markers{position:absolute;inset:0;pointer-events:none}
  .marker{position:absolute;pointer-events:auto;cursor:grab;transform:translate(-50%,-50%) scale(var(--inv,1))}
  .marker .dot{display:block;width:12px;height:12px;border-radius:50%;border:2px solid #0009;margin:0 auto}
  .marker.entry .dot{background:var(--entry)} .marker.cover .dot{background:var(--cover)} .marker.region .dot{background:var(--region)}
  .marker .lbl{position:absolute;left:16px;top:-6px;white-space:nowrap;font-size:11px;color:#dfe6f5;background:#000a;padding:1px 5px;border-radius:4px;border:1px solid #ffffff22}
  .marker.sel .dot{outline:2px solid #fff;outline-offset:2px}
  .ring{position:absolute;pointer-events:none;border:2px dashed #ffb54566;border-radius:50%;transform:translate(-50%,-50%)}
  #hud{position:absolute;left:12px;bottom:12px;color:#aeb7c9;font-size:11px;font-variant-numeric:tabular-nums;background:#0009;border:1px solid #ffffff18;padding:5px 9px;border-radius:6px;pointer-events:none}
  dialog{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:10px;padding:14px;width:min(720px,90vw)}
  dialog textarea{width:100%;height:320px;background:#0f1319;color:#d8ffd8;border:1px solid var(--line);border-radius:8px;padding:8px;font:12px/1.5 ui-monospace,Consolas,monospace;resize:vertical}
  dialog .row{justify-content:flex-end;margin-top:8px}
  .saved #stage::after{content:"已保存";position:absolute;top:12px;right:12px;background:#1f6f45;color:#eafff3;padding:4px 10px;border-radius:6px;font-size:12px}
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <header>
      <h1>打点 · <span id="mapName"></span></h1>
      <div class="sub" id="subInfo"></div>
    </header>
    <section class="toolbar">
      <div class="kinds">
        <button data-kind="entry" class="on">1 敌方进入点</button>
        <button data-kind="cover">2 掩体点</button>
        <button data-kind="region">3 区域中心</button>
      </div>
      <div class="row">
        <label>区域</label>
        <select id="regionSel"></select>
        <input id="newRegion" placeholder="新区域名" style="flex:1">
        <button id="addRegion">＋</button>
      </div>
      <div class="row">
        <label>z <input id="zInput" type="number" step="1"></label>
        <label>半径 <input id="radiusInput" type="number" step="10"></label>
        <button id="fitBtn" title="复位视图 (0)">复位</button>
      </div>
      <div class="hint">左键加点 · 拖动移动 · 右键删除 · 滚轮缩放 · 空白处拖动平移</div>
    </section>
    <section id="gateBox"></section>
    <section id="list"></section>
    <footer>
      <button id="saveBtn" class="primary">保存 (Ctrl+S)</button>
      <button id="saveExitBtn">保存并关闭</button>
      <button id="exportBtn">导出 YAML</button>
      <div id="status"></div>
    </footer>
  </aside>
  <main id="view">
    <div id="stage">
      <img id="radar" alt="">
      <div id="grid" hidden></div>
      <div id="markers"></div>
    </div>
    <div id="hud"></div>
  </main>
</div>

<dialog id="exportDlg">
  <div class="sub">把下面内容存成 <code>config/points.yaml</code>（或让 <code>csc maps apply --map __MAP__ --points &lt;文件&gt;</code> 读它）</div>
  <textarea id="exportText" spellcheck="false"></textarea>
  <div class="row">
    <button id="copyBtn">复制</button>
    <button id="downloadBtn">下载</button>
    <button id="closeDlg">关闭</button>
  </div>
</dialog>

<script>
const PAYLOAD = __PAYLOAD__;
const X = PAYLOAD.transform || {pos_x:0,pos_y:0,scale:1,rotation_deg:0,width:1024,height:1024};
const GATE = PAYLOAD.gate || {min_entry:3,min_cover:1};
const CLAMP_XY = 20000, CLAMP_Z = 5000;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = v => Math.round(Number(v) * 10) / 10;

const state = {
  map: PAYLOAD.map,
  kind: (PAYLOAD.defaults && PAYLOAD.defaults.kind) || 'entry',
  region: (PAYLOAD.defaults && PAYLOAD.defaults.region) || 'A',
  z: Number((PAYLOAD.defaults && PAYLOAD.defaults.z) || 0),
  regions: JSON.parse(JSON.stringify(PAYLOAD.regions || {})),
  points: (PAYLOAD.points || []).map(p => ({
    kind: p.kind, region: p.region, name: p.name,
    pos: [Number(p.pos[0]), Number(p.pos[1]), Number(p.pos[2])],
    source: p.source || 'points'
  })),
  selected: -1,
  nextId: 1,
};

/* ---------- 坐标换算（与 csc/radar.py 同一公式） ---------- */
function rotatePoint(px, py, deg) {
  if (!deg) return [px, py];
  const cx = X.width / 2, cy = X.height / 2, a = deg * Math.PI / 180;
  const dx = px - cx, dy = py - cy;
  return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)];
}
function radarToWorld(px, py) {
  const [cx, cy] = rotatePoint(px, py, -X.rotation_deg);
  return [X.pos_x + cx * X.scale, X.pos_y - cy * X.scale];
}
function worldToRadar(x, y) {
  const px = (x - X.pos_x) / X.scale, py = (X.pos_y - y) / X.scale;
  return rotatePoint(px, py, X.rotation_deg);
}

/* ---------- 视图（平移/缩放） ---------- */
const stage = $('stage'), view = $('view'), markers = $('markers'), hud = $('hud');
const viewState = {x: 0, y: 0, z: 1};

stage.style.width = X.width + 'px';
stage.style.height = X.height + 'px';
if (PAYLOAD.radar) { $('radar').src = PAYLOAD.radar; } else { $('radar').hidden = true; $('grid').hidden = false; }
markers.style.setProperty('--inv', '1');
$('mapName').textContent = state.map;
$('subInfo').textContent = `雷达 ${X.width}×${X.height} · scale ${X.scale} · pos(${X.pos_x}, ${X.pos_y})`
  + (PAYLOAD.version ? ` · awpy-data ${PAYLOAD.version}` : '')
  + (PAYLOAD.lower ? ' · lower' : '');

function applyView() {
  stage.style.transform = `translate(${viewState.x}px, ${viewState.y}px) scale(${viewState.z})`;
  markers.style.setProperty('--inv', String(1 / viewState.z));
  updateHud();
}
function fitView() {
  const r = view.getBoundingClientRect();
  const z = Math.min(r.width / X.width, r.height / X.height) * 0.94;
  viewState.z = z || 1;
  viewState.x = (r.width - X.width * viewState.z) / 2;
  viewState.y = (r.height - X.height * viewState.z) / 2;
  applyView();
}
function toStage(ev) {
  const r = stage.getBoundingClientRect();
  return [(ev.clientX - r.left) / viewState.z, (ev.clientY - r.top) / viewState.z];
}
function updateHud() {
  const counts = gateCounts();
  const parts = Object.keys(counts).sort().map(r => {
    const c = counts[r];
    return `${r}: 入口 ${c.entry}/${GATE.min_entry} · 掩体 ${c.cover}/${GATE.min_cover}`;
  });
  hud.textContent = `缩放 ${viewState.z.toFixed(2)}×  |  ${parts.join('   ')}  |  世界 x∈[${fmt(PAYLOAD.world_bounds ? PAYLOAD.world_bounds[0] : 0)}, ${fmt(PAYLOAD.world_bounds ? PAYLOAD.world_bounds[1] : 0)}]`;
}

/* ---------- 点管理 ---------- */
function gateCounts() {
  const out = {};
  const ensure = r => (out[r] = out[r] || {entry: 0, cover: 0, center: false});
  Object.keys(state.regions).forEach(ensure);
  state.points.forEach(p => {
    const c = ensure(p.region);
    if (p.kind === 'entry') c.entry++;
    else if (p.kind === 'cover') c.cover++;
    else c.center = true;
  });
  return out;
}
function autoName(kind, region) {
  const prefix = kind === 'entry' ? 'entry' : kind === 'cover' ? 'cover' : 'region';
  const used = new Set(state.points.filter(p => p.kind === kind && p.region === region).map(p => p.name));
  let i = 1;
  while (used.has(`${prefix}_${i}`)) i++;
  return `${prefix}_${i}`;
}
function addPointAt(px, py) {
  const [wx, wy] = radarToWorld(px, py);
  const pos = [fmt(Math.max(-CLAMP_XY, Math.min(CLAMP_XY, wx))), fmt(Math.max(-CLAMP_XY, Math.min(CLAMP_XY, wy))), fmt(state.z)];
  let idx;
  if (state.kind === 'region') {
    const existing = state.points.findIndex(p => p.kind === 'region' && p.region === state.region);
    if (existing >= 0) state.points.splice(existing, 1);
    idx = state.points.push({kind: 'region', region: state.region, name: state.region, pos, source: 'points'}) - 1;
    state.regions[state.region] = state.regions[state.region] || {};
    state.regions[state.region].center = pos;
  } else {
    idx = state.points.push({kind: state.kind, region: state.region, name: autoName(state.kind, state.region), pos, source: 'points'}) - 1;
  }
  state.selected = idx;
  renderAll();
  return idx;
}
function deletePoint(idx) {
  if (idx < 0 || idx >= state.points.length) return;
  const p = state.points[idx];
  if (p.source === 'yaml') {
    setStatus('这个点写在 config/maps.yaml 里，编辑器删不掉；改 yaml 或换个名字覆盖它', true);
    return;
  }
  state.points.splice(idx, 1);
  state.selected = Math.min(idx, state.points.length - 1);
  renderAll();
}
function selectPoint(idx) {
  state.selected = idx;
  if (idx >= 0) {
    const p = state.points[idx];
    state.kind = p.kind;
    state.region = p.region;
    state.z = p.pos[2];
    const [px, py] = worldToRadar(p.pos[0], p.pos[1]);
    const r = view.getBoundingClientRect();
    viewState.x = r.width / 2 - px * viewState.z;
    viewState.y = r.height / 2 - py * viewState.z;
    applyView();
  }
  renderAll();
}

/* ---------- 渲染 ---------- */
function renderMarkers() {
  markers.innerHTML = '';
  Object.keys(state.regions).forEach(region => {
    const spec = state.regions[region] || {};
    if (!spec.center) return;
    const [px, py] = worldToRadar(spec.center[0], spec.center[1]);
    const radius = Number(spec.radius || 1100) / X.scale;
    const ring = document.createElement('div');
    ring.className = 'ring';
    ring.style.left = px + 'px';
    ring.style.top = py + 'px';
    ring.style.width = (radius * 2) + 'px';
    ring.style.height = (radius * 2) + 'px';
    markers.appendChild(ring);
  });
  state.points.forEach((p, i) => {
    const [px, py] = worldToRadar(p.pos[0], p.pos[1]);
    const el = document.createElement('div');
    el.className = 'marker ' + p.kind + (i === state.selected ? ' sel' : '');
    el.style.left = px + 'px';
    el.style.top = py + 'px';
    el.dataset.idx = String(i);
    el.innerHTML = `<span class="dot"></span><span class="lbl">${esc(p.name)}</span>`;
    el.addEventListener('pointerdown', ev => { ev.stopPropagation(); ev.preventDefault(); startDragMarker(ev, i); });
    el.addEventListener('contextmenu', ev => { ev.preventDefault(); ev.stopPropagation(); deletePoint(i); });
    markers.appendChild(el);
  });
}
function renderPanel() {
  const counts = gateCounts();
  const gate = Object.keys(counts).sort().map(region => {
    const c = counts[region];
    const okEntry = c.entry >= GATE.min_entry, okCover = c.cover >= GATE.min_cover;
    const chips = [
      `<span class="chip ${okEntry ? 'ok' : 'bad'}">入口 ${c.entry}/${GATE.min_entry}</span>`,
      `<span class="chip ${okCover ? 'ok' : 'bad'}">掩体 ${c.cover}/${GATE.min_cover}</span>`,
      `<span class="chip ${c.center ? 'ok' : 'bad'}">${c.center ? '有中心' : '无中心'}</span>`,
    ].join('');
    return `<div><b style="color:#cfd6e6;font-size:12px">${esc(region)}</b> ${chips}</div>`;
  }).join('');
  $('gateBox').innerHTML = gate || '<div class="sub">还没有区域</div>';

  const byRegion = {};
  state.points.forEach((p, i) => { (byRegion[p.region] = byRegion[p.region] || []).push([p, i]); });
  const order = {entry: 0, cover: 1, region: 2};
  const html = Object.keys(byRegion).sort().map(region => {
    const rows = byRegion[region].sort((a, b) => (order[a[0].kind] - order[b[0].kind]) || a[0].name.localeCompare(b[0].name));
    const body = rows.map(([p, i]) => `
      <div class="pt ${p.kind} ${i === state.selected ? 'sel' : ''}" data-idx="${i}">
        <span class="dot"></span>
        <input class="name" value="${esc(p.name)}" data-name="${i}" spellcheck="false">
        <span class="xyz">${fmt(p.pos[0])}, ${fmt(p.pos[1])}, ${fmt(p.pos[2])}</span>
        <span class="src">${p.source === 'yaml' ? 'yaml' : '工具'}</span>
        <button data-del="${i}" title="删除">✕</button>
      </div>`).join('');
    return `<div class="grp"><h3>区域 <b>${esc(region)}</b> <span>${rows.length} 个点</span></h3>${body}</div>`;
  }).join('');
  $('list').innerHTML = html || '<div class="sub" style="padding:10px">左键在雷达图上点位置即可加第一个点</div>';

  document.querySelectorAll('#list .pt').forEach(el => {
    el.addEventListener('click', ev => {
      if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'BUTTON') return;
      selectPoint(Number(el.dataset.idx));
    });
  });
  document.querySelectorAll('#list [data-del]').forEach(btn => {
    btn.addEventListener('click', ev => { ev.stopPropagation(); deletePoint(Number(btn.dataset.del)); });
  });
  document.querySelectorAll('#list input.name').forEach(inp => {
    inp.addEventListener('change', () => {
      const i = Number(inp.dataset.name);
      state.points[i].name = inp.value.trim() || state.points[i].name;
      renderMarkers(); renderPanel();
    });
    inp.addEventListener('click', ev => ev.stopPropagation());
  });
  syncControls();
}
function syncControls() {
  document.querySelectorAll('.kinds button').forEach(b => b.classList.toggle('on', b.dataset.kind === state.kind));
  const sel = $('regionSel');
  sel.innerHTML = Object.keys(state.regions).sort().map(r => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
  if (state.regions[state.region]) sel.value = state.region;
  $('zInput').value = state.z;
  const spec = state.regions[state.region] || {};
  $('radiusInput').value = spec.radius || 1100;
}
function renderAll() { renderMarkers(); renderPanel(); updateHud(); }
function setStatus(msg, isErr) {
  const el = $('status');
  el.textContent = msg || '';
  el.className = isErr ? 'err' : 'ok';
}

/* ---------- 交互：平移 / 缩放 / 加点 / 拖点 ---------- */
let panning = null, dragged = false, dragMarker = null;

view.addEventListener('pointerdown', ev => {
  if (ev.button !== 0 && ev.button !== 1) return;
  panning = {sx: ev.clientX, sy: ev.clientY, ox: viewState.x, oy: viewState.y};
  dragged = false;
  view.setPointerCapture(ev.pointerId);
});
view.addEventListener('pointermove', ev => {
  if (panning) {
    const dx = ev.clientX - panning.sx, dy = ev.clientY - panning.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragged = true;
    viewState.x = panning.ox + dx;
    viewState.y = panning.oy + dy;
    applyView();
  } else if (!dragMarker) {
    const [px, py] = toStage(ev);
    const [wx, wy] = radarToWorld(px, py);
    hud.dataset.cursor = `${fmt(wx)}, ${fmt(wy)}`;
    if (/[0-9]/.test(hud.textContent)) {
      hud.textContent = hud.textContent.replace(/\s*\|\s*光标.*$/, '');
    }
    hud.textContent += `   |   光标 ${fmt(wx)}, ${fmt(wy)}`;
  }
});
view.addEventListener('pointerup', ev => {
  if (panning) {
    const wasDrag = dragged;
    panning = null;
    if (!wasDrag) {
      const [px, py] = toStage(ev);
      if (px >= 0 && py >= 0 && px <= X.width && py <= X.height) addPointAt(px, py);
    }
  }
});
view.addEventListener('contextmenu', ev => ev.preventDefault());
view.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = stage.getBoundingClientRect();
  const [sx, sy] = [(ev.clientX - r.left) / viewState.z, (ev.clientY - r.top) / viewState.z];
  const factor = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
  const next = Math.min(12, Math.max(0.15, viewState.z * factor));
  const vr = view.getBoundingClientRect();
  viewState.x = (ev.clientX - vr.left) - sx * next;
  viewState.y = (ev.clientY - vr.top) - sy * next;
  viewState.z = next;
  applyView();
}, {passive: false});

function startDragMarker(ev, idx) {
  dragMarker = {idx, moved: false};
  view.setPointerCapture(ev.pointerId);
  const move = e => {
    const [px, py] = toStage(e);
    const [wx, wy] = radarToWorld(Math.max(0, Math.min(X.width, px)), Math.max(0, Math.min(X.height, py)));
    const p = state.points[idx];
    p.pos = [fmt(wx), fmt(wy), p.pos[2]];
    if (p.kind === 'region') {
      state.regions[p.region] = state.regions[p.region] || {};
      state.regions[p.region].center = p.pos;
      renderMarkers();
    } else {
      renderMarkers();
    }
    dragMarker.moved = true;
    updateHud();
  };
  const up = () => {
    view.removeEventListener('pointermove', move);
    view.removeEventListener('pointerup', up);
    state.selected = idx;
    dragMarker = null;
    renderAll();
  };
  view.addEventListener('pointermove', move);
  view.addEventListener('pointerup', up, {once: true});
}

/* ---------- 控件 ---------- */
document.querySelectorAll('.kinds button').forEach(btn => {
  btn.addEventListener('click', () => { state.kind = btn.dataset.kind; syncControls(); });
});
$('regionSel').addEventListener('change', ev => { state.region = ev.target.value; syncControls(); });
$('zInput').addEventListener('change', ev => { state.z = Number(ev.target.value); applySelectedZ(); });
$('radiusInput').addEventListener('change', ev => {
  state.regions[state.region] = state.regions[state.region] || {};
  state.regions[state.region].radius = Number(ev.target.value);
  renderMarkers(); updateHud();
});
$('addRegion').addEventListener('click', () => {
  const name = ($('newRegion').value || '').trim().toUpperCase();
  if (!name) return;
  state.regions[name] = state.regions[name] || {center: null, radius: 1100};
  state.region = name;
  $('newRegion').value = '';
  renderAll();
});
$('fitBtn').addEventListener('click', fitView);

function applySelectedZ() {
  if (state.selected < 0) return;
  const p = state.points[state.selected];
  p.pos[2] = fmt(state.z);
  renderAll();
}

document.addEventListener('keydown', ev => {
  const tag = (ev.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  if (ev.key === '1') { state.kind = 'entry'; syncControls(); }
  else if (ev.key === '2') { state.kind = 'cover'; syncControls(); }
  else if (ev.key === '3') { state.kind = 'region'; syncControls(); }
  else if (ev.key === '0') { fitView(); }
  else if (ev.key === 'Escape') { state.selected = -1; renderAll(); }
  else if ((ev.key === 'Delete' || ev.key === 'Backspace') && state.selected >= 0) { ev.preventDefault(); deletePoint(state.selected); }
  else if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 's') { ev.preventDefault(); save(false); }
});

/* ---------- 保存 ---------- */
function buildPayload() {
  const regions = {}, entry_points = {}, cover_points = {};
  Object.keys(state.regions).forEach(region => {
    const spec = state.regions[region] || {};
    const center = state.points.find(p => p.kind === 'region' && p.region === region);
    const out = {radius: Number(spec.radius || 1100)};
    if (center) out.center = center.pos;
    regions[region] = out;
  });
  state.points.forEach(p => {
    if (p.kind === 'region') return;
    const bucket = p.kind === 'entry' ? entry_points : cover_points;
    (bucket[p.region] = bucket[p.region] || []).push({name: p.name, pos: p.pos});
  });
  return {
    map: state.map, regions, entry_points, cover_points,
    meta: {radar: PAYLOAD.version || null, transform: X},
  };
}
function toYaml() {
  const body = buildPayload();
  const lines = ['# 由 csc maps edit 导出', 'maps:'];
  lines.push(`  ${state.map}:`);
  lines.push('    regions:');
  Object.keys(body.regions).forEach(r => {
    const spec = body.regions[r];
    lines.push(`      ${r}:`);
    if (spec.center) lines.push(`        center: [${spec.center.map(fmt).join(', ')}]`);
    lines.push(`        radius: ${fmt(spec.radius)}`);
  });
  ['entry_points', 'cover_points'].forEach(key => {
    lines.push(`    ${key}:`);
    const groups = body[key];
    const names = Object.keys(groups).sort();
    if (!names.length) { lines.push('      {}'); return; }
    names.forEach(r => {
      lines.push(`      ${r}:`);
      groups[r].forEach(pt => {
        lines.push(`        - name: ${JSON.stringify(pt.name)}`);
        lines.push(`          pos: [${pt.pos.map(fmt).join(', ')}]`);
      });
    });
  });
  return lines.join('\n') + '\n';
}
async function save(exit) {
  const body = buildPayload();
  if (exit) body.exit = true;
  if (!PAYLOAD.serve) { openExport(); return; }
  setStatus('保存中…');
  try {
    const resp = await fetch('/api/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await resp.json();
    if (!data.ok) { setStatus('保存失败：' + (data.error || resp.status), true); return; }
    const problems = data.problems || [];
    setStatus('已保存 → ' + data.path + (problems.length ? `（${problems.length} 条提示，见上方）` : ''), problems.length > 0);
    if (problems.length) {
      renderPanel();
      $('gateBox').innerHTML += problems.map(t => `<div class="chip bad" style="display:block">${esc(t)}</div>`).join('');
    }
    if (exit) {
      document.body.classList.add('saved');
      setStatus('已保存并关闭服务，可以关掉这个页面了。');
    }
  } catch (err) {
    setStatus('保存失败：' + err, true);
  }
}
function openExport() {
  $('exportText').value = toYaml();
  $('exportDlg').showModal();
}
$('saveBtn').addEventListener('click', () => save(false));
$('saveExitBtn').addEventListener('click', () => save(true));
$('exportBtn').addEventListener('click', openExport);
$('closeDlg').addEventListener('click', () => $('exportDlg').close());
$('copyBtn').addEventListener('click', async () => {
  const text = $('exportText');
  try { await navigator.clipboard.writeText(text.value); setStatus('已复制到剪贴板'); }
  catch { text.select(); document.execCommand('copy'); setStatus('已复制（兼容模式）'); }
});
$('downloadBtn').addEventListener('click', () => {
  const blob = new Blob([$('exportText').value], {type: 'text/yaml;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `points-${state.map}.yaml`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
});

/* ---------- 启动 ---------- */
renderAll();
fitView();
if (!PAYLOAD.serve) {
  // 离线模式没有后端可关，"保存并关闭"没意义
  $('saveExitBtn').disabled = true;
  $('saveExitBtn').title = '离线模式：请用「导出 YAML」';
  $('saveBtn').textContent = '保存 = 导出 YAML';
}
setStatus(PAYLOAD.serve ? '已连接本地保存服务：Ctrl+S 直接写入 config/points.yaml' : '离线模式：用「导出 YAML」保存');
</script>
</body>
</html>
"""
