// Coordinates stay in original-image pixels. Painting and hit targets use screen pixels.
// Each reviewed dataset gets its own draft, so older boxes cannot overwrite a new review.
// Previous browser drafts are kept under their previous keys.
const storageKey = 'barcode-ml-next-batch-annotations-v2-' + annotationRevision;
const classInfo = {b:['Barcode','#25b9ff'], l:['Shipping label','#39d879'], p:['Parcel','#ff5757'], e:['Excluded sticker','#ffb82e']};
const $ = id => document.getElementById(id);
let data = structuredClone(originalData);
const keys = Object.keys(data), history = new Map();
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
  if (saved && keys.every(key => saved[key]?.annotation)) {
    // Restore boxes only; image paths and dimensions always come from the gallery.
    for (const key of keys) data[key].annotation = saved[key].annotation;
  }
} catch (error) { console.warn('Could not restore draft', error); }
const editor = $('editor'), image = $('canvas-image'), canvas = $('box-canvas');
const ctx = canvas.getContext('2d');
let current = -1, selected = null, gesture = null, mode = 'select', pendingFrame = 0;
let view = {width:0, height:0, sx:1, sy:1, dpr:1};
const entry = () => data[keys[current]];
const annotations = () => entry().annotation;
const selectedBox = () => selected ? annotations()[selected.type]?.[selected.index] : null;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function persist() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(data));
    $('save-status').textContent = 'Draft saved in this browser. Export JSON when finished.';
  } catch { $('save-status').textContent = 'Browser storage unavailable. Export JSON to keep your changes.'; }
}
function remember(before) {
  const key = keys[current], items = history.get(key) || [];
  items.push(before); if (items.length > 50) items.shift(); history.set(key, items);
}
function commit(before) {
  if (JSON.stringify(before) !== JSON.stringify(annotations())) { remember(before); persist(); }
  updatePanel(); requestPaint();
}
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'annotations.manual.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function setMode(next) {
  mode = next;
  $('select-mode').setAttribute('aria-pressed', String(mode === 'select'));
  $('draw-mode').setAttribute('aria-pressed', String(mode === 'draw'));
  canvas.style.cursor = mode === 'draw' ? 'crosshair' : 'default';
}

// Resize the backing canvas only when the workspace changes, never during a drag.
function fitImage() {
  if (current < 0 || !image.complete || !image.naturalWidth || editor.hidden) return;
  const stage = $('image-stage'), scale = Math.min((stage.clientWidth-32)/entry().width, (stage.clientHeight-32)/entry().height);
  if (scale <= 0) return;
  view.width = entry().width*scale; view.height = entry().height*scale;
  view.sx = scale; view.sy = scale; view.dpr = window.devicePixelRatio || 1;
  $('canvas-frame').style.width = view.width+'px'; $('canvas-frame').style.height = view.height+'px';
  canvas.width = Math.round(view.width*view.dpr); canvas.height = Math.round(view.height*view.dpr);
  requestPaint();
}
function handles(box) {
  const [x1,y1,x2,y2] = box, xm = (x1+x2)/2, ym = (y1+y2)/2;
  return {nw:[x1,y1], n:[xm,y1], ne:[x2,y1], e:[x2,ym], se:[x2,y2], s:[xm,y2], sw:[x1,y2], w:[x1,ym]};
}
function requestPaint() {
  if (!pendingFrame) pendingFrame = requestAnimationFrame(() => { pendingFrame = 0; paint(); });
}
function paint() {
  if (current < 0 || editor.hidden) return;
  ctx.setTransform(view.dpr,0,0,view.dpr,0,0); ctx.clearRect(0,0,view.width,view.height);
  for (const type of ['p','l','b','e']) {
    (annotations()[type] || []).forEach((box,index) => {
      const [x1,y1,x2,y2] = box.map((v,i) => v*(i%2 ? view.sy : view.sx));
      const active = selected?.type === type && selected.index === index;
      ctx.strokeStyle = classInfo[type][1]; ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      if (active) { ctx.fillStyle = classInfo[type][1]+'18'; ctx.fillRect(x1,y1,x2-x1,y2-y1); }
      ctx.font = '12px system-ui'; const text = classInfo[type][0], ty = Math.max(15,y1-4);
      ctx.lineWidth = 3; ctx.strokeStyle = '#0c1117'; ctx.strokeText(text,x1+3,ty);
      ctx.fillStyle = classInfo[type][1]; ctx.fillText(text,x1+3,ty);
    });
  }
  const box = selectedBox();
  if (box) for (const [x,y] of Object.values(handles(box))) {
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#102235'; ctx.lineWidth = 1.5;
    ctx.fillRect(x*view.sx-4,y*view.sy-4,8,8); ctx.strokeRect(x*view.sx-4,y*view.sy-4,8,8);
  }
  if (gesture?.kind === 'draw') {
    const a = gesture.start, b = gesture.latest;
    ctx.strokeStyle = classInfo[$('class').value][1]; ctx.lineWidth = 2; ctx.setLineDash([6,4]);
    ctx.strokeRect(a.x*view.sx,a.y*view.sy,(b.x-a.x)*view.sx,(b.y-a.y)*view.sy); ctx.setLineDash([]);
  }
}
function updatePanel() {
  const list = $('box-list'); list.replaceChildren();
  if (!selectedBox()) selected = null;
  $('delete-selected').disabled = !selected; $('undo').disabled = !history.get(keys[current])?.length;
  for (const type of ['p','l','b','e']) (annotations()[type] || []).forEach((box,index) => {
    const row = document.createElement('div');
    row.className = 'box-row'+(selected?.type === type && selected.index === index ? ' selected' : '');
    const label = document.createElement('span'); label.style.color = classInfo[type][1]; label.textContent = classInfo[type][0];
    const coordinates = document.createElement('small'); coordinates.className = 'coordinates'; coordinates.textContent = box.map(Math.round).join(', ');
    label.append(coordinates); row.append(label); row.onclick = () => { selected = {type,index}; setMode('select'); updatePanel(); requestPaint(); };
    const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Remove '+classInfo[type][0];
    remove.onclick = event => { event.stopPropagation(); selected = {type,index}; deleteSelected(); };
    row.append(remove); list.append(row);
  });
}
function deleteSelected() {
  if (!selectedBox()) return;
  const before = structuredClone(annotations()); annotations()[selected.type].splice(selected.index,1); selected = null; commit(before);
}
function openImage(key) {
  cancelGesture(); current = keys.indexOf(key); selected = null; editor.hidden = false; setMode('select');
  document.body.style.overflow = 'hidden'; $('editor-title').textContent = key+' · '+(current+1)+' / '+keys.length;
  $('editor-note').textContent = entry().review_note || entry().annotation.note || 'Click a box to edit it.';
  image.onload = fitImage; image.src = entry().src; updatePanel(); fitImage();
}
function point(event) {
  const rect = canvas.getBoundingClientRect();
  return {x:clamp((event.clientX-rect.left)/rect.width*entry().width,0,entry().width), y:clamp((event.clientY-rect.top)/rect.height*entry().height,0,entry().height)};
}
function handleAt(p) {
  const box = selectedBox(); if (!box) return null;
  const nearest = Object.entries(handles(box)).map(([name,[x,y]]) => ({name,distance:Math.hypot((p.x-x)*view.sx,(p.y-y)*view.sy)}))
    .sort((a,b) => a.distance-b.distance)[0];
  return nearest.distance <= 9 ? nearest.name : null;
}
function hitTest(p) {
  const hits = [];
  for (const type of ['b','l','e','p']) (annotations()[type] || []).forEach((box,index) => {
    const [x1,y1,x2,y2] = box, tolerance = 5/view.sx;
    if (p.x < x1-tolerance || p.x > x2+tolerance || p.y < y1-tolerance || p.y > y2+tolerance) return;
    const edge = Math.min(Math.abs(p.x-x1),Math.abs(p.x-x2),Math.abs(p.y-y1),Math.abs(p.y-y2))*view.sx;
    hits.push({type,index,edge,area:(x2-x1)*(y2-y1)});
  });
  // Borders win over interiors, then the smallest box wins inside nested objects.
  hits.sort((a,b) => Number(b.edge<=5)-Number(a.edge<=5) || a.area-b.area);
  return hits[0] || null;
}
const cursors = {nw:'nwse-resize',se:'nwse-resize',ne:'nesw-resize',sw:'nesw-resize',n:'ns-resize',s:'ns-resize',e:'ew-resize',w:'ew-resize'};
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0 || gesture) return;
  event.preventDefault(); canvas.focus({preventScroll:true});
  const p = point(event), before = structuredClone(annotations());
  if (mode === 'draw') { selected = null; gesture = {kind:'draw',start:p,latest:p,before}; }
  else {
    const handle = handleAt(p), hit = handle ? selected : hitTest(p);
    if (!hit) { selected = null; updatePanel(); requestPaint(); return; }
    selected = {type:hit.type,index:hit.index};
    gesture = {kind:handle ? 'resize' : 'move',handle,start:p,latest:p,original:[...selectedBox()],before};
  }
  gesture.pointerId = event.pointerId; canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = gesture.kind === 'move' ? 'grabbing' : gesture.kind === 'resize' ? cursors[gesture.handle] : 'crosshair';
  updatePanel(); requestPaint();
});
function updateGesture(event) {
  if (!gesture || event.pointerId !== gesture.pointerId) return;
  const p = point(event); gesture.latest = p;
  const dx = p.x-gesture.start.x, dy = p.y-gesture.start.y;
  if (!gesture.moved && Math.hypot(dx*view.sx,dy*view.sy)<3) return;
  gesture.moved = true;
  if (gesture.kind !== 'draw') {
    const b = gesture.original, box = [...b], w = b[2]-b[0], h = b[3]-b[1];
    if (gesture.kind === 'move') {
      box[0] = clamp(b[0]+dx,0,entry().width-w); box[1] = clamp(b[1]+dy,0,entry().height-h);
      box[2] = box[0]+w; box[3] = box[1]+h;
    } else {
      const side = gesture.handle;
      if (side.includes('w')) box[0] = clamp(b[0]+dx,0,b[2]-8);
      if (side.includes('e')) box[2] = clamp(b[2]+dx,b[0]+8,entry().width);
      if (side.includes('n')) box[1] = clamp(b[1]+dy,0,b[3]-8);
      if (side.includes('s')) box[3] = clamp(b[3]+dy,b[1]+8,entry().height);
    }
    annotations()[selected.type][selected.index] = box;
  }
  requestPaint();
}
canvas.addEventListener('pointermove', event => {
  if (gesture) { updateGesture(event); return; }
  const p = point(event), handle = mode === 'select' && handleAt(p);
  canvas.style.cursor = mode === 'draw' ? 'crosshair' : handle ? cursors[handle] : hitTest(p) ? 'move' : 'default';
});
canvas.addEventListener('pointerup', event => {
  if (!gesture || event.pointerId !== gesture.pointerId) return;
  updateGesture(event);
  const done = gesture;
  if (done.kind === 'draw' && done.moved) {
    const a = done.start, b = done.latest, box = [Math.min(a.x,b.x),Math.min(a.y,b.y),Math.max(a.x,b.x),Math.max(a.y,b.y)].map(Math.round);
    if (box[2]-box[0]>=8 && box[3]-box[1]>=8) {
      const type = $('class').value; (annotations()[type] ??= []).push(box); selected = {type,index:annotations()[type].length-1}; setMode('select');
    }
  } else if (done.moved && selectedBox()) annotations()[selected.type][selected.index] = selectedBox().map(Math.round);
  gesture = null; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  canvas.style.cursor = mode === 'draw' ? 'crosshair' : 'move'; commit(done.before);
});
function cancelGesture() {
  if (!gesture) return;
  const canceled = gesture; gesture = null;
  entry().annotation = canceled.before;
  if (canvas.hasPointerCapture(canceled.pointerId)) canvas.releasePointerCapture(canceled.pointerId);
  updatePanel(); requestPaint();
}
canvas.addEventListener('pointercancel',cancelGesture); canvas.addEventListener('lostpointercapture',cancelGesture);
$('select-mode').onclick = () => setMode('select'); $('draw-mode').onclick = () => setMode('draw');
$('delete-selected').onclick = deleteSelected;
$('undo').onclick = () => { const before = history.get(keys[current])?.pop(); if (before) { entry().annotation = before; selected = null; persist(); updatePanel(); requestPaint(); } };
$('previous').onclick = () => openImage(keys[(current-1+keys.length)%keys.length]);
$('next').onclick = () => openImage(keys[(current+1)%keys.length]);
$('close').onclick = () => { cancelGesture(); editor.hidden = true; document.body.style.overflow = ''; };
$('export').onclick = download; $('export-editor').onclick = download;
document.querySelectorAll('.edit-image').forEach(button => button.onclick = () => openImage(button.dataset.key));
new ResizeObserver(fitImage).observe($('image-stage'));
document.addEventListener('keydown', event => {
  if (editor.hidden || /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
  if (gesture) { if (event.key === 'Escape') cancelGesture(); return; }
  if (event.key.toLowerCase() === 'v') setMode('select');
  if (event.key.toLowerCase() === 'b') setMode('draw');
  if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); deleteSelected(); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); $('undo').click(); }
  if (event.key === 'Escape') { if (selected) { selected = null; updatePanel(); requestPaint(); } else $('close').click(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); $('previous').click(); }
  if (event.key === 'ArrowRight') { event.preventDefault(); $('next').click(); }
});
$('import').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const imported = JSON.parse(await file.text());
    if (keys.some(key => !imported[key]?.annotation)) throw new Error('Missing image annotation records.');
    for (const key of keys) data[key].annotation = imported[key].annotation;
    selected = null; history.clear(); persist(); if (current>=0) { updatePanel(); requestPaint(); }
  } catch (error) { alert('Could not import: '+error.message); }
  event.target.value = '';
};
