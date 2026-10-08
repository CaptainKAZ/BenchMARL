// BenchMARL/liveview/test_live_view.js —— 无头测试 live_view.html 的观察窗绘制逻辑（不需要浏览器）
// 用法：node BenchMARL/liveview/test_live_view.js
// 做三件事：① 用真实 live_env.json 逐局逐帧调用 drawEpisode，抓异常与非有限坐标
//          ② 模拟拖动进度条 / 切局 / tick 播放
//          ③ 校验球员/篮筐/投篮点的绘制半径是否符合场景真实值（agent 0.3m / basket 0.1m / spot 1.2m）
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const OPC = __dirname;
const ROOT = path.resolve(OPC, '..', '..');   // 仓库根
const html = fs.readFileSync(path.join(OPC, 'live_view.html'), 'utf8');
const m = html.match(/<script>\s*([\s\S]*?)<\/script>\s*<\/body>/);
if (!m) { console.error('没找到主 <script> 块'); process.exit(1); }
const js = m[1];

const livePath = path.join(ROOT, 'BenchMARL/outputs/live/live_env.json');
if (!fs.existsSync(livePath)) { console.error('缺 live_env.json：' + livePath); process.exit(1); }
const payload = JSON.parse(fs.readFileSync(livePath, 'utf8'));

globalThis.CUR = 'init';
globalThis.problems = [];
const problems = globalThis.problems;
const rec = (msg) => problems.push(`[${globalThis.CUR}] ${msg}`);

// ---- 半径记录：把 arc 调用按“当前实体”归类（用 arc 半径 / S 换回米）----
globalThis.arcLog = [];

function makeCtx() {
  const ctx = {};
  // Canvas2D 各方法的必填参数个数（少传会像真浏览器一样抛错）
  const ARITY = { clearRect: 4, fillRect: 4, strokeRect: 4, moveTo: 2, lineTo: 2, arc: 5, fill: 0, stroke: 0,
                  setLineDash: 1, fillText: 3, rect: 4, save: 0, restore: 0, closePath: 0, beginPath: 0,
                  translate: 2, rotate: 1, scale: 2, quadraticCurveTo: 4, bezierCurveTo: 6, ellipse: 7,
                  arcTo: 5, strokeText: 3 };
  const wrap = (name) => (...a) => {
    const need = ARITY[name];
    if (need !== undefined && a.length < need)
      rec(`ctx.${name} 参数不足: ${a.length} 个（应为 ${need}）: ${JSON.stringify(a.map(v => typeof v === 'number' ? +v.toFixed(2) : v))}`);
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (typeof v === 'number' && !Number.isFinite(v)) rec(`ctx.${name} 第${i}个参数非有限: ${v}`);
    }
    if (name === 'arc') {
      if (!(a[2] > 0)) rec(`arc 半径<=0: ${a[2]}`);
      globalThis.arcLog.push({ r: a[2], x: a[0], y: a[1] });
    }
  };
  for (const n of ['clearRect','fillRect','strokeRect','beginPath','moveTo','lineTo','arc','fill','stroke',
                   'setLineDash','fillText','strokeText','rect','save','restore','closePath','translate','rotate','scale',
                   'quadraticCurveTo','bezierCurveTo','ellipse','arcTo']) ctx[n] = wrap(n);
  // measureText：真浏览器会按字体宽度返回；这里按字符数近似（CJK 权重 1.0，ASCII 0.6）
  ctx.measureText = (t) => {
    const str = String(t);
    let w = 0;
    for (const ch of str) w += (ch.charCodeAt(0) > 0x2e80 ? 1.0 : 0.6) * 10;
    return { width: w };
  };
  for (const n of ['globalAlpha','strokeStyle','fillStyle','lineWidth','font','textAlign','lineCap','lineJoin'])
    ctx[n] = '';
  return ctx;
}
const ctx = makeCtx();

const els = {};
function mkEl(id) {
  return {
    id, style: {}, classList: { toggle(){}, add(){}, remove(){}, contains(){ return false; } },
    innerHTML: '', textContent: '', value: '0', checked: true, max: '0', min: '0',
    clientWidth: 900, clientHeight: 900, width: 0, height: 0, scrollTop: 0, scrollHeight: 0,
    onclick: null, oninput: null, onchange: null, _ls: {},
    getContext: () => ctx, appendChild(){}, focus() {},
    addEventListener(t, fn) { this._ls[t] = fn; },
    removeEventListener(){},
  };
}
function el(id) { return els[id] || (els[id] = mkEl(id)); }

global.document = {
  getElementById: el,
  createElement: (t) => mkEl('new-' + t + '-' + Math.random()),
  querySelectorAll: () => [],
  addEventListener(){},
  body: mkEl('body'),
};
global.window = { _ls: {}, addEventListener(t, fn) { this._ls[t] = fn; } };
global.ResizeObserver = class { observe() {} };
global.requestAnimationFrame = () => 0;
global.setInterval = () => 0;
global.localStorage = { getItem: () => null, setItem() {} };
global.Chart = class { constructor() {} update() {} };
global.fetch = async (url) => {
  const u = String(url);
  if (u.startsWith('live_env.json')) return { ok: true, status: 200, json: async () => payload, text: async () => '' };
  if (u.startsWith('curve.json'))   return { ok: true, status: 200, json: async () => ({ files: [], n: 0, points: [], mtime: '-' }), text: async () => '' };
  if (u.startsWith('logs.json'))    return { ok: true, status: 200, json: async () => [], text: async () => '' };
  if (u.startsWith('status.json'))  return { ok: true, status: 200, json: async () => ({}), text: async () => '' };
  if (u.startsWith('log'))          return { ok: true, status: 200, json: async () => ({}), text: async () => '' };
  return { ok: false, status: 404, json: async () => ({}), text: async () => '' };
};

const probe = `
;(async () => {
  await loadLive();
  if (!D) { console.log('FAIL: 没有加载到观察窗数据'); return; }
  let throws = 0;
  const call = (name, fn) => { CUR = name; try { fn(); } catch (e) { throws++; console.log('THROW ' + name + ': ' + e.message); } };
  let framesNoAgents = 0, framesChecked = 0;
  for (let i = 0; i < D.episodes.length; i++) {
    const ep = D.episodes[i];
    call('selectEp' + i, () => selectEp(i));
    for (let tt = 0; tt < ep.frames.length + 5; tt++) {
      const before = arcLog.length;
      CUR = 'ep' + i + '/t' + tt;
      try { drawEpisode(ep, tt); } catch (e) { throws++; console.log('THROW ' + CUR + ': ' + e.message); }
      const S0 = S;
      const rWant = Math.max(AGENT_R * S0, 6);
      const got = arcLog.slice(before).filter(a => Math.abs(a.r - rWant) < 1e-6).length;
      framesChecked++;
      if (got < 4) { framesNoAgents++; if (framesNoAgents <= 3) console.log('NO-AGENT-FRAME ' + CUR + ' arcs=' + got + ' S=' + S0); }
    }
    const sl = document.getElementById('slider');
    call('play#' + i, () => document.getElementById('play').onclick());
    call('slider-mid#' + i, () => sl._ls.input({ target: { value: String(Math.floor(ep.frames.length / 2)) } }));
    CUR = 'slider-mid-check#' + i;
    if (t !== Math.floor(ep.frames.length / 2)) console.log('SLIDER-VALUE-MISMATCH ep' + i + ': t=' + t);
    call('slider-over#' + i, () => sl._ls.input({ target: { value: '999999' } }));
    CUR = 'slider-over-check#' + i;
    if (t !== ep.frames.length - 1) console.log('SLIDER-CLAMP-FAIL ep' + i + ': t=' + t + ' want=' + (ep.frames.length - 1));
    call('pointerdown#' + i, () => sl._ls.pointerdown && sl._ls.pointerdown());
    call('pointerup#' + i, () => window._ls.pointerup && window._ls.pointerup());
    call('restart#' + i, () => document.getElementById('restart').onclick());
    CUR = 'restart-check#' + i;
    if (t !== 0) console.log('RESTART-FAIL ep' + i + ': t=' + t);
  }
  let tickThrows = 0;
  for (let k = 0; k < 60; k++) {
    CUR = 'tick#' + k;
    try { tick(k * 120); } catch (e) { tickThrows++; console.log('THROW tick#' + k + ': ' + e.message); }
  }
  const sVal = S;
  console.log('episodes=' + D.episodes.length + '  S=' + sVal);
  console.log('drawEpisode throws=' + throws + '  tick throws=' + tickThrows);
  console.log('frames=' + framesChecked + '  没画出 4 个球员的帧=' + framesNoAgents);
  console.log('errline=' + JSON.stringify(document.getElementById('errline').textContent));
  console.log('problems=' + problems.length);
  console.log(problems.slice(0, 30).join('\\n'));
  if (sVal) {
    const rs = arcLog.map(a => +(a.r / sVal).toFixed(3));
    const uniq = [...new Set(rs)].sort((a, b) => a - b);
    console.log('arc 半径(米) 取值: ' + uniq.slice(0, 12).join(', ') + (uniq.length > 12 ? ' …共' + uniq.length + '种' : ''));
    console.log('arc 次数=' + arcLog.length);
  }
})();
`;

vm.runInThisContext(js + probe, { filename: 'live_view.html.js' });
