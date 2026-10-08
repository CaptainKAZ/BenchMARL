#!/usr/bin/env python3
"""
BenchMARL/liveview/live_html_check.py —— 用无头浏览器（Playwright）验收 live_view.html

用法：
    python3 BenchMARL/liveview/live_html_check.py                     # 默认测 http://127.0.0.1:8765/（面板服务）
    LIVE_CHECK_BASE=http://127.0.0.1:8766/ python3 ...       # 指定地址
    （面板没起时会自动起一个临时静态服务兜底，只测观察窗部分）

产物：
    liveview/live_check_full.png（默认 .opencode/，可用 LIVE_CHECK_SHOTS 覆盖）     整页截图
    liveview/live_check_canvas.png（默认 .opencode/）   球场画布截图（看球员/读条/遮挡圈）
退出码：0 = 全通过；1 = 有 FAIL；2 = 环境缺 Playwright/浏览器

检查项（逐条打印 PASS/FAIL）：
  ① 页面加载、画布初始化、画布非空、4 名球员颜色像素存在（A1 橙 / D1 蓝）
  ② 10 个回合 chip、#errline 为空、状态行已更新
  ③ 动画循环在推进（rAF 没死）
  ④ 播放/暂停按钮文案切换、重播归零
  ⑤ 拖进度条：中场/越界后画面仍有球员；越界被夹到本局末帧；进度条 max = 本局帧数-1
  ⑥ 点击 chip 切局、→ 键切局
  ⑦ 无 JS 报错（pageerror / console.error）
"""
import json
import os
import sys
import threading
import time
import tempfile
from pathlib import Path

OPC = Path(__file__).resolve().parent          # BenchMARL/liveview/
ROOT = OPC.parent.parent                         # 仓库根（含 BenchMARL/outputs）
SHOTS = Path(os.environ.get("LIVE_CHECK_SHOTS", str(ROOT / ".opencode")))
SHOTS.mkdir(parents=True, exist_ok=True)
RESULTS = []


def check(name, ok, extra=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""), flush=True)


def http_ok(url, timeout=1.5):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start_temp_server():
    """面板服务没起时的兜底：起一个只含观察窗素材的临时静态服务。"""
    import http.server
    import socketserver

    port = int(os.environ.get("LIVE_CHECK_TMP_PORT", "8766"))
    tmp = Path(tempfile.mkdtemp(prefix="lvcheck-"))
    assets = {
        "index.html": OPC / "live_view.html",
        "live_view.html": OPC / "live_view.html",
        "chart.umd.min.js": OPC / "chart.umd.min.js",
        "live_env.json": ROOT / "BenchMARL" / "outputs" / "live" / "live_env.json",
    }
    for name, target in assets.items():
        if target.exists():
            (tmp / name).symlink_to(target)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp), **kw)

        def log_message(self, *a):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, tmp, f"http://127.0.0.1:{port}/"


SNAP_JS = """() => {
  const c = document.getElementById('big'), X = c.getContext('2d');
  const d = X.getImageData(0, 0, c.width, c.height).data;
  let nonBg = 0, a1 = 0, d1 = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    if (Math.abs(r-251) > 6 || Math.abs(g-251) > 6 || Math.abs(b-247) > 6) nonBg++;
    if (Math.abs(r-230) < 26 && Math.abs(g-85) < 26 && Math.abs(b-13) < 26) a1++;
    if (Math.abs(r-49) < 26 && Math.abs(g-130) < 26 && Math.abs(b-189) < 26) d1++;
  }
  const chips = [...document.querySelectorAll('#eps .chip')];
  return {
    cw: c.width, ch: c.height, nonBg, a1, d1,
    play: document.getElementById('play').textContent.trim(),
    tlab: document.getElementById('tlab').textContent.trim(),
    sliderMax: +document.getElementById('slider').max,
    chips: chips.length,
    selIdx: chips.findIndex(x => x.classList.contains('sel')),
    errline: document.getElementById('errline').textContent,
    statusline: document.getElementById('statusline').textContent,
  };
}"""


def main():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print("缺 Playwright：", e)
        print("安装： pip install playwright && python3 -m playwright install chromium")
        return 2

    base = os.environ.get("LIVE_CHECK_BASE", "http://127.0.0.1:8765/")
    httpd = tmpdir = None
    if not http_ok(base + "status.json"):
        print(f"[info] {base} 不是面板服务，改用临时静态服务")
        httpd, tmpdir, base = start_temp_server()
        if not http_ok(base):
            print("临时静态服务没起来")
            return 2
    print("被测地址 =", base)
    if not base.endswith("/"):
        base += "/"

    live_env = ROOT / "BenchMARL" / "outputs" / "live" / "live_env.json"
    max_ep_len = 0
    if live_env.exists():
        try:
            payload = json.loads(live_env.read_text())
            max_ep_len = max((len(e["frames"]) for e in payload.get("episodes", [])), default=0)
        except Exception:
            pass

    js_errors = []
    code = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
                "--force-device-scale-factor=1", "--hide-scrollbars",
            ])
            ctx = browser.new_context(viewport={"width": 1680, "height": 950}, device_scale_factor=1)
            page = ctx.new_page()
            page.on("pageerror", lambda e: js_errors.append("pageerror: " + str(e)))
            page.on("console", lambda m: js_errors.append("console.error: " + m.text) if m.type == "error" else None)

            page.goto(base, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_selector("#play", timeout=8000)
            try:
                page.wait_for_function(
                    "() => { const c=document.getElementById('big'); "
                    "return c && c.width>0 && document.getElementById('eps').children.length>0; }",
                    timeout=15000)
            except Exception:
                pass

            check("页面加载（有 #play 按钮）", True)
            check("有观察窗数据 live_env.json", live_env.exists(),
                  "" if live_env.exists() else "(缺文件，观察窗部分会失败)")

            s = page.evaluate(SNAP_JS)
            check("画布已初始化", s["cw"] > 100 and s["ch"] > 100, f"canvas={s['cw']}x{s['ch']}")
            check("画布非空", s["nonBg"] > 2000, f"非背景像素={s['nonBg']}")
            check("球员画出（A1 橙 + D1 蓝）", s["a1"] > 30 and s["d1"] > 30, f"A1={s['a1']} D1={s['d1']}")
            check("10 局 chip", s["chips"] == 10, f"chips={s['chips']}")
            check("#errline 为空", not s["errline"], s["errline"])
            check("状态行已更新", s["statusline"] and s["statusline"] != "连接中…", s["statusline"])

            t1 = s["tlab"]
            time.sleep(2.6)
            s2 = page.evaluate(SNAP_JS)
            check("动画循环在推进", t1 != s2["tlab"], f"{t1} -> {s2['tlab']}")

            p0 = page.evaluate(SNAP_JS)["play"]
            page.click("#play")
            p1 = page.evaluate(SNAP_JS)["play"]
            page.click("#play")
            p2 = page.evaluate(SNAP_JS)["play"]
            check("播放按钮切换文案", p0 in ("▶ 播放", "⏸ 暂停") and p1 != p0 and p2 == p0,
                  f"{p0} -> {p1} -> {p2}")

            page.evaluate("() => document.getElementById('restart').click()")
            r1 = page.evaluate(SNAP_JS)
            check("重播归零", r1["tlab"].startswith("step 0/"), r1["tlab"])

            page.evaluate("""() => { const sl=document.getElementById('slider');
                sl.value=String(Math.max(1, Math.floor(+sl.max*0.6)));
                sl.dispatchEvent(new Event('input',{bubbles:true})); }""")
            d1s = page.evaluate(SNAP_JS)
            check("拖进度条后画面仍有球员", d1s["a1"] > 30 and d1s["d1"] > 30,
                  f"A1={d1s['a1']} D1={d1s['d1']} tlab={d1s['tlab']}")

            page.evaluate("""() => { const sl=document.getElementById('slider');
                sl.value='999999';
                sl.dispatchEvent(new Event('input',{bubbles:true})); }""")
            d2s = page.evaluate(SNAP_JS)
            check("进度条越界被夹住且仍有球员", d2s["a1"] > 30, f"tlab={d2s['tlab']} A1={d2s['a1']}")
            if max_ep_len:
                check("进度条 max 按当前局（不是全局最长局）",
                      d2s["sliderMax"] + 1 <= max_ep_len,
                      f"max={d2s['sliderMax']} 全局最长局={max_ep_len}")

            page.evaluate("() => document.querySelectorAll('#eps .chip')[4].click()")
            c1 = page.evaluate(SNAP_JS)
            check("点击 chip 切局", c1["selIdx"] == 4, f"selIdx={c1['selIdx']}")
            page.keyboard.press("ArrowRight")
            c2 = page.evaluate(SNAP_JS)
            check("→ 键切局", c2["selIdx"] == 5, f"selIdx={c2['selIdx']}")

            # ⑧ 终局横幅：三行都不溢出、在画布内、内容含结束原因/奖励/肇事方
            banner = page.evaluate("""() => {
                const codes = [2, 13, 15, 3, 5, 14];
                const i = D.episodes.findIndex(e => codes.indexOf(e.code) >= 0);
                selectEp(i >= 0 ? i : 0); playing = false; setPlayLabel();
                const ep = curEp(); t = ep.frames.length - 1; setSlider(ep); render();
                if (!lastBanner) return null;
                const b = lastBanner;
                return {outcome: ep.outcome, offender: ep.offender, inner: b.inner,
                        w: [b.w1, b.w2, b.w3], box: [b.bxp, b.byp, b.bw, b.bh],
                        cw: cvs.width, ch: cvs.height, lines: [b.line1, b.line2, b.line3],
                        sizes: [b.s1, b.s2, b.s3], ys: [b.y1, b.y2, b.y3]};
            }""")
            if banner:
                fits = all(w <= banner["inner"] + 0.5 for w in banner["w"])
                inside = banner["box"][1] >= 0 and banner["box"][1] + banner["box"][3] <= banner["ch"] + 0.5
                check("终局横幅不溢出（三行都放得下）", fits,
                      "inner=%.0f w=%s sizes=%s" % (banner["inner"], ["%.0f" % w for w in banner["w"]], banner["sizes"]))
                check("终局横幅在画布内", inside, "box=%s canvas=%sx%s" % (banner["box"], banner["cw"], banner["ch"]))
                # 纵向：首行上沿 ≤ 卡片顶 +? 、末行下沿 ≤ 卡片底 −?（上下各留 ≥1px）
                top, bot = banner["box"][1], banner["box"][1] + banner["box"][3]
                y1, y2, y3 = banner["ys"]; s1, s2, s3 = banner["sizes"]
                vmarg_ok = (y1 - 0.55*s1 >= top + 1) and (y2 + 0.55*s2 <= bot - 1) and (y3 + 0.55*s3 <= bot - 1)
                check("终局横幅三行都落在卡片内（纵向 margin）", vmarg_ok,
                      "y1..3=%s 行占比=%.1f/%.1f/%.1f 卡片=[%.1f,%.1f] 上留=%.1f 下留=%.1f"
                      % (["%.1f" % y for y in banner["ys"]], 0.55*s1, 0.55*s2, 0.55*s3, top, bot,
                         (y1 - 0.55*s1) - top, bot - (y3 + 0.55*s3)))
                check("终局横幅内容完整", banner["lines"][0].startswith("本局结束") and "A1" in banner["lines"][1]
                      and ("肇事" in banner["lines"][2] or "剩" in banner["lines"][2]),
                      " | ".join(banner["lines"]) + " | offender=%s" % (banner["offender"],))
            else:
                check("终局横幅不溢出（三行都放得下）", False, "lastBanner 为空（没渲染到终局帧）")

            full_png = SHOTS / "live_check_full.png"
            canvas_png = SHOTS / "live_check_canvas.png"
            page.screenshot(path=str(full_png))
            box = page.evaluate("""() => { const r=document.getElementById('big').getBoundingClientRect();
                return {x:r.x, y:r.y, width:r.width, height:r.height}; }""")
            page.screenshot(path=str(canvas_png), clip=box)
            print("截图 =", full_png, canvas_png)

            # ⑨ 像素实测（画布坐标系内直接 getImageData）：横幅文字与卡片上下边的实际间距
            try:
                px = page.evaluate("""() => {
                    if (!lastBanner) return null;
                    const b = lastBanner;
                    const x = Math.round(b.bxp), y = Math.round(b.byp);
                    const w = Math.round(b.bw), h = Math.round(b.bh);
                    const c2 = document.getElementById('big').getContext('2d');
                    const d = c2.getImageData(x, y, w, h).data;
                    const insetX = 16;                       // 避开圆角
                    const rows = [];
                    for (let j = 0; j < h; j++) {
                        let c = 0;
                        for (let i = insetX; i < w - insetX; i++) {
                            const k = (j * w + i) * 4;
                            if (d[k] > 228 && d[k+1] > 228 && d[k+2] > 228) c++;   // 白字
                        }
                        rows.push(c);
                    }
                    let first = -1, last = -1;
                    for (let j = 0; j < h; j++) if (rows[j] > 3) { if (first < 0) first = j; last = j; }
                    return {h: h, first: first, last: last, top: first, bot: h - 1 - last};
                }""")
                if px and px["first"] >= 0:
                    check("像素实测：横幅文字上下留白 ≥6px", px["top"] >= 6 and px["bot"] >= 6,
                          "卡片高=%d 上留=%dpx 下留=%dpx（画布坐标实测）" % (px["h"], px["top"], px["bot"]))
                else:
                    check("像素实测：横幅文字上下留白 ≥6px", False, "没取到白字 ink（first=%s）" % (px and px["first"],))
            except Exception as e:
                check("像素实测：横幅文字上下留白 ≥6px", False, "分析失败: %r" % (e,))

            check("无 JS 报错", not js_errors, " | ".join(js_errors[:4]))
            browser.close()
    except Exception as e:
        check("运行异常", False, repr(e))
    finally:
        if httpd:
            try:
                httpd.shutdown()
            except Exception:
                pass

    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} 通过 ====")
    if bad:
        print("失败项：" + "；".join(bad))
    code = 0 if not bad else 1
    return code


if __name__ == "__main__":
    sys.exit(main())
