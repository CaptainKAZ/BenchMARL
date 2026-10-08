#!/usr/bin/env python3
"""BenchMARL/liveview/mobile_html_check.py —— 面板移动端适配验收（Playwright）

用法：python3 BenchMARL/liveview/mobile_html_check.py      # 默认测 http://127.0.0.1:8765/
按语：三档视口各截图到 /tmp/opencode/mobile_<W>x<H>.png
样例：LD_LIBRARY_PATH=/tmp/opencode/chromelibs/root/usr/lib/x86_64-linux-gnu python3 ...
检查：手机竖屏 390x844（纵向堆叠）/ 手机横屏 844x390（两栏不裁切）/ 桌面 1600x900（不回归）
      —— 无横向溢出、画布已渲染且不超宽、曲线有数据、日志高度、触控目标≥34px、无 JS 报错。
"""
import json
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"

PROBE = """() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const r = el => { const b = el.getBoundingClientRect();
    return {x:Math.round(b.x), y:Math.round(b.y), w:Math.round(b.width), h:Math.round(b.height)}; };
  const px = (cv, pred) => { try { const x = cv.getContext('2d');
      const d = x.getImageData(0,0,cv.width,cv.height).data; let n=0;
      for (let i=0;i<d.length;i+=97) if (pred(d[i],d[i+1],d[i+2])) n++; return n; } catch(e){ return -1; } };
  const cvs = document.getElementById('big'), cc = document.getElementById('curveChart');
  return {
    vw, vh, scrollW: document.documentElement.scrollWidth,
    status: (document.getElementById('statusline').textContent||'').trim().slice(0,52),
    canvas: {rect:r(cvs), px:cvs.width+'x'+cvs.height, drawn: px(cvs, (a,b,c)=> a<250||b<250||c<250)},
    curve:  {rect:r(cc), px:cc.width+'x'+cc.height, ink: px(cc, (a,b,c)=> a>120||b>120||c>120),
             n:(typeof curveData!=='undefined'&&curveData&&curveData.points)?curveData.points.length:null},
    logbox: r(document.getElementById('logbox')),
    play:   r(document.getElementById('play')),
    side:   r(document.getElementById('side')),
    live:   r(document.getElementById('pane-live')),
    curvePane: r(document.getElementById('pane-curve')),
    chips:  document.querySelectorAll('#eps .chip').length,
  };
}"""

CASES = [
    ("手机竖屏 390x844", 390, 844, True),
    ("手机横屏 844x390", 844, 390, True),
    ("桌面 1600x900", 1600, 900, False),
]

fails = []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    for name, w, h, mobile in CASES:
        errors = []
        ctx = b.new_context(viewport={"width": w, "height": h}, is_mobile=mobile,
                            has_touch=mobile, device_scale_factor=(3 if mobile else 1))
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))
        pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        pg.goto(URL, wait_until="load")
        pg.wait_for_timeout(3500)
        d = pg.evaluate(PROBE)
        print(f"\n---- {name} ----")
        print(json.dumps(d, ensure_ascii=False))
        checks = {
            "无横向溢出": d["scrollW"] <= d["vw"] + 1,
            "画布已渲染": d["canvas"]["drawn"] > 30 and int(d["canvas"]["px"].split("x")[0]) > 150,
            "画布不超宽": d["canvas"]["rect"]["w"] <= d["vw"] + 1,
            "曲线可见且有数据": d["curve"]["rect"]["h"] >= 90 and d["curve"]["ink"] > 15 and (d["curve"]["n"] or 0) > 0,
            "日志区高度足够": d["logbox"]["h"] >= 90,
            "按钮触控目标>=34": (d["play"]["h"] >= 34) if mobile else True,
            "面板不超视口高": (d["logbox"]["y"] + d["logbox"]["h"] <= d["vh"] + 60) if h < w else True,
            "10 局列表存在": d["chips"] == 10,
            "无 JS 报错": len(errors) == 0,
        }
        if mobile and h > w:  # 仅竖屏要求纵向堆叠（观察窗在曲线之上）
            checks["窄屏纵向堆叠"] = (d["live"]["y"] + d["live"]["h"]) <= d["curvePane"]["y"] + 2
        for k, v in checks.items():
            print(("PASS  " if v else "FAIL  ") + k)
            if not v:
                fails.append(f"{name}: {k}")
        pg.screenshot(path=f"/tmp/opencode/mobile_{w}x{h}.png", full_page=mobile and w < h)
        if errors:
            print("ERRORS:", json.dumps(errors[:6], ensure_ascii=False))
        ctx.close()
    b.close()

print("\n==== 汇总 ====")
print("ALL PASS" if not fails else "FAILS:\n  " + "\n  ".join(fails))
sys.exit(0 if not fails else 1)
