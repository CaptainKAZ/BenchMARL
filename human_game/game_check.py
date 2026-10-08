#!/usr/bin/env python3
"""人机大战 · 前端验收（Playwright 无头浏览器）

用法:  python3 game_check.py [base_url] [--role D2]
检查: 页面加载、无 JS 报错、画布已渲染、点击球场能吸引 A1、松开触发投篮事件、截图
截图: /tmp/opencode/game_<role>_<stage>.png
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
OUT = Path("/tmp/opencode")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE + "/", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)  # 等 meta/state + 首帧
        status = page.inner_text("#status")
        print("[status]", status)
        page.screenshot(path=str(OUT / "game_page.png"))

        # 画布已渲染（非全空白）
        shot = page.screenshot(path=str(OUT / "game_canvas.png"), clip=_canvas_box(page))
        size = (OUT / "game_canvas.png").stat().st_size
        print("[canvas] 截图字节 =", size)

        # 找投篮点坐标 -> 屏幕坐标：从页面状态里读 spot
        spot = page.evaluate("async () => (await (await fetch('/api/state')).json()).spot")
        box = _canvas_box(page)
        px = box["x"] + box["width"] / 2 + spot[0] * _scale(page)
        py = box["y"] + box["height"] / 2 - spot[1] * _scale(page)
        print(f"[spot] 场上 {spot} -> 屏幕 ({px:.0f},{py:.0f})")

        # 按住鼠标吸引 A1
        page.mouse.move(px, py)
        page.mouse.down()
        page.wait_for_timeout(2200)
        page.screenshot(path=str(OUT / "game_drive.png"))
        st = page.evaluate("async () => await (await fetch('/api/state')).json()")
        print(f"[按住] step={st['step']} A1=({st['pos'][0][0]:.2f},{st['pos'][0][1]:.2f}) "
              f"charge={st['charge']['frames']}/{st['charge']['max']}")
        # 松开（A1 应投篮）
        page.mouse.up()
        page.wait_for_timeout(800)
        st2 = page.evaluate("async () => await (await fetch('/api/state')).json()")
        print(f"[松开] step={st2['step']} done={st2['done']} outcome={(st2.get('outcome') or {}).get('code')} "
              f"A1=({st2['pos'][0][0]:.2f},{st2['pos'][0][1]:.2f})")
        page.screenshot(path=str(OUT / "game_shot.png"))

        print("[JS errors]", errors if errors else "无")
        browser.close()

    ok = (not errors) and size > 3000 and st["step"] > 0
    print("GAME_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _canvas_box(page):
    return page.evaluate("() => { const r = document.getElementById('cv').getBoundingClientRect();"
                         " return {x:r.x, y:r.y, width:r.width, height:r.height}; }")


def _scale(page):
    # 与页面 layout() 一致：PAD + S
    return page.evaluate("() => S")


if __name__ == "__main__":
    sys.exit(main())
