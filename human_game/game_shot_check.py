#!/usr/bin/env python3
"""人机大战 · 前端投篮全流程截图（Playwright 无头浏览器，陪练模式 + 绕人策略）

用法:  LD_LIBRARY_PATH=/tmp/opencode/pwdeps/root/usr/lib/x86_64-linux-gnu \
       python3 game_shot_check.py [base_url]
说明:  陪练模式下其他球员站桩，但高速撞上去仍会判攻方犯规 —— 脚本复刻"人类小心操作"：
       前方 1.6m 内有站桩球员就侧向绕行（贴近时慢速侧移），否则全速指向投篮点。
产出:  /tmp/opencode/game_charge.png（蓄满读条瞬间）、game_banner.png（终局横幅）
"""
import math
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
OUT = Path("/tmp/opencode")


def main():
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE + "/", wait_until="domcontentloaded")
        page.wait_for_timeout(1800)
        page.click("#btnBots")           # 进入陪练模式
        page.wait_for_timeout(500)

        def st():
            return page.evaluate("async () => await (await fetch('/api/state')).json()")

        def to_screen(p):
            box = page.evaluate("() => { const r = document.getElementById('cv').getBoundingClientRect();"
                                " return {x:r.x, y:r.y, w:r.width, h:r.height}; }")
            s = page.evaluate("() => S")
            return (box["x"] + box["w"] / 2 + p[0] * s, box["y"] + box["h"] / 2 - p[1] * s)

        def aim_point(s):
            """返回本步指针应指向的点（限速巡航 2.1m/s；前方有障碍则侧绕/贴身慢移）"""
            a1, spot = s["pos"][0], s["spot"]
            vx, vy = spot[0] - a1[0], spot[1] - a1[1]
            n = math.hypot(vx, vy) + 1e-6
            ux, uy = vx / n, vy / n
            blocker, bd = None, 1e9
            for j in (1, 2, 3):
                p = s["pos"][j]
                dx, dy = p[0] - a1[0], p[1] - a1[1]
                d = math.hypot(dx, dy)
                forward = (dx * ux + dy * uy) / (d + 1e-6)
                if forward > 0.4 and d < bd:
                    blocker, bd = p, d
            if blocker and bd < 1.6:
                px, py = -uy, ux                       # 垂直于前进方向
                side = 1.0 if (blocker[0] - a1[0]) * (-uy) + (blocker[1] - a1[1]) * ux < 0 else -1.0
                if bd < 0.7:                            # 已贴脸：侧向慢移
                    return [a1[0] + px * side * 0.12, a1[1] + py * side * 0.12]
                return [a1[0] + (blocker[0] + px * side * 1.3 - a1[0]) * 0.35,
                        a1[1] + (blocker[1] + py * side * 1.3 - a1[1]) * 0.35]
            return [a1[0] + ux * min(n, 0.6), a1[1] + uy * min(n, 0.6)]   # ≤2.1 m/s 巡航

        def clearance(s):
            """A1 → 投篮点 直线走廊上，其他人到线段的最近距离（米）"""
            a1, spot = s["pos"][0], s["spot"]
            vx, vy = spot[0] - a1[0], spot[1] - a1[1]
            L2 = vx * vx + vy * vy + 1e-9
            best = 1e9
            for j in (1, 2, 3):
                px, py = s["pos"][j]
                t = max(0.0, min(1.0, ((px - a1[0]) * vx + (py - a1[1]) * vy) / L2))
                cx, cy = a1[0] + t * vx, a1[1] + t * vy
                best = min(best, math.hypot(px - cx, py - cy))
            return best

        import os
        dbg = bool(os.environ.get("SHOT_DEBUG"))

        charged, last = False, None
        for episode in range(1, 9):
            page.mouse.up()
            page.wait_for_timeout(3200)           # 等页面在 done 后 2.4s 的自动重开走完，避免它把 down 状态清掉
            page.click("#btnReset")
            page.wait_for_timeout(600)
            s = st()
            if s["done"]:
                page.click("#btnReset"); page.wait_for_timeout(600); s = st()
            # 选一条走廊干净的线路再开跑（陪练模式下其他人站桩，选好就一劳永逸）
            for _ in range(8):
                if clearance(s) > 0.95:
                    break
                page.click("#btnReset"); page.wait_for_timeout(500); s = st()
            if dbg:
                print(f"  -- 开局: noBots={s['noBots']} step={s['step']} clearance={clearance(s):.2f} "
                      f"pos={[[round(v, 2) for v in p] for p in s['pos']]}")
            aim = aim_point(s)
            page.mouse.move(*to_screen(aim))
            page.mouse.down()
            reason = None
            charged, shot = False, False
            for t in range(420):
                s = st(); last = s
                if s["done"]:
                    reason = f"码 {(s.get('outcome') or {}).get('code')}"
                    break
                a1, spot = s["pos"][0], s["spot"]
                d_spot = math.hypot(a1[0] - spot[0], a1[1] - spot[1])
                if d_spot <= 0.30:
                    # 新语义：松开鼠标 = 圈内刹车蓄力（按住才是跟随）→ 读满自动出手
                    page.mouse.up()
                    for _ in range(50):
                        s = st(); last = s
                        if s["charge"]["frames"] >= 8 and not charged:
                            charged = True
                            page.screenshot(path=str(OUT / "game_charge.png"))
                        if s["done"]:
                            shot = True
                            reason = f"码 {(s.get('outcome') or {}).get('code')}"
                            break
                        page.wait_for_timeout(110)
                    break
                aim = aim_point(s)
                if dbg and t % 15 == 0:
                    ds = {n: round(math.hypot(s["pos"][j][0] - a1[0], s["pos"][j][1] - a1[1]), 2)
                          for j, n in enumerate(("A1", "A2", "D1", "D2"))}
                    print(f"    t={t:3d} step={s['step']:3d} A1=({a1[0]:+.2f},{a1[1]:+.2f}) "
                          f"dSpot={d_spot:.2f} charge={s['charge']['frames']:2d} "
                          f"dist={ {k: v for k, v in ds.items() if k != 'A1'} } aim=({aim[0]:+.2f},{aim[1]:+.2f}) spot=({spot[0]:+.2f},{spot[1]:+.2f})")
                page.mouse.move(*to_screen(aim))
                page.wait_for_timeout(110)
            page.mouse.up()
            page.wait_for_timeout(800)
            s = st()
            print(f"[第{episode}局] charged={charged} shot={shot} done={s['done']} "
                  f"码={(s.get('outcome') or {}).get('code')} score={s['score']}")
            if shot:
                break
            print(f"    本局被 {reason} 终止，重开再来")

        s = st()
        page.screenshot(path=str(OUT / "game_banner.png"))
        oc = s.get("outcome") or {}
        print(f"[终局] done={s['done']} outcome={oc.get('code')} {oc.get('name')} "
              f"humanWin={oc.get('humanWin')} score={s['score']}")
        print("[JS errors]", errors if errors else "无")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
