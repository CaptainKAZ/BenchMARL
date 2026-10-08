"""给观察窗某一局的关键帧截图（用于人工核对：球员内部黑圆 / 终局肇事方圈）。

用法（宿主机，需面板在跑）：
  LD_LIBRARY_PATH=/tmp/opencode/chromelibs/root/usr/lib/x86_64-linux-gnu \
      python3 BenchMARL/liveview/snap_end_frame.py --ep 3 --at end --out .opencode/snap_ep3_end.png

参数：
  --ep   第几局（1 起，对应 chip 左侧编号）
  --at   帧位置：end（终局）/ mid（中间）/ 具体帧号
  --base 面板地址（默认 http://127.0.0.1:8765/）
"""
import argparse
import os

from playwright.sync_api import sync_playwright

LIBS = "/tmp/opencode/chromelibs/root/usr/lib/x86_64-linux-gnu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--at", default="end")
    ap.add_argument("--out", default=".opencode/snap_frame.png")  # 相对 CWD，建议指到 .opencode/
    ap.add_argument("--base", default="http://127.0.0.1:8765/")
    args = ap.parse_args()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, env=env)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.base, wait_until="domcontentloaded")
        page.wait_for_selector("#canvasbox canvas", timeout=15000)
        page.wait_for_function("() => (typeof D !== 'undefined') && D && D.episodes && D.episodes.length > 0",
                               timeout=15000)
        idx = max(0, args.ep - 1)
        page.evaluate(f"() => {{ selectEp({idx}); }}")
        page.evaluate("() => { playing = false; setPlayLabel(); }")
        if args.at == "end":
            expr = "() => { const ep = curEp(); t = ep.frames.length - 1; setSlider(ep); render(); }"
        elif args.at == "mid":
            expr = "() => { const ep = curEp(); t = Math.floor((ep.frames.length - 1) * 0.6); setSlider(ep); render(); }"
        else:
            expr = f"() => {{ const ep = curEp(); t = Math.min({int(args.at)}, ep.frames.length - 1); setSlider(ep); render(); }}"
        page.evaluate(expr)
        page.wait_for_timeout(400)
        info = page.evaluate(
            "() => { const ep = curEp(); return {outcome: ep.outcome, code: ep.code, offender: ep.offender,"
            " n: ep.frames.length, t, blockF: (typeof blockFactorA1 === 'function') ? blockFactorA1(ep.frames[t], ep) : null}; }"
        )
        print("[snap]", info)
        page.locator("#canvasbox canvas").screenshot(path=args.out)
        browser.close()
    print("[snap] wrote", args.out)


if __name__ == "__main__":
    main()
