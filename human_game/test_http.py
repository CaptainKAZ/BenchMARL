#!/usr/bin/env python3
"""人机大战 · host 侧 HTTP 端到端冒烟（纯标准库）

用法:  python3 test_http.py [base_url]      # 默认 http://127.0.0.1:8766
覆盖:  /api/meta、/api/reset（含陪练模式 noBots）、/api/step（吸引→蓄力→松开出手）、换角色
说明:  "带 bot" 的抢投很容易被判攻方犯规（正面高速碰撞=主动方，这是游戏难度的一部分），
       因此把"出手链路"的硬断言放在陪练模式下做；带 bot 的尝试只做信息性统计。
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"


def api(path, payload=None, timeout=20):
    url = BASE + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def play_into_spot(spot, tx=None, waypoint_y=-1.0, max_steps=200):
    """按住吸引 A1 进入投篮圈（新语义：按住 = 跟随，不蓄力）；调用方需已 reset。
    返回 (state, steps, done_code)"""
    aim_x = spot[0] if tx is None else tx
    st = None
    for i in range(1, max_steps + 1):
        aim_y = waypoint_y if i <= 25 else spot[1]
        st = api("/api/step", {"x": aim_x, "y": aim_y, "down": True})
        if st.get("done"):
            return st, None, (st.get("outcome") or {}).get("code")
        a1 = st["pos"][0]
        if ((a1[0] - spot[0]) ** 2 + (a1[1] - spot[1]) ** 2) ** 0.5 <= 0.25:
            return st, i, None
    return st, None, None


def main():
    checks = []

    def ck(name, cond, info=""):
        checks.append((name, bool(cond), info))
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    meta = api("/api/meta")
    ck("meta 可用", meta.get("ok") and meta.get("spotR") == 0.9, str(meta))

    # ---- 阶段1：陪练模式（noBots）—— 出手链路硬断言 ----
    st0 = api("/api/reset", {"role": "A1", "noBots": True})
    ck("reset(A1, 陪练模式)", st0.get("ok") and st0.get("noBots") is True and st0.get("step") == 0)
    spot = st0["spot"]
    print(f"     spot=({spot[0]:.2f},{spot[1]:.2f})  A1 起点=({st0['pos'][0][0]:.2f},{st0['pos'][0][1]:.2f})")

    st, into, code = play_into_spot(spot)
    ck("按住吸引进圈（陪练）", into is not None,
       f"用 {into} 步" + (f"（被码 {code} 打断）" if code else ""))
    if into is not None:
        a1pos = st["pos"][0]
        dist = ((a1pos[0] - spot[0]) ** 2 + (a1pos[1] - spot[1]) ** 2) ** 0.5
        ck("进圈时在圈内", dist <= 0.9 + 1e-6, f"dist={dist:.3f}")
        ck("按住期间不蓄力（新语义）",
           (st["charge"]["frames"] or 0) == 0 and (st.get("progress") or 0) <= 1e-9,
           f"charge={st['charge']['frames']} prog={st.get('progress')}")
        # 松手 = 圈内刹车蓄力 → 读满自动出手
        max_ch, done_code, st2 = 0, None, st
        for _ in range(25):
            s2 = api("/api/step", {"x": spot[0], "y": spot[1], "down": False})
            max_ch = max(max_ch, (s2.get("charge") or {}).get("frames") or 0)
            st2 = s2
            if s2.get("done"):
                done_code = (s2.get("outcome") or {}).get("code")
                break
        ck("松手后读条累计到 >=8", max_ch >= 8, f"max={max_ch}")
        ck("读满自动出手终局（码 1 命中 / 11 被盖）", done_code in (1, 11), f"code={done_code}")

    # ---- 阶段2：带 bot 抢投（信息性，不作断言）----
    print("\n     [信息] 带 bot 正常模式直冲投篮点：")
    codes = []
    for attempt, off in enumerate([0.0, 2.0, -2.0], start=1):
        st = api("/api/reset", {"role": "A1", "noBots": False})
        spot = st["spot"]
        tx = max(-3.7, min(3.7, spot[0] + off))
        _, into, code = None, None, None
        for i in range(1, 201):
            aim_y = -1.0 if i <= 25 else spot[1]
            st = api("/api/step", {"x": tx, "y": aim_y, "down": True})
            if st.get("done"):
                code = (st.get("outcome") or {}).get("code")
                break
            a1 = st["pos"][0]
            if ((a1[0] - spot[0]) ** 2 + (a1[1] - spot[1]) ** 2) ** 0.5 <= 0.25:
                into = i
                break
        codes.append(code if code is not None else "INTO_SPOT")
        print(f"       尝试{attempt}（进场 x={tx:+.1f}）: " +
              (f"进圈于第 {into} 步" if into else f"码 {code}"))
    ck("[信息] 带 bot 尝试至少留下有效终局/进圈记录", len(codes) == 3, f"结果={codes}")

    # ---- 阶段3：换角色（D2）----
    st = api("/api/reset", {"role": "D2"})
    ck("切换到 D2", st.get("ok") and st.get("role") == "D2" and st.get("humanIdx") == 3)
    st = api("/api/step", {"x": 0, "y": 0, "down": True})
    ck("D2 可操作", st.get("ok") and st.get("step") == 1)

    # 归还默认（带 bot）
    api("/api/reset", {"role": "A1", "noBots": False})
    return finish(checks)


def finish(checks):
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n==== HTTP 冒烟 {n_pass}/{len(checks)} 通过 ====")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
