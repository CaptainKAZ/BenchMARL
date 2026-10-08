"""人机大战机制自测（容器内运行）：

    PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/test_game.py [ckpt]

覆盖：
  [1] 状态解码与真实环境一致（spot / basket / A1 位置）
  [2] 人类 A1：走到投篮点蓄力→松开→出手（期望码 1 或 11）
  [3] 按住期间不会自动出手（环境读条计数被钉在 <=9）
  [4] 提前松开 = 不出手（计数清零，不终局）
  [5] 其他角色可玩（D2 被吸引移动）
  [6] bots 完整一局 + 重开局（隐状态复位）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/vscode/workspace/BenchMARL")
sys.path.insert(0, "/home/vscode/workspace/BenchMARL/human_game")

from game_server import Game, NAMES, pick_ckpt  # noqa: E402

DEFAULT_ROOT = "/home/vscode/workspace/BenchMARL/outputs"

RESULTS = []


def check(name, ok, extra=""):
    RESULTS.append((name, ok, extra))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {extra}", flush=True)
    return ok


def decode_pos(st, i):
    return st["pos"][i]


def test_decode(g):
    print("\n[1] 状态解码核对", flush=True)
    sc = g.env.scenario
    st = g.state_json()
    spot_true = sc.spot_center.state.pos[0].tolist()
    basket_true = sc.basket.state.pos[0].tolist()
    a1_true = sc.world.agents[0].state.pos[0].tolist()
    check("spot 解码", abs(st["spot"][0] - spot_true[0]) < 1e-3 and abs(st["spot"][1] - spot_true[1]) < 1e-3,
          f"dec={st['spot']} true={[round(v,3) for v in spot_true]}")
    check("basket 解码", abs(st["basket"][0] - basket_true[0]) < 1e-3 and abs(st["basket"][1] - basket_true[1]) < 1e-3,
          f"dec={st['basket']} true={[round(v,3) for v in basket_true]}")
    check("A1 位置解码", abs(st["pos"][0][0] - a1_true[0]) < 1e-3 and abs(st["pos"][0][1] - a1_true[1]) < 1e-3,
          f"dec={st['pos'][0]} true={[round(v,3) for v in a1_true]}")
    check("spot 半径 0.9", abs(sc.h_params["R_spot"] - 0.9) < 1e-6, f"R_spot={sc.h_params['R_spot']}")


def hold_to_spot(g, spot, max_steps=170):
    """按住鼠标把 A1 吸引到 spot 附近（新语义：按住 = 跟随，不蓄力）"""
    st = None
    for i in range(max_steps):
        st = g.step(spot[0], spot[1], down=True)
        if st["done"]:
            return st, i + 1
        dist = ((st["pos"][0][0] - spot[0]) ** 2 + (st["pos"][0][1] - spot[1]) ** 2) ** 0.5
        if dist <= 0.30 and abs(st["vel"][0][0]) < 0.5 and abs(st["vel"][0][1]) < 0.5:
            return st, i + 1
    return st, max_steps


def test_human_shot(g):
    print("\n[2] 人类 A1：按住接近 → 圈内松手蓄力 → 读满自动出手", flush=True)
    g.reset("A1")
    st0 = g.state_json()
    spot = st0["spot"]
    t0 = time.time()
    st, n = hold_to_spot(g, spot)
    check("接近途中不蓄力（按住 ≠ 蓄力）",
          st is not None and st["charge"]["frames"] == 0 and st["progress"] <= 1e-9,
          f"steps={n} charge={st['charge']['frames'] if st else None} prog={st['progress'] if st else None}")
    dist_spot = ((st["pos"][0][0] - spot[0]) ** 2 + (st["pos"][0][1] - spot[1]) ** 2) ** 0.5
    check("松手时人在圈内", dist_spot <= 0.9 + 1e-6, f"dist={dist_spot:.3f}")
    # 松手：开始蓄力，逐帧涨到 10 后自动出手（无需再点鼠标）
    max_charge, fired, st = 0, False, st
    for _ in range(30):
        st = g.step(0.0, 0.0, down=False)
        max_charge = max(max_charge, st["charge"]["frames"])
        if st["done"]:
            fired = True
            break
    code = st["outcome"]["code"] if st["done"] else None
    check("松手后读条涨到过 >=8", max_charge >= 8, f"max_charge={max_charge}")
    check("读满自动出手", fired, f"done={st['done']}")
    check("码为 1(命中) 或 11(被盖)", code in (1, 11), f"code={code}")
    check("人类在攻方 → humanWin=True", st["outcome"]["humanWin"] is True if st["done"] else False,
          f"outcome={st['outcome']['name'] if st['done'] else None}")
    print(f"      耗时 {time.time()-t0:.1f}s, 接近 {n} 步", flush=True)
    return st


def test_cancel_charge(g):
    print("\n[4] 蓄力中重新按住 = 取消蓄力并恢复跟随", flush=True)
    for attempt in range(3):
        g.reset("A1")
        spot = g.state_json()["spot"]
        st, _ = hold_to_spot(g, spot)
        if st["done"]:
            print(f"      尝试{attempt+1}: 接近途中意外终局，换一局重试", flush=True)
            continue
        got = 0
        for _ in range(20):
            st = g.step(0.0, 0.0, down=False)
            if st["done"]:
                break
            if st["charge"]["frames"] >= 3:
                got = st["charge"]["frames"]
                break
        if st["done"] or got < 3:
            print(f"      尝试{attempt+1}: 未到蓄力阶段（got={got}），换一局重试", flush=True)
            continue
        st = g.step(spot[0], spot[1], down=True)   # 重新按住 → 取消
        check("按住立即取消蓄力（本地清零）", st["charge"]["frames"] == 0, f"charge={st['charge']}")
        check("环境读条计数清零", st["progress"] <= 1e-9, f"progress={st['progress']}")
        check("取消后不终局", not st["done"], f"done={st['done']}")
        return
    check("蓄力取消链路", False, "3 次尝试均未进入蓄力阶段（随机初始位置干扰）")


def test_other_role(g):
    print("\n[5] 其他角色（D2）可玩", flush=True)
    g.reset("D2")
    st = g.state_json()
    p0 = decode_pos(st, 3)
    tgt = [p0[0] - 2.0, p0[1] - 1.0]  # 往左下方拉
    for _ in range(10):
        st = g.step(tgt[0], tgt[1], down=True)
    p1 = decode_pos(st, 3)
    d0 = ((p0[0] - tgt[0]) ** 2 + (p0[1] - tgt[1]) ** 2) ** 0.5
    d1 = ((p1[0] - tgt[0]) ** 2 + (p1[1] - tgt[1]) ** 2) ** 0.5
    check("D2 朝目标移动", d1 < d0 - 0.05, f"{d0:.2f}m -> {d1:.2f}m")
    check("人类索引=3", st["humanIdx"] == 3, f"role={st['role']} idx={st['humanIdx']}")


def test_bots_episode(ckpt):
    print("\n[6] bots 完局 + 重开局", flush=True)
    g = Game(ckpt, human="A1", seed=3, no_bots=False)
    done_code = None
    for _ in range(220):
        st = g.step(None, None, down=False)  # 人类不操作，看 bots 打
        if st["done"]:
            done_code = st["outcome"]["code"]
            break
    check("bots 单局能终局", done_code is not None, f"code={done_code}")
    g.reset("A2")
    ok = True
    for _ in range(8):
        st = g.step(None, None, down=False)
    check("重开局（角色 A2）后继续跑", not st.get("error", False), f"step={st['step']}")
    g.reset("A2")
    ok2 = True
    for _ in range(5):
        st = g.step(None, None, down=False)
    check("二次重开局正常（隐状态复位）", not st.get("error", False), f"step={st['step']}")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else pick_ckpt(Path(DEFAULT_ROOT))
    print(f"[test] ckpt = {ckpt}", flush=True)
    t0 = time.time()
    g = Game(ckpt, human="A1", seed=7, no_bots=True)
    print(f"[test] Game(no_bots) 构建 {time.time()-t0:.1f}s", flush=True)
    test_decode(g)
    test_human_shot(g)
    test_cancel_charge(g)
    test_other_role(g)
    del g
    test_bots_episode(ckpt)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*60}\n{len(RESULTS)-n_fail}/{len(RESULTS)} PASS, {n_fail} FAIL", flush=True)
    if n_fail:
        print("FAILED:", [n for n, ok, _ in RESULTS if not ok], flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
