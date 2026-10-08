"""调试：终局后 reset 的第一次走位是否正常（test_game[4] 失败原因）"""
import sys
sys.path.insert(0, "/home/vscode/workspace/BenchMARL")
sys.path.insert(0, "/home/vscode/workspace/BenchMARL/human_game")
from game_server import Game, pick_ckpt  # noqa

ck = pick_ckpt(__import__("pathlib").Path("/home/vscode/workspace/BenchMARL/outputs"))
g = Game(ck, human="A1", seed=7, no_bots=True)
st = g.state_json()
print("episode1 spot =", [round(v, 2) for v in st["spot"]], "A1 =", [round(v, 2) for v in st["pos"][0]])
spot = st["spot"]

# episode 1：走满 → 松开出手
for i in range(160):
    st = g.step(spot[0], spot[1], down=True)
    if st["charge"]["ready"]:
        break
st = g.step(0.0, 0.0, down=False)
print("ep1 done =", st["done"], "code =", st["outcome"]["code"] if st["done"] else None,
      "steps =", st["step"])

# reset 后逐帧观察
r = g.reset("A1")
print("reset -> A1 =", [round(v, 2) for v in r["pos"][0]], "spot =", [round(v, 2) for v in r["spot"]],
      "done =", r["done"], "charge =", r["charge"])
for i in range(170):
    st = g.step(spot[0], spot[1], down=True)
    sc = g.env.scenario
    a1 = sc.world.agents[0]
    if i < 14 or i % 10 == 9:
        print(f"  f{i:03d} A1={[round(v,2) for v in st['pos'][0]]} d_spot="
              f"{((st['pos'][0][0]-spot[0])**2+(st['pos'][0][1]-spot[1])**2)**0.5:.2f} "
              f"charge={st['charge']['frames']} prog={st['progress']} done={st['done']} "
              f"| vel={[round(v,2) for v in a1.state.vel[0].tolist()]} "
              f"delay={int(sc.delay_counter[0])} "
              f"outcome={st['outcome']['name'] if st['done'] else None}")
    if st["done"]:
        break
    if st["charge"]["ready"]:
        print(f"  -> ready at f{i}")
        break
print("DEBUG_DONE")
