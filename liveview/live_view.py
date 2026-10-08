#!/usr/bin/env python3
"""Live 观察窗 —— 每个训练迭代随机挑 N 局完整回合，写成前端可渲染的 JSON。

数据源（都带**精确** termination_reason）：
1) LiveViewCallback : 训练进程内回调（clear_restore.py 里 LIVE_VIEW=1 时挂上）。
   直接吃本轮 rollout 的 batch（`("next","agents","info","termination_reason")` 在 batch 里），
   不读 replay buffer（buffer 被 _get_excluded_keys 过滤，没有终局码）。
2) --from-ckpt : 离线用 checkpoint 的策略跑一小段真实 rollout，同样带终局码（验证前端用）。

JSON 每帧 19 个数：[A1x,A1y,A1vx,A1vy, A2..., D1..., D2..., progress, in_spot, t_remaining]
全局 state 布局（已归一化，见 layup.py:get_global_state）：
  0:2 A1 pos(/[W/2,L/2])  2:4 A1 vel(/v_max)  4 in_spot  5 读条进度
  6:10 A2(pos,vel)  10:14 D1  14:18 D2  18:20 spot  20:22 basket  22 t_remaining/t_limit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

import torch
import numpy as np

POS_DIV = (4.0, 7.5)   # W/2, L/2
V_DIV = 5.0            # v_max
T_LIMIT = 20.0         # t_limit
AGENT_OFFSETS = (0, 6, 10, 14)   # A1, A2, D1, D2
N_EPISODES = 10        # 每个迭代随机挑几局

# 精确终局码（layup_jit.py 里的注释为准）
CODE_NAMES = {
    1: "码1 · 投篮命中（获胜）",
    2: "码2 · 防守犯规（获胜）",
    3: "码3 · 防守撞墙（获胜）",
    4: "码4 · 防守越线（获胜）",
    5: "码5 · 防守友军误伤（获胜）",
    11: "码11 · 投篮被盖",
    12: "码12 · 进攻超时",
    13: "码13 · 攻方犯规",
    14: "码14 · 攻方撞墙",
    15: "码15 · 攻方友军误伤",
}
WIN_CODES = {1, 2, 3, 4, 5}


def code_label(code):
    try:
        c = int(code)
    except Exception:
        return "码?"
    return CODE_NAMES.get(c, f"码{c} · 未知")


# 碰撞类终局码（需要在终局圈出犯规/碰撞方）
COLLISION_CODES = {2, 3, 5, 13, 14, 15}


def pick_offenders(code, rewards, pos):
    """按终局码 + 终局奖励签名（+位置兜底）找出要圈出的"肇事方"。

    - 码2 防守犯规：犯规的防守者（终局奖励最负的那个）
    - 码13 攻方犯规：犯规的进攻者
    - 码5 / 码15 同队友军误伤：同队两人一起（用户要求"相同方自己碰的就一起圈"）
    - 码3 / 码14 撞墙：撞墙的那个人
    返回 agent 索引列表（0=A1, 1=A2, 2=D1, 3=D2）。
    """
    try:
        c = int(code)
    except Exception:
        return []
    if c == 5:
        return [2, 3]
    if c == 15:
        return [0, 1]
    if c == 2:
        side, other = (2, 3), (0, 1)
    elif c == 13:
        side, other = (0, 1), (2, 3)
    elif c == 3:
        side, other = (2, 3), None
    elif c == 14:
        side, other = (0, 1), None
    else:
        return []
    # 1) 奖励签名：犯规方吃了大额负分
    if rewards is not None:
        cand = [(i, float(rewards[i])) for i in side]
        cand = [x for x in cand if math.isfinite(x[1])]
        if cand:
            i_min, r_min = min(cand, key=lambda x: x[1])
            if r_min < -1.0:
                return [i_min]
    # 2) 位置兜底：犯规 = 离对方最近 / 撞墙 = 最靠墙
    if pos is not None and len(pos) > max(side):
        if other is not None:
            best, bd = None, 1e9
            for i in side:
                d = min(math.dist(pos[i], pos[j]) for j in other)
                if d < bd:
                    best, bd = i, d
            return [best] if best is not None else []
        best, bx = None, -1.0
        for i in side:
            if abs(pos[i][0]) > bx:
                best, bx = i, abs(pos[i][0])
        return [best] if best is not None else []
    return []


def _access(st, key):
    """兼容 TensorDict / dict / OrderedDict 的取值；key 为 str 或 tuple 路径。"""
    if key is None:
        return None
    if isinstance(key, tuple):
        cur = st
        for k in key:
            cur = _access(cur, k)
            if cur is None:
                return None
        return cur
    try:
        v = st.get(key)
        if v is not None:
            return v
    except Exception:
        pass
    try:
        return st[key]
    except Exception:
        return None


def _pick(td, paths, shape=None, dim=None):
    """按优先级尝试多个 key 路径，返回第一个形状合格的张量（或 None）。"""
    for p in paths:
        v = _access(td, p)
        if v is None or not hasattr(v, "shape"):
            continue
        if shape is not None and tuple(v.shape[:2]) != tuple(shape[:2]):
            continue
        if dim is not None and v.dim() != dim:
            continue
        return v
    return None


def find_episodes(done, is_init, n_eps, iteration):
    """在 [E,T] 的 done / is_init 里找出**完整回合**（起点 -> 终点），随机挑 n_eps 个。

    用独立生成器（不碰全局 torch RNG，避免影响训练随机性）；种子由 iteration 决定，
    这样同一迭代重复刷新得到同一批局（前端不会每 2s 跳来跳去）。
    """
    d = done[..., 0].bool().cpu().numpy()
    E, T = d.shape
    if is_init is not None:
        ini = is_init[..., 0].bool().cpu().numpy()
    else:                                  # 没给 is_init：done 的下一步就是新局起点
        ini = np.zeros_like(d)
        ini[:, 1:] = d[:, :-1]
        ini[:, 0] = True
    episodes = []
    for e in range(E):
        starts = np.nonzero(ini[e])[0]
        ends = np.nonzero(d[e])[0]
        if starts.size == 0 or ends.size == 0:
            continue
        for s in starts:
            after = ends[ends >= s]
            if after.size:
                episodes.append((e, int(s), int(after[0])))
    if not episodes:
        return []
    gen = torch.Generator()
    gen.manual_seed((int(iteration) * 2654435761 + 12345) % (2 ** 31 - 1))
    perm = torch.randperm(len(episodes), generator=gen).tolist()
    return [episodes[i] for i in perm[:n_eps]]


def build_episode(state, reward, codes, row, t0, t1):
    """切出一个完整回合 [t0, t1]（含起点与终局）并序列化。"""
    s = state[row, t0:t1 + 1].float().cpu()      # [L, 23]
    T = int(s.shape[0])
    end = T - 1                                  # 完整回合：末帧就是终局

    spot = s[0, 18:20] * torch.tensor(POS_DIV)
    basket = s[0, 20:22] * torch.tensor(POS_DIV)

    frames = []
    for t in range(T):
        f = []
        for o in AGENT_OFFSETS:
            f.append(float(s[t, o]) * POS_DIV[0])
            f.append(float(s[t, o + 1]) * POS_DIV[1])
            f.append(float(s[t, o + 2]) * V_DIV)
            f.append(float(s[t, o + 3]) * V_DIV)
        f.append(float(s[t, 5]))                  # progress
        f.append(float(s[t, 4]))                  # in_spot
        f.append(float(s[t, 22]) * T_LIMIT)       # t_remaining (s)
        frames.append([round(v, 4) for v in f])

    prog_last = float(s[end, 5])
    r_all = [float(reward[row, t0 + end, i]) for i in range(reward.shape[-1])] if reward is not None else []
    while len(r_all) < 4:
        r_all.append(0.0)
    r_all = r_all[:4]
    r_a1_last, r_a2_last = r_all[0], r_all[1]
    t_rem_last = float(s[end, 22]) * T_LIMIT
    pos_end = [[float(s[end, o]), float(s[end, o + 1])] for o in AGENT_OFFSETS]

    code = None
    if codes is not None:                        # 精确终局码（每步都在，末帧即终局）
        try:
            code = int(codes[row, t0 + end].item())
            if code == 0:                        # 0 = 没有终局（被截断的回合不该出现）
                code = int(codes[row, t0 + end].reshape(-1)[0].item())
        except Exception:
            code = None
    offenders = pick_offenders(code, r_all, pos_end) if code is not None else []
    return {
        "row": int(row),
        "start_step": int(t0),
        "n": T,
        "end_step": end,
        "spot": [round(float(spot[0]), 3), round(float(spot[1]), 3)],
        "basket": [round(float(basket[0]), 3), round(float(basket[1]), 3)],
        "code": code,
        "win": (code in WIN_CODES) if code is not None else None,
        "outcome": code_label(code) if code is not None else "无终局码",
        "rewards": [round(v, 2) for v in r_all],
        "offender": offenders,
        "prog_last": round(prog_last, 3),
        "r_a1_last": round(r_a1_last, 2),
        "r_a2_last": round(r_a2_last, 2),
        "t_rem_last": round(t_rem_last, 2),
        "frames": frames,
    }


def build_payload(td, iteration, n_eps=N_EPISODES):
    """从 rollout/batch TensorDict 生成 payload（state/done/is_init/reward/codes 全在里面）。"""
    state = _pick(td, ["state", ("next", "state")], dim=3)
    if state is None:
        return None
    state = state.detach()
    E, T = int(state.shape[0]), int(state.shape[1])
    if E == 0 or T == 0:
        return None

    done = _pick(td, [("next", "done")], dim=3)
    if done is None:
        done = _pick(td, [("next", "done")], dim=2)
        done = done.unsqueeze(-1) if done is not None else None
    if done is None or tuple(done.shape[:2]) != (E, T):
        return None

    is_init = _pick(td, ["is_init"], dim=3)
    if is_init is None:
        is_init = _pick(td, ["is_init"], dim=2)
        is_init = is_init.unsqueeze(-1) if is_init is not None else None
    if is_init is not None and tuple(is_init.shape[:2]) != (E, T):
        is_init = None

    reward = _pick(td, [("next", "agents", "reward")], dim=4)
    if reward is not None:
        reward = reward.detach()[..., 0] if reward.shape[-1] == 1 else reward.detach()
        if tuple(reward.shape[:2]) != (E, T):
            reward = None

    codes = None
    for p in [("next", "agents", "info", "termination_reason"), ("agents", "info", "termination_reason")]:
        v = _access(td, p)
        if v is None or not hasattr(v, "shape"):
            continue
        v = v.detach()
        if v.dim() == 4:                 # [E,T,A,1]（每个 agent 广播同一个码）
            v = v[..., 0, 0]
        elif v.dim() == 3:               # [E,T,A]
            v = v[..., 0]
        if v.dim() == 2 and tuple(v.shape[:2]) == (E, T):
            codes = v
            break

    picked = find_episodes(done, is_init, n_eps, iteration)
    episodes = [build_episode(state, reward, codes, r, a, b) for (r, a, b) in picked]

    spot0 = state[0, 0, 18:20].float().cpu() * torch.tensor(POS_DIV) if E else torch.zeros(2)
    return {
        "iter": int(iteration),
        "t_limit": T_LIMIT,
        "dt": T_LIMIT / max(T, 1),
        "has_codes": codes is not None,
        "names": ["A1", "A2", "D1", "D2"],
        "episodes": episodes,
    }


def write_json(path, payload):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 训练进程内回调
# --------------------------------------------------------------------------- #
try:
    from benchmarl.experiment.callback import Callback
except Exception:  # 离线/独立运行时没有 benchmarl
    class Callback:  # type: ignore
        def __init__(self):
            self.experiment = None


class LiveViewCallback(Callback):
    def __init__(self, out_dir: str = "outputs/live", iter_offset: int = 0, n_eps: int = N_EPISODES):
        super().__init__()
        self.out_dir = out_dir
        self.iter_offset = iter_offset
        self.n_eps = n_eps
        self._n = 0
        os.makedirs(out_dir, exist_ok=True)

    def on_batch_collected(self, batch):
        try:
            it = self.iter_offset + self._n
            payload = build_payload(batch, it, self.n_eps)
            if payload is None:
                print("[LiveView] batch 缺少 state/done，跳过", flush=True)
                return
            path = os.path.join(self.out_dir, "live_env.json")
            write_json(path, payload)
            self._n += 1
            tags = " ".join(f"{e['code'] if e['code'] is not None else '?'}" for e in payload["episodes"])
            print(f"[LiveView] iter={it} 写入 {len(payload['episodes'])} 局（终局码 {tags}）-> {path}", flush=True)
        except Exception as e:  # 绝不影响训练
            print(f"[LiveView] skipped: {type(e).__name__}: {e}", flush=True)


# --------------------------------------------------------------------------- #
# 离线：用 checkpoint 策略跑真实 rollout（拿到精确终局码）
# --------------------------------------------------------------------------- #
def offline_from_ckpt(ckpt, out, iter_label=0, n_eps=N_EPISODES, n_envs=32, round_seconds=20):
    os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
    from benchmarl.algorithms import MappoConfig
    from benchmarl.environments import LayupTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models import SequenceModelConfig
    from benchmarl.models.attention import AttentionConfig
    from benchmarl.models.gru import GruConfig

    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_n_envs_per_worker = int(n_envs)
    cfg.on_policy_collected_frames_per_batch = int(n_envs) * 200
    cfg.max_n_iters = 1
    cfg.evaluation = False
    cfg.render = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/live_offline"
    os.makedirs(cfg.save_folder, exist_ok=True)
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    # 注意：不要走 cfg.restore_file —— 完整 checkpoint 里带 collector 状态（帧计数已达上限），
    # 恢复后 collector 迭代器立刻 StopIteration。改为"全新 Experiment + 手动装载权重 + 同步采样器"。
    cfg.restore_file = None

    task = LayupTask.LAYUP.get_from_yaml()
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    model_config = SequenceModelConfig(
        model_configs=[AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
                       GruConfig.get_from_yaml("benchmarl/conf/model/layers/gru.yaml")],
        intermediate_sizes=[256])
    critic_model_config = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    exp = Experiment(task=task, algorithm_config=acfg, model_config=model_config,
                     critic_model_config=critic_model_config, seed=0, config=cfg, callbacks=[])
    try:
        # 手动装载权重（与实验内部恢复等价：losses 的参数与 exp.policy 共享，随后同步到采样器）
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        for group in exp.group_map.keys():
            exp.losses[group].load_state_dict(ck[f"loss_{group}"])
        exp.collector.update_policy_weights_()
        exp.policy.eval()
        batch = next(iter(exp.collector))
        payload = build_payload(batch, iter_label, n_eps)
        if payload is None:
            print("[live_view] rollout batch 缺少 state/done")
            return None
        write_json(out, payload)
        outs = ", ".join(f"{e['outcome']}(row{e['row']})" for e in payload["episodes"])
        print(f"[live_view] wrote {out} ({len(payload['episodes'])} 局, has_codes={payload['has_codes']}): {outs}")
        return payload
    finally:
        try:
            exp.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _serve(directory, port):
    import http.server
    import socketserver

    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler

    class Quiet(handler):
        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("0.0.0.0", port), Quiet) as httpd:
        print(f"[live_view] serving {directory} at http://localhost:{port}/index.html", flush=True)
        httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ckpt", help="离线：用 checkpoint 策略跑真实 rollout 生成 JSON（带精确终局码）")
    ap.add_argument("--iter", type=int, default=0, help="离线模式的 iter 标签")
    ap.add_argument("--eps", type=int, default=N_EPISODES, help="每个迭代挑几局")
    ap.add_argument("--envs", type=int, default=32, help="离线 rollout 的并行环境数")
    ap.add_argument("--out", default="outputs/live/live_env.json")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--dir", default="outputs/live")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.from_ckpt:
        offline_from_ckpt(args.from_ckpt, args.out, args.iter, args.eps, args.envs)
    if args.serve:
        _serve(args.dir, args.port)


if __name__ == "__main__":
    main()
