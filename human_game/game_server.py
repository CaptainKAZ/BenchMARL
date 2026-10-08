#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[人机大战] 单环境小游戏 —— 后端推理 + 物理环境服务

在容器内运行（需要 torch/VMAS/BenchMARL 环境）：
  PYTHONPATH=/home/vscode/workspace/BenchMARL python BenchMARL/human_game/game_server.py --port 8766

设计
----
- 单环境（num_envs=1），200 步 / 20 秒一局；4 个 agent 由同一份训练策略驱动。
- 人类可接管任意一个 player（A1/A2/D1/D2），其余 3 个走策略推理（确定性/评测口径）。
- 操作方式：**按住鼠标 = 把该 player 吸引到鼠标指向的目标点**（比例导引：速度 ∝ 距离，
  近点自动减速停下）；松开 = 停止吸引（圈外）。
- A1 特别规则（用户定义·第二版）：**按住 = 跟随鼠标移动（不蓄力）；在投篮圈内松手 =
  原地刹车蓄力，读满 10 帧自动出手**（按住可随时取消）。实现上直接把"松手且在圈内"
  翻译成环境的 press 信号，由环境自身的读条/出手判定驱动，不改训练环境任何逻辑。
- 服务端只做：策略前向 → 覆盖人类那一行动作 → env.step → 回 JSON 状态。
  纯 CPU、单环境，和训练共处一个容器时对训练的干扰很小（可用 OMP_NUM_THREADS 限制）。
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from tensordict.nn import set_composite_lp_aggregate

# 复合动作策略 / 离散 mask 的 log_prob 处理：与训练、crossplay 同款
set_composite_lp_aggregate(False).set()

from torchrl.envs.utils import (  # noqa: E402
    ExplorationType,
    MarlGroupMapType,
    set_exploration_type,
    step_mdp,
)

from benchmarl.algorithms import MappoConfig  # noqa: E402
from benchmarl.environments import LayupTask  # noqa: E402
from benchmarl.experiment import ExperimentConfig  # noqa: E402
from benchmarl.models.attention import AttentionConfig  # noqa: E402
from benchmarl.models.common import SequenceModelConfig  # noqa: E402
from benchmarl.models.gru import GruConfig  # noqa: E402
from benchmarl.utils import _add_rnn_transforms  # noqa: E402

GRU_KEY = ("agents", "_hidden_gru_1")          # SequenceModelConfig([Attention, GRU]) 的隐状态键
NAMES = ["A1", "A2", "D1", "D2"]
ROLE_INDEX = {"A1": 0, "A2": 1, "D1": 2, "D2": 3}
WIN_CODES = {1, 2, 3, 4, 5}                    # 这些码 = 进攻方获胜
FOUL_CODES = {2, 3, 4, 5, 13, 14, 15}         # 这些码 = 有"肇事者"（碰撞/撞墙/越线/友军误伤）
CODE_NAMES = {
    1: "投篮命中", 2: "防守犯规", 3: "对手撞墙", 4: "对手越线", 5: "对手友军误伤",
    11: "投篮被盖", 12: "进攻超时", 13: "攻方犯规", 14: "攻方撞墙", 15: "攻方友军误伤",
}
POS_DIV = torch.tensor([4.0, 7.5])             # 归一化分母 (W/2, L/2)
REL_DIV = torch.tensor([8.0, 15.0])
V_MAX = 5.0
T_LIMIT = 20.0
MAX_STEPS = 200
SPOT_R = 0.9
ARRIVE_K = 8.0                                 # P 增益：短距离下按 v=Kp*err 收敛
BRAKE_A = 1.5                                  # 刹车规划加速度（略低于环境 a_max=3，留裕量）：
                                               # 速度上限取 sqrt(2*a*dist)，保证刹得住、不过冲
STOP_R = 0.03                                  # 到点死区（米）——只有球员半径的 1/10，视觉上就是"跑到鼠标底下"
TARGET_MX = 3.5                                # 目标点 x 上限（场边留缓冲，鼠标甩出场外不会撞墙送死）
TARGET_MY = 6.8                                # 目标点 y 上限
FF_MAX = 4.0                                   # 指针速度前馈上限（m/s）：拖动鼠标时球员提前跟上
KD_OVER = 1.2                                  # 相对指针超速时的反向刹车增益（防冲过头）
WALL_HX = 3.7                                  # 场地 x 方向"触墙面"（半宽 4 - 球员半径 0.3）
WALL_HY = 7.2                                  # 场地 y 方向触墙面
WALL_A = 1.0                                   # 贴墙刹车规划加速度（保守值：接近速度 ~0.45 m/s，不会触发 >0.5 撞墙判负）
WALL_M = 0.1                                   # 贴墙保留余量（米）
SHOT_STILL_FRAMES = 10                         # 读条门槛（硬约束，不可降）


# ----------------------------------------------------------------------------
# 从 crossplay_v3.py 复刻的"装载单组策略"工具（前缀匹配 + 忽略 torch.compile 包装）
# ----------------------------------------------------------------------------
def strip_orig(k: str) -> str:
    return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def extract_actor(ck, group="agents"):
    pfx = "actor_network_params."
    sd = {}
    for k, v in ck[f"loss_{group}"].items():
        if isinstance(v, torch.Tensor) and k.startswith(pfx):
            sd[strip_orig(k[len(pfx):])] = v
    return sd


def install(policy, actor_sd, prefix=None):
    psd = policy.state_dict()
    nkeys = list(psd.keys())
    if prefix is None:
        best, best_hit = None, -1
        for pfx in ("module.0.", "module.1.", ""):
            hits = 0
            for k, v in actor_sd.items():
                nk = strip_orig(k)
                c = [pk for pk in nkeys if pk.startswith(pfx) and strip_orig(pk).endswith(nk)]
                if len(c) == 1 and tuple(psd[c[0]].shape) == tuple(v.shape):
                    hits += 1
            if hits > best_hit:
                best, best_hit = pfx, hits
        prefix = best
    chosen = {}
    hit = miss = amb = 0
    for k, v in actor_sd.items():
        nk = strip_orig(k)
        cands = [pk for pk in nkeys if pk.startswith(prefix) and strip_orig(pk).endswith(nk)]
        if len(cands) == 1 and tuple(psd[cands[0]].shape) == tuple(v.shape):
            chosen[cands[0]] = v
            hit += 1
        elif not cands:
            miss += 1
        else:
            amb += 1
    full = dict(psd)
    full.update(chosen)
    policy.load_state_dict(full, strict=True)
    return prefix, hit, miss, amb


def shell_for(env, algo_cfg, model_cfg, critic_cfg, cfg, task, with_mask=True):
    return SimpleNamespace(
        config=cfg, algorithm_config=algo_cfg, model_config=model_cfg,
        critic_model_config=critic_cfg, task=task, group_map=task.group_map(env),
        continuous_actions=True, seed=0, on_policy=algo_cfg.on_policy(),
        observation_spec=task.observation_spec(env), action_spec=task.action_spec(env),
        info_spec=task.info_spec(env), state_spec=task.state_spec(env),
        action_mask_spec=(task.action_mask_spec(env) if with_mask else None),
    )


def find_latest_ckpt(root: Path, min_age_s: float = 45.0):
    """找最近一个"写完"的单组 checkpoint（跳过正在写的最新档）。"""
    cands = sorted(root.glob("**/checkpoints/checkpoint_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    now = time.time()
    for p in cands:
        if now - p.stat().st_mtime >= min_age_s:
            return p
    return cands[0] if cands else None


def ckpt_is_single_group(path: Path) -> bool:
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        ok = ("loss_agents" in ck.keys()) and ("buffer_agents" in ck.keys())
        del ck
        return ok
    except Exception:
        return False


def pick_ckpt(root: Path, prefer=None):
    if prefer:
        return Path(prefer)
    newest = find_latest_ckpt(root)
    if newest is None:
        raise SystemExit(f"[game] 找不到任何 checkpoint: {root}")
    if ckpt_is_single_group(newest):
        return newest
    # 最新档不是单组架构（旧 run）→ 按时间往回找第一个单组的
    cands = sorted(root.glob("**/checkpoints/checkpoint_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if ckpt_is_single_group(p):
            return p
    raise SystemExit("[game] 没有找到兼容的单组 checkpoint")


# ----------------------------------------------------------------------------
# 游戏核心
# ----------------------------------------------------------------------------
class Game:
    def __init__(self, ckpt: Path, human: str = "A1", seed: int = 0, no_bots: bool = False):
        self.lock = threading.RLock()
        self.no_bots = no_bots
        self.human = human if human in ROLE_INDEX else "A1"
        self.human_idx = ROLE_INDEX[self.human]
        self.ckpt = str(ckpt)
        self.score = {"human": 0, "bot": 0, "episodes": 0}

        print(f"[game] 构建环境与策略（ckpt={ckpt}）...", flush=True)
        t0 = time.time()
        self.task = LayupTask.LAYUP.get_from_yaml()
        cfg = ExperimentConfig.get_from_yaml()
        cfg.train_device = "cpu"
        cfg.sampling_device = "cpu"
        self.cfg = cfg
        self.max_steps = int(self.task.max_steps(None))
        model_cfg = SequenceModelConfig(
            [AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
             GruConfig.get_from_yaml()],
            intermediate_sizes=[256],
        )
        critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
        a_cfg = MappoConfig.get_from_yaml()
        a_cfg.share_param_actor = True
        a_cfg.share_param_critic = False

        env_spec = self.task.get_env_fun(num_envs=1, continuous_actions=True,
                                         seed=seed, device=torch.device("cpu"))()
        shell = shell_for(env_spec, a_cfg, model_cfg, critic_cfg, cfg, self.task, with_mask=True)
        algo = a_cfg.get_algorithm(shell)
        self.env = _add_rnn_transforms(lambda: env_spec, self.task.group_map(env_spec), model_cfg)()
        self.policy = algo.get_policy_for_collection().to("cpu")
        self.policy.eval()

        ck = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=False)
        pfx, hit, miss, amb = install(self.policy, extract_actor(ck, "agents"))
        del ck
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"[game] 策略装载: prefix={pfx} hit={hit} miss={miss} amb={amb} | "
              f"{n_params/1e6:.2f}M 参数 | {time.time()-t0:.1f}s", flush=True)
        if miss or amb:
            print(f"[game] ⚠️ 权重未完全装载（miss={miss} amb={amb}），策略行为可能异常", flush=True)

        self.td = None
        self.reset_locked(self.human)

    # ---------------- 状态解析 ----------------
    def _counter(self):
        return int(self.env.scenario.a1_still_frames_counter[0].item())

    def _set_counter(self, v: int):
        self.env.scenario.a1_still_frames_counter[0] = int(v)

    def _state_vec(self):
        st = self.td.get("state")
        if st is None:
            return None
        st = st.reshape(-1, 23)[0].to(torch.float32)
        return st

    def state_json(self):
        with self.lock:
            st = self._state_vec()
            if st is None:
                return {"ok": False, "msg": "state 缺失"}
            pos = [[float(st[0]) * 4, float(st[1]) * 7.5],
                   [float(st[6]) * 4, float(st[7]) * 7.5],
                   [float(st[10]) * 4, float(st[11]) * 7.5],
                   [float(st[14]) * 4, float(st[15]) * 7.5]]
            vel = [[float(st[2]) * 5, float(st[3]) * 5],
                   [float(st[8]) * 5, float(st[9]) * 5],
                   [float(st[12]) * 5, float(st[13]) * 5],
                   [float(st[16]) * 5, float(st[17]) * 5]]
            spot = [float(st[18]) * 4, float(st[19]) * 7.5]
            basket = [float(st[20]) * 4, float(st[21]) * 7.5]
            t_rem = float(st[22]) * T_LIMIT
            progress = float(st[5])
            in_spot = bool(st[4] > 0.5)
            try:
                block = self.env.scenario.a1_block_factor
                block = float(block[0].item()) if hasattr(block, "__getitem__") else float(block)
            except Exception:
                block = 0.0
            return {
                "ok": True,
                "role": self.human, "humanIdx": self.human_idx,
                "noBots": bool(self.no_bots),
                "names": NAMES,
                "step": self.step_i, "maxSteps": self.max_steps,
                "tRem": round(t_rem, 3), "progress": round(progress, 3), "inSpot": in_spot,
                "pos": pos, "vel": vel, "spot": spot, "basket": basket,
                "blockA1": round(block, 3),
                "charge": {"frames": self.charge, "max": SHOT_STILL_FRAMES,
                           "ready": self.charge >= SHOT_STILL_FRAMES},
                "down": bool(self.last_down), "target": self.target,
                "rewards": [round(r, 2) for r in self.rewards],
                "done": bool(self.done), "outcome": self.outcome,
                "score": dict(self.score),
                "ckpt": Path(self.ckpt).name,
            }

    # ---------------- 一局 ----------------
    def reset_locked(self, human=None):
        if human is not None and human in ROLE_INDEX:
            self.human = human
        self.human_idx = ROLE_INDEX[self.human]
        self.td = self.env.reset()
        self.step_i = 0
        self.done = False
        self.outcome = None
        self.rewards = [0.0] * 4
        self.charge = 0
        self.last_down = False
        self.target = None
        self.target_prev = None
        self._v_ff = (0.0, 0.0)
        return self.td

    def reset(self, human=None, no_bots=None):
        with self.lock:
            if no_bots is not None:
                self.no_bots = bool(no_bots)
            self.reset_locked(human)
            return self.state_json()

    def _human_pos_tensor(self):
        st = self._state_vec()
        base = {0: 0, 1: 6, 2: 10, 3: 14}[self.human_idx]
        return torch.tensor([float(st[base]) * 4, float(st[base + 1]) * 7.5])

    def _human_vel_tensor(self):
        st = self._state_vec()
        base = {0: 0, 1: 6, 2: 10, 3: 14}[self.human_idx]
        return torch.tensor([float(st[base + 2]) * 5, float(st[base + 3]) * 5])

    def _target_ff(self, x, y):
        """指针速度前馈估计（m/s，模拟时间 dt=0.1s/步）：拖动鼠标时球员提前跟。

        每步做 0.6 新 + 0.4 旧的平滑并限幅到 FF_MAX；位置没动则衰减。
        """
        import math

        cur = (float(x), float(y))
        prev = self.target_prev
        self.target_prev = cur
        if prev is None:
            self._v_ff = (0.0, 0.0)
            return torch.zeros(2)
        raw = ((cur[0] - prev[0]) / 0.1, (cur[1] - prev[1]) / 0.1)
        v = (0.6 * raw[0] + 0.4 * self._v_ff[0], 0.6 * raw[1] + 0.4 * self._v_ff[1])
        n = math.hypot(v[0], v[1])
        if n > FF_MAX:
            v = (v[0] * FF_MAX / n, v[1] * FF_MAX / n)
        elif n < 0.02:
            v = (0.0, 0.0)
        self._v_ff = v
        return torch.tensor(v)

    def _v_des2(self, x, y):
        """到点控制器 + 指针速度前馈：尽量把球员驱动到鼠标位置（快速拖动也跟得上）。

        期望相对速度 = min(Kp*err, sqrt(2*BRAKE_A*err), v_max)；
        相对指针超速时按 KD_OVER 反向刹车，避免冲过头。
        """
        import math

        # 目标点先夹取到场内安全范围（鼠标甩出场外时球员贴边而不是送死）
        x = max(-TARGET_MX, min(TARGET_MX, float(x)))
        y = max(-TARGET_MY, min(TARGET_MY, float(y)))
        cur = self._human_pos_tensor()
        d = torch.tensor([x, y]) - cur
        dist = float(d.norm())
        v_ff = self._target_ff(x, y)
        if dist < STOP_R:
            return self._wall_brake(cur, v_ff.clamp(-V_MAX, V_MAX))
        dirv = d / (dist + 1e-9)
        vel = self._human_vel_tensor()
        v_rel_par = float(torch.dot(vel - v_ff, dirv))     # 相对指针的接近速度
        v_ref = min(ARRIVE_K * dist, math.sqrt(max(0.0, 2.0 * BRAKE_A * dist)), V_MAX)
        mag = v_ref - KD_OVER * max(0.0, v_rel_par - v_ref)
        v_des = v_ff + dirv * mag
        n = float(v_des.norm())
        if n > V_MAX:
            v_des = v_des * (V_MAX / n)
        return self._wall_brake(cur, v_des)

    @staticmethod
    def _wall_brake(pos, v_des):
        """贴墙刹车：只压"朝墙"的速度分量，沿墙/离墙方向不受影响。

        规划加速度 WALL_A 与余量 WALL_M 保证触墙速度 < 0.5 m/s（不会触发撞墙判负）。
        """
        import math

        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(v_des[0]), float(v_des[1])
        if vx < 0:
            cap = math.sqrt(max(0.0, 2.0 * WALL_A * (x - (-WALL_HX) - WALL_M)))
            vx = -min(-vx, cap)
        else:
            cap = math.sqrt(max(0.0, 2.0 * WALL_A * (WALL_HX - x - WALL_M)))
            vx = min(vx, cap)
        if vy < 0:
            cap = math.sqrt(max(0.0, 2.0 * WALL_A * (y - (-WALL_HY) - WALL_M)))
            vy = -min(-vy, cap)
        else:
            cap = math.sqrt(max(0.0, 2.0 * WALL_A * (WALL_HY - y - WALL_M)))
            vy = min(vy, cap)
        return torch.tensor([vx, vy])

    def step(self, x=None, y=None, down=False):
        with self.lock:
            if self.td is None or self.done:
                return self.state_json()
            down = bool(down)
            # [新语义] 按住 = 吸引跟随（不蓄力）；松手 = 圈内刹车蓄力（读满 10 帧自动出手）、圈外停下
            in_spot = False
            if self.human == "A1":
                try:
                    in_spot = bool(self.env.scenario.is_in_spot_a1[0].item())
                except Exception:
                    in_spot = False
            charge_signal = (self.human == "A1") and (not down) and in_spot

            if not down:
                self.target = None
            else:
                self.target = [round(float(x), 3), round(float(y), 3)]

            # --- 策略前向 + 覆盖人类动作 ---
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                # 新episode首帧：显式清零 GRU 隐状态（与 crossplay 驱动一致，双保险）
                is_init = self.td.get("is_init")
                if is_init is not None:
                    ii = is_init.bool().reshape(-1)
                    h = self.td.get(GRU_KEY, None)
                    if h is not None and bool(ii.any()):
                        h[ii] = 0.0
                self.policy(self.td)
                act = self.td[("agents", "action")]
                cont, disc = act["continuous"], act["discrete"]
                if down and x is not None:
                    cont[0, self.human_idx] = self._v_des2(float(x), float(y))
                else:
                    cont[0, self.human_idx] = torch.zeros(2)
                if self.human == "A1":
                    if disc.dim() >= 3:
                        disc[0, self.human_idx, 0] = 1 if charge_signal else 0
                    else:
                        disc[0, self.human_idx] = 1 if charge_signal else 0
                else:
                    if disc.dim() >= 3:
                        disc[0, self.human_idx, 0] = 0
                    else:
                        disc[0, self.human_idx] = 0
                if self.no_bots:   # 测试用：其余 agent 空动作
                    for i in range(4):
                        if i != self.human_idx:
                            cont[0, i] = 0.0
                            if disc.dim() >= 3:
                                disc[0, i, 0] = 0
                            else:
                                disc[0, i] = 0
                out = self.env.step(self.td)

                # 先取 "next" 子树（step_mdp 前）
                done_t = out.get(("next", "done"))
                rew_t = out.get(("next", "agents", "reward"))
                code_t = out.get(("next", "agents", "info", "termination_reason"))
                self.td = step_mdp(out)

            self.step_i += 1
            self.last_down = down
            if self.human == "A1":
                self.charge = min(self._counter(), SHOT_STILL_FRAMES)
            else:
                self.charge = 0

            if rew_t is not None:
                r = rew_t.reshape(-1, 4)
                if r.shape[0] > 0:
                    self.rewards = [float(v) for v in r[0]]
            done = bool(done_t.reshape(-1)[0]) if done_t is not None else False
            if done:
                code = 0
                if code_t is not None:
                    c = code_t.reshape(-1, 4, code_t.shape[-1])[0] if code_t.dim() >= 3 else code_t.reshape(-1, 4)
                    for v in c.reshape(4, -1)[:, 0].tolist():
                        if int(v) > 0:
                            code = int(v)
                            break
                human_side = self.human in ("A1", "A2")
                if code in WIN_CODES:
                    human_win = human_side
                elif code >= 11:
                    human_win = not human_side
                else:
                    human_win = None
                offender = (int(min(range(4), key=lambda i: self.rewards[i]))
                            if code in FOUL_CODES else None)
                self.done = True
                self.score["episodes"] += 1
                if human_win:
                    self.score["human"] += 1
                elif human_win is False:
                    self.score["bot"] += 1
                self.outcome = {
                    "code": code, "name": CODE_NAMES.get(code, f"码{code}"),
                    "humanWin": human_win, "rewards": [round(r, 2) for r in self.rewards],
                    "offender": offender,
                    "offenderName": NAMES[offender] if offender is not None else None,
                }
            return self.state_json()


# ----------------------------------------------------------------------------
# HTTP 服务
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    game: Game = None
    html_path: Path = None
    server_version = "LayupGame/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html", "/game.html"):
            try:
                body = Path(self.html_path).read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        elif self.path == "/api/meta":
            self._json({
                "ok": True, "names": NAMES, "maxSteps": self.game.max_steps,
                "tLimit": T_LIMIT, "vMax": V_MAX, "spotR": SPOT_R,
                "field": [8.0, 15.0], "ckpt": Path(self.game.ckpt).name,
            })
        elif self.path.startswith("/api/state"):
            self._json(self.game.state_json())
        elif self.path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            payload = {}
        try:
            if self.path == "/api/step":
                self._json(self.game.step(payload.get("x"), payload.get("y"), payload.get("down", False)))
            elif self.path == "/api/reset":
                self._json(self.game.reset(payload.get("role"), payload.get("noBots")))
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint 路径（默认自动挑最新的单组档）")
    ap.add_argument("--root", default="/home/vscode/workspace/BenchMARL/outputs")
    ap.add_argument("--role", default="A1", choices=list(ROLE_INDEX))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-bots", action="store_true", help="测试用：其余 agent 空动作")
    ap.add_argument("--selftest", type=int, default=0, help="跑 N 步脚本化自检后退出")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("GAME_THREADS", "2")))
    ckpt = pick_ckpt(Path(args.root), args.ckpt)
    game = Game(ckpt, human=args.role, seed=args.seed, no_bots=args.no_bots)

    if args.selftest:
        st = game.state_json()
        print(f"[selftest] init: spot={st['spot']} pos={st['pos'][0]} tRem={st['tRem']}")
        for i in range(args.selftest):
            down = (i % 20) < 15
            tgt = st["spot"] if down else (0.0, 0.0)
            st = game.step(tgt[0], tgt[1], down)
            if i % 5 == 0 or st["done"]:
                print(f"[selftest] step={st['step']} tRem={st['tRem']} inSpot={st['inSpot']} "
                      f"prog={st['progress']} charge={st['charge']['frames']} "
                      f"pos0=({st['pos'][0][0]:.2f},{st['pos'][0][1]:.2f}) done={st['done']} "
                      f"outcome={st['outcome']['name'] if st['outcome'] else None}")
            if st["done"]:
                game.reset_locked()
                st = game.state_json()
        print("[selftest] OK")
        return

    Handler.game = game
    Handler.html_path = Path(__file__).resolve().parent / "game.html"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[game] 服务已就绪: http://{args.host}:{args.port}/  (人类角色={game.human}, ckpt={Path(ckpt).name})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
