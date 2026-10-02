"""Muon 优化器（Jordan et al. 2024）+ AdamW 的混合封装。

Muon 的做法：对 >=2 维的权重矩阵做 "动量 + Newton-Schulz 正交化" 更新，
把动量矩阵的奇异值压到 ~1 后再按 lr 缩放，因此每个矩阵参数的更新尺度
不随层宽漂移，实测（语言模型预训练）能用少得多的步数收敛。
其余参数（bias / LayerNorm / 标量）继续走 AdamW —— 这是社区标准配方。

与 BenchMARL 的对接：
- BenchMARL 给 "每个 loss 一个 torch.optim.Optimizer"（experiment.py），
  所以这里把两种更新规则塞进同一个 Optimizer 的两个 param group，
  对外仍是标准的 Optimizer 接口（step / zero_grad / state_dict / param_groups）。
- 打开方式：环境变量 `USE_MUON=1`。
  可调：`MUON_LR`（默认 0.005）、`MUON_MOMENTUM`（0.95）、`MUON_NS_STEPS`（5）、
        `MUON_WEIGHT_DECAY`（0.0）、`MUON_SCALE_MODE`（shape | match_rms | none）。
"""

import math
import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 五次迭代，求 G 的"零次幂"（正交化：奇异值压向 1）。

    G: (m, n) 的动量矩阵（任意数值范围）；返回同形状矩阵，其奇异值 ≈ 1。
    为了提速，迭代在 bf16 下做（与 Keller Jordan 参考实现一致），最后转回原 dtype。
    """
    assert G.ndim == 2, "Muon 只处理二维权重矩阵"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.clone()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _zeropower_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """批量版 Newton-Schulz：G 为 (K, m, n)，对每张矩阵独立归一化后一起做迭代。

    与逐矩阵版数学等价（归一化按矩阵做，转置判定在桶内一致），
    但把 K 次小 matmul 合成一次 bmm，显著减少 kernel 启动开销。
    """
    X = G.bfloat16() if G.is_cuda else G.clone()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class MuonWithAdamW(torch.optim.Optimizer):
    """Muon（矩阵参数）+ AdamW（其余参数）混合优化器。

    参数按 `ndim >= 2` 自动分组；`scale_mode` 控制 Muon 更新的尺度对齐：
      - "shape"      : update *= max(1, rows/cols) ** 0.5   （Muon 原始配方）
      - "match_rms"  : update *= 0.2 * sqrt(max(rows, cols))（对齐 AdamW 的更新 RMS 量级）
      - "none"       : 不缩放
    """

    def __init__(
        self,
        params,
        lr: float = 5e-5,
        muon_lr: float = 0.005,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        muon_weight_decay: float = 0.0,
        scale_mode: str = "shape",
    ):
        params = [p for p in params]
        muon_params = [p for p in params if p.ndim >= 2]
        aux_params = [p for p in params if p.ndim < 2]

        groups = []
        if len(muon_params) > 0:
            groups.append(
                dict(
                    params=muon_params,
                    lr=muon_lr,
                    use_muon=True,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                    weight_decay=muon_weight_decay,
                    scale_mode=scale_mode,
                )
            )
        if len(aux_params) > 0:
            groups.append(
                dict(
                    params=aux_params,
                    lr=lr,
                    use_muon=False,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
            )
        super().__init__(groups, defaults=dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                # ---------- Muon: 动量 -> 正交化 -> 缩放（按形状分桶批量做 NS） ----------
                buckets = {}
                for p in group["params"]:
                    if p.grad is None or p.grad.ndim != 2:
                        continue
                    buckets.setdefault(tuple(p.grad.shape), []).append(p)
                for shape, ps in buckets.items():
                    # 动量在各自的 buffer 上原地更新（与逐矩阵版语义一致），
                    # 只有最贵的 Newton-Schulz 才分桶批量做。
                    upd_list = []
                    for p in ps:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p.grad)
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(p.grad)
                        if group["nesterov"]:
                            upd_list.append(p.grad.add(buf, alpha=group["momentum"]))
                        else:
                            upd_list.append(buf.clone())
                    update = _zeropower_batched(torch.stack(upd_list), steps=group["ns_steps"])
                    scale_mode = group.get("scale_mode", "shape")
                    if scale_mode == "shape":
                        scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    elif scale_mode == "match_rms":
                        scale = 0.2 * math.sqrt(max(shape[0], shape[1]))
                    else:
                        scale = 1.0
                    if group["weight_decay"] != 0.0:
                        decay = 1.0 - group["lr"] * group["weight_decay"]
                        for p in ps:
                            p.mul_(decay)
                    lr = group["lr"]
                    for p, u in zip(ps, update.unbind(0)):
                        p.add_(u, alpha=-lr * scale)
            else:
                # ---------- AdamW（与 torch.optim.AdamW 语义一致） ----------
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = group["lr"] / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                    if group["weight_decay"] != 0.0:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
