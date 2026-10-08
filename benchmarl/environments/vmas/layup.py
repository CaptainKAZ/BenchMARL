from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING

    # --- [冒烟实验] 环境初始化与观测历史配置 ---
    # 固定初始位置（A2 / D1 / D2 / 投篮点），默认关闭保持原有随机行为
    fixed_init: bool = False
    fixed_spot: bool = False
    # 为无记忆的 MLP 提供固定窗口历史: 沿特征维拼接最近 history_frames 帧观测
    # history_frames=0 表示不启用; history_stride 目前仅支持 1 (每帧采样)
    history_frames: int = 0
    history_stride: int = 1

    # --- [感知噪声] 队友/对手观测的高斯噪声（模拟真实感知误差）---
    # sigma_pos = perception_noise_floor + k_perception_noise * 距离   (米)
    # sigma_vel = perception_noise_vel_floor + k_perception_noise_vel * 距离 (米/秒)
    # 全部置 0 可关闭噪声（回到"全知观测"）。
    k_perception_noise: float = 0.02
    perception_noise_floor: float = 0.05
    k_perception_noise_vel: float = 0.01
    perception_noise_vel_floor: float = 0.02
