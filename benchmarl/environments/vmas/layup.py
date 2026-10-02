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
