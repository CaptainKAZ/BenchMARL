# Critic Visualization Guide

## Overview

`visualize_critic_fixed.py` 可视化智能体的critic值函数热图，展示不同位置对critic评估的影响。

## 基本用法

```bash
# 默认：可视化attacker的critic，观察A1移动的影响
python visualize_critic_fixed.py

# Debug模式（只运行50步）
python visualize_critic_fixed.py --debug
```

## 主要参数

### 1. 选择要移动的智能体 (`--agent`)

控制热图显示哪个智能体移动到不同位置时的critic值：

```bash
# 观察A1（持球球员）移动的影响
python visualize_critic_fixed.py --agent a1

# 观察A2（挡拆球员）移动的影响
python visualize_critic_fixed.py --agent a2

# 观察D1（防守者1）移动的影响
python visualize_critic_fixed.py --agent d1

# 观察D2（防守者2）移动的影响
python visualize_critic_fixed.py --agent d2
```

### 2. 选择Critic组 (`--critic-group`)

选择使用哪个组的critic来评估：

```bash
# 使用attacker的critic（默认）
python visualize_critic_fixed.py --critic-group attacker

# 使用defender的critic
python visualize_critic_fixed.py --critic-group defender
```

### 3. 选择特定Agent的Critic (`--critic-agent`)

如果critic不是shared模式（每个agent有独立的critic），可以指定使用哪个agent的critic：

```bash
# 使用第0个agent的critic（默认）
python visualize_critic_fixed.py --critic-agent 0

# 使用第1个agent的critic
python visualize_critic_fixed.py --critic-agent 1
```

**注意**：在当前配置中，attacker和defender的critic都是shared模式，此参数会被忽略。

## 组合示例

### 示例1：看A2在哪里能提高A1的得分机会

```bash
python visualize_critic_fixed.py --agent a2 --critic-group attacker --debug
```

热图显示：A2移动到不同位置时，attacker critic的评估值。高值区域表示A2在那里更有利。

### 示例2：看D1在哪里防守最有效

```bash
python visualize_critic_fixed.py --agent d1 --critic-group defender --debug
```

热图显示：D1移动到不同位置时，defender critic的评估值。高值区域表示D1在那里防守更成功。

### 示例3：从defender视角看A1威胁

```bash
python visualize_critic_fixed.py --agent a1 --critic-group defender --debug
```

热图显示：A1在不同位置时，defender认为的威胁程度（低值=高威胁）。

## 其他参数

### 动态范围调节

默认情况下，colormap范围在episode开始前计算并固定。如果critic值在运行过程中变化很大，可能导致饱和。使用动态范围可以解决这个问题：

```bash
# 启用动态范围调节（每步更新）
python visualize_critic_fixed.py --dynamic-range --debug

# 每5步更新一次范围（更平滑但响应慢）
python visualize_critic_fixed.py --dynamic-range --range-update-freq 5 --debug

# 每20步更新（最平滑）
python visualize_critic_fixed.py --dynamic-range --range-update-freq 20
```

**动态范围工作方式**：
- 追踪观察到的critic值的最小值和最大值
- 使用指数移动平均平滑更新范围（避免跳变）
- 每N步更新一次colormap范围
- 在打印输出中显示当前范围

**何时使用**：
- Critic值在episode中变化很大时
- 初始范围估计不准确导致颜色饱和
- 想要看清细微的值变化

**注意**：动态更新可能导致颜色含义随时间变化，但能避免饱和问题。

### 手动设置范围

如果你知道期望的值域范围，可以手动指定：

```bash
# 手动设置colormap范围
python visualize_critic_fixed.py --vmin -2.0 --vmax 2.0 --debug
```

这会禁用自动范围计算，直接使用你指定的范围。适合对比多个runs或确保一致的颜色映射。

### 其他可调参数

```bash
# 调整网格精度（更细=更慢）
python visualize_critic_fixed.py --precision 0.1

# 改变colormap
python visualize_critic_fixed.py --cmap viridis

# 调整透明度
python visualize_critic_fixed.py --alpha 0.8

# 运行多个episode
python visualize_critic_fixed.py --episodes 3
```

## 理解热图

- **颜色**：
  - 红色（高值）：critic认为这是好位置
  - 蓝色（低值）：critic认为这是差位置

- **Attacker Critic**：
  - 高值区域 = 进攻有利的位置
  - 通常射门区域和开阔位置值高

- **Defender Critic**：
  - 高值区域 = 防守有利的位置
  - 通常能封堵射门路线的位置值高

## 输出说明

程序会显示：
- 当前agent位置和对应的critic值
- 每10步更新一次位置和值
- Episode结束时显示总结（初始值、最终值、变化量）

## 性能建议

- 使用 `--debug` 快速测试（只运行50步）
- 使用 `--precision 0.2` 或更大值来加速（默认0.15）
- GPU会用于critic计算，渲染在CPU上进行
