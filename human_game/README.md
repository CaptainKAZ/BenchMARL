# 人机大战（human_game）

用浏览器和训练出来的 bot 打 2v2：你可以操作 4 名球员中的任何一名，其余由 bot 接管（对打/配合）。

```
浏览器 ──> host:8766 ──> 中继容器 layup-game-relay ──> 训练容器 robocon2025-marl:8766
                  TCP 转发（纯标准库）            game_server.py（推理 + VMAS 物理）
```

> rootless Docker 下 host 无法直连容器 IP（docker0 linkdown），所以用了一个专用中继容器把 `8766` 转发进训练容器。

## 启动 / 停止

```bash
bash BenchMARL/human_game/start_game.sh                 # 默认你操作 A1，bot 载入最新 checkpoint
bash BenchMARL/human_game/start_game.sh --role D1       # 指定初始角色
bash BenchMARL/human_game/start_game.sh --ckpt outputs/xxx/.../checkpoint_57000000.pt
```

打开 **http://localhost:8766/**（局域网/手机：`http://<WSL的IP>:8766/`，如 `192.168.31.145`）。

停止：`docker exec robocon2025-marl pkill -f "[g]ame_server.py"`（中继容器可留着不动）。

## 操作

| 操作 | 含义 |
|---|---|
| **按下鼠标并移动** | 把你操控的球员"吸引"到鼠标指向的目标点（松开 = 停下/在圈内则开始蓄力）。到点控制器：速度 = min(8·err, √(2·1.5·err), 5 m/s) + 指针速度前馈；死区 3cm，贴墙自动刹车（沿墙移动不受影响，鼠标甩出场外也不会撞墙判负） |
| **你是 A1** | **按住 = 跟随移动**；把 A1 带进绿圈后**松手 = 原地刹车蓄力**（进度条 10/10），读满**自动出手**；期间按住鼠标可取消 |
| **顶栏角色按钮** | 随时切换你操作的球员（A1/A2/D1/D2），自动重开一局 |
| **顶栏 🤖 对手 / 🎯 陪练** | 陪练模式：其他球员原地站桩、不干扰，适合练投篮走位 |
| **速度 2× / 1× / 0.5×** | 画面倍速（物理步长固定 0.1s，只改刷新节奏） |
| **重开** | 立刻重开一局；终局后也会自动重开 |

提示：**别高速撞人**——相对速度 > 0.5 m/s 的碰撞算主动犯规（码13 撞防守 / 码15 撞队友），直冲投篮点基本必犯；要绕开挡路的人（稍微偏离直线，或用陪练模式先练手感）。

## 文件

| 文件 | 说明 |
|---|---|
| `game_server.py` | 后端：加载最新 checkpoint → 策略推理 + VMAS 手逐步进 + HTTP API（`/api/meta`、`/api/state`、`/api/step`、`/api/reset`） |
| `game.html` | 前端（自包含 canvas）：球场渲染、鼠标吸引、读条条、终局横幅、比分板、角色/倍速/陪练开关 |
| `tcp_relay.py` | 中继容器的 TCP 转发脚本（容器 → 容器） |
| `start_game.sh` | 一键启动（容器内 server + 中继 + 端到端自检） |
| `test_http.py` | 后端 HTTP 冒烟（9 项：meta/reset/陪练/蓄力链路/角色切换…） |
| `test_game.py` | 后端机制自测（18 项：解码、松手蓄力自动出手、蓄力取消、其他角色、bot 完局…） |
| `game_check.py` | 前端验收（Playwright：页面/画布/鼠标驱动/无 JS 错误） |
| `game_shot_check.py` | 前端投篮全流程截图（陪练模式，绕行策略，产出 `/tmp/opencode/game_charge.png`、`game_banner.png`） |

前端验收脚本在 **host** 上跑（容器里没有 playwright）：

```bash
LD_LIBRARY_PATH=/tmp/opencode/pwdeps/root/usr/lib/x86_64-linux-gnu \
  python3 BenchMARL/human_game/game_check.py        # 或 game_shot_check.py
```

## 机制备注

- 速度：前端 1× = 实时（物理步长 0.1s，单步服务端 ~13ms、经中继往返 ~17ms，余量充足）；2× = 50ms/步。
- 读条：**松手**且 A1 在圈内（`dist ≤ 0.9m` 且 `y>0`）时开始累计（环境 press 信号 = "松手且在圈内"），速度足够小（< 0.2 m/s）后逐帧涨读条，**满 10 帧自动出手**；按住期间 = 正常跟随、不蓄力；圈外松手 = 停下（后端 action mask 保证圈外按不出键）。
- 开局 A1 有 ~10 帧（1s）的"发球延迟"，期间不受控，属环境设计。
- bot 走**确定性**（评测口径）策略；服务每次启动自动挑 `outputs/**/checkpoints/` 里最新的一个单组架构 checkpoint，可在 `game_server.py` 的 `--ckpt` 指定。
- 终局码与含义见页面横幅（码1 命中 / 码11 被盖 / 码12 超时 / 码2 防守犯规 / 码13 攻方犯规 / 码3,4,5,14,15 撞墙、越线、友军误伤）。
- "肇事"只在犯规/撞墙/越线类终局显示。
