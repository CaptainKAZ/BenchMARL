# liveview —— 训练观察窗 / 前端面板

一套训练可视化工具：**观察窗**（每迭代 10 局完整回合的真实 rollout）+ **胜率曲线**（Chart.js，浏览器端渲染）+ **训练日志**，
三者同一个网页、同一个端口。

```
liveview/
├── live_view.py          # ① 数据：离线从 checkpoint 生成 live_env.json；训练中作为 Callback 直接吃 batch
├── live_view.html        # ② 前端：球场 canvas + 10 局切换 + 曲线 + 日志（Chart.js）
├── serve_dashboard.py    # ③ 面板服务（标准库 http.server，单端口 8765，无需额外依赖）
├── start_dashboard.sh    # 启动/重启面板（自动杀掉占端口的旧进程）
├── chart.umd.min.js      # Chart.js 4.4.1（本地内置，不联网）
├── install_cjk_fonts.sh  # 给 WSL host 装中文字体（截图/渲染用，从 Windows 字体复制）
├── live_html_check.py    # 验收：Playwright 真浏览器 22 项检查（含像素级横幅留白）
├── mobile_html_check.py  # 验收：移动端三档视口（竖屏/横屏/桌面）适配检查
├── snap_end_frame.py     # 单帧截图（人工核对遮挡黑圆 / 终局肇事圈）
└── test_live_view.js     # 验收：node 无头逻辑自检（不需要浏览器）
```

## 用法

```bash
# 1) 面板（宿主机）：观察窗 + 曲线 + 日志  →  http://localhost:8765/
bash BenchMARL/liveview/start_dashboard.sh          # 可选参数：端口，默认 8765

# 2) 观察窗数据（二选一）
#    训练中：LIVE_VIEW=1 启动训练（回调每迭代写 outputs/live/live_env.json）
#    离线  ：从 checkpoint 生成 10 局（严格终局码）
docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl \
    python /home/vscode/workspace/BenchMARL/liveview/live_view.py \
    --from-ckpt outputs/<run>/checkpoints/checkpoint_XXXX.pt --iter <N> \
    --out outputs/live/live_env.json --eps 10
```

## 验收（改动前端后跑这两个）

```bash
# 逻辑级（node，秒级）：绘制不抛错、半径符合真实值、控制流可用
node BenchMARL/liveview/test_live_view.js

# 浏览器级（Playwright 22 项）：球员像素/播放/重播/拖进度条/chip 切局/横幅排版…
#   若缺库：先用 install_cjk_fonts.sh 装字体；chromium 缺 so → 见下方 LD_LIBRARY_PATH
LD_LIBRARY_PATH=/tmp/opencode/chromelibs/root/usr/lib/x86_64-linux-gnu \
    python3 BenchMARL/liveview/live_html_check.py
#   截图默认写 .opencode/（可用 LIVE_CHECK_SHOTS=<dir> 覆盖）

# 移动端适配（手机竖屏 390x844 / 横屏 844x390 / 桌面 1600x900 三档，各 9-10 项）
LD_LIBRARY_PATH=/tmp/opencode/chromelibs/root/usr/lib/x86_64-linux-gnu \
    python3 BenchMARL/liveview/mobile_html_check.py
#   截图写 /tmp/opencode/mobile_<W>x<H>.png
```

## 移动端（手机看面板）

- 布局：`<=860px` 宽度自动改为**纵向滚动**（观察窗 → 曲线 → 日志），按钮/滑条按触控尺寸放大；
  手机**横屏**（高度 `<=560px`）恢复左右两栏，避免高视口把日志挤没。
- 手机访问：与 Windows 同一 Wi-Fi 时用 **http://192.168.31.145:8765/**（WSL 为 mirrored 网络，该 IP 即 Windows 侧 IP）。
  若打不开，管理员 PowerShell 放行一次：
  `New-NetFirewallRule -DisplayName "WSL layup dashboard 8765" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -Profile Private`

## 终端渲染对照（vmas 真值）

| 元素 | 画法 |
|---|---|
| 球员 | 实心圆 r=0.3m（agent_radius），A1/A2 橙、D1/D2 蓝，名字居中 |
| 篮筐 | r=0.1m 实心圆 + 0.35m 视觉圈 + 篮板条 |
| 投篮点 | r=0.9m 绿色圆圈 |
| 读条 | A1 头顶绿条（仅 `prog>0` 显示）；圈内未按 = 橙色虚线空框 |
| 遮挡 | A1 出手通道被挡：**球员内部半透明黑圆**（半径 = block_factor×0.3m，与训练渲染一致） |
| 终局 | 底部彩色卡片：码 N·原因 / 四人终局奖励 / 肇事方（碰撞类圈出，≥2 人加外圈）+ 剩余秒 + step |
