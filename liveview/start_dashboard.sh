#!/usr/bin/env bash
# 启动/重启 Layup 训练面板（观察窗 + 胜率曲线 + 日志，同一端口 8765）
# 用法：bash BenchMARL/liveview/start_dashboard.sh [port]
set -u
HOST=/home/proton/robocon2025_marl_devcontainer
PORT=${1:-8765}

OLD=$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "${OLD:-}" ]; then echo ">> 停掉占 $PORT 的旧进程 pid=$OLD"; kill "$OLD" 2>/dev/null || true; sleep 1; fi

nohup python3 "$HOST/BenchMARL/liveview/serve_dashboard.py" --port "$PORT" --bind 0.0.0.0 \
  > /tmp/opencode/dashboard.log 2>&1 &
sleep 2
cat /tmp/opencode/dashboard.log
echo ">> 打开： http://localhost:$PORT/   （Windows 侧；备用 http://$(hostname -I | awk '{print $1}'):$PORT/ ）"
for p in "/" "/status.json" "/live_env.json" "/curve.json" "/chart.umd.min.js"; do
  printf "   %-16s %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$p")"
done
