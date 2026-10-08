#!/usr/bin/env bash
# 人机大战 · 一键启动（容器内 game_server + 中继容器）
#
# 架构（rootless Docker 下 host 不能直连容器 IP）:
#   浏览器 → host:8766 → 中继容器 layup-game-relay → 训练容器 robocon2025-marl:8766
#
# 用法: bash start_game.sh [--ckpt <path>] [--role A1|A2|D1|D2]
set -u

TRAIN_CT=robocon2025-marl
RELAY_CT=layup-game-relay
PORT=8766
IMAGE=robocon2025-marl:latest
REPO=/home/proton/robocon2025_marl_devcontainer

CKPT=""
ROLE=A1
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --role) ROLE="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 2;;
  esac
done

echo "== [1/4] 检查训练容器 =="
if ! docker ps --format '{{.Names}}' | grep -q "^${TRAIN_CT}$"; then
  echo "容器 ${TRAIN_CT} 未运行，尝试 docker start..."
  docker start "${TRAIN_CT}" >/dev/null 2>&1 || { echo "!! 启动失败，请先 bash .opencode/start_container.sh"; exit 1; }
fi
docker ps --filter "name=${TRAIN_CT}" --format '  {{.Names}} | {{.Status}}'

echo "== [2/4] 重启容器内 game_server =="
docker exec "${TRAIN_CT}" pkill -f "[g]ame_server.py" 2>/dev/null
sleep 2
EXTRA=""
[ -n "${CKPT}" ] && EXTRA="--ckpt ${CKPT}"
docker exec -d -w /home/vscode/workspace/BenchMARL "${TRAIN_CT}" bash -lc \
  "GAME_THREADS=4 OMP_NUM_THREADS=4 python human_game/game_server.py --root outputs --port ${PORT} --role ${ROLE} ${EXTRA} > outputs/game_server.log 2>&1"
READY=0
for i in $(seq 1 45); do
  sleep 2
  if docker exec -w /home/vscode/workspace/BenchMARL "${TRAIN_CT}" \
      python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:${PORT}/api/meta',timeout=2).status)" 2>/dev/null | grep -q 200; then
    READY=1; echo "  game_server 就绪 (${i}×2s)"; break
  fi
done
if [ "${READY}" != "1" ]; then
  echo "!! game_server 未就绪，日志尾部:"
  docker exec -w /home/vscode/workspace/BenchMARL "${TRAIN_CT}" tail -5 outputs/game_server.log
  exit 1
fi
docker exec -w /home/vscode/workspace/BenchMARL "${TRAIN_CT}" grep -a "策略装载\|服务已就绪" outputs/game_server.log | tail -2

echo "== [3/4] 检查中继容器 =="
TRAIN_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${TRAIN_CT}" 2>/dev/null)
TARGET="${TRAIN_IP}:${PORT}"
echo "  训练容器当前 IP: ${TRAIN_IP}"
# 容器 IP 在服务器重启后可能变化（172.17.0.2 → .3 ...），中继目标不一致就重建
CUR_ARGS=$(docker inspect -f '{{.Args}}' "${RELAY_CT}" 2>/dev/null || true)
if [ -n "${CUR_ARGS}" ] && ! echo "${CUR_ARGS}" | grep -q "${TRAIN_IP}:${PORT}"; then
  echo "  中继目标已过期（${CUR_ARGS}），重建中继容器"
  docker rm -f "${RELAY_CT}" >/dev/null
fi
if ! docker ps -a --format '{{.Names}}' | grep -q "^${RELAY_CT}$"; then
  echo "  创建中继容器 ${RELAY_CT} → ${TARGET} ..."
  docker run -d --name "${RELAY_CT}" --privileged --restart unless-stopped \
    -p ${PORT}:${PORT} -v "${REPO}:/home/vscode/workspace" "${IMAGE}" \
    python3 /home/vscode/workspace/BenchMARL/human_game/tcp_relay.py --listen ${PORT} --target "${TARGET}" >/dev/null
elif ! docker ps --format '{{.Names}}' | grep -q "^${RELAY_CT}$"; then
  docker start "${RELAY_CT}" >/dev/null
fi
sleep 2
docker ps --filter "name=${RELAY_CT}" --format '  {{.Names}} | {{.Status}} | {{.Ports}}'

echo "== [4/4] 端到端自检 =="
if curl -s -m 5 "http://127.0.0.1:${PORT}/api/meta" | grep -q '"ok": true'; then
  echo "  ✓ 经中继访问正常"
else
  echo "  !! 经中继访问失败：检查中继日志 docker logs ${RELAY_CT}"
  docker logs --tail 5 "${RELAY_CT}" 2>&1 | sed 's/^/    /'
  exit 1
fi

LANIP=$(hostname -I | awk '{print $1}')
echo
echo "================================================"
echo "  打开浏览器开玩："
echo "    本机:     http://localhost:${PORT}/"
echo "    局域网:   http://${LANIP}:${PORT}/"
echo "  角色参数 --role A1|A2|D1|D2（默认 A1，也可在页面顶栏切换）"
echo "  停止: docker exec ${TRAIN_CT} pkill -f \"[g]ame_server.py\""
echo "================================================"
