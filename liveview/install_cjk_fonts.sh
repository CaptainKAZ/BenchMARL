#!/usr/bin/env bash
# 给 WSL host 安装中文字体（供 Playwright/Chromium 截图 + 前端渲染使用）
# 复刻自 2026-10-04 的实操：Windows 微软雅黑/黑体 → ~/.local/share/fonts（无需 sudo）
set -e
DEST="$HOME/.local/share/fonts"
mkdir -p "$DEST"
for f in /mnt/c/Windows/Fonts/msyh.ttc /mnt/c/Windows/Fonts/msyhbd.ttc /mnt/c/Windows/Fonts/simhei.ttf; do
  [ -f "$f" ] || { echo "跳过（不存在）: $f"; continue; }
  cp -n "$f" "$DEST/" 2>/dev/null || true
done
fc-cache -f >/dev/null
echo "已安装到 $DEST:"
ls -la "$DEST" | tail -n +2 | awk '{printf "  %.1fMB %s\n", $5/1048576, $9}'
echo "fc-match Microsoft YaHei -> $(fc-match 'Microsoft YaHei')"
echo "fc-match SimHei          -> $(fc-match SimHei)"
