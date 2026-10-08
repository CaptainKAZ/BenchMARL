#!/usr/bin/env python3
"""Layup 训练面板（host 侧，纯标准库）：一个端口同时提供 观察窗 / 胜率曲线 / 日志。

路由
  /                      -> 本文件同目录的 live_view.html（每次请求重新读取，改完刷新即可）
  /live_env.json         -> BenchMARL/outputs/live/live_env.json（训练进程每迭代写）
  /curve.json            -> 解析日志统计块得到的曲线序列（前端 Chart.js 直接渲染，不再用 matplotlib）
  /chart.umd.min.js      -> 本目录自带的前端绘图库 Chart.js（离线可用）
  /logs.json             -> outputs/train_v*.log 列表（名/大小/修改时间）
  /log?n=200&file=xxx.log-> 指定日志（默认最新）末尾 n 行
  /status.json           -> 训练是否在跑 / tqdm 进度 / 最新日志 / 观察窗数据新旧

用法：python3 BenchMARL/liveview/serve_dashboard.py --port 8765 [--bind 0.0.0.0]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent          # BenchMARL/liveview/
HOST = HERE.parent.parent                        # 仓库根
OUT = HERE.parent / "outputs"                    # BenchMARL/outputs
LIVE = OUT / "live"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it\]")
CODE_RE = re.compile(r"码\s*(\d+)\):\s*(\d+)\s*次\s*\(([\d.]+)%\)")
WIN_RE = re.compile(r"胜率:\s*([\d.]+)%")

# ---- 曲线（前端渲染）：从日志统计块解析出每个批次的一行数据 ----
CURVE_CODES = (1, 2, 3, 4, 5, 11, 12, 13, 14, 15)
CURVE_DEFAULT_GLOBS = ("train_v40c.log", "train_v41*.log", "train_v42*.log", "train_v43*.log")
_CURVE_CACHE = {"key": None, "t": 0.0, "data": None}


def newest_log() -> Path | None:
    logs = [Path(p) for p in glob.glob(str(OUT / "train_v*.log"))]
    logs = [p for p in logs if p.is_file()]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def training_running() -> int:
    try:
        r = subprocess.run(["pgrep", "-fc", "clear_restore.py"], capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def tail_lines(p: Path, n: int) -> str:
    """高效取末尾 n 行（避免整文件读入）。"""
    if not p or not p.is_file():
        return ""
    n = max(1, min(n, 5000))
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        blk = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(blk, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
        lines = data.splitlines()
    return "\n".join(l.decode("utf-8", "replace") for l in lines[-n:])


def curve_logs(q) -> list[Path]:
    """决定解析哪些日志：?files=a.log,b.log 或默认的当前 run 组合。"""
    raw = (q.get("files") or [""])[0].strip()
    if raw:
        paths = []
        for name in raw.split(","):
            name = name.strip()
            if "/" in name or not name.endswith(".log"):
                continue
            p = OUT / name
            if p.is_file():
                paths.append(p)
    else:
        found = []
        for pat in CURVE_DEFAULT_GLOBS:
            found += glob.glob(str(OUT / pat))
        paths = [Path(p) for p in found if Path(p).is_file()]
        if not paths:
            f = newest_log()
            paths = [f] if f else []
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths


def parse_curve(paths) -> dict:
    """把若干日志的「回合结束原因统计」块解析成一条连续序列（跨文件按修改时间累计批次号）。"""
    pts, files, x = [], [], 0
    for p in paths:
        files.append(p.name)
        txt = p.read_text(errors="replace")
        for seg in txt.split("回合结束原因统计")[1:]:
            seg = seg[:4000]
            wm = WIN_RE.search(seg)
            if not wm:            # 半截块（还没打完）直接跳过
                continue
            codes = {}
            for cm in CODE_RE.finditer(seg):
                codes[int(cm.group(1))] = float(cm.group(3))
            x += 1
            pt = {"x": x, "file": p.name, "win": float(wm.group(1)),
                  "auto_win": round(sum(codes.get(c, 0.0) for c in (1, 2, 3, 4, 5)), 2)}
            for c in CURVE_CODES:
                pt["c%d" % c] = codes.get(c)
            pts.append(pt)
    w = 5
    for i, pt in enumerate(pts):          # 5 窗滑动平均
        vals = [q["win"] for q in pts[max(0, i - w + 1):i + 1]]
        pt["win_ma"] = round(sum(vals) / len(vals), 2)
    return {"files": files, "n": len(pts), "points": pts,
            "mtime": datetime.now().strftime("%H:%M:%S")}


def curve_json(paths) -> dict:
    key = tuple((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in paths)
    now = time.time()
    if _CURVE_CACHE["key"] == key and now - _CURVE_CACHE["t"] < 20 and _CURVE_CACHE["data"] is not None:
        return _CURVE_CACHE["data"]
    data = parse_curve(paths)
    _CURVE_CACHE.update(key=key, t=now, data=data)
    return data


class Handler(BaseHTTPRequestHandler):
    server_version = "LayupDashboard/1.0"

    def log_message(self, *a):  # 静默
        pass

    def _send(self, code, body: bytes, ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            f = HERE / "live_view.html"
            if not f.is_file():
                return self._send(404, b"live_view.html not found")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

        if u.path == "/live_env.json":
            f = LIVE / "live_env.json"
            if not f.is_file():
                return self._json({"error": "还没生成 live_env.json（LIVE_VIEW=1 的训练跑起来后就有了）"}, 404)
            return self._send(200, f.read_bytes(), "application/json; charset=utf-8")

        if u.path == "/chart.umd.min.js":
            f = HERE / "chart.umd.min.js"
            if not f.is_file():
                return self._send(404, b"chart.umd.min.js not found")
            return self._send(200, f.read_bytes(), "application/javascript; charset=utf-8")

        if u.path == "/curve.json":
            paths = curve_logs(q)
            data = curve_json(paths)
            return self._json({**data, "requested": [p.name for p in paths]})

        if u.path == "/logs.json":
            logs = [Path(p) for p in glob.glob(str(OUT / "train_v*.log")) if Path(p).is_file()]
            logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return self._json([{"name": p.name, "size": p.stat().st_size,
                                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M:%S")}
                               for p in logs[:30]])

        if u.path == "/log":
            name = (q.get("file") or [""])[0]
            if name:
                if "/" in name or not name.endswith(".log"):   # 防目录穿越
                    return self._send(400, b"bad file name")
                f = OUT / name
            else:
                f = newest_log()
            n = int((q.get("n") or ["200"])[0])
            return self._send(200, tail_lines(f, n).encode(), "text/plain; charset=utf-8",
                              {"X-Log-File": f.name if f else "-"})

        if u.path == "/status.json":
            f = newest_log()
            txt = tail_lines(f, 4000) if f else ""
            m = None
            for m in TQDM_RE.finditer(txt):
                pass
            prog = {"cur": int(m.group(1)), "total": int(m.group(2)), "elapsed": m.group(3),
                    "eta": m.group(4), "s_it": float(m.group(5))} if m else None
            # 最近一个统计块
            codes = {}
            win = None
            for block in txt.split("回合结束原因统计")[-1:]:
                for cm in CODE_RE.finditer(block):
                    codes[cm.group(1)] = float(cm.group(3))
                wm = WIN_RE.search(block)
                win = float(wm.group(1)) if wm else None
            le = LIVE / "live_env.json"
            live = None
            if le.is_file():
                try:
                    j = json.loads(le.read_text())
                    live = {"iter": j.get("iter"), "episodes": len(j.get("episodes", [])),
                            "age_s": round(time.time() - le.stat().st_mtime, 1)}
                except Exception:
                    pass
            return self._json({"training_procs": training_running(),
                               "log": f.name if f else None,
                               "log_mtime": datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S") if f else None,
                               "progress": prog, "last_block_codes": codes, "last_block_win": win,
                               "live": live,
                               "server_time": datetime.now().strftime("%H:%M:%S")})

        return self._send(404, b"not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"[dashboard] http://{a.bind}:{a.port}/  (root={OUT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
