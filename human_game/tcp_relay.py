#!/usr/bin/env python3
"""人机大战 · 端口中继容器入口（纯标准库，无三方依赖）

背景：rootless Docker 下 host 无法直连容器 IP（docker0 linkdown），
主容器 robocon2025-marl 又没有发布端口。本脚本跑在一个专用的"中继"容器里：
它自己在容器内监听 8766，并把每条 TCP 连接转发到主容器 172.17.0.2:8766。
中继容器用 `docker run -p 8766:8766` 创建 ⇒ host/手机浏览器可直连 8766。

由 start_game.sh 自动创建/启动，一般不需要手工运行。
"""
import argparse
import socket
import threading


def pipe(src: socket.socket, dst: socket.socket):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client: socket.socket, target):
    try:
        upstream = socket.create_connection(target, timeout=5)
    except OSError:
        client.close()
        return
    client.settimeout(None)
    upstream.settimeout(None)
    threads = [
        threading.Thread(target=pipe, args=(client, upstream), daemon=True),
        threading.Thread(target=pipe, args=(upstream, client), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8766)
    ap.add_argument("--target", default="172.17.0.2:8766", help="主容器 game_server 地址")
    args = ap.parse_args()
    host, port = args.target.rsplit(":", 1)
    port = int(port)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.listen))
    srv.listen(64)
    print(f"[relay] 0.0.0.0:{args.listen} -> {host}:{port} 就绪", flush=True)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client, (host, port)), daemon=True).start()


if __name__ == "__main__":
    main()
