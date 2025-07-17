#!/usr/bin/env python3
import os
import time
import glob

def find_latest_file(directory, pattern='*'):
    """查找目录下最新的文件（按修改时间）"""
    files = glob.glob(os.path.join(directory, pattern))
    files = [f for f in files if os.path.isfile(f)]  # 过滤掉目录
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # 按修改时间排序
    return files[0]  # 返回最新的文件

def update_symlink(directory, symlink_path, pattern='*'):
    """更新符号链接（强制使用绝对路径）"""
    latest_file = find_latest_file(directory, pattern)
    if not latest_file:
        print(f"警告：目录 {directory} 下没有匹配的文件！")
        return False

    symlink_path = os.path.abspath(symlink_path)  # 符号链接目标路径转为绝对路径
    latest_file = os.path.abspath(latest_file)  # 最新文件路径转为绝对路径


    # 如果符号链接已存在，先删除
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.unlink(symlink_path)

    # 创建新的符号链接（绝对路径）
    os.symlink(latest_file, symlink_path)
    print(f"更新符号链接：{symlink_path} -> {latest_file}")
    return True

def main():
    # 配置参数
    WATCH_DIR = "outputs/2025-06-27/05-28-46/mappo_layup_mlp__9269b3f4_25_06_27-05_28_46/mappo_layup_mlp__9269b3f4_25_06_27-05_28_46/videos"  # 要监控的目录
    SYMLINK_PATH = "outputs/videos_latest.mp4"  # 符号链接路径
    FILE_PATTERN = "*.mp4"  # 文件匹配模式（例如 "*.log" 只匹配日志文件）
    INTERVAL = 20  # 检查间隔（秒）

    print(f"开始监控目录 {WATCH_DIR}，每 {INTERVAL} 秒更新符号链接 {SYMLINK_PATH}...")

    try:
        while True:
            update_symlink(WATCH_DIR, SYMLINK_PATH, FILE_PATTERN)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n监控已停止。")

if __name__ == "__main__":
    main()