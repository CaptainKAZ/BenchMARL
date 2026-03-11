#!/bin/bash

# --- 配置区域 ---
PYTHON_FILE="clear_restore.py"  # 你的 Python 文件名
MAX_RUNS=10              # 最大运行次数
TRAINING_MODE="$1"     # 训练模式: cold, cont, atk-c, def-c, atk-a, def-a, both-a
# --- --- --- ---

n=0

echo "开始任务，计划运行 $MAX_RUNS 次。"

# 修改循环条件：当 n 小于 MAX_RUNS 时继续执行
while [ $n -lt $MAX_RUNS ]
do
    # 显示当前进度
    current_count=$((n + 1))
    echo "--------------------------------------"
    echo "[$(date '+%H:%M:%S')] 正在执行第 $current_count / $MAX_RUNS 次运行..."
    echo "训练模式: $TRAINING_MODE"

    # 执行 Python 程序
    # 第一次运行使用配置的模式，之后自动使用 cont 模式继续训练
    if [ $n -eq 0 ]; then
        echo "首次运行，使用模式: $TRAINING_MODE"
        python3 "$PYTHON_FILE" -m "$TRAINING_MODE"
    else
        echo "自动恢复，使用模式: cont"
        python3 "$PYTHON_FILE" -m cont
    fi

    # 增加计数器
    n=$((n + 1))

    # 如果还没达到最大次数，则计算并执行等待
    if [ $n -lt $MAX_RUNS ]; then
        WAIT_TIME=$((10 + 10 * (n-1))) # 这里维持你之前的公式逻辑
        echo "运行结束。等待 $WAIT_TIME 秒后进行下一次运行..."
        sleep $WAIT_TIME
    else
        echo "--------------------------------------"
        echo "[$(date '+%H:%M:%S')] 已完成全部 $MAX_RUNS 次运行，脚本退出。"
    fi
done