import torch
import numpy as np
import sys

# =================配置区=================
# 你的 checkpoint 路径
CHECKPOINT_PATH = "outputs/2025-12-17_17-16-42/ensemblealgorithm_layup_ensemblemodel__cfac0534_25_12_17-17_16_42/checkpoints/checkpoint_750000.pt" 
# 阈值：如果权重绝对值超过这个数，就报警
WARNING_THRESHOLD = 10.0  
# 阈值：如果优化器动量(Variance)超过这个数，说明之前梯度炸过
OPTIMIZER_THRESHOLD = 100.0 
# =======================================

def analyze_tensor(name, tensor, threshold):
    if not isinstance(tensor, torch.Tensor):
        return None
    
    # 转换为 float 以防某些类型不支持统计
    t = tensor.float().cpu()
    
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    
    min_val = t.min().item()
    max_val = t.max().item()
    abs_max = t.abs().max().item()
    mean_val = t.mean().item()
    std_val = t.std().item() if t.numel() > 1 else 0.0

    status = "✅ 正常"
    if has_nan or has_inf:
        status = "☠️ 损坏 (NaN/Inf)"
    elif abs_max > threshold:
        status = f"⚠️ 膨胀 (Max > {threshold})"
    
    return {
        "name": name,
        "status": status,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "abs_max": abs_max,
        "shape": t.shape
    }

def recursive_inspect(data, prefix="", threshold=10.0):
    results = []
    if isinstance(data, dict):
        for k, v in data.items():
            results.extend(recursive_inspect(v, prefix + k + ".", threshold))
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            results.extend(recursive_inspect(v, prefix + str(i) + ".", threshold))
    elif isinstance(data, torch.Tensor):
        # 这是一个 Tensor，分析它
        res = analyze_tensor(prefix[:-1], data, threshold) # 去掉末尾的点
        if res: results.append(res)
    return results

def main():
    print(f"正在加载 Checkpoint: {CHECKPOINT_PATH} ...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print("加载成功，开始扫描...\n")
    
    # 1. 扫描模型权重 (通常在 keys 中包含 'loss' 或 'policy')
    # 根据 BenchMARL experiment.py，权重可能在 'loss_group_name' 中
    print("=== 🔍 模型权重分析 (Model Weights) ===")
    weight_issues = []
    all_tensors = recursive_inspect(checkpoint, threshold=WARNING_THRESHOLD)
    
    # 过滤一下，只看看起来像权重的（跳过一些标量统计数据）
    # 这里的过滤逻辑比较宽泛，主要为了抓大鱼
    for res in all_tensors:
        # 忽略优化器状态，单独分析
        # if "optimizer" in res['name'] or "state_dict" not in res['name']: 
        #     continue
            
        print(f"[{res['status']}] {res['name']:<60} | Range: [{res['min']:.4f}, {res['max']:.4f}] | Mean: {res['mean']:.4f} shape {res['shape']}")
        
        if "⚠️" in res['status'] or "☠️" in res['status']:
            weight_issues.append(res)

    print("\n" + "="*80)
    
    # 2. 扫描优化器状态 (Optimizer State) - 这是最重要的早期预警！
    # 如果 Adam 的 exp_avg_sq 很大，说明梯度曾经炸过
    print("\n=== 🔍 优化器状态分析 (Optimizer State) ===")
    print("注意：如果 exp_avg_sq (二阶动量) 很大，说明之前的梯度非常大。")
    
    optimizer_issues = []
    for res in all_tensors:
        # 寻找优化器相关的 key
        if "optimizer" in res['name'] and "state" in res['name']:
            # Adam 的状态通常叫 exp_avg (一阶) 和 exp_avg_sq (二阶/方差)
            status = res['status']
            # 对优化器状态使用更宽松的阈值，或者专门检查 exp_avg_sq
            if "exp_avg_sq" in res['name'] and res['max'] > OPTIMIZER_THRESHOLD:
                status = f"🔥 危险 (Huge Variance > {OPTIMIZER_THRESHOLD})"
            
            if "危险" in status or "损坏" in status or "膨胀" in status:
                print(f"[{status}] {res['name']:<60} | Max: {res['max']:.4f}")
                optimizer_issues.append(res)

    # 3. 总结建议
    print("\n" + "="*80)
    print("=== 📊 诊断报告 ===")
    
    if any(r['has_nan'] for r in all_tensors):
        print("❌ 结论：Checkpoint 已损坏。包含 NaN，不可用。请寻找更早的存档。")
    elif weight_issues or optimizer_issues:
        print("⚠️ 结论：Checkpoint 处于“亚健康”状态 (Pre-Explosion)。")
        print(f"   发现 {len(weight_issues)} 个权重参数数值过大。")
        print(f"   发现 {len(optimizer_issues)} 个优化器状态数值异常。")
        print("   建议：")
        print("   1. 可以尝试读取，但必须立刻大幅降低 Learning Rate (例如 * 0.1)。")
        print("   2. 必须开启强力的 Gradient Clipping (1.0 或 0.5)。")
        print("   3. 检查是否有特定的层 (如 self_embed) 已经开始飙升。")
    else:
        print("✅ 结论：Checkpoint 数据健康。权重在正常范围内。可以放心加载。")

if __name__ == "__main__":
    main()