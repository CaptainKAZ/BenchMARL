import torch
import torch.nn as nn
from benchmarl.experiment.callback import Callback

class HealthCheckCallback(Callback):
    """
    [修复版] 神经网络健康体检回调函数
    
    修复内容：
    - 增加了对 torch.vmap (MultiAgentMLP) 的兼容性处理。
    - 如果检测到层在 vmap 环境下运行，会自动跳过"激活值检查"并打印警告，但仍保留"梯度检查"。
    """
    
    def __init__(
        self, 
        log_every_n_steps: int = 50, 
        dormancy_threshold: float = 0.1, 
        grad_threshold: float = 1e-7,
        gelu_inactive_threshold: float = 1e-3
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.dormancy_threshold = dormancy_threshold
        self.grad_threshold = grad_threshold
        self.gelu_inactive_threshold = gelu_inactive_threshold
        
        self.tracked_layers = {}
        self.activation_stats = {} 
        self.step_counter = 0
        self.disabled_layers = set() # 记录因 vmap 而禁用的层

    def on_setup(self):
        print(f"\n[HealthCheck] 初始化体检回调... (Log freq: {self.log_every_n_steps})")
        
        for group in self.experiment.group_map.keys():
            self.tracked_layers[group] = []
            
            # 监听 Policy (Actor)
            policy = self.experiment.algorithm.get_policy_for_loss(group)
            self._register_hooks(policy, group, prefix="actor")
            
            # 监听 Critic (如果存在)
            if self.experiment.algorithm.has_critic:
                critic = self.experiment.algorithm.get_critic(group)
                self._register_hooks(critic, group, prefix="critic")

    def _register_hooks(self, module, group, prefix):
        count = 0
        for name, layer in module.named_modules():
            if isinstance(layer, nn.Linear):
                safe_name = name.replace('.', '_')
                layer_name = f"{group}/{prefix}_{safe_name}" if safe_name else f"{group}/{prefix}_output"
                
                self.tracked_layers[group].append({
                    "layer": layer,
                    "name": layer_name
                })
                
                self._init_layer_stats(layer)
                layer.register_forward_hook(self._get_hook(layer, layer_name))
                count += 1
        print(f"[HealthCheck] Group '{group}' {prefix}: 监控了 {count} 个线性层。")

    def _init_layer_stats(self, layer):
        self.activation_stats[layer] = {
            "sum_abs": 0.0,
            "inactive_count": 0,
            "total_samples": 0
        }

    def _get_hook(self, layer, layer_name):
        def hook(module, input, output):
            # 如果该层已被标记为禁用 (vmap冲突)，直接返回
            if layer in self.disabled_layers: return
            if layer not in self.activation_stats: return
            
            try:
                with torch.no_grad():
                    # output shape: [batch, ..., features]
                    flattened = output.reshape(-1, output.shape[-1])
                    
                    # --- 关键修改：尝试计算并获取 .item() ---
                    # 如果在 vmap 上下文中，.item() 会抛出 RuntimeError
                    
                    # 1. 累加绝对值
                    abs_sum = flattened.abs().sum(dim=0)
                    
                    # 2. 统计静默神经元 (GELU check)
                    is_inactive_sum = (flattened.abs() < self.gelu_inactive_threshold).sum()

                    # 3. 尝试写入 (这里会触发 vmap 错误)
                    # 我们必须在这里做加法，因为不能把 Tracer Tensor 带出 vmap
                    self.activation_stats[layer]["sum_abs"] += abs_sum.detach().cpu()
                    self.activation_stats[layer]["inactive_count"] += is_inactive_sum.item()
                    self.activation_stats[layer]["total_samples"] += flattened.shape[0]

            except RuntimeError as e:
                # 捕获 vmap 错误
                if "vmap" in str(e) or "BatchedTensor" in str(e):
                    if layer not in self.disabled_layers:
                        print(f"\n[HealthCheck Warning] ⚠️ 无法监控层 '{layer_name}' 的激活值。")
                        print(f"  原因: 该层正在 torch.vmap 上下文中运行 (MAPPO/MultiAgentMLP 的特性)。")
                        print(f"  措施: 已禁用该层的【激活值】监控，但【梯度】监控仍然有效！")
                        self.disabled_layers.add(layer)
                        # 清理该层统计，避免后续报错
                        if layer in self.activation_stats:
                            del self.activation_stats[layer]
                else:
                    raise e # 如果是其他错误，继续抛出
        return hook

    def on_train_end(self, training_td, group):
        self.step_counter += 1
        if self.step_counter % self.log_every_n_steps == 0:
            self._log_health_stats(group)
            self._reset_all_stats(group)

    def _log_health_stats(self, group):
        log_data = {}
        layers_info = self.tracked_layers.get(group, [])
        optimizers = self.experiment.optimizers.get(group, {})
        
        # --- 1. 激活值健康度 (跳过已被禁用的层) ---
        for info in layers_info:
            layer = info["layer"]
            name = info["name"]
            
            # 如果层被禁用或没有数据，跳过
            if layer in self.disabled_layers or layer not in self.activation_stats:
                continue
                
            stats = self.activation_stats.get(layer)
            if stats["total_samples"] == 0: continue

            # [指标] Inactive Ratio & Dormancy Ratio
            total_elements = stats["total_samples"] * layer.out_features
            inactive_ratio = stats["inactive_count"] / total_elements
            log_data[f"health_act/{name}_inactive_ratio"] = inactive_ratio

            avg_abs_act = stats["sum_abs"] / stats["total_samples"]
            layer_mean = avg_abs_act.mean()
            if layer_mean > 1e-9:
                normalized_act = avg_abs_act / layer_mean
                dormant_ratio = (normalized_act <= self.dormancy_threshold).float().mean().item()
            else:
                dormant_ratio = 1.0
            
            log_data[f"health_act/{name}_dormancy_ratio"] = dormant_ratio

        # --- 2. 梯度健康度 (完全不受 vmap 影响，重点关注！) ---
        dead_grad_metrics = []
        for opt_name, opt in optimizers.items():
            for i, param_group in enumerate(opt.param_groups):
                for p_idx, param in enumerate(param_group['params']):
                    if param.grad is None: continue
                    
                    state = opt.state[param]
                    if 'exp_avg_sq' in state:
                        exp_avg_sq = state['exp_avg_sq']
                        
                        # 检查所有矩阵参数 (Attention, GRU, MLP)
                        if len(param.shape) >= 2:
                            # 对除了第一维之外的维度求平均
                            dims_to_mean = list(range(1, len(param.shape)))
                            neuron_grad_energy = exp_avg_sq.mean(dim=dims_to_mean)
                            
                            # Dead Gradient Ratio
                            dead_ratio = (neuron_grad_energy < self.grad_threshold).float().mean().item()
                            
                            shape_str = "x".join(map(str, param.shape))
                            log_id = f"{group}/{opt_name}_p{p_idx}_{shape_str}"
                            
                            log_data[f"health_grad/{log_id}_dead_ratio"] = dead_ratio
                            dead_grad_metrics.append(dead_ratio)

        # 打印摘要
        max_dormancy = max([v for k,v in log_data.items() if "dormancy" in k] or [0])
        max_dead_grad = max(dead_grad_metrics) if dead_grad_metrics else 0
        
        print(f"\n[HealthCheck Step {self.step_counter}] Group {group}:")
        if max_dormancy == 0 and len(self.disabled_layers) > 0:
             print(f"  - Max Dormancy Ratio: N/A (Disabled due to vmap)")
        else:
             print(f"  - Max Dormancy Ratio: {max_dormancy:.2%}")
        print(f"  - Max Dead Grad Ratio: {max_dead_grad:.2%} (Warning if > 5%)")
        
        self.experiment.logger.log(log_data, step=self.step_counter)

    def _reset_all_stats(self, group):
        for info in self.tracked_layers.get(group, []):
            layer = info["layer"]
            if layer not in self.disabled_layers:
                self._init_layer_stats(layer)