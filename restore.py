#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from tabnanny import check
from benchmarl.hydra_config import reload_experiment_from_file
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
import glob
import os


def find_latest_file(path, pattern='*'):
    """
    查找指定路径下最新的文件
    
    参数:
        path (str): 要搜索的目录路径
        pattern (str): 文件匹配模式，默认为所有文件
    
    返回:
        str: 最新文件的完整路径，如果没有文件则返回None
    """
    # 获取所有匹配的文件列表
    files = glob.glob(os.path.join(path, pattern))
    
    # 过滤掉目录，只保留文件
    files = [f for f in files if os.path.isfile(f)]
    
    if not files:
        return None
    
    # 按修改时间排序文件（最新的排在最前面）
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files[0]

newest_checkpoint = "outputs/2025-06-27/05-28-46/mappo_layup_mlp__9269b3f4_25_06_27-05_28_46/checkpoints/checkpoint_19251200.pt"
# checkpoint_path = "outputs/2025-06-27/05-28-46/mappo_layup_mlp__9269b3f4_25_06_27-05_28_46/checkpoints"
checkpoint_path = "outputs/2025-07-01/17-04-36/mappo_layup_mlp__752ba5de_25_07_01-17_04_36/checkpoints"
# checkpoint_path = "outputs/2025-07-02/16-18-35/mappo_layup_mlp__56026589_25_07_02-16_18_35/checkpoints"

if __name__ == "__main__":
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    experiment_config.restore_file = find_latest_file(checkpoint_path,"*.pt")
    

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.LAYUP.get_from_yaml()

    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = MappoConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()