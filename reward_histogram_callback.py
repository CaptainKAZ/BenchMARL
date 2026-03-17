"""
奖励直方图记录回调 (按Agent分组)

使用方法:
    from reward_histogram_callback import RewardHistogramCallback

    experiment = Experiment(
        ...,
        callbacks=[
            WinRateReportDebounced(),
            RewardHistogramCallback(log_interval=10),
        ]
    )
"""

import torch
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from typing import Dict, List


class RewardHistogramCallback(Callback):
    """
    记录奖励分布直方图的回调函数 (按每个Agent单独记录)

    Args:
        log_interval (int): 每隔多少次迭代记录一次直方图 (默认: 10)
        log_terminal_reward (bool): 是否记录 terminal_reward 直方图 (默认: True)
        log_episode_reward (bool): 是否记录 episode_reward 直方图 (默认: True)
        bins (str): 直方图的分箱方式 (默认: 'auto')
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_terminal_reward: bool = True,
        log_episode_reward: bool = True,
        bins: str = "auto",
    ):
        super().__init__()
        self.log_interval = log_interval
        self.log_terminal_reward = log_terminal_reward
        self.log_episode_reward = log_episode_reward
        self.bins = bins
        self.iteration_count = 0

        self.terminal_reward_buffer: Dict[str, List[torch.Tensor]] = {}
        self.episode_reward_buffer: Dict[str, List[torch.Tensor]] = {}

        print(
            f"[RewardHistogram] Callback initialized with log_interval={log_interval}"
        )

    def on_setup(self):
        for group, agents in self.experiment.group_map.items():
            for agent in agents:
                self.terminal_reward_buffer[agent] = []
                self.episode_reward_buffer[agent] = []
        print(
            f"[RewardHistogram] Setup complete. Tracking agents: {list(self.terminal_reward_buffer.keys())}"
        )

    def on_batch_collected(self, batch: TensorDictBase):
        global_done = batch.get(("next", "done")).squeeze(-1).bool()

        if global_done.sum() == 0:
            return

        for group, agents in self.experiment.group_map.items():
            n_agents = len(agents)

            if self.log_terminal_reward:
                try:
                    terminal_reward = batch.get(
                        ("next", group, "info", "terminal_reward")
                    )
                    if terminal_reward is not None:
                        done_terminal = terminal_reward[global_done]
                        if done_terminal.numel() > 0:
                            for agent_idx, agent in enumerate(agents):
                                if (
                                    done_terminal.dim() >= 2
                                    and done_terminal.shape[-2] >= n_agents
                                ):
                                    agent_reward = done_terminal[..., agent_idx, :]
                                else:
                                    agent_reward = done_terminal
                                if agent_reward.numel() > 0:
                                    self.terminal_reward_buffer[agent].append(
                                        agent_reward.detach().cpu().flatten()
                                    )
                except KeyError:
                    pass

            if self.log_episode_reward:
                try:
                    episode_reward = batch.get(("next", group, "episode_reward"))
                    if episode_reward is not None:
                        done_episode = episode_reward[global_done]
                        if done_episode.numel() > 0:
                            for agent_idx, agent in enumerate(agents):
                                if (
                                    done_episode.dim() >= 2
                                    and done_episode.shape[-2] >= n_agents
                                ):
                                    agent_reward = done_episode[..., agent_idx, :]
                                else:
                                    agent_reward = done_episode
                                if agent_reward.numel() > 0:
                                    self.episode_reward_buffer[agent].append(
                                        agent_reward.detach().cpu().flatten()
                                    )
                except KeyError:
                    pass

    def on_train_end(self, training_td: TensorDictBase, group: str):
        groups = list(self.experiment.group_map.keys())
        if group == groups[-1]:
            self.iteration_count += 1

            if self.iteration_count % self.log_interval == 0:
                self._log_histograms()
                self._clear_buffers()

    def _log_histograms(self):
        step = self.experiment.n_iters_performed

        for logger in self.experiment.logger.loggers:
            from torchrl.record import TensorboardLogger, WandbLogger

            if not isinstance(logger, (TensorboardLogger, WandbLogger)):
                continue

            for group, agents in self.experiment.group_map.items():
                for agent in agents:
                    if self.log_terminal_reward and self.terminal_reward_buffer[agent]:
                        terminal_data = torch.cat(self.terminal_reward_buffer[agent])
                        if terminal_data.numel() > 0:
                            try:
                                logger.log_histogram(
                                    name=f"rewards/{agent}/terminal_reward_histogram",
                                    data=terminal_data.numpy(),
                                    step=step,
                                    bins=self.bins,
                                )
                                print(
                                    f"[RewardHistogram] {agent} terminal_reward: "
                                    f"n={terminal_data.numel()}, "
                                    f"mean={terminal_data.mean():.2f}, "
                                    f"std={terminal_data.std():.2f}"
                                )
                            except Exception as e:
                                print(
                                    f"[RewardHistogram] Warning: Failed to log {agent} terminal_reward histogram: {e}"
                                )

                    if self.log_episode_reward and self.episode_reward_buffer[agent]:
                        episode_data = torch.cat(self.episode_reward_buffer[agent])
                        if episode_data.numel() > 0:
                            try:
                                logger.log_histogram(
                                    name=f"rewards/{agent}/episode_reward_histogram",
                                    data=episode_data.numpy(),
                                    step=step,
                                    bins=self.bins,
                                )
                                print(
                                    f"[RewardHistogram] {agent} episode_reward: "
                                    f"n={episode_data.numel()}, "
                                    f"mean={episode_data.mean():.2f}, "
                                    f"std={episode_data.std():.2f}"
                                )
                            except Exception as e:
                                print(
                                    f"[RewardHistogram] Warning: Failed to log {agent} episode_reward histogram: {e}"
                                )

    def _clear_buffers(self):
        for agent in self.terminal_reward_buffer.keys():
            self.terminal_reward_buffer[agent] = []
            self.episode_reward_buffer[agent] = []


class RewardDistributionCallback(Callback):
    """
    按终止原因分类记录奖励分布 (按每个Agent单独记录)

    Args:
        log_interval (int): 记录间隔
        log_by_termination_reason (bool): 是否按终止原因分类记录 terminal_reward
    """

    TERMINATION_REASONS = {
        1: "shot_made",
        2: "opponent_foul",
        3: "opponent_wall",
        4: "opponent_overmidline",
        5: "opponent_friendly_fire",
        11: "shot_blocked",
        12: "timeout",
        13: "team_foul",
        14: "team_wall",
        15: "team_friendly_fire",
    }

    WIN_CODES = {1, 2, 3, 4, 5}
    LOSS_CODES = {11, 12, 13, 14, 15}

    def __init__(
        self,
        log_interval: int = 10,
        log_by_termination_reason: bool = True,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.log_by_termination_reason = log_by_termination_reason
        self.iteration_count = 0

        self.terminal_rewards_by_reason: Dict[str, Dict[int, List[torch.Tensor]]] = {}
        self.episode_reward_buffer: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    def on_setup(self):
        for group, agents in self.experiment.group_map.items():
            for agent in agents:
                self.terminal_rewards_by_reason[agent] = {
                    reason: [] for reason in self.TERMINATION_REASONS.keys()
                }
                self.episode_reward_buffer[agent] = {
                    "win": [],
                    "loss": [],
                }

    def on_batch_collected(self, batch: TensorDictBase):
        global_done = batch.get(("next", "done")).squeeze(-1).bool()

        if global_done.sum() == 0:
            return

        try:
            termination_reason = batch.get(
                ("next", "attacker", "info", "termination_reason")
            )
            if termination_reason is not None:
                reasons_flat = termination_reason[global_done].squeeze(-1).long()
                reasons = reasons_flat[:, 0]
            else:
                reasons = None
        except KeyError:
            reasons = None

        for group, agents in self.experiment.group_map.items():
            n_agents = len(agents)

            try:
                terminal_reward = batch.get(("next", group, "info", "terminal_reward"))
                if terminal_reward is not None:
                    done_terminal = terminal_reward[global_done]

                    if done_terminal.dim() == 3:
                        done_terminal = done_terminal.mean(dim=1)

                    done_terminal = done_terminal.squeeze(-1)

                    for agent_idx, agent in enumerate(agents):
                        if (
                            done_terminal.dim() >= 2
                            and done_terminal.shape[-1] >= n_agents
                        ):
                            agent_reward = done_terminal[..., agent_idx]
                        else:
                            agent_reward = done_terminal

                        if reasons is not None and self.log_by_termination_reason:
                            for reason_code in self.TERMINATION_REASONS.keys():
                                mask = reasons == reason_code
                                if mask.any():
                                    self.terminal_rewards_by_reason[agent][
                                        reason_code
                                    ].append(agent_reward[mask].detach().cpu())
            except KeyError:
                pass

            try:
                episode_reward = batch.get(("next", group, "episode_reward"))
                if episode_reward is not None:
                    done_episode = episode_reward[global_done]

                    if done_episode.dim() == 3:
                        done_episode = done_episode.mean(dim=1)
                    done_episode = done_episode.squeeze(-1)

                    for agent_idx, agent in enumerate(agents):
                        if (
                            done_episode.dim() >= 2
                            and done_episode.shape[-1] >= n_agents
                        ):
                            agent_reward = done_episode[..., agent_idx]
                        else:
                            agent_reward = done_episode

                        if reasons is not None:
                            win_mask = torch.zeros_like(reasons, dtype=torch.bool)
                            for code in self.WIN_CODES:
                                win_mask |= reasons == code

                            if win_mask.any():
                                self.episode_reward_buffer[agent]["win"].append(
                                    agent_reward[win_mask].detach().cpu()
                                )

                            loss_mask = torch.zeros_like(reasons, dtype=torch.bool)
                            for code in self.LOSS_CODES:
                                loss_mask |= reasons == code

                            if loss_mask.any():
                                self.episode_reward_buffer[agent]["loss"].append(
                                    agent_reward[loss_mask].detach().cpu()
                                )
            except KeyError:
                pass

    def on_train_end(self, training_td: TensorDictBase, group: str):
        groups = list(self.experiment.group_map.keys())
        if group == groups[-1]:
            self.iteration_count += 1

            if self.iteration_count % self.log_interval == 0:
                self._log_histograms()
                self._clear_buffers()

    def _log_histograms(self):
        step = self.experiment.n_iters_performed

        for logger in self.experiment.logger.loggers:
            from torchrl.record import TensorboardLogger, WandbLogger

            if not isinstance(logger, (TensorboardLogger, WandbLogger)):
                continue

            for group, agents in self.experiment.group_map.items():
                for agent in agents:
                    for reason_code, reason_name in self.TERMINATION_REASONS.items():
                        if self.terminal_rewards_by_reason[agent][reason_code]:
                            data = torch.cat(
                                self.terminal_rewards_by_reason[agent][reason_code]
                            )
                            if data.numel() > 0:
                                try:
                                    logger.log_histogram(
                                        name=f"rewards/{agent}/terminal_by_reason/{reason_name}",
                                        data=data.numpy(),
                                        step=step,
                                        bins="auto",
                                    )
                                except Exception as e:
                                    print(
                                        f"[RewardDistribution] Warning: Failed to log {agent} {reason_name} histogram: {e}"
                                    )

                    for outcome in ["win", "loss"]:
                        if self.episode_reward_buffer[agent][outcome]:
                            data = torch.cat(self.episode_reward_buffer[agent][outcome])
                            if data.numel() > 0:
                                try:
                                    logger.log_histogram(
                                        name=f"rewards/{agent}/episode_reward_{outcome}",
                                        data=data.numpy(),
                                        step=step,
                                        bins="auto",
                                    )
                                except Exception as e:
                                    print(
                                        f"[RewardDistribution] Warning: Failed to log {agent} {outcome} histogram: {e}"
                                    )

    def _clear_buffers(self):
        for agent in self.terminal_rewards_by_reason.keys():
            for reason in self.TERMINATION_REASONS.keys():
                self.terminal_rewards_by_reason[agent][reason] = []
            self.episode_reward_buffer[agent] = {"win": [], "loss": []}
