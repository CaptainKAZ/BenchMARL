#!/usr/bin/env python3
"""
Standalone script to visualize critic value function heatmap.
Loads checkpoint and renders A1's value landscape overlayed on VMAS environment.
"""

import torch
import numpy as np
import os
import glob
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from tensordict import TensorDict


# --- Definitions required for unpickling the checkpoint ---

# Dummy class to satisfy unpickling. We don't need the logic for visualization.
class WinRateReportDebounced(Callback):
    def __init__(self, *args, **kwargs):
        pass
    def on_setup(self):
        pass
    def on_batch_collected(self, batch):
        pass


def find_latest_checkpoint(pattern: str):
    """
    Find newest checkpoint matching glob pattern.

    Args:
        pattern: Glob pattern for checkpoint files

    Returns:
        str: Path to latest checkpoint, or None if not found
    """
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def create_critic_value_function(critic_network, env, device, field_params, critic_input_keys, agent_to_vary=0):
    """
    Create value function wrapper that evaluates critic at different agent positions.

    The function maintains all other entities at their current state while varying
    the specified agent's position across the grid.

    Args:
        critic_network: Attacker value network from experiment
        env: VMAS layup environment
        device: torch device
        field_params: Dict with W, L, v_max for normalization
        critic_input_keys: List of recurrent hidden state keys (e.g., "_recurrent_state_h")
        agent_to_vary: Which agent's position to vary (0=A1, 1=A2, 2=D1, 3=D2)

    Returns:
        Callable: (n_points, 2) -> (n_points,) numpy function
    """
    # Extract normalization parameters
    max_x = field_params["W"] / 2.0
    max_y = field_params["L"] / 2.0
    max_v = field_params["v_max"]
    pos_divisor = torch.tensor([max_x, max_y], device=device)

    # Get the underlying model if compiled with torch._dynamo
    if hasattr(critic_network, "_orig_mod"):
        critic_model = critic_network._orig_mod
    else:
        critic_model = critic_network

    # Initialize recurrent hidden states by calling critic once with dummy data
    # This creates proper zero-initialized hidden states for GRU/LSTM layers
    with torch.no_grad():
        dummy_state = env._env.scenario.get_global_state()[0:1]  # [1, 23]
        dummy_td = TensorDict({
            "state": dummy_state.unsqueeze(1),  # [1, 1, 23] - add sequence dim
            "is_init": torch.ones(1, 1, 1, device=device, dtype=torch.bool)
        }, batch_size=[1, 1], device=device)
        _ = critic_network(dummy_td)

        # Extract initialized hidden states (e.g., GRU hidden states)
        # Note: if critic_input_keys is empty, there are no recurrent states
        hidden_state_template = {k: dummy_td.get(k) for k in critic_input_keys} if critic_input_keys else {}

    def value_fn(positions_np):
        """
        Evaluate critic value at grid positions.

        Args:
            positions_np: (n_points, 2) numpy array of x,y world coordinates

        Returns:
            (n_points,) numpy array of value estimates (flattened for rendering)
        """
        n_points = positions_np.shape[0]

        # Get current global state from environment
        # Global state structure (23D total):
        # - flat_agent_states: 16D (4 agents × [x, y, vx, vy]) - ALREADY NORMALIZED
        # - spot_pos: 2D - ALREADY NORMALIZED
        # - is_in_spot_a1: 1D
        # - a1_shoot_process: 1D
        # - basket_pos: 2D - ALREADY NORMALIZED
        # - time_obs: 1D
        current_state = env._env.scenario.get_global_state()  # [batch_dim, 23]

        # Extract first environment if batched
        if current_state.dim() > 1 and current_state.shape[0] > 1:
            current_state = current_state[0:1]  # [1, 23]

        # Expand state to match number of grid points
        batched_states = current_state.expand(n_points, -1).clone()  # [n_points, 23]

        # Normalize grid positions (positions_np is in world coordinates, need to normalize)
        positions_tensor = torch.from_numpy(positions_np).float().to(device)
        normalized_positions = positions_tensor / pos_divisor  # [n_points, 2]

        # Replace the specified agent's position in the global state
        # Agent states are: A1[x,y,vx,vy], A2[x,y,vx,vy], D1[x,y,vx,vy], D2[x,y,vx,vy]
        # agent_to_vary: 0=A1, 1=A2, 2=D1, 3=D2
        start_idx = agent_to_vary * 4  # Each agent has 4 dims: [x, y, vx, vy]
        batched_states[:, start_idx:start_idx+2] = normalized_positions

        # Forward through critic
        with torch.no_grad():
            # Prepare input TensorDict for critic
            # Include all required keys: state, is_init, and hidden states
            critic_input_dict = {
                "state": batched_states.unsqueeze(1),  # [n_points, 1, 23] - add sequence dim
                "is_init": torch.ones(n_points, 1, 1, device=device, dtype=torch.bool)
            }

            # Add recurrent hidden state keys, expanding from template
            for key, template in hidden_state_template.items():
                # Expand from [1, seq, ...] to [n_points, seq, ...]
                expanded = template.expand(n_points, *template.shape[1:])
                critic_input_dict[key] = expanded

            critic_input = TensorDict(critic_input_dict, batch_size=[n_points, 1], device=device)

            # Get value estimates
            output = critic_network(critic_input)
            values = output["state_value"]  # [n_points, 1, 1] or [n_points, 1]

            # Squeeze to get [n_points]
            while values.dim() > 1:
                values = values.squeeze(-1)

        return values.cpu().numpy()

    return value_fn


def visualize_critic_value_landscape(
    checkpoint_path: str,
    num_episodes: int = 5,
    grid_precision: float = 0.15,
    cmap_name: str = "coolwarm",
    cmap_alpha: float = 0.6,
    value_range: tuple = None,
    debug_mode: bool = False,
    agent_to_vary: int = 0,
    critic_group: str = "attacker",
    critic_agent_index: int = 0,
    dynamic_range: bool = False,
    range_update_freq: int = 10
):
    """
    Load checkpoint and run visualization with critic heatmap overlay.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        num_episodes: Number of episodes to visualize
        grid_precision: Grid resolution in meters (0.15 = 15cm)
        cmap_name: Matplotlib colormap name (e.g., 'coolwarm', 'viridis', 'RdYlGn')
        cmap_alpha: Heatmap transparency (0.0 = invisible, 1.0 = opaque)
        value_range: (vmin, vmax) for colormap normalization, or None for auto
        debug_mode: If True, run only 50 steps per episode and exit without waiting
        agent_to_vary: Which agent to vary position (0=A1, 1=A2, 2=D1, 3=D2)
        critic_group: Which group's critic to visualize ("attacker" or "defender")
        critic_agent_index: If critic not shared, which agent's critic (0-based within group)
        dynamic_range: If True, update colormap range during episode based on observed values
        range_update_freq: Update range every N steps (only if dynamic_range=True)
    """
    agent_names = ["A1 (ball handler)", "A2 (screener)", "D1 (defender)", "D2 (defender)"]
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Visualizing: {critic_group.upper()} critic")
    print(f"Varying agent position: {agent_names[agent_to_vary]}\n")

    # Load experiment
    exp = Experiment.reload_from_file(checkpoint_path)
    exp.seed = 42
    torch.manual_seed(42)

    # Get critic network
    print("\nExtracting critic network...")
    # exp.losses[group_name] is a ClipPPOLoss instance
    # The critic network is stored in the .critic_network attribute
    if critic_group not in exp.losses:
        available_groups = list(exp.losses.keys())
        raise ValueError(f"Group '{critic_group}' not found in losses. Available: {available_groups}")

    group_critic = exp.losses[critic_group].critic_network

    # Check if critic is shared or per-agent
    # First unwrap if it's a compiled/optimized module
    if hasattr(group_critic, "_orig_mod"):
        unwrapped_critic = group_critic._orig_mod
    else:
        unwrapped_critic = group_critic

    print(f"Critic type: {type(group_critic)}")
    print(f"Unwrapped critic type: {type(unwrapped_critic)}")

    # Try to access specific agent's critic if not shared
    # Check for ModuleList, ModuleDict, or similar container types
    from torch.nn import ModuleList, ModuleDict
    if isinstance(unwrapped_critic, (ModuleList, ModuleDict, list, dict)):
        # Non-shared critic - multiple critics in a container
        print(f"Non-shared critic detected. Using critic index {critic_agent_index}")
        critic_network = unwrapped_critic[critic_agent_index]
    else:
        # Shared critic
        print(f"Shared critic detected (ignoring critic_agent_index={critic_agent_index})")
        critic_network = group_critic

    critic_network.eval()
    device = next(critic_network.parameters()).device

    # Get the underlying uncompiled model if needed
    if hasattr(critic_network, "_orig_mod"):
        critic_model = critic_network._orig_mod
    else:
        critic_model = critic_network

    print(f"Critic loaded on device: {device}")
    print(f"Critic architecture: {type(critic_network)}")
    print(f"Critic in_keys: {critic_model.in_keys}")

    # Pre-collect critic input keys (excluding observation, state, and is_init)
    # These are the recurrent hidden state keys (e.g., "_recurrent_state_h", "_recurrent_state_c")
    critic_input_keys = [k for k in critic_model.in_keys if k not in ("observation", "state", "is_init")]
    print(f"Critic hidden state keys: {critic_input_keys}")

    # Create single evaluation environment
    print("\nCreating environment...")
    task = exp.task
    env = task.get_env_fun(
        num_envs=1,
        continuous_actions=True,
        seed=42,
        device=device
    )()

    # Extract field parameters for normalization
    scenario = env._env.scenario
    field_params = {
        "W": scenario.h_params["W"],
        "L": scenario.h_params["L"],
        "v_max": scenario.h_params["v_max"]
    }
    print(f"Field parameters: W={field_params['W']}, L={field_params['L']}, v_max={field_params['v_max']}")

    # Field bounds for layup (from CLAUDE.md: 8m × 15m)
    field_bounds = ((-4.0, 4.0), (-7.5, 7.5))  # (x_range, y_range)

    # Create value function wrapper
    print("\nCreating critic value function wrapper...")
    value_fn = create_critic_value_function(critic_network, env, device, field_params, critic_input_keys, agent_to_vary)

    print(f"\nStarting visualization:")
    print(f"  Varying agent: {agent_names[agent_to_vary]}")
    print(f"  Grid precision: {grid_precision}m")
    print(f"  Field bounds: {field_bounds}")
    print(f"  Colormap: {cmap_name}")
    print(f"  Alpha: {cmap_alpha}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Dynamic range: {dynamic_range}")
    if debug_mode:
        print("  [DEBUG] Will run only 50 steps per episode")

    # Determine initial value range for colormap normalization
    if value_range is None and not dynamic_range:
        print("\nComputing initial value range for colormap normalization...")
        # Use full grid for more accurate range estimation
        x_range = field_bounds[0]
        y_range = field_bounds[1]
        x_full = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) / grid_precision) + 1)
        y_full = np.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) / grid_precision) + 1)
        X_full, Y_full = np.meshgrid(x_full, y_full)
        full_positions = np.stack([X_full.flatten(), Y_full.flatten()], axis=-1)

        # Reset environment to get initial state for sampling
        obs = env.reset()
        full_values = value_fn(full_positions)
        value_range = (float(np.min(full_values)), float(np.max(full_values)))
        print(f"  Full grid value range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")
    elif value_range is None and dynamic_range:
        # For dynamic range, start with a reasonable default
        value_range = (-1.0, 1.0)
        print(f"  Dynamic range mode: Starting with [{value_range[0]:.2f}, {value_range[1]:.2f}]")
    else:
        print(f"  Manual value range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")

    # Convert to list for dynamic updates
    current_range = list(value_range)

    # Track observed values for dynamic range
    observed_min = float('inf')
    observed_max = float('-inf')

    # Run episodes with rendering
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        obs = env.reset()

        # Initialize state for recurrent policy
        # We need to maintain a running TensorDict that contains:
        # 1. Observations
        # 2. Hidden states (initially missing, policy will create them)
        # 3. is_init flag - must have shape (batch, 1) to become (batch, seq, 1) after unsqueeze
        current_td = obs.clone()
        current_td.set("is_init", torch.ones((current_td.shape[0], 1), device=device, dtype=torch.bool))

        done = False
        step = 0

        # Store initial critic value statistics
        initial_agent_pos = env._env.scenario.world.agents[agent_to_vary].state.pos[0].cpu().numpy()
        initial_value = value_fn(initial_agent_pos.reshape(1, -1))[0]
        print(f"\nInitial state:")
        print(f"  {agent_names[agent_to_vary]} position: ({initial_agent_pos[0]:6.2f}, {initial_agent_pos[1]:6.2f})")
        print(f"  Critic value: {initial_value:8.4f}")

        while not done:
            # Add sequence dimension for GRU: [batch] -> [batch, seq=1]
            td_seq = current_td.unsqueeze(-1)

            # Get action from policy
            with torch.no_grad():
                # policy returns action and next hidden states
                out_seq = exp.policy(td_seq)

            # Squeeze back to [batch]
            out_td = out_seq.squeeze(-1)

            # Step environment
            # env.step expects action in input tensordict and populates "next" key
            env_out = env.step(out_td)

            # Get current agent position for display (unnormalized world coordinates)
            current_agent = env._env.scenario.world.agents[agent_to_vary]
            agent_pos = current_agent.state.pos[0].cpu().numpy()  # [2] unnormalized position
            agent_x, agent_y = agent_pos

            # Evaluate critic at current agent position
            critic_value = value_fn(agent_pos.reshape(1, -1))[0]

            # Update observed range for dynamic mode
            if dynamic_range:
                observed_min = min(observed_min, critic_value)
                observed_max = max(observed_max, critic_value)

                # Update range every N steps
                if step % range_update_freq == 0 and step > 0:
                    # Use exponential moving average for smooth updates
                    alpha = 0.3  # Smoothing factor
                    current_range[0] = alpha * observed_min + (1 - alpha) * current_range[0]
                    current_range[1] = alpha * observed_max + (1 - alpha) * current_range[1]

                    # Print range updates less frequently to avoid clutter
                    if step % max(range_update_freq * 10, 10) == 0:
                        print(f"    [Range update] New range: [{current_range[0]:.2f}, {current_range[1]:.2f}]")

            # Use current_range for rendering
            render_range = tuple(current_range) if dynamic_range else value_range

            # Render the environment with critic heatmap overlay
            # Pass the value function directly to render() which will call plot_function internally
            env.render(
                mode="human",
                agent_index_focus=None,
                visualize_when_rgb=True,
                plot_position_function=value_fn,
                plot_position_function_precision=grid_precision,
                plot_position_function_range=field_bounds,
                plot_position_function_cmap_range=render_range,
                plot_position_function_cmap_alpha=cmap_alpha,
                plot_position_function_cmap_name=cmap_name
            )

            if step % 10 == 0:  # Print every 10 steps to reduce clutter
                range_str = f" | Range: [{current_range[0]:.2f}, {current_range[1]:.2f}]" if dynamic_range else ""
                print(f"  Step {step:3d} | {agent_names[agent_to_vary]} pos: ({agent_x:6.2f}, {agent_y:6.2f}) | Critic value: {critic_value:8.4f}{range_str}")

            # Check termination
            done = env_out.get(("next", "done")).item()
            step += 1

            # Debug mode: exit after 50 steps
            if debug_mode and step >= 50:
                print(f"  [DEBUG] Reached 50 steps, exiting episode early")
                break

            # Prepare for next step
            # We take the output of the policy (which has hidden states)
            # and update it with the new observations from "next"
            current_td = out_td.clone()
            current_td.update(env_out.get("next"))

            # Reset is_init for subsequent steps
            current_td.set("is_init", torch.zeros((current_td.shape[0], 1), device=device, dtype=torch.bool))

        # Get final statistics
        final_agent_pos = env._env.scenario.world.agents[agent_to_vary].state.pos[0].cpu().numpy()
        final_value = value_fn(final_agent_pos.reshape(1, -1))[0]

        print(f"\nEpisode summary:")
        print(f"  Duration: {step} steps")
        print(f"  Initial value: {initial_value:8.4f}")
        print(f"  Final value: {final_value:8.4f}")
        print(f"  Value change: {final_value - initial_value:+8.4f}")

    env.close()
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize critic value function heatmap overlay on VMAS environment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/**/checkpoints/*.pt",
        help="Checkpoint path or glob pattern"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.15,
        help="Grid resolution in meters (smaller = finer but slower)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name (coolwarm, viridis, RdYlGn, etc.)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Heatmap transparency (0.0-1.0)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for colormap (None for auto)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for colormap (None for auto)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: run only 50 steps per episode"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="a1",
        choices=["a1", "a2", "d1", "d2"],
        help="Which agent's position to vary (a1, a2, d1, d2)"
    )
    parser.add_argument(
        "--critic-group",
        type=str,
        default="attacker",
        choices=["attacker", "defender"],
        help="Which group's critic to visualize"
    )
    parser.add_argument(
        "--critic-agent",
        type=int,
        default=0,
        help="If critic is not shared, which agent's critic to use (0-based index within group)"
    )
    parser.add_argument(
        "--dynamic-range",
        action="store_true",
        help="Dynamically update colormap range during episode based on observed values"
    )
    parser.add_argument(
        "--range-update-freq",
        type=int,
        default=1,
        help="Update range every N steps when using dynamic range (default: 1 for every step)"
    )

    args = parser.parse_args()

    # Map agent name to index
    agent_map = {"a1": 0, "a2": 1, "d1": 2, "d2": 3}
    agent_to_vary = agent_map[args.agent.lower()]

    # Find checkpoint
    if "*" in args.checkpoint:
        checkpoint_path = find_latest_checkpoint(args.checkpoint)
        if checkpoint_path is None:
            print(f"No checkpoint found matching: {args.checkpoint}")
            exit(1)
    else:
        checkpoint_path = args.checkpoint

    # Prepare value range
    value_range = None
    if args.vmin is not None and args.vmax is not None:
        value_range = (args.vmin, args.vmax)

    # Run visualization
    visualize_critic_value_landscape(
        checkpoint_path=checkpoint_path,
        num_episodes=args.episodes,
        grid_precision=args.precision,
        cmap_name=args.cmap,
        cmap_alpha=args.alpha,
        value_range=value_range,
        debug_mode=args.debug,
        agent_to_vary=agent_to_vary,
        critic_group=args.critic_group,
        critic_agent_index=args.critic_agent,
        dynamic_range=args.dynamic_range,
        range_update_freq=args.range_update_freq
    )
