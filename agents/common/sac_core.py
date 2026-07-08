"""Shared SAC training loop for multi-constraint CMDP agents."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.common.buffer import ReplayBuffer
from agents.common.networks import Actor, SoftQNetwork
from tcl.envs import SinusoidalCMDP
from tcl.utils import seed_everything


def make_sinusoidal_env(args: Any) -> gym.Env:
    """Create a SinusoidalCMDP from a standard Args namespace."""
    return SinusoidalCMDP(
        horizon=args.horizon,
        b0=args.b0,
        amplitude=args.amplitude,
        omega=args.omega,
        random_phase_at_reset=args.random_phase_at_reset,
    )


def probe_k_costs(args: Any, env_factory: Callable[[Any], gym.Env]) -> int:
    """Probe the environment to discover the number of constraints K."""
    env = env_factory(args)
    env.reset(seed=args.seed)
    _, _, _, _, info = env.step(env.action_space.sample())
    k = int(np.asarray(info["costs"]).shape[0])
    env.close()
    return k


class SACTrainer:
    """Generic SAC training loop for multi-constraint CMDPs.

    Hook parameters (all optional, None = identity / no-op):

      reward_fn(batch, global_step) -> Tensor (B,1)
          Shaped reward used for the critic target. None = use raw batch["rewards"].
      step_hook(r, costs_np, done) -> float
          Transform the reward before storing in the replay buffer (e.g., HPRS).
          None = store raw reward.
      dual_update_fn(batch, gradient_steps) -> None
          Update dual variables (Lagrange multipliers). Called after actor/alpha
          update and before the soft target-network update.
      extra_log_fn(writer, batch, global_step, gradient_steps) -> None
          Called at every gradient step for agent-specific logging.
      episode_end_fn(writer, global_step) -> None
          Called once per episode end for agent-specific episode-level logging.
      extra_return_fn() -> dict
          Extra key/value pairs merged into the return dict of run().
    """

    def __init__(
        self,
        args: Any,
        env_factory: Callable[[Any], gym.Env],
        k_costs: int,
        *,
        reward_fn: Callable[[dict, int], torch.Tensor] | None = None,
        step_hook: Callable[[float, np.ndarray, bool], float] | None = None,
        dual_update_fn: Callable[[dict, int], None] | None = None,
        extra_log_fn: (
            Callable[[SummaryWriter, dict, int, int], None] | None
        ) = None,
        episode_end_fn: Callable[[SummaryWriter, int], None] | None = None,
        extra_return_fn: Callable[[], dict] | None = None,
    ) -> None:
        self.args = args
        self.env_factory = env_factory
        self.k_costs = k_costs
        self.reward_fn = reward_fn
        self.step_hook = step_hook
        self.dual_update_fn = dual_update_fn
        self.extra_log_fn = extra_log_fn
        self.episode_end_fn = episode_end_fn
        self.extra_return_fn = extra_return_fn

    def run(self) -> dict[str, float]:
        args = self.args
        k_costs = self.k_costs

        seed_everything(args.seed, deterministic_torch=args.torch_deterministic)
        device = torch.device(
            "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
        )

        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        log_path = Path(args.log_dir) / run_name
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_path))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n"
            + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
        )

        env = self.env_factory(args)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        action_low = env.action_space.low
        action_high = env.action_space.high

        actor = Actor(obs_dim, act_dim, action_low, action_high).to(device)
        qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
        qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
        qf1_tgt = SoftQNetwork(obs_dim, act_dim).to(device)
        qf2_tgt = SoftQNetwork(obs_dim, act_dim).to(device)
        qf1_tgt.load_state_dict(qf1.state_dict())
        qf2_tgt.load_state_dict(qf2.state_dict())

        q_optim = torch.optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
        )
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)

        if args.autotune_alpha:
            target_entropy = -float(act_dim)
            log_alpha = torch.tensor(
                math.log(args.alpha_init), requires_grad=True, device=device
            )
            alpha_optim: torch.optim.Optimizer | None = torch.optim.Adam(
                [log_alpha], lr=args.q_lr
            )
        else:
            log_alpha = torch.tensor(math.log(args.alpha_init), device=device)
            alpha_optim = None
            target_entropy = None

        buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, k_costs)
        rng = np.random.default_rng(args.seed)

        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_costs = np.zeros(k_costs, dtype=np.float64)
        episode_steps = 0
        episode_action_sum = np.zeros(act_dim, dtype=np.float64)
        episodes_done = 0
        gradient_steps = 0
        start_time = time.time()

        for global_step in range(args.total_timesteps):
            if global_step < args.learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    a_t, _, _ = actor.sample(s_t)
                    action = a_t.cpu().numpy().flatten()

            episode_action_sum += action

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            costs_np = np.asarray(info["costs"], dtype=np.float32)

            r_buf = (
                self.step_hook(float(reward), costs_np, done)
                if self.step_hook
                else float(reward)
            )
            buffer.add(obs, action, r_buf, costs_np, next_obs, terminated)

            episode_return += float(reward)
            episode_costs += costs_np
            episode_steps += 1

            obs = next_obs
            if done:
                writer.add_scalar(
                    "rollout/episode_return", episode_return, global_step
                )
                for k in range(k_costs):
                    writer.add_scalar(
                        f"rollout/episode_cost_{k}", episode_costs[k], global_step
                    )
                writer.add_scalar(
                    "rollout/episode_steps", episode_steps, global_step
                )
                ep_action_mean = float(
                    episode_action_sum.mean() / max(episode_steps, 1)
                )
                writer.add_scalar(
                    "rollout/action_mean", ep_action_mean, global_step
                )
                if self.episode_end_fn:
                    self.episode_end_fn(writer, global_step)
                episodes_done += 1
                episode_return = 0.0
                episode_costs[:] = 0.0
                episode_steps = 0
                episode_action_sum[:] = 0.0
                obs, _ = env.reset()

            if global_step < args.learning_starts:
                continue

            batch = buffer.sample(args.batch_size, rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            alpha_val = log_alpha.exp().detach()

            r_shaped = (
                self.reward_fn(batch, global_step)
                if self.reward_fn
                else batch["rewards"]
            )

            with torch.no_grad():
                next_action, next_log_prob, _ = actor.sample(batch["next_obs"])
                q1_next = qf1_tgt(batch["next_obs"], next_action)
                q2_next = qf2_tgt(batch["next_obs"], next_action)
                q_next = torch.min(q1_next, q2_next) - alpha_val * next_log_prob
                q_target = r_shaped + args.gamma * (1.0 - batch["dones"]) * q_next

            q1_pred = qf1(batch["obs"], batch["actions"])
            q2_pred = qf2(batch["obs"], batch["actions"])
            q_loss = (
                F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            )
            q_optim.zero_grad(set_to_none=True)
            q_loss.backward()
            q_optim.step()

            if gradient_steps % args.policy_frequency == 0:
                pi_action, log_prob, _ = actor.sample(batch["obs"])
                q1_pi = qf1(batch["obs"], pi_action)
                q2_pi = qf2(batch["obs"], pi_action)
                actor_loss = (
                    alpha_val * log_prob - torch.min(q1_pi, q2_pi)
                ).mean()
                actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optim.step()

                if args.autotune_alpha:
                    with torch.no_grad():
                        _, log_prob_a, _ = actor.sample(batch["obs"])
                    alpha_loss = -(
                        log_alpha.exp() * (log_prob_a + target_entropy)
                    ).mean()
                    alpha_optim.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_optim.step()

            if self.dual_update_fn:
                self.dual_update_fn(batch, gradient_steps)

            if gradient_steps % args.target_network_frequency == 0:
                for p, p_t in zip(
                    qf1.parameters(), qf1_tgt.parameters(), strict=False
                ):
                    p_t.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
                for p, p_t in zip(
                    qf2.parameters(), qf2_tgt.parameters(), strict=False
                ):
                    p_t.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

            gradient_steps += 1

            if global_step % 500 == 0:
                writer.add_scalar("train/q_loss", q_loss.item(), global_step)
                writer.add_scalar(
                    "train/alpha_entropy", alpha_val.item(), global_step
                )
                writer.add_scalar(
                    "train/sps",
                    global_step / (time.time() - start_time),
                    global_step,
                )

            if self.extra_log_fn:
                self.extra_log_fn(writer, batch, global_step, gradient_steps)

        actor_save_path = log_path.parent.parent / "actor.pt"
        torch.save(actor.state_dict(), actor_save_path)
        writer.close()
        env.close()

        result: dict[str, float] = {
            "episodes_done": float(episodes_done),
            "gradient_steps": float(gradient_steps),
            "log_path": str(log_path),
        }
        if self.extra_return_fn:
            result.update(self.extra_return_fn())
        return result
