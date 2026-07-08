"""SAC with a learnable Lagrangian multiplier (RCPO-style) — single-file.

Reference implementation used to validate Proposition 2 of the TCL paper.
The Lagrangian-augmented reward fed to the critic is

    r_aug(s, a) = r(s, a) - lambda * g(s, a)

where g(s, a) = info["cost"] is the per-step constraint cost and
lambda >= 0 is a learnable scalar updated by primal-dual gradient ascent:

    lambda <- ReLU(lambda + eta_lambda * mean_batch(g)).

Hyperparameters are exposed via tyro. The script logs episodic return,
episodic cost, the dual trajectory lambda(t), and the SAC losses to
TensorBoard.

Usage:
    uv run python agents/sac_lagrangian.py \\
        --total-timesteps 200000 --lambda-lr 1e-3 --omega 0.0436

References:
    Haarnoja et al., Soft Actor-Critic, ICML 2018.
    Tessler, Mankowitz, Mannor, RCPO, ICLR 2019.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from agents.common import Actor, ReplayBuffer, SoftQNetwork
from agents.common.args import SACBaseArgs
from tcl.envs import SinusoidalCMDP
from tcl.utils import seed_everything

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + Lagrangian."""

    exp_name: str = "sac_lagrangian"

    # Lagrangian
    lambda_init: float = 0.0
    lambda_lr: float = 1e-3
    lambda_update_frequency: int = 1  # in gradient steps
    cost_baseline: float = 0.0  # subtract from cost in dual update (0 = none)

    # trajectory dump (for Prop 2 amplitude analysis)
    dump_trajectory_path: str | None = None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def make_env(args: Args) -> gym.Env:
    return SinusoidalCMDP(
        horizon=args.horizon,
        b0=args.b0,
        amplitude=args.amplitude,
        omega=args.omega,
        random_phase_at_reset=args.random_phase_at_reset,
    )


def train(
    args: Args,
    env_factory: Callable[[Args], gym.Env] | None = None,
) -> dict[str, float]:
    seed_everything(args.seed, deterministic_torch=args.torch_deterministic)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    run_name = (
        f"{args.exp_name}__seed{args.seed}__omega{args.omega:.4f}"
        f"__lr{args.lambda_lr:.0e}__{int(time.time())}"
    )
    log_path = Path(args.log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    env = (env_factory or make_env)(args)
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

    q_optim = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune_alpha:
        target_entropy = -float(act_dim)
        log_alpha = torch.tensor(math.log(args.alpha_init), requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.q_lr)
    else:
        log_alpha = torch.tensor(math.log(args.alpha_init), device=device)
        alpha_optim = None
        target_entropy = None

    # Lagrangian multiplier (kept as a plain tensor; manual gradient ascent).
    lam = torch.tensor(args.lambda_init, device=device)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    rng = np.random.default_rng(args.seed)

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_cost = 0.0
    episode_steps = 0
    episodes_done = 0
    gradient_steps = 0

    # Trajectory arrays for offline amplitude analysis.
    traj_steps = np.empty(args.total_timesteps, dtype=np.int64)
    traj_lambda = np.empty(args.total_timesteps, dtype=np.float64)
    traj_cost = np.empty(args.total_timesteps, dtype=np.float64)

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # ------------------------------------------------------------------
        # Action selection
        # ------------------------------------------------------------------
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                a_t, _, _ = actor.sample(s_t)
                action = a_t.cpu().numpy().flatten()

        # ------------------------------------------------------------------
        # Env step
        # ------------------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cost = float(info["cost"])
        buffer.add(obs, action, float(reward), cost, next_obs, terminated)

        traj_steps[global_step] = global_step
        traj_lambda[global_step] = lam.item()
        traj_cost[global_step] = cost

        episode_return += float(reward)
        episode_cost += cost
        episode_steps += 1

        obs = next_obs
        if done:
            writer.add_scalar("rollout/episode_return", episode_return, global_step)
            writer.add_scalar("rollout/episode_cost", episode_cost, global_step)
            writer.add_scalar("rollout/episode_steps", episode_steps, global_step)
            writer.add_scalar("rollout/episodes_done", episodes_done, global_step)
            episodes_done += 1
            episode_return = 0.0
            episode_cost = 0.0
            episode_steps = 0
            obs, _ = env.reset()

        # ------------------------------------------------------------------
        # Learning updates
        # ------------------------------------------------------------------
        if global_step < args.learning_starts:
            continue

        batch = buffer.sample(args.batch_size, rng)
        batch = {k: v.to(device) for k, v in batch.items()}

        alpha_val = log_alpha.exp().detach()

        # Critic update with Lagrangian-augmented reward.
        with torch.no_grad():
            next_action, next_log_prob, _ = actor.sample(batch["next_obs"])
            q1_next = qf1_tgt(batch["next_obs"], next_action)
            q2_next = qf2_tgt(batch["next_obs"], next_action)
            q_next = torch.min(q1_next, q2_next) - alpha_val * next_log_prob
            r_aug = batch["rewards"] - lam * batch["costs"]
            q_target = r_aug + args.gamma * (1.0 - batch["dones"]) * q_next

        q1_pred = qf1(batch["obs"], batch["actions"])
        q2_pred = qf2(batch["obs"], batch["actions"])
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optim.step()

        # Actor / alpha updates at reduced frequency.
        if gradient_steps % args.policy_frequency == 0:
            pi_action, log_prob, _ = actor.sample(batch["obs"])
            q1_pi = qf1(batch["obs"], pi_action)
            q2_pi = qf2(batch["obs"], pi_action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha_val * log_prob - q_pi).mean()

            actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optim.step()

            if args.autotune_alpha:
                with torch.no_grad():
                    _, log_prob_for_alpha, _ = actor.sample(batch["obs"])
                alpha_loss = -(log_alpha.exp() * (log_prob_for_alpha + target_entropy)).mean()
                alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optim.step()

        # Lagrangian multiplier update (primal-dual ascent).
        if gradient_steps % args.lambda_update_frequency == 0:
            mean_cost = (batch["costs"].mean() - args.cost_baseline).item()
            lam = torch.clamp(lam + args.lambda_lr * mean_cost, min=0.0)

        # Target network soft update.
        if gradient_steps % args.target_network_frequency == 0:
            for p, p_t in zip(qf1.parameters(), qf1_tgt.parameters(), strict=False):
                p_t.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
            for p, p_t in zip(qf2.parameters(), qf2_tgt.parameters(), strict=False):
                p_t.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

        gradient_steps += 1

        # ------------------------------------------------------------------
        # Periodic logging
        # ------------------------------------------------------------------
        if global_step % 500 == 0:
            writer.add_scalar("train/q_loss", q_loss.item(), global_step)
            writer.add_scalar("train/alpha_entropy", alpha_val.item(), global_step)
            writer.add_scalar("train/lambda", lam.item(), global_step)
            writer.add_scalar("train/sps", global_step / (time.time() - start_time), global_step)

        # Log lambda at high frequency for amplitude measurement.
        writer.add_scalar("dual/lambda", lam.item(), global_step)
        writer.add_scalar("dual/cost_batch_mean", batch["costs"].mean().item(), global_step)

    writer.close()
    env.close()

    if args.dump_trajectory_path is not None:
        out = Path(args.dump_trajectory_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            steps=traj_steps,
            lam=traj_lambda,
            cost=traj_cost,
            omega=np.array(args.omega),
            lambda_lr=np.array(args.lambda_lr),
            amplitude=np.array(args.amplitude),
            b0=np.array(args.b0),
            horizon=np.array(args.horizon),
            seed=np.array(args.seed),
        )

    return {
        "lambda_final": lam.item(),
        "episodes_done": float(episodes_done),
        "gradient_steps": float(gradient_steps),
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    result = train(args)
    print(f"Training complete. lambda_final={result['lambda_final']:.4f}")
    print(f"Logs: {result['log_path']}")
