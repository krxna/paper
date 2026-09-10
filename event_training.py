"""Profile-gated Attention+PPO retraining on the shared execution engine."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict

import torch

import config
from controller_profile import DEFAULT_PROFILE_PATH, require_profile
from execution_engine import EXECUTION_MODEL_VERSION
from neural import (
    Actor, AttentionEncoder, Critic, behavior_clone_fodas, compute_gae,
    ppo_update,
)
from scenario import ScenarioConfig, make_policy, run_scenario


DEFAULT_EPISODES = 50
PROGRESS_PATH = Path("figures/event_training_progress.json")


def _atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _render_progress(history, output_dir: Path) -> None:
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    episodes = [row["episode"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(episodes, [100 * row["drop_rate"] for row in history], marker="o")
    axes[0].set(xlabel="completed PPO episode", ylabel="drop rate (%)",
                title="Training outcome")
    axes[1].plot(episodes, [row["total_loss"] for row in history], marker="o")
    axes[1].set(xlabel="completed PPO episode", ylabel="PPO total loss",
                title="Optimization progress")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "event_training_progress.png", dpi=200)
    fig.savefig(output_dir / "event_training_progress.pdf")
    plt.close(fig)


def _atomic_torch(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _signature(profile: Dict, episodes: int, base_seed: int) -> Dict:
    return {
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "controller_profile_hash": profile["profile_hash"],
        "controller_workload_hash": profile["workload_hash"],
        "episodes": episodes,
        "base_seed": int(base_seed),
        "behavior_clone_steps": config.BC_PRETRAIN_STEPS,
    }


def train(
    *, profile: Dict, output: Path, episodes: int = DEFAULT_EPISODES,
    resume: bool = False, base_seed: int = config.SEED,
) -> Dict:
    signature = _signature(profile, episodes, base_seed)
    encoder, actor, critic = AttentionEncoder(), Actor(), Critic()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.parameters())
        + list(critic.parameters()), lr=config.LEARNING_RATE)
    completed = 0
    bc = {}
    history = []
    payload = {}
    if resume:
        if not output.exists():
            raise FileNotFoundError(f"resume checkpoint missing: {output}")
        try:
            saved = torch.load(output, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(output, map_location="cpu")
        if saved.get("signature") != signature:
            raise ValueError("training checkpoint signature mismatch")
        encoder.load_state_dict(saved["encoder"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = int(saved["completed_episodes"])
        bc = dict(saved.get("behavior_cloning", {}))
        history = list(saved.get("history", []))
        payload = saved
    else:
        config.set_global_seed(base_seed)
        _atomic_json(PROGRESS_PATH, {
            "status": "in_progress",
            "phase": "behavior_cloning",
            "completed_episodes": 0,
            "total_episodes": episodes,
            "percent": 0.0,
            "estimated_remaining_s": None,
            "message": "Behavior cloning is the initial atomic stage.",
        })
        bc = behavior_clone_fodas(encoder, actor)
        _atomic_torch(output, {
            "signature": signature,
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "controller_profile_hash": profile["profile_hash"],
            "completed_episodes": 0,
            "behavior_cloning": bc,
            "history": history,
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
        })

    # At the configured 180 tasks/s this duration yields roughly the historical
    # 2500-task episode while preserving a time-based execution horizon.
    episode_duration_s = config.TASKS_PER_EPISODE / config.TASK_ARRIVAL_RATE
    run_started = time.monotonic()
    durations = [float(row.get("duration_s", 0)) for row in history
                 if float(row.get("duration_s", 0)) > 0]
    _atomic_json(PROGRESS_PATH, {
        "status": "complete" if completed >= episodes else "in_progress",
        "phase": "ppo_training",
        "completed_episodes": completed, "total_episodes": episodes,
        "percent": 100 * completed / episodes,
        "estimated_remaining_s": (
            0.0 if completed >= episodes else
            sum(durations) / len(durations) * (episodes - completed)
            if durations else None),
        "checkpoint": str(output),
        "progress_figure": str(output.parent / "event_training_progress.png"),
    })
    if completed >= episodes:
        print(
            f"[PPO {completed:02d}/{episodes:02d}] training already complete; "
            f"using checkpoint {output}",
            flush=True,
        )
        return payload

    for episode in range(completed, episodes):
        seed = base_seed + 1000 + episode
        config.set_global_seed(seed)
        encoder.train()
        actor.train()
        critic.train()
        adapter = make_policy(
            "Attention+PPO", seed, (encoder, actor), critic=critic)
        cfg = ScenarioConfig(
            seed=seed,
            target_load=0.96,
            policy="Attention+PPO",
            controller_placement="native",
            warmup_s=0.0,
            measurement_s=episode_duration_s,
            training=True,
        )
        started = time.monotonic()
        result = run_scenario(cfg, adapter, profile)
        rollout = result.rollout
        losses = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "total_loss": 0.0}
        if rollout:
            rewards = [row["reward"] for row in rollout]
            values = [row["value"] for row in rollout]
            next_values = values[1:] + [0.0]
            dones = [False] * len(rollout)
            dones[-1] = True
            advantages, targets = compute_gae(
                rewards, values, next_values, dones)
            batch = {
                "H": [row["H"] for row in rollout],
                "c": [row["c"] for row in rollout],
                "soc": [row["soc"] for row in rollout],
                "cand_masks": [row["cand_mask"] for row in rollout],
                "f_p": [row["f_p"] for row in rollout],
                "f_b": [row["f_b"] for row in rollout],
                "logp_old": [row["logp_old"] for row in rollout],
                "advantages": advantages,
                "vtargets": targets,
            }
            losses = ppo_update(actor, critic, optimizer, batch)
        completed = episode + 1
        episode_duration = time.monotonic() - started
        history.append({
            "episode": completed,
            "drop_rate": result.drop_rate,
            "productive": result.capacity.utilization()["productive"],
            "wasted": result.capacity.utilization()["wasted"],
            "total_loss": losses["total_loss"],
            "duration_s": episode_duration,
        })
        durations.append(episode_duration)
        eta = sum(durations) / len(durations) * (episodes - completed)
        payload = {
            "signature": signature,
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "controller_profile_hash": profile["profile_hash"],
            "completed_episodes": completed,
            "behavior_cloning": bc,
            "history": history,
            "last_metrics": {
                **losses,
                "drop_rate": result.drop_rate,
                "productive": result.capacity.utilization()["productive"],
                "wasted": result.capacity.utilization()["wasted"],
            },
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        _atomic_torch(output, payload)
        _render_progress(history, output.parent)
        _atomic_json(PROGRESS_PATH, {
            "status": "complete" if completed == episodes else "in_progress",
            "phase": "ppo_training",
            "completed_episodes": completed,
            "total_episodes": episodes,
            "percent": 100 * completed / episodes,
            "last_episode_duration_s": episode_duration,
            "elapsed_this_run_s": time.monotonic() - run_started,
            "estimated_remaining_s": eta,
            "checkpoint": str(output),
            "progress_figure": str(output.parent / "event_training_progress.png"),
        })
        print(
            f"[PPO {completed:02d}/{episodes:02d}] "
            f"drop={100*result.drop_rate:.2f}% "
            f"loss={losses['total_loss']:.4f} "
            f"time={episode_duration:.1f}s ETA={eta/60:.1f}m checkpointed",
            flush=True)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--output", type=Path, default=Path("figures/event_ppo_final.pt"))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--base-seed", type=int, default=config.SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes != DEFAULT_EPISODES:
        parser.error("the final methodology requires exactly 50 PPO episodes")
    # This is intentionally the first stateful step.
    profile = require_profile(args.profile)
    try:
        train(
            profile=profile, output=args.output,
            episodes=args.episodes, resume=args.resume,
            base_seed=args.base_seed)
    except KeyboardInterrupt:
        current = {}
        if PROGRESS_PATH.exists():
            current = json.loads(PROGRESS_PATH.read_text())
        current["status"] = "paused"
        current["message"] = "Resume with: python3 event_training.py --resume"
        _atomic_json(PROGRESS_PATH, current)
        print("\nPaused safely; the last completed episode checkpoint is intact.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
