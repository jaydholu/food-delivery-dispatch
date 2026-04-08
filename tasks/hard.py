"""
HARD task — 6 drivers, 15 initial orders, traffic + dynamic order spawning.

Dynamic spawning means the agent must continuously balance in-progress
deliveries against newly arriving demand.
This is the most realistic and challenging configuration.

Difficulty characteristics:
  - 6 drivers, 15 initial orders (up to 30 total with dynamic spawning)
  - Traffic zones with 1.5x-3.0x slowdown
  - Tight deadlines (25-70 steps)
  - Dynamic order arrival (~12% chance per step)
  - 300 max steps
  - Strict penalties for inactivity (threshold: 15 steps)
  - Higher order failure penalty
  - Fastest escalation of wait penalties (threshold=2)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from models import EpisodeResult
from server.food_delivery_dispatch_environment import (
    HARD_CONFIG,
    HARD_REWARDS,
    DriverStatus,
    EnvConfig,
    FoodDeliveryEnvironment,
)
from tasks.grader import format_grade_report, grade_episode


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_hard_env() -> FoodDeliveryEnvironment:
    """
    Construct and return the HARD task environment.

    HARD settings:
      - 6 drivers
      - 15 initial orders (dynamic spawning up to 30)
      - Traffic zones enabled
      - Tight deadlines (25-70 steps)
      - 300 max steps
      - Inactivity threshold: 15 steps (strict)
      - Consecutive wait escalation after just 2 waits

    Returns:
        Configured FoodDeliveryEnvironment instance.
    """
    env = FoodDeliveryEnvironment(task="hard")
    env._rwt = HARD_REWARDS
    return env


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_hard(
    policy_fn: Any,
    num_episodes: int = 5,
    seed_offset: int = 200,
    verbose: bool = True,
) -> tuple[float, list[EpisodeResult]]:
    """
    Evaluate a policy on the HARD task over multiple episodes.

    Args:
        policy_fn:    Callable(observation, env) → FoodDeliveryAction.
        num_episodes: Number of evaluation episodes.
        seed_offset:  Shift seeds for independent evaluation runs.
        verbose:      Print per-episode reports.

    Returns:
        mean_score: Average normalised score across episodes.
        results:    List of EpisodeResult dataclasses.
    """
    scores: list[float] = []
    all_results: list[EpisodeResult] = []

    for ep in range(num_episodes):
        env = make_hard_env()
        env._cfg = EnvConfig(
            num_drivers=HARD_CONFIG.num_drivers,
            num_orders=HARD_CONFIG.num_orders,
            max_steps=HARD_CONFIG.max_steps,
            order_deadline_min=HARD_CONFIG.order_deadline_min,
            order_deadline_max=HARD_CONFIG.order_deadline_max,
            enable_traffic=HARD_CONFIG.enable_traffic,
            dynamic_orders=HARD_CONFIG.dynamic_orders,
            dynamic_order_rate=HARD_CONFIG.dynamic_order_rate,
            max_total_orders=HARD_CONFIG.max_total_orders,
            seed=seed_offset + ep,
            inactivity_threshold=HARD_CONFIG.inactivity_threshold,
        )

        obs = env.reset()
        total_reward = 0.0
        idle_steps = 0
        done = False

        while not done:
            action = policy_fn(obs, env)
            obs = env.step(action)
            total_reward += obs.last_reward
            idle_steps += sum(
                1 for d in env._drivers if d.status == DriverStatus.IDLE
            )
            done = obs.done

        score, result = grade_episode(
            orders=env._orders,
            total_reward=total_reward,
            total_steps=env._current_step,
            idle_driver_steps=idle_steps,
            max_steps=env._cfg.max_steps,
        )
        scores.append(score)
        all_results.append(result)

        if verbose:
            print(format_grade_report(score, result, f"HARD — Episode {ep + 1}"))

    mean_score = float(np.mean(scores))
    if verbose:
        print(f"  [HARD] Mean Score: {mean_score:.4f}\n")

    return mean_score, all_results
