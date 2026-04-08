"""
EASY task — 2 drivers, 3 orders, no traffic, generous deadlines.

Designed for initial policy development and sanity-checking.
The small state space allows near-exhaustive search and should yield
high scores (> 0.80) with even simple heuristic policies.

Difficulty characteristics:
  - Only 2 drivers and 3 orders (very manageable)
  - No traffic zones
  - Long deadlines (50-120 steps) giving plenty of time
  - 150 max steps
  - Moderate wait penalties to train correct behavior
"""

from __future__ import annotations

from typing import Any
import numpy as np

from models import EpisodeResult
from server.food_delivery_dispatch_environment import (
    EASY_CONFIG,
    EASY_REWARDS,
    DriverStatus,
    EnvConfig,
    FoodDeliveryEnvironment,
)
from tasks.grader import format_grade_report, grade_episode


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_easy_env() -> FoodDeliveryEnvironment:
    """
    Construct and return the EASY task environment.

    EASY settings:
      - 2 drivers
      - 3 initial orders
      - No traffic
      - Generous deadlines (50-120 steps)
      - 150 max steps
      - Inactivity threshold: 30 steps
      - Moderate wait penalties

    Returns:
        Configured FoodDeliveryEnvironment instance.
    """
    env = FoodDeliveryEnvironment(task="easy")
    env._rwt = EASY_REWARDS
    return env


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_easy(
    policy_fn: Any,
    num_episodes: int = 5,
    seed_offset: int = 0,
    verbose: bool = True,
) -> tuple[float, list[EpisodeResult]]:
    """
    Evaluate a policy on the EASY task over multiple episodes.

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
        env = make_easy_env()
        env._cfg = EnvConfig(
            num_drivers=EASY_CONFIG.num_drivers,
            num_orders=EASY_CONFIG.num_orders,
            max_steps=EASY_CONFIG.max_steps,
            order_deadline_min=EASY_CONFIG.order_deadline_min,
            order_deadline_max=EASY_CONFIG.order_deadline_max,
            enable_traffic=EASY_CONFIG.enable_traffic,
            dynamic_orders=EASY_CONFIG.dynamic_orders,
            seed=seed_offset + ep,
            inactivity_threshold=EASY_CONFIG.inactivity_threshold,
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
            print(format_grade_report(score, result, f"EASY — Episode {ep + 1}"))

    mean_score = float(np.mean(scores))
    if verbose:
        print(f"  [EASY] Mean Score: {mean_score:.4f}\n")

    return mean_score, all_results
