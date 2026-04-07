"""
EASY task — 2 drivers, 3 orders, no traffic, no dynamic spawning.

This task is designed for initial policy development and sanity-checking.
The small state space allows near-exhaustive search and should yield high scores (> 0.80) with even simple heuristic policies.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from environment.environment import FoodDeliveryEnv
from environment.models import EnvironmentConfig
from environment.reward import RewardConfig
from tasks.grader import EpisodeResult, format_grade_report, grade_episode


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

EASY_CONFIG = EnvironmentConfig(
    num_drivers=2,
    num_orders=3,
    max_steps=150,
    map_size=1.0,
    order_deadline_min=40,
    order_deadline_max=100,
    enable_traffic=False,
    dynamic_orders=False,
    seed=0,
)

EASY_REWARD_CONFIG = RewardConfig(
    delivery_success=10.0,
    early_bonus=5.0,
    late_penalty=2.0,
    idle_penalty=0.05,
    inefficiency_penalty=0.3,
    order_failure_penalty=8.0,
    assignment_reward=0.5,
    pickup_reward=1.0,
    max_idle_penalty=5.0
)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_easy_env() -> FoodDeliveryEnv:
    """
    Construct and return the EASY task environment.

    Returns:
        Configured FoodDeliveryEnv instance.
    """
    return FoodDeliveryEnv(
        config=EASY_CONFIG,
        reward_config=EASY_REWARD_CONFIG
    )


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade_easy(
    policy_fn: Any,
    num_episodes: int = 5,
    seed_offset: int = 0,
    verbose: bool = True
) -> tuple[float, list[EpisodeResult]]:
    """
    Evaluate a policy on the EASY task over multiple episodes.

    Args:
        policy_fn:    Callable(observation, env) → int action.
        num_episodes: Number of evaluation episodes.
        seed_offset:  Shift seeds for independent evaluation runs.
        verbose:      Print per-episode reports.

    Returns:
        mean_score: Average normalised score across episodes.
        results:    List of EpisodeResult dataclasses.
    """
    env = make_easy_env()
    scores: list[float] = []
    all_results: list[EpisodeResult] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        idle_steps = 0
        done = False

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            idle_steps += sum(1 for d in env.drivers if d.status.value == "idle")
            done = terminated or truncated

        score, result = grade_episode(
            orders=env.orders,
            total_reward=total_reward,
            total_steps=env.current_step,
            idle_driver_steps=idle_steps,
            max_steps=EASY_CONFIG.max_steps,
        )
        scores.append(score)
        all_results.append(result)

        if verbose:
            print(format_grade_report(score, result, f"EASY — Episode {ep + 1}"))

    mean_score = float(np.mean(scores))
    if verbose:
        print(f"  [EASY] Mean Score: {mean_score:.4f}\n")

    return mean_score, all_results
