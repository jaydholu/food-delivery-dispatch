"""
EASY task — 2 drivers, 3 orders, no traffic, no dynamic spawning.

This task is designed for initial policy development and sanity-checking.
The small state space allows near-exhaustive search and should yield high
scores (> 0.80) with even simple heuristic policies.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from models import EpisodeResult
from server.food_delivery_openenv_environment import (
    EASY_CONFIG,
    DriverStatus,
    EnvConfig,
    FoodDeliveryEnvironment,
    RewardWeights,
)
from tasks.grader import format_grade_report, grade_episode


# ---------------------------------------------------------------------------
# Task reward weights (override defaults for easy mode)
# ---------------------------------------------------------------------------

EASY_REWARD_CONFIG = RewardWeights(
    delivery_success=10.0,
    early_bonus_max=5.0,
    late_penalty_per_step=2.0,
    idle_penalty_base=0.05,
    inefficiency_penalty=0.3,
    order_failure=8.0,
    assignment_reward=0.5,
    pickup_reward=1.0,
    idle_penalty_cap=5.0,
)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_easy_env() -> FoodDeliveryEnvironment:
    """
    Construct and return the EASY task environment.

    Returns:
        Configured FoodDeliveryEnvironment instance.
    """
    env = FoodDeliveryEnvironment(task="easy")
    env._rwt = EASY_REWARD_CONFIG
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

    The policy receives the raw FoodDeliveryObservation returned by the
    environment and the environment instance itself, and must return a
    FoodDeliveryAction (or a dict that can be passed to env.step).

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
