"""
HARD task — 6 drivers, 15 initial orders, traffic + dynamic order spawning.

Dynamic spawning means the agent must continuously balance in-progress
deliveries against newly arriving demand.
This is the most realistic and challenging configuration; the greedy
nearest-driver baseline scores ~0.81.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from models import EpisodeResult
from server.food_delivery_openenv_environment import (
    HARD_CONFIG,
    DriverStatus,
    EnvConfig,
    FoodDeliveryEnvironment,
    RewardWeights,
)
from tasks.grader import format_grade_report, grade_episode

# ---------------------------------------------------------------------------
# Task reward weights
# ---------------------------------------------------------------------------

HARD_REWARD_CONFIG = RewardWeights(
    delivery_success=10.0,
    early_bonus_max=4.0,
    early_threshold=8,
    late_penalty_per_step=3.0,
    idle_penalty_base=0.15,
    inefficiency_penalty=0.6,
    order_failure=9.0,
    assignment_reward=0.5,
    pickup_reward=1.0,
    idle_penalty_cap=2.5,
)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_hard_env() -> FoodDeliveryEnvironment:
    """
    Construct and return the HARD task environment.

    Returns:
        Configured FoodDeliveryEnvironment instance.
    """
    env = FoodDeliveryEnvironment(task="hard")
    env._rwt = HARD_REWARD_CONFIG
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
