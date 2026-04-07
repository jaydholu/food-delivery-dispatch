"""
HARD task — 6 drivers, 15 initial orders, traffic + dynamic order spawning.

Dynamic spawning means the agent must continuously balance in-progress deliveries against newly arriving demand.
This is the most realistic and challenging configuration; the greedy nearest-driver baseline scores ~0.81.
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

HARD_CONFIG = EnvironmentConfig(
    num_drivers=6,
    num_orders=15,
    max_steps=300,
    map_size=1.0,
    order_deadline_min=25,
    order_deadline_max=70,
    enable_traffic=True,
    dynamic_orders=True,
    dynamic_order_rate=0.12,
    max_total_orders=30,
    seed=2,
)

HARD_REWARD_CONFIG = RewardConfig(
    delivery_success=10.0,
    early_bonus=4.0,
    late_penalty=3.0,
    idle_penalty=0.15,
    inefficiency_penalty=0.6,
    order_failure_penalty=9.0,
    assignment_reward=0.5,
    pickup_reward=1.0,
    early_threshold=8,
    max_idle_penalty=2.5,
)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def make_hard_env() -> FoodDeliveryEnv:
    """
    Construct and return the HARD task environment.

    Returns:
        Configured FoodDeliveryEnv instance.
    """
    return FoodDeliveryEnv(
        config=HARD_CONFIG,
        reward_config=HARD_REWARD_CONFIG
    )


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
        policy_fn:    Callable(observation, env) → int action.
        num_episodes: Number of evaluation episodes.
        seed_offset:  Shift seeds for independent evaluation runs.
        verbose:      Print per-episode reports.

    Returns:
        mean_score: Average normalised score across episodes.
        results:    List of EpisodeResult dataclasses.
    """
    env = make_hard_env()
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
            max_steps=HARD_CONFIG.max_steps,
        )
        scores.append(score)
        all_results.append(result)

        if verbose:
            print(format_grade_report(score, result, f"HARD — Episode {ep + 1}"))

    mean_score = float(np.mean(scores))
    if verbose:
        print(f"  [HARD] Mean Score: {mean_score:.4f}\n")

    return mean_score, all_results
