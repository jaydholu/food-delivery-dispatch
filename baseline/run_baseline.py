"""
Baseline policy — Greedy Nearest-Driver Dispatch.

Strategy:
    At each step, identify the closest idle driver to each pending order.
    Assign the globally closest (driver, order) pair.

This greedy heuristic serves as a reproducible lower-bound benchmark.
All three tasks are evaluated and scores are printed to stdout.

Usage:
    python -m baseline.run_baseline
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment.environment import FoodDeliveryEnv
from environment.models import DriverStatus, OrderStatus
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard


# ---------------------------------------------------------------------------
# Greedy policy implementation
# ---------------------------------------------------------------------------

# NOTE: This baseline uses full environment state (env.drivers, env.orders) for simplicity. RL agents should rely only on observations.
def greedy_policy(observation: dict, env: FoodDeliveryEnv) -> int:
    """
    Nearest-driver greedy dispatch policy.

    For each pending order, compute the Euclidean distance to every idle driver.
    Select the (driver, order) pair with minimum distance and encode it as a 
    discrete action. If no valid pair exists, return 0 (no-op).

    Args:
        observation: Observation dict from the environment (unused; we read
                     live state from ``env`` directly for clarity).
        env:         The live environment instance.

    Returns:
        Integer action from env.action_space.
    """
    # Collect idle drivers
    idle_drivers = [d for d in env.drivers if d.status == DriverStatus.IDLE]

    # Collect pending orders
    pending_orders = [o for o in env.orders if o.status == OrderStatus.PENDING]

    if not idle_drivers or not pending_orders:
        return 0  # no-op

    best_action = 0
    best_dist = float("inf")

    for driver in idle_drivers:
        for order in pending_orders:
            dist = driver.position.distance_to(order.pickup_position)
            if dist < best_dist:
                best_dist = dist

                # Encode action: k = driver_idx * MAX_ORDERS + order_idx + 1
                driver_idx = env.drivers.index(driver)

                # Find order index in env.orders (up to MAX_ORDERS)
                try:
                    order_idx = env.orders.index(order)
                except ValueError:
                    continue

                if order_idx >= env._max_orders:
                    continue
                best_action = driver_idx * env._max_orders + order_idx + 1

    return best_action


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_all_tasks(num_episodes: int = 5) -> dict[str, float]:
    """
    Run the greedy baseline on all three tasks and return scores.

    Args:
        num_episodes: Episodes per task for averaging.

    Returns:
        Dict mapping task names to mean scores.
    """
    results: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("  FOOD DELIVERY DISPATCH — GREEDY BASELINE EVALUATION")
    print("=" * 60)

    # --- EASY ---
    print("\n[1/3] Running EASY task ...")
    easy_score, _ = grade_easy(
        policy_fn=greedy_policy,
        num_episodes=num_episodes,
        seed_offset=0,
        verbose=True,
    )
    results["easy"] = easy_score

    # --- MEDIUM ---
    print("\n[2/3] Running MEDIUM task ...")
    medium_score, _ = grade_medium(
        policy_fn=greedy_policy,
        num_episodes=num_episodes,
        seed_offset=100,
        verbose=True,
    )
    results["medium"] = medium_score

    # --- HARD ---
    print("\n[3/3] Running HARD task ...")
    hard_score, _ = grade_hard(
        policy_fn=greedy_policy,
        num_episodes=num_episodes,
        seed_offset=200,
        verbose=True,
    )
    results["hard"] = hard_score

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  FINAL BASELINE SCORES")
    print("=" * 60)
    for task, score in results.items():
        bar = "█" * int(score * 30)
        print(f"  {task.upper():8s} | {bar:<30s} | {score:.4f}")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run greedy baseline on all Food Delivery tasks.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to average per task (default: 5).")
    args = parser.parse_args()

    scores = run_all_tasks(num_episodes=args.episodes)
    sys.exit(0)
