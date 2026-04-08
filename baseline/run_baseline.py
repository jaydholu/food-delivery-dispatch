"""
Baseline policy - Greedy Nearest-Driver Dispatch.

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

# Correct imports: environment lives in server/, models in root
from server.food_delivery_dispatch_environment import (
    FoodDeliveryEnvironment,
    DriverStatus,
    OrderStatus,
)
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard


# ---------------------------------------------------------------------------
# Greedy policy implementation
# ---------------------------------------------------------------------------

# NOTE: This baseline uses full environment state directly for simplicity.
# RL agents should rely only on observations.
def greedy_policy(observation, env: FoodDeliveryEnvironment) -> dict:
    """
    Nearest-driver greedy dispatch policy.

    For each pending order, compute the Euclidean distance to every idle driver.
    Select the (driver, order) pair with minimum distance and return a
    FoodDeliveryAction-compatible dict. If no valid pair exists, return a wait action.

    Args:
        observation: Observation from the environment (unused; we read
                     live state from ``env`` directly for clarity).
        env:         The live FoodDeliveryEnvironment instance.

    Returns:
        dict action compatible with env.step().
    """
    # Collect idle drivers
    idle_drivers = [d for d in env._drivers if d.status == DriverStatus.IDLE]

    # Collect pending orders
    pending_orders = [o for o in env._orders if o.status == OrderStatus.PENDING]

    if not idle_drivers or not pending_orders:
        return {"action_type": "wait"}

    best_driver = None
    best_order = None
    best_dist = float("inf")

    for driver in idle_drivers:
        for order in pending_orders:
            dist = driver.pos.dist(order.pickup)
            if dist < best_dist:
                best_dist = dist
                best_driver = driver
                best_order = order

    if best_driver is None or best_order is None:
        return {"action_type": "wait"}

    from models import FoodDeliveryAction
    return FoodDeliveryAction(
        action_type="assign",
        driver_id=best_driver.driver_id,
        order_id=best_order.order_id,
    )


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
    print("  FOOD DELIVERY DISPATCH - GREEDY BASELINE EVALUATION")
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
        bar = "" * int(score * 30)
        print(f"  {task.upper():8s} | {bar:<30s} | {score:.4f}")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run greedy baseline on all Food Delivery tasks."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to average per task (default: 5).",
    )
    args = parser.parse_args()

    scores = run_all_tasks(num_episodes=args.episodes)
    sys.exit(0)
