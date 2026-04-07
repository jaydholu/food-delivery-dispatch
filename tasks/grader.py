"""
Grader module — standardised task evaluation for all difficulty tiers.

Each task calls ``grade_episode`` after running a policy for one episode.
The returned score in [0.0, 1.0] is the canonical hackathon metric.
"""

from __future__ import annotations

from models import EpisodeResult
from server.food_delivery_openenv_environment import Order, OrderStatus

def grade_episode(
    orders: list[Order],
    total_reward: float,
    total_steps: int,
    idle_driver_steps: int,
    max_steps: int,
) -> tuple[float, EpisodeResult]:
    """
    Compute the normalised score ∈ [0.0, 1.0] for a completed episode.

    Scoring formula:
        score = 0.50 × delivery_rate + 0.25 × on_time_rate + 0.15 × reward_rate + 0.10 × efficiency_rate

    where:
        delivery_rate   = delivered / total_orders
        on_time_rate    = on_time / max(delivered, 1)
        reward_rate     = clamp(total_reward / reward_ceiling, 0, 1)
        efficiency_rate = 1 - clamp(idle_driver_steps / idle_ceiling, 0, 1)

    Args:
        orders:            All orders in the episode (final state).
        total_reward:      Cumulative reward over the episode.
        total_steps:       Total environment steps taken.
        idle_driver_steps: Sum of idle steps across all drivers.
        max_steps:         Episode step limit from EnvConfig.

    Returns:
        score:  Float in [0.0, 1.0].
        result: EpisodeResult dataclass with raw stats.
    """
    total_orders = len(orders)
    delivered_on_time = sum(
        1
        for o in orders
        if o.status == OrderStatus.DELIVERED
        and o.delivered_at is not None
        and o.delivered_at <= o.deadline
    )
    delivered_late = sum(
        1
        for o in orders
        if o.status == OrderStatus.DELIVERED
        and o.delivered_at is not None
        and o.delivered_at > o.deadline
    )
    failed = sum(1 for o in orders if o.status == OrderStatus.FAILED)
    delivered = delivered_on_time + delivered_late

    result = EpisodeResult(
        total_orders=total_orders,
        delivered_on_time=delivered_on_time,
        delivered_late=delivered_late,
        failed=failed,
        total_reward=total_reward,
        total_steps=total_steps,
        idle_driver_steps=idle_driver_steps,
    )

    if total_orders == 0:
        return 0.0, result

    # Component rates
    delivery_rate = delivered / total_orders
    on_time_rate = delivered_on_time / max(delivered, 1)

    # Reward ceiling: rough maximum possible reward
    reward_ceiling = total_orders * 20.0
    reward_rate = max(0.0, min(total_reward / max(reward_ceiling, 1.0), 1.0))

    # Efficiency: penalise idle time relative to worst-case all-idle scenario
    idle_ceiling = max_steps * 10  # 10 = max drivers in hard task
    efficiency_rate = 1.0 - min(idle_driver_steps / max(idle_ceiling, 1), 1.0)

    score = 0.50 * delivery_rate + 0.25 * on_time_rate + 0.15 * reward_rate + 0.10 * efficiency_rate
    score = max(0.0, min(score, 1.0))

    return score, result


def format_grade_report(score: float, result: EpisodeResult, task_name: str) -> str:
    """
    Return a human-readable grade report string.

    Args:
        score:     Normalised score in [0.0, 1.0].
        result:    EpisodeResult with raw stats.
        task_name: Human-readable task label.

    Returns:
        Formatted multi-line string.
    """
    return (
        f"\n{'=' * 50}\n"
        f"  Task: {task_name}\n"
        f"{'=' * 50}\n"
        f"  Score            :  {score:.4f} / 1.0000\n"
        f"  Total Orders     :  {result.total_orders}\n"
        f"  Delivered (OT)   :  {result.delivered_on_time}\n"
        f"  Delivered (Late) :  {result.delivered_late}\n"
        f"  Failed           :  {result.failed}\n"
        f"  Total Reward     :  {result.total_reward:.2f}\n"
        f"  Total Steps      :  {result.total_steps}\n"
        f"  Idle Drv-Steps   :  {result.idle_driver_steps}\n"
        f"{'=' * 50}\n"
    )
