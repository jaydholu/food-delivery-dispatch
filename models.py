"""
Data models for the Food Delivery Dispatch OpenEnv Environment.

Action:      FoodDeliveryAction   - structured dispatch command
Observation: FoodDeliveryObservation - full environment snapshot (no numpy)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal
from dataclasses import dataclass

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class FoodDeliveryAction(Action):
    """
    A single dispatch command issued by the agent (or LLM) each step.

    Four mutually exclusive action types:
      - assign   : assign a specific driver to a specific order
      - reject   : reject/cancel a pending order (e.g. too far, no driver)
      - wait     : do nothing this tick (no-op; incurs idle penalty)
      - batch    : assign multiple (driver_id, order_id) pairs at once

    Only the fields relevant to the chosen action_type need to be populated.
    """

    action_type: Literal["assign", "reject", "wait", "batch"] = Field(
        ...,
        description=(
            "Type of dispatch action. "
            "'assign' assigns one driver to one order. "
            "'reject' cancels a pending order. "
            "'wait' is a no-op. "
            "'batch' assigns multiple pairs simultaneously."
        ),
    )

    # --- assign / reject fields ---
    order_id: int | None = Field(
        default=None,
        description="Target order ID (required for 'assign' and 'reject').",
    )
    driver_id: int | None = Field(
        default=None,
        description="Target driver ID (required for 'assign').",
    )

    # --- batch assign fields ---
    assignments: List[Dict[str, int]] | None = Field(
        default=None,
        description=(
            "List of {driver_id, order_id} dicts for 'batch' action. "
            "Example: [{'driver_id': 0, 'order_id': 2}, ...]"
        ),
    )


# ---------------------------------------------------------------------------
# Observation sub-models
# ---------------------------------------------------------------------------

class DriverInfo(Action):
    """Snapshot of a single driver."""

    driver_id: int
    x: float = Field(..., description="Normalised x position [0, 1]")
    y: float = Field(..., description="Normalised y position [0, 1]")
    status: str = Field(..., description="idle | picking_up | delivering | offline")
    assigned_order_id: int | None = Field(default=None)
    idle_steps: int = Field(default=0)
    total_deliveries: int = Field(default=0)
    speed: float = Field(default=0.05)


class OrderInfo(Action):
    """Snapshot of a single order."""

    order_id: int
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float
    status: str = Field(..., description="pending | assigned | picked_up | delivered | failed")
    deadline: int = Field(..., description="Simulation step by which delivery must complete")
    steps_until_deadline: int = Field(default=0)
    priority: float = Field(default=0.5, description="Priority weight [0, 1]")
    assigned_driver_id: int | None = Field(default=None)
    distance_to_nearest_idle_driver: float | None = Field(default=None)


class TrafficZoneInfo(Action):
    """Snapshot of a single traffic congestion zone."""

    center_x: float
    center_y: float
    radius: float
    slowdown_multiplier: float
    active: bool


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FoodDeliveryObservation(Observation):
    """
    Full environment snapshot returned after every reset() / step().

    All numeric values are plain Python floats/ints - no numpy arrays.
    The agent (or LLM) can read this as clean JSON.
    """

    # --- Episode metadata ---
    episode_id: str = Field(default="", description="Unique episode identifier")
    current_step: int = Field(default=0)
    max_steps: int = Field(default=200)
    steps_remaining: int = Field(default=200)

    # --- Drivers ---
    drivers: List[DriverInfo] = Field(default_factory=list)
    num_idle_drivers: int = Field(default=0)
    num_active_drivers: int = Field(default=0)

    # --- Orders ---
    orders: List[OrderInfo] = Field(default_factory=list)
    num_pending_orders: int = Field(default=0)
    num_active_orders: int = Field(default=0)
    num_delivered_orders: int = Field(default=0)
    num_failed_orders: int = Field(default=0)

    # --- Traffic ---
    traffic_zones: List[TrafficZoneInfo] = Field(default_factory=list)

    # --- Last step feedback ---
    last_reward: float = Field(default=0.0)
    last_action_valid: bool = Field(default=True)
    last_action_message: str = Field(default="")

    # --- Episode aggregates (running totals for LLM context) ---
    cumulative_reward: float = Field(default=0.0)
    delivery_rate: float = Field(default=0.0, description="delivered / total_orders so far")
    on_time_rate: float = Field(default=0.0, description="on_time / delivered so far")

    # --- Terminal ---
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class EpisodeResult:
    total_orders: int
    delivered_on_time: int
    delivered_late: int
    failed: int
    total_reward: float
    total_steps: int
    idle_driver_steps: int
