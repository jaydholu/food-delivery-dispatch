"""
Food Delivery Dispatch — OpenEnv Client.

Wraps the OpenEnv EnvClient to communicate with the Food Delivery server
over WebSocket, exposing reset() / step() with typed action / observation.
"""

from __future__ import annotations

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DriverInfo,
    FoodDeliveryAction,
    FoodDeliveryObservation,
    OrderInfo,
    TrafficZoneInfo,
)


class FoodDeliveryEnv(
    EnvClient[FoodDeliveryAction, FoodDeliveryObservation, State]
):
    """
    Client for the Food Delivery Dispatch OpenEnv Environment.

    Maintains a persistent WebSocket connection to the environment server
    for low-latency multi-step interaction.

    Quick start (Docker):
        >>> env = FoodDeliveryEnv.from_docker_image("food_delivery_openenv-env:latest")
        >>> try:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(f"Pending orders: {obs.num_pending_orders}")
        ...
        ...     action = FoodDeliveryAction(
        ...         action_type="assign",
        ...         driver_id=0,
        ...         order_id=0,
        ...     )
        ...     result = env.step(action)
        ...     print(f"Reward: {result.reward}  Done: {result.done}")
        ... finally:
        ...     env.close()

    Quick start (running server):
        >>> with FoodDeliveryEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     result = env.step(FoodDeliveryAction(action_type="wait"))
    """

    # ------------------------------------------------------------------
    # Outgoing payload
    # ------------------------------------------------------------------

    def _step_payload(self, action: FoodDeliveryAction) -> Dict:
        """Serialise FoodDeliveryAction to JSON-safe dict."""
        payload: Dict = {"action_type": action.action_type}

        if action.action_type in ("assign", "reject"):
            if action.order_id is not None:
                payload["order_id"] = action.order_id

            if action.action_type == "assign" and action.driver_id is not None:
                payload["driver_id"] = action.driver_id

        elif action.action_type == "batch":
            payload["assignments"] = action.assignments or []

        return payload

    # ------------------------------------------------------------------
    # Incoming payload
    # ------------------------------------------------------------------

    def _parse_result(self, payload: Dict) -> StepResult[FoodDeliveryObservation]:
        """Parse server JSON response into typed StepResult."""
        obs_data    = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        
        return StepResult(
            observation = observation,
            reward      = payload.get("reward"),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse /state response."""
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_observation(obs_data: Dict, payload: Dict) -> FoodDeliveryObservation:
        """Convert raw dict to FoodDeliveryObservation."""

        def _parse_drivers(raw: List[Dict]) -> List[DriverInfo]:
            out = []
            for d in raw:
                out.append(DriverInfo(
                    driver_id          = d.get("driver_id", 0),
                    x                  = d.get("x", 0.0),
                    y                  = d.get("y", 0.0),
                    status             = d.get("status", "idle"),
                    assigned_order_id  = d.get("assigned_order_id"),
                    idle_steps         = d.get("idle_steps", 0),
                    total_deliveries   = d.get("total_deliveries", 0),
                    speed              = d.get("speed", 0.05),
                ))
            return out

        def _parse_orders(raw: List[Dict]) -> List[OrderInfo]:
            out = []
            for o in raw:
                out.append(OrderInfo(
                    order_id                         = o.get("order_id", 0),
                    pickup_x                         = o.get("pickup_x", 0.0),
                    pickup_y                         = o.get("pickup_y", 0.0),
                    dropoff_x                        = o.get("dropoff_x", 0.0),
                    dropoff_y                        = o.get("dropoff_y", 0.0),
                    status                           = o.get("status", "pending"),
                    deadline                         = o.get("deadline", 0),
                    steps_until_deadline             = o.get("steps_until_deadline", 0),
                    priority                         = o.get("priority", 0.5),
                    assigned_driver_id               = o.get("assigned_driver_id"),
                    distance_to_nearest_idle_driver  = o.get("distance_to_nearest_idle_driver"),
                ))
            return out

        def _parse_traffic(raw: List[Dict]) -> List[TrafficZoneInfo]:
            out = []
            for z in raw:
                out.append(TrafficZoneInfo(
                    center_x            = z.get("center_x", 0.0),
                    center_y            = z.get("center_y", 0.0),
                    radius              = z.get("radius", 0.1),
                    slowdown_multiplier = z.get("slowdown_multiplier", 1.5),
                    active              = z.get("active", True),
                ))
            return out

        return FoodDeliveryObservation(
            episode_id           = obs_data.get("episode_id", ""),
            current_step         = obs_data.get("current_step", 0),
            max_steps            = obs_data.get("max_steps", 200),
            steps_remaining      = obs_data.get("steps_remaining", 200),
            drivers              = _parse_drivers(obs_data.get("drivers", [])),
            num_idle_drivers     = obs_data.get("num_idle_drivers", 0),
            num_active_drivers   = obs_data.get("num_active_drivers", 0),
            orders               = _parse_orders(obs_data.get("orders", [])),
            num_pending_orders   = obs_data.get("num_pending_orders", 0),
            num_active_orders    = obs_data.get("num_active_orders", 0),
            num_delivered_orders = obs_data.get("num_delivered_orders", 0),
            num_failed_orders    = obs_data.get("num_failed_orders", 0),
            traffic_zones        = _parse_traffic(obs_data.get("traffic_zones", [])),
            last_reward          = obs_data.get("last_reward", 0.0),
            last_action_valid    = obs_data.get("last_action_valid", True),
            last_action_message  = obs_data.get("last_action_message", ""),
            cumulative_reward    = obs_data.get("cumulative_reward", 0.0),
            delivery_rate        = obs_data.get("delivery_rate", 0.0),
            on_time_rate         = obs_data.get("on_time_rate", 0.0),
            done                 = payload.get("done", False),
            reward               = payload.get("reward") or obs_data.get("reward", 0.0),
            metadata             = obs_data.get("metadata", {}),
        )
