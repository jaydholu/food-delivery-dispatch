"""
Food Delivery Dispatch — Pure OpenEnv Environment Implementation.

The agent acts as a dispatch controller: each step it issues one of four
action types (assign, reject, wait, batch). The simulation advances time,
moves drivers, resolves pickups and deliveries, handles deadlines, and
optionally spawns new orders (HARD mode).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import List, Tuple, cast
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DriverInfo,
        FoodDeliveryAction,
        FoodDeliveryObservation,
        OrderInfo,
        TrafficZoneInfo,
    )
except ImportError:
    from models import (
        DriverInfo,
        FoodDeliveryAction,
        FoodDeliveryObservation,
        OrderInfo,
        TrafficZoneInfo,
    )


# ---------------------------------------------------------------------------
# Internal domain enums & dataclasses
# ---------------------------------------------------------------------------

class DriverStatus(str, Enum):
    IDLE = "idle"
    PICKING_UP = "picking_up"
    DELIVERING = "delivering"
    OFFLINE = "offline"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class Pos:
    x: float
    y: float

    def dist(self, other: "Pos") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Driver:
    driver_id: int
    pos: Pos
    status: DriverStatus = DriverStatus.IDLE
    assigned_order: int | None = None
    speed: float = 0.05
    idle_steps: int = 0
    total_deliveries: int = 0
    total_late: int = 0
    consecutive_waits: int = 0


@dataclass
class Order:
    order_id: int
    pickup: Pos
    dropoff: Pos
    created_at: int
    deadline: int
    status: OrderStatus = OrderStatus.PENDING
    assigned_driver: int | None = None
    picked_up_at: int | None = None
    delivered_at: int | None = None
    priority: float = 0.5

    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.ASSIGNED)

    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.DELIVERED, OrderStatus.FAILED)


@dataclass
class TrafficZone:
    center: Pos
    radius: float
    multiplier: float
    active: bool = True


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    num_drivers: int = 4
    num_orders: int = 8
    max_steps: int = 200
    order_deadline_min: int = 30
    order_deadline_max: int = 80
    enable_traffic: bool = True
    dynamic_orders: bool = False
    dynamic_order_rate: float = 0.12
    max_total_orders: int = 30
    seed: int = 42
    inactivity_threshold: int = 20      # Inactivity threshold: penalize if no delivery in this many steps


# Default configs for each task tier
EASY_CONFIG = EnvConfig(
    num_drivers=2,
    num_orders=3,
    max_steps=150,
    order_deadline_min=50,
    order_deadline_max=120,
    enable_traffic=False,
    dynamic_orders=False,
    seed=0,
    inactivity_threshold=30,
)

MEDIUM_CONFIG = EnvConfig(
    num_drivers=4,
    num_orders=8,
    max_steps=200,
    order_deadline_min=30,
    order_deadline_max=80,
    enable_traffic=True,
    dynamic_orders=False,
    seed=1,
    inactivity_threshold=20,
)

HARD_CONFIG = EnvConfig(
    num_drivers=6,
    num_orders=15,
    max_steps=300,
    order_deadline_min=25,
    order_deadline_max=70,
    enable_traffic=True,
    dynamic_orders=True,
    dynamic_order_rate=0.12,
    max_total_orders=30,
    seed=2,
    inactivity_threshold=15,
)

TASK_CONFIGS = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}


# ---------------------------------------------------------------------------
# Reward weights — tuned to discourage waiting and encourage deliveries
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    delivery_success: float = 10.0
    early_bonus_max: float = 5.0
    early_threshold: int = 10
    late_penalty_per_step: float = 2.0
    partial_credit: float = 3.0
    order_failure: float = 8.0
    idle_penalty_base: float = 0.1
    idle_penalty_growth: float = 0.05
    idle_penalty_cap: float = 2.0
    assignment_reward: float = 0.5
    inefficiency_penalty: float = 0.5
    pickup_reward: float = 1.0
    reject_penalty: float = 1.0
    invalid_action_penalty: float = 5.0
    useless_wait_penalty: float = 0.2
    consecutive_wait_penalty: float = 2.0
    consecutive_wait_threshold: int = 3
    inactivity_penalty: float = 1.0
    efficiency_bonus_scale: float = 0.5


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class FoodDeliveryEnvironment(Environment):
    """
    Multi-driver, multi-order food delivery dispatch RL environment.

    Improvements over base version:
      - Strong wait penalties to discourage inaction
      - Inactivity penalty when no delivery occurs for N steps
      - Consecutive wait escalation penalty
      - Much stronger invalid action penalty (5.0 vs 0.2)
      - Efficiency bonus tied to delivery rate / steps
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "medium") -> None:
        self._task = task
        self._cfg = TASK_CONFIGS.get(task, MEDIUM_CONFIG)
        self._rwt = RewardWeights()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(self._cfg.seed)

        # Episode state
        self._drivers: List[Driver] = []
        self._orders: List[Order] = []
        self._traffic: List[TrafficZone] = []
        self._current_step: int = 0
        self._order_counter: int = 0
        self._cumulative_rew: float = 0.0
        self._delivered_ot: int = 0
        self._delivered_late: int = 0
        self._delivered_all: int = 0
        self._failed: int = 0
        self._consecutive_global_waits: int = 0
        self._steps_since_last_delivery: int = 0   # inactivity tracking

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> FoodDeliveryObservation:
        cfg = self._cfg
        self._rng = random.Random(cfg.seed)
        self._current_step = 0
        self._order_counter = 0
        self._cumulative_rew = 0.0
        self._delivered_ot = 0
        self._delivered_late = 0
        self._delivered_all = 0
        self._failed = 0
        self._consecutive_global_waits = 0
        self._steps_since_last_delivery = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._drivers = [
            Driver(
                driver_id=i,
                pos=self._rand_pos(),
                speed=self._rng.uniform(0.04, 0.07),
            )
            for i in range(cfg.num_drivers)
        ]

        self._orders = [self._spawn_order() for _ in range(cfg.num_orders)]

        self._traffic = []
        if cfg.enable_traffic:
            for _ in range(self._rng.randint(2, 3)):
                self._traffic.append(TrafficZone(
                    center=self._rand_pos(),
                    radius=self._rng.uniform(0.1, 0.2),
                    multiplier=self._rng.uniform(1.5, 3.0),
                ))

        return self._build_obs(reward=0.0, done=False, action_valid=True, action_msg="Environment reset.")

    def step(self, action: FoodDeliveryAction) -> FoodDeliveryObservation:
        self._state.step_count += 1
        self._current_step += 1
        self._steps_since_last_delivery += 1

        reward = 0.0
        action_valid = True
        action_msg = ""

        # 1. Apply dispatch action
        dispatch_reward, action_valid, action_msg = self._apply_action_safe(action)
        reward += dispatch_reward

        # 2. Move drivers
        self._move_drivers()

        # 3. Resolve pickups / deliveries
        pickup_reward, delivery_reward = self._resolve_arrivals()
        reward += pickup_reward + delivery_reward

        # 4. Expire overdue orders
        failure_penalty = self._check_deadlines()
        reward += failure_penalty

        # 5. Dynamic order spawning (HARD only)
        if self._cfg.dynamic_orders:
            self._maybe_spawn_orders()

        # 6. Idle driver penalty
        idle_penalty = self._compute_idle_penalty()
        reward += idle_penalty

        # 7. Inactivity penalty — punish extended periods without delivery
        if self._steps_since_last_delivery >= self._cfg.inactivity_threshold:
            inactivity_steps = self._steps_since_last_delivery - self._cfg.inactivity_threshold
            # Scale penalty by how long we've been inactive
            scaled = min(inactivity_steps * 0.05, self._rwt.inactivity_penalty)
            reward -= scaled

        # 8. Efficiency bonus — reward delivery rate relative to steps taken
        if self._delivered_all > 0 and self._current_step > 0:
            efficiency = self._delivered_all / max(self._current_step, 1)
            reward += efficiency * self._rwt.efficiency_bonus_scale

        # 9. Accumulate
        self._cumulative_rew += reward

        # 10. Check termination
        done = self._is_done()

        return self._build_obs(reward=reward, done=done, action_valid=action_valid, action_msg=action_msg)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Safe action processing
    # ------------------------------------------------------------------

    def _apply_action_safe(self, action: FoodDeliveryAction) -> Tuple[float, bool, str]:
        try:
            return self._apply_action(action)
        
        except Exception as exc:
            # Strong penalty for crashing actions
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Action error: {exc}"

    def _apply_action(self, action: FoodDeliveryAction) -> Tuple[float, bool, str]:
        at = action.action_type

        if at == "wait":
            return self._handle_wait()

        if at == "assign":
            return self._do_assign(action.driver_id, action.order_id)

        if at == "reject":
            return self._do_reject(action.order_id)

        if at == "batch":
            if not action.assignments:
                penalty = -self._rwt.invalid_action_penalty
                return penalty, False, "batch action requires 'assignments' list."
            
            total_r = 0.0
            msgs = []
            any_valid = False
            for pair in action.assignments:
                try:
                    did = int(pair.get("driver_id", -1))
                    oid = int(pair.get("order_id", -1))
                except (TypeError, ValueError):
                    msgs.append("Invalid id types in batch pair")
                    continue

                r, v, m = self._do_assign(did, oid)
                total_r += r
                msgs.append(m)
                if v:
                    any_valid = True
            self._consecutive_global_waits = 0

            return total_r, any_valid, " | ".join(msgs)

        # Unknown action type — strong penalty
        penalty = -self._rwt.invalid_action_penalty

        return penalty, False, f"Unknown action_type: {at!r}"

    def _handle_wait(self) -> Tuple[float, bool, str]:
        """
        Handle wait action.

        Penalties applied:
          1. Base useless-wait penalty when work is available
          2. Escalating consecutive-wait penalty after threshold
        """
        pending = [o for o in self._orders if o.status == OrderStatus.PENDING]
        idle = [d for d in self._drivers if d.status == DriverStatus.IDLE]

        if pending and idle:
            # Useless wait: work is available but agent chose to do nothing
            self._consecutive_global_waits += 1

            # Base penalty, grows with consecutive waits (capped)
            base_penalty = self._rwt.useless_wait_penalty * min(self._consecutive_global_waits, 10)

            # Extra escalation penalty after threshold
            extra_penalty = 0.0
            if self._consecutive_global_waits > self._rwt.consecutive_wait_threshold:
                extra_penalty = self._rwt.consecutive_wait_penalty

            total_penalty = -(base_penalty + extra_penalty)
            return (
                total_penalty,
                True,
                f"Useless wait (consecutive={self._consecutive_global_waits}). "
                f"Penalty={total_penalty:.2f}",
            )
        else:
            # Appropriate wait — no work available
            self._consecutive_global_waits = 0
            return 0.0, True, "Wait (no work available — appropriate)."

    def _do_assign(self, driver_id: int | None, order_id: int | None) -> Tuple[float, bool, str]:
        if driver_id is None or order_id is None:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, "assign requires driver_id and order_id."

        try:
            driver_id = int(driver_id)
            order_id = int(order_id)
        except (TypeError, ValueError):
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, "driver_id and order_id must be integers."

        driver = self._driver_by_id(driver_id)
        order = self._order_by_id(order_id)

        if driver is None:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Driver {driver_id} does not exist."

        if order is None:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Order {order_id} does not exist."

        if driver.status != DriverStatus.IDLE:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Driver {driver_id} not idle (status={driver.status})."

        if order.status != OrderStatus.PENDING:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Order {order_id} not pending (status={order.status})."

        # Valid assignment
        driver.status = DriverStatus.PICKING_UP
        driver.assigned_order = order_id
        driver.idle_steps = 0
        driver.consecutive_waits = 0
        order.status = OrderStatus.ASSIGNED
        order.assigned_driver = driver_id

        self._consecutive_global_waits = 0

        dist = driver.pos.dist(order.pickup)
        ineff = -self._rwt.inefficiency_penalty * dist
        reward = self._rwt.assignment_reward + ineff

        return reward, True, f"Assigned driver {driver_id} → order {order_id} (dist={dist:.3f})."

    def _do_reject(self, order_id: int | None) -> Tuple[float, bool, str]:
        if order_id is None:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, "reject requires order_id."

        try:
            order_id = int(order_id)
        except (TypeError, ValueError):
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, "order_id must be an integer."

        order = self._order_by_id(order_id)
        if order is None:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Order {order_id} does not exist."

        if order.status != OrderStatus.PENDING:
            penalty = -self._rwt.invalid_action_penalty
            return penalty, False, f"Order {order_id} is not pending."

        order.status = OrderStatus.FAILED
        self._failed += 1

        return -self._rwt.reject_penalty, True, f"Order {order_id} rejected."

    # ------------------------------------------------------------------
    # Simulation mechanics
    # ------------------------------------------------------------------

    def _move_drivers(self) -> None:
        order_map = {o.order_id: o for o in self._orders}

        for d in self._drivers:
            if d.status == DriverStatus.IDLE or d.assigned_order is None:
                continue

            order = order_map.get(d.assigned_order)
            if order is None:
                d.status = DriverStatus.IDLE
                d.assigned_order = None
                continue

            target = order.pickup if d.status == DriverStatus.PICKING_UP else order.dropoff
            speed = self._effective_speed(d)
            self._move_toward(d, target, speed)

    def _effective_speed(self, d: Driver) -> float:
        spd = d.speed
        for zone in self._traffic:
            if zone.active and d.pos.dist(zone.center) <= zone.radius:
                spd /= zone.multiplier

        return max(spd, 0.001)

    @staticmethod
    def _move_toward(d: Driver, target: Pos, speed: float) -> None:
        dx = target.x - d.pos.x
        dy = target.y - d.pos.y
        dist = sqrt(dx * dx + dy * dy)

        if dist <= speed:
            d.pos = Pos(target.x, target.y)
        else:
            d.pos = Pos(d.pos.x + dx / dist * speed, d.pos.y + dy / dist * speed)

    def _resolve_arrivals(self) -> Tuple[float, float]:
        order_map = {o.order_id: o for o in self._orders}
        pickup_reward = 0.0
        delivery_reward = 0.0
        tol = 1e-4

        for d in self._drivers:
            if d.assigned_order is None:
                continue
            
            order = order_map.get(d.assigned_order)
            if order is None:
                continue

            if d.status == DriverStatus.PICKING_UP:
                if d.pos.dist(order.pickup) <= tol:
                    d.status = DriverStatus.DELIVERING
                    order.status = OrderStatus.PICKED_UP
                    order.picked_up_at = self._current_step
                    pickup_reward += self._rwt.pickup_reward

            elif d.status == DriverStatus.DELIVERING:
                if d.pos.dist(order.dropoff) <= tol:
                    d.status = DriverStatus.IDLE
                    d.assigned_order = None
                    d.total_deliveries += 1
                    order.status = OrderStatus.DELIVERED
                    order.delivered_at = self._current_step
                    self._delivered_all += 1
                    self._steps_since_last_delivery = 0  # Reset inactivity counter

                    steps_remaining = order.deadline - self._current_step
                    if steps_remaining >= 0:
                        self._delivered_ot += 1
                        delivery_reward += self._rwt.delivery_success
                        if steps_remaining >= self._rwt.early_threshold:
                            frac = min(steps_remaining / max(order.deadline, 1), 1.0)
                            delivery_reward += self._rwt.early_bonus_max * frac
                    else:
                        self._delivered_late += 1
                        d.total_late += 1
                        lateness = abs(steps_remaining)
                        delivery_reward += self._rwt.partial_credit
                        delivery_reward -= self._rwt.late_penalty_per_step * min(lateness, 10)

        return pickup_reward, delivery_reward

    def _check_deadlines(self) -> float:
        penalty = 0.0
        for order in self._orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.ASSIGNED):
                continue
            if self._current_step <= order.deadline:
                continue

            order.status = OrderStatus.FAILED
            self._failed += 1
            penalty -= self._rwt.order_failure

            if order.assigned_driver is not None:
                d = self._driver_by_id(order.assigned_driver)
                if d:
                    d.status = DriverStatus.IDLE
                    d.assigned_order = None

        return penalty

    def _compute_idle_penalty(self) -> float:
        penalty = 0.0
        for d in self._drivers:
            if d.status == DriverStatus.IDLE:
                d.idle_steps += 1
                raw = self._rwt.idle_penalty_base * (
                    1 + d.idle_steps * self._rwt.idle_penalty_growth
                )
                penalty -= min(raw, self._rwt.idle_penalty_cap)
            else:
                d.idle_steps = 0
        return penalty

    def _maybe_spawn_orders(self) -> None:
        if len(self._orders) >= self._cfg.max_total_orders:
            return
        if self._rng.random() < self._cfg.dynamic_order_rate:
            self._orders.append(self._spawn_order())

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._current_step >= self._cfg.max_steps:
            return True

        active = [o for o in self._orders if o.is_active()]
        if active:
            return False

        if self._cfg.dynamic_orders and len(self._orders) < self._cfg.max_total_orders:
            return False

        return True

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        reward: float,
        done: bool,
        action_valid: bool,
        action_msg: str,
    ) -> FoodDeliveryObservation:
        total_orders = len(self._orders)
        delivery_rate = self._delivered_all / max(total_orders, 1)
        on_time_rate = self._delivered_ot / max(self._delivered_all, 1)

        driver_infos = [
            DriverInfo(
                driver_id=int(d.driver_id),
                x=round(float(d.pos.x), 4),
                y=round(float(d.pos.y), 4),
                status=str(d.status.value),
                assigned_order_id=int(d.assigned_order) if d.assigned_order is not None else None,
                idle_steps=int(d.idle_steps),
                total_deliveries=int(d.total_deliveries),
                speed=round(float(d.speed), 4),
            )
            for d in self._drivers
        ]

        idle_drivers = [d for d in self._drivers if d.status == DriverStatus.IDLE]
        order_infos = []
        for o in self._orders:
            nearest_dist: float | None = None
            if o.status == OrderStatus.PENDING and idle_drivers:
                nearest_dist = round(
                    float(min(d.pos.dist(o.pickup) for d in idle_drivers)), 4
                )

            order_infos.append(OrderInfo(
                order_id=int(o.order_id),
                pickup_x=round(float(o.pickup.x), 4),
                pickup_y=round(float(o.pickup.y), 4),
                dropoff_x=round(float(o.dropoff.x), 4),
                dropoff_y=round(float(o.dropoff.y), 4),
                status=str(o.status.value),
                deadline=int(o.deadline),
                steps_until_deadline=int(max(o.deadline - self._current_step, 0)),
                priority=round(float(o.priority), 4),
                assigned_driver_id=int(o.assigned_driver) if o.assigned_driver is not None else None,
                distance_to_nearest_idle_driver=nearest_dist,
            ))

        traffic_infos = [
            TrafficZoneInfo(
                center_x=round(float(z.center.x), 4),
                center_y=round(float(z.center.y), 4),
                radius=round(float(z.radius), 4),
                slowdown_multiplier=round(float(z.multiplier), 4),
                active=bool(z.active),
            )
            for z in self._traffic
        ]

        return FoodDeliveryObservation(
            episode_id=str(cast(str, self._state.episode_id)),
            current_step=int(self._current_step),
            max_steps=int(self._cfg.max_steps),
            steps_remaining=int(max(self._cfg.max_steps - self._current_step, 0)),
            drivers=driver_infos,
            num_idle_drivers=int(len(idle_drivers)),
            num_active_drivers=int(sum(1 for d in self._drivers if d.status != DriverStatus.IDLE)),
            orders=order_infos,
            num_pending_orders=int(sum(1 for o in self._orders if o.status == OrderStatus.PENDING)),
            num_active_orders=int(sum(1 for o in self._orders if o.is_active())),
            num_delivered_orders=int(self._delivered_all),
            num_failed_orders=int(self._failed),
            traffic_zones=traffic_infos,
            last_reward=round(float(reward), 4),
            last_action_valid=bool(action_valid),
            last_action_message=str(action_msg),
            cumulative_reward=round(float(self._cumulative_rew), 4),
            delivery_rate=round(float(delivery_rate), 4),
            on_time_rate=round(float(on_time_rate), 4),
            done=bool(done),
            reward=round(float(reward), 4),
            metadata={
                "task": str(self._task),
                "delivered_on_time": int(self._delivered_ot),
                "delivered_late": int(self._delivered_late),
                "failed": int(self._failed),
                "step": int(self._current_step),
                "steps_since_last_delivery": int(self._steps_since_last_delivery),
                "consecutive_waits": int(self._consecutive_global_waits),
            },
        )

    # ------------------------------------------------------------------
    # World generation helpers
    # ------------------------------------------------------------------

    def _rand_pos(self) -> Pos:
        return Pos(x=self._rng.uniform(0.0, 1.0), y=self._rng.uniform(0.0, 1.0))

    def _spawn_order(self) -> Order:
        oid = self._order_counter
        self._order_counter += 1
        offset = self._rng.randint(self._cfg.order_deadline_min, self._cfg.order_deadline_max)

        return Order(
            order_id=oid,
            pickup=self._rand_pos(),
            dropoff=self._rand_pos(),
            created_at=self._current_step,
            deadline=self._current_step + offset,
            priority=self._rng.uniform(0.2, 1.0),
        )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _driver_by_id(self, did: int) -> Driver | None:
        for d in self._drivers:
            if d.driver_id == did:
                return d
        return None

    def _order_by_id(self, oid: int) -> Order | None:
        for o in self._orders:
            if o.order_id == oid:
                return o
        return None

    # ------------------------------------------------------------------
    # Score utility
    # ------------------------------------------------------------------

    def compute_score(self) -> float:
        """Normalised score in [0, 1]."""
        total = len(self._orders)
        if total == 0:
            return 0.0

        delivery_rate = self._delivered_all / total
        on_time_rate = self._delivered_ot / max(self._delivered_all, 1)

        reward_ceiling = total * 20.0
        reward_rate = max(0.0, min(self._cumulative_rew / max(reward_ceiling, 1), 1.0))

        idle_ceiling = self._cfg.max_steps * 10
        total_idle = sum(d.idle_steps for d in self._drivers)
        efficiency_rate = 1.0 - min(total_idle / max(idle_ceiling, 1), 1.0)

        score = (
            0.50 * delivery_rate
            + 0.25 * on_time_rate
            + 0.15 * reward_rate
            + 0.10 * efficiency_rate
        )

        return round(max(0.0, min(score, 1.0)), 4)
