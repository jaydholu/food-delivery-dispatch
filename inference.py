"""
Food Delivery Dispatch - LLM-Based Inference Agent (HuggingFace Router).
An LLM (via HuggingFace router API) observes the environment state each step
and decides which dispatch action to take. The agent is prompted with strong
behavioral guidance to actively assign drivers and avoid waiting.

STRICT Log format (NO extra output allowed):
    [START] task=<task> env=<image> model=<model>
    [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<n> score=<float> rewards=<list>

Environment variables:
    HF_TOKEN       - HuggingFace token (primary API key)
    API_KEY        - Alternative API key
    API_BASE_URL   - API base URL (default: https://router.huggingface.co/v1)
    MODEL_NAME     - Model to use (default: Qwen/Qwen2.5-72B-Instruct)
    IMAGE_NAME     - Docker image name (default: food_delivery_dispatch-env:latest)
    TASK           - easy | medium | hard (default: medium)
    MAX_RETRIES    - LLM retry attempts (default: 3)
"""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath("."))

import json
import time
import traceback
from typing import Any, Dict, List

from openai import OpenAI


# ---------------------------------------------------------------------------
# Environment variables - HuggingFace router
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
IMAGE_NAME = os.getenv("IMAGE_NAME", "food_delivery_dispatch-env:latest")
TASK = os.getenv("TASK", "medium")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


# Score calculation constants
MAX_STEPS_MAP = {"easy": 150, "medium": 200, "hard": 300}
MAX_STEPS = MAX_STEPS_MAP.get(TASK, 200)


# ---------------------------------------------------------------------------
# Strict logging helpers - EXACT required format, NO extra output
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: str | None = None) -> None:
    action_str = json.dumps(action) if not isinstance(action, str) else action
    err_str = "null" if (error is None or error == "") else json.dumps(str(error))
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    succ_str = "true" if success else "false"
    print(
        f"[END] success={succ_str} steps={steps}"
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
        end="\n\n",
    )


# ---------------------------------------------------------------------------
# System prompt - strong behavioral guidance to prevent wait-spamming
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert food delivery dispatch controller managing a fleet of drivers.

CRITICAL MISSION: Maximize completed on-time deliveries. Every idle moment costs you points.

WARNING: WAITING IS EXPENSIVE
- Each useless "wait" when drivers and orders are available incurs PENALTIES
- Consecutive waits compound: after 3 waits in a row, penalties DOUBLE
- Going 15-30 steps without a delivery triggers an INACTIVITY PENALTY
- Invalid actions (wrong IDs, wrong status) cost -3 to -5 points each

AVAILABLE ACTIONS (choose the most efficient):
1. BATCH (PREFERRED - assign multiple pairs at once):
   {"action_type": "batch", "assignments": [{"driver_id": 0, "order_id": 2}, {"driver_id": 1, "order_id": 5}]}

2. ASSIGN (single driver to single order):
   {"action_type": "assign", "driver_id": <int>, "order_id": <int>}

3. REJECT (only if order is truly unreachable before deadline):
   {"action_type": "reject", "order_id": <int>}

4. WAIT (ONLY use when ALL drivers are already assigned AND no pending orders):
   {"action_type": "wait"}

   
DISPATCH STRATEGY (follow strictly):

STEP 1 - Check for idle drivers AND pending orders:
   If BOTH exist: ALWAYS assign. Never wait. Use BATCH for efficiency.
   If no idle drivers: wait is acceptable (all drivers are working)
   If no pending orders: wait is acceptable (nothing to assign)

STEP 2 - Assignment priority (sort pending orders by urgency):
   Most urgent first: smallest steps_until_deadline
   Among equally urgent: pick order closest to an idle driver
   Use distance_to_nearest_idle_driver field for quick lookup

STEP 3 - Driver selection:
   Pick the idle driver with smallest distance to the order's pickup_x/pickup_y
   Calculate: sqrt((driver_x - pickup_x)^2 + (driver_y - pickup_y)^2)

STEP 4 - Batch all possible assignments in one action:
   If 3 idle drivers and 5 pending orders  assign all 3 in one batch
   Never assign one at a time if multiple can be batched

WHAT TO AVOID (these hurt your score):
1. Waiting when idle drivers + pending orders exist
2. Using "wait" repeatedly (penalties compound)
3. Assigning a driver that is not "idle"
4. Assigning an order that is not "pending"
5. Using IDs that don't exist in the observation
6. Leaving urgent orders unassigned until deadline

OUTPUT FORMAT:

Respond with ONLY a valid JSON object. No explanation, no markdown, no text.
Examples:
  {"action_type": "batch", "assignments": [{"driver_id": 0, "order_id": 1}]}
  {"action_type": "assign", "driver_id": 2, "order_id": 4}
  {"action_type": "wait"}
"""


# ---------------------------------------------------------------------------
# Observation  prompt
# ---------------------------------------------------------------------------

def build_user_prompt(obs_dict: Dict, step: int, history: List[Dict]) -> str:
    """Construct a highly structured user message from the current observation."""

    # Sort pending orders by urgency (fewest steps until deadline first)
    pending_orders = sorted(
        [o for o in obs_dict.get("orders", []) if o.get("status") == "pending"],
        key=lambda o: o.get("steps_until_deadline", 9999),
    )
    idle_drivers = [d for d in obs_dict.get("drivers", []) if d.get("status") == "idle"]
    active_drivers = [
        d for d in obs_dict.get("drivers", []) if d.get("status") != "idle"
    ]

    # Format idle drivers with positions
    idle_lines = [
        f"   Driver {d['driver_id']}: pos=({d['x']:.3f}, {d['y']:.3f}), speed={d.get('speed', 0.05):.3f}"
        for d in idle_drivers
    ]

    # Format pending orders with urgency info
    pending_lines = []
    for o in pending_orders:
        urgency = (
            "URGENT"
            if o.get("steps_until_deadline", 999) < 20
            else ("SOON" if o.get("steps_until_deadline", 999) < 40 else "OK")
        )
        pending_lines.append(
            f"   Order {o['order_id']} [{urgency}]: "
            f"deadline_in={o.get('steps_until_deadline', '?')} steps, "
            f"pickup=({o['pickup_x']:.3f},{o['pickup_y']:.3f}), "
            f"dropoff=({o['dropoff_x']:.3f},{o['dropoff_y']:.3f}), "
            f"nearest_driver_dist={o.get('distance_to_nearest_idle_driver', '?')}, "
            f"priority={o.get('priority', 0.5):.2f}"
        )

    # Format active (busy) drivers for context
    active_lines = [
        f"   Driver {d['driver_id']}: status={d.get('status')}, "
        f"assigned_order={d.get('assigned_order_id', 'none')}"
        for d in active_drivers
    ]

    # Recent history
    recent = history[-6:] if len(history) >= 6 else history
    history_lines = [
        f"  step={h['step']} action={h['action_type']} reward={h['reward']:.2f}"
        for h in recent
    ]

    # Compute consecutive waits from history
    recent_waits = sum(1 for h in reversed(recent) if h.get("action_type") == "wait")

    # Decision guidance
    if idle_drivers and pending_orders:
        guidance = (
            f" ACTION REQUIRED: {len(idle_drivers)} idle driver(s) + "
            f"{len(pending_orders)} pending order(s). "
            f"You MUST assign - waiting will be penalized!"
        )

        if len(pending_orders) >= 1 and len(idle_drivers) >= 1:
            # Suggest the most urgent assignment
            most_urgent = pending_orders[0]
            guidance += (
                f"\n   Most urgent order: Order {most_urgent['order_id']} "
                f"(deadline in {most_urgent.get('steps_until_deadline', '?')} steps)"
            )
    elif not idle_drivers:
        guidance = "All drivers busy - wait is acceptable."
    elif not pending_orders:
        guidance = "No pending orders - wait is acceptable."
    else:
        guidance = "Consider your options carefully."

    if recent_waits >= 2:
        guidance += f"\n   WARNING: {recent_waits} recent waits detected - penalties are compounding!"

    lines = [
        f" STEP {step} / {obs_dict.get('max_steps', '?')} ",
        f"  Steps remaining   : {obs_dict.get('steps_remaining', '?')}",
        f"  Delivered so far  : {obs_dict.get('num_delivered_orders', 0)} "
        f"(on-time rate: {obs_dict.get('on_time_rate', 0.0):.0%})",
        f"  Failed orders     : {obs_dict.get('num_failed_orders', 0)}",
        f"  Cumulative reward : {obs_dict.get('cumulative_reward', 0.0):.2f}",
        f"  Delivery rate     : {obs_dict.get('delivery_rate', 0.0):.0%}",
        "",
        f"  {guidance}",
        "",
        f"IDLE DRIVERS ({len(idle_drivers)}):",
    ]

    lines += idle_lines if idle_lines else ["  (none - all drivers are busy)"]

    lines += ["", f"BUSY DRIVERS ({len(active_drivers)}):"]
    lines += active_lines if active_lines else ["  (none)"]
    lines += ["", f"PENDING ORDERS ({len(pending_orders)}) - sorted by urgency:"]

    lines += (
        pending_lines
        if pending_lines
        else ["  (none - all orders assigned or complete)"]
    )

    traffic_zones = obs_dict.get("traffic_zones", [])
    if traffic_zones:
        lines += ["", f"TRAFFIC ZONES ({len(traffic_zones)}):"]
        for z in traffic_zones:
            lines.append(
                f"   center=({z['center_x']:.2f},{z['center_y']:.2f}), "
                f"radius={z['radius']:.2f}, slowdown={z['slowdown_multiplier']:.1f}x"
            )

    lines += ["", "RECENT HISTORY:"]
    lines += history_lines if history_lines else ["  (no history yet)"]
    lines += ["", " DECIDE NOW - return ONLY a JSON action "]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM action decision with greedy fallback
# ---------------------------------------------------------------------------

def safe_default_action(obs_dict: Dict) -> Dict:
    """
    Greedy nearest-driver fallback used when LLM fails.
    Always assigns if possible - never waits unnecessarily.
    """

    idle_drivers = [d for d in obs_dict.get("drivers", []) if d.get("status") == "idle"]

    pending_orders = sorted(
        [o for o in obs_dict.get("orders", []) if o.get("status") == "pending"],
        key=lambda o: o.get("steps_until_deadline", 9999),
    )

    if not idle_drivers or not pending_orders:
        return {"action_type": "wait"}

    assignments: List[Dict[str, int]] = []
    assigned_orders: set = set()

    for driver in idle_drivers:
        best_order = None
        best_dist = float("inf")

        for order in pending_orders:
            if order["order_id"] in assigned_orders:
                continue

            dx = driver["x"] - order["pickup_x"]
            dy = driver["y"] - order["pickup_y"]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_order = order

        if best_order:
            assignments.append(
                {
                    "driver_id": driver["driver_id"],
                    "order_id": best_order["order_id"],
                }
            )
            assigned_orders.add(best_order["order_id"])

    if not assignments:
        return {"action_type": "wait"}

    if len(assignments) == 1:
        return {
            "action_type": "assign",
            "driver_id": assignments[0]["driver_id"],
            "order_id": assignments[0]["order_id"],
        }

    return {"action_type": "batch", "assignments": assignments}


def decide_action(client: OpenAI, obs_dict: Dict, step: int, history: List[Dict]) -> Dict:
    """
    Ask the LLM to choose an action.

    Falls back to greedy nearest-driver if LLM fails after MAX_RETRIES.
    The greedy fallback itself never waits unnecessarily, ensuring the
    agent always makes progress even when the LLM is unreliable.
    """
    user_msg = build_user_prompt(obs_dict, step, history)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,  # Lower temperature for more deterministic dispatch
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            if not raw:
                raise ValueError("Empty response from LLM")

            # Strip markdown fences
            if raw.startswith("```"):
                parts = raw.split("```")

                if len(parts) >= 2:
                    raw = parts[1]

                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

            # Find JSON object
            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start >= 0 and end > start:
                raw = raw[start:end]

            action = json.loads(raw)
            if not isinstance(action, dict):
                raise ValueError("Response is not a JSON object")

            if "action_type" not in action:
                raise ValueError("Missing action_type in LLM response")

            valid_types = {"assign", "reject", "wait", "batch"}
            if action["action_type"] not in valid_types:
                raise ValueError(f"Invalid action_type: {action['action_type']}")

            # Safety check: if LLM wants to wait when work is available,
            # override with greedy assignment
            if action["action_type"] == "wait":
                idle = [d for d in obs_dict.get("drivers", []) if d.get("status") == "idle"]
                pending = [o for o in obs_dict.get("orders", []) if o.get("status") == "pending"]

                if idle and pending:
                    # Override the wait with a greedy assignment
                    return safe_default_action(obs_dict)

            return action

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)

    # All retries exhausted - use greedy fallback (never waits unnecessarily)
    return safe_default_action(obs_dict)


# ---------------------------------------------------------------------------
# Score calculator
# ---------------------------------------------------------------------------

def compute_normalized_score(rewards: List[float], max_steps: int) -> float:
    """Normalized score in [0, 1] based on total reward accumulated."""
    if not rewards:
        return 0.0

    total_reward = sum(rewards)
    MAX_TOTAL_REWARD = max_steps * 10.0

    if MAX_TOTAL_REWARD <= 0:
        return 0.0

    score = total_reward / MAX_TOTAL_REWARD
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

async def run_inference() -> None:
    """Main async entry point - runs easy, medium, and hard tasks in sequence."""

    from models import FoodDeliveryAction
    from client import FoodDeliveryEnv

    #  run all three tasks instead of a single task
    TASKS = ["easy", "medium", "hard"]

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_scores: List[float] = []

    for task in TASKS:
        max_steps = MAX_STEPS_MAP.get(task, 200)

        # Per-task state - fully reset for each iteration
        all_rewards: List[float] = []
        steps_taken: int = 0
        success: bool = False
        score: float = 0.0
        history: List[Dict] = []
        env: FoodDeliveryEnv | None = None

        log_start(task=task, env=IMAGE_NAME, model=MODEL_NAME)

        try:
            env = await FoodDeliveryEnv.from_docker_image(
                IMAGE_NAME,
                env_vars={"FOOD_DELIVERY_TASK": task},
            )

            reset_result = await env.reset()
            obs = reset_result.observation
            obs_dict = obs.model_dump()
            max_steps = obs_dict.get("max_steps", max_steps)

            done = False

            while not done and steps_taken < max_steps:
                steps_taken += 1
                error_msg: str | None = None

                action_dict = decide_action(llm, obs_dict, steps_taken, history)
                action_type = action_dict.get("action_type", "wait")
                reward = 0.0

                try:
                    action = FoodDeliveryAction(**action_dict)
                    result = await env.step(action)

                    obs = result.observation
                    obs_dict = obs.model_dump()
                    reward = result.reward or 0.0
                    done = result.done or obs_dict.get("done", False)

                    if not obs_dict.get("last_action_valid", True):
                        error_msg = obs_dict.get("last_action_message", "invalid action")

                except Exception as exc:
                    error_msg = str(exc)
                    reward = 0.0
                    done = False

                all_rewards.append(reward)

                log_step(step=steps_taken, action=action_dict, reward=reward, done=done, error=error_msg)

                history.append(
                    {
                        "step": steps_taken,
                        "action_type": action_type,
                        "reward": reward,
                    }
                )

            score = compute_normalized_score(all_rewards, max_steps)
            success = score > 0.0

        except Exception as exc:

            print(
                f"[ERROR] task={task} {exc}\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )

            success = False
            score = compute_normalized_score(all_rewards, max_steps)

        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    pass

            log_end(success=success, steps=steps_taken, score=score, rewards=all_rewards)

        task_scores.append(score)

    #  print average score across all tasks
    if task_scores:
        avg_score = sum(task_scores) / len(task_scores)
        print(f"[FINAL] avg_score = {avg_score:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_inference())
