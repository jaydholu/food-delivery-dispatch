"""
Food Delivery Dispatch — LLM-Based Inference Agent (HuggingFace Router).

An LLM (via HuggingFace router API) observes the environment state each step
and decides which dispatch action to take.  The agent is prompted with the
full observation (pending orders, idle drivers, traffic, history) and must
return a valid JSON action.

STRICT Log format (NO extra output allowed):
    [START] task=<task> env=<image> model=<model>
    [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<n> score=<float> rewards=<list>

Environment variables:
    HF_TOKEN       — HuggingFace token (primary API key)
    API_KEY        — Alternative API key
    API_BASE_URL   — API base URL (default: https://router.huggingface.co/v1)
    MODEL_NAME     — Model to use (default: Qwen/Qwen2.5-72B-Instruct)
    IMAGE_NAME     — Docker image name (default: food_delivery_openenv-env:latest)
    TASK           — easy | medium | hard (default: medium)
    MAX_RETRIES    — LLM retry attempts (default: 3)
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
# Environment variables — HuggingFace router
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "food_delivery_openenv-env:latest")
TASK         = os.getenv("TASK", "medium")
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))

# Score calculation constants
MAX_STEPS_MAP = {"easy": 150, "medium": 200, "hard": 300}
MAX_STEPS = MAX_STEPS_MAP.get(TASK, 200)

# ---------------------------------------------------------------------------
# Strict logging helpers — EXACT required format, NO extra output
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: str | None = None) -> None:
    action_str = json.dumps(action) if not isinstance(action, str) else action
    err_str    = "null" if (error is None or error == "") else json.dumps(str(error))
    done_str   = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    succ_str    = "true" if success else "false"
    print(
        f"[END] success={succ_str} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert food delivery dispatch controller.
Your job is to assign delivery drivers to customer orders optimally.

GOAL: Maximise the number of on-time deliveries, minimise idle drivers,
and avoid order failures (deadline expiry).

AVAILABLE ACTION TYPES:
1. assign  — assign one idle driver to one pending order
   {"action_type": "assign", "driver_id": <int>, "order_id": <int>}

2. batch   — assign multiple pairs at once (most efficient)
   {"action_type": "batch", "assignments": [{"driver_id": 0, "order_id": 2}, ...]}

3. reject  — cancel an order (use only when no driver can reach before deadline)
   {"action_type": "reject", "order_id": <int>}

4. wait    — do nothing this step (only use when all orders are assigned)
   {"action_type": "wait"}

STRATEGY TIPS:
- Prefer "batch" to assign all idle drivers to pending orders in one step.
- Prioritise orders with fewer steps_until_deadline (most urgent first).
- Choose the nearest idle driver to each pickup location.
- Never leave idle drivers when there are pending orders — idle penalty accumulates.
- Only "reject" if steps_until_deadline < estimated travel time.

OUTPUT: Respond with ONLY a valid JSON object — no explanation, no markdown.
"""

# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------

def build_user_prompt(obs_dict: Dict, step: int, history: List[Dict]) -> str:
    """Construct the user message from the current observation."""
    idle_drivers = [
        f"  Driver {d['driver_id']}: pos=({d['x']:.3f},{d['y']:.3f})"
        for d in obs_dict.get("drivers", [])
        if d.get("status") == "idle"
    ]

    pending_orders = sorted(
        [o for o in obs_dict.get("orders", []) if o.get("status") == "pending"],
        key=lambda o: o.get("steps_until_deadline", 9999),
    )
    pending_lines = [
        f"  Order {o['order_id']}: pickup=({o['pickup_x']:.3f},{o['pickup_y']:.3f}) "
        f"dropoff=({o['dropoff_x']:.3f},{o['dropoff_y']:.3f}) "
        f"deadline_in={o.get('steps_until_deadline', '?')} steps "
        f"nearest_driver_dist={o.get('distance_to_nearest_idle_driver', '?')} "
        f"priority={o.get('priority', 0.5):.2f}"
        for o in pending_orders
    ]

    recent = history[-5:] if len(history) >= 5 else history
    history_lines = [
        f"  step={h['step']} action={h['action_type']} reward={h['reward']:.2f}"
        for h in recent
    ]

    lines = [
        f"=== Step {step} / {obs_dict.get('max_steps', '?')} ===",
        f"Steps remaining   : {obs_dict.get('steps_remaining', '?')}",
        f"Pending orders    : {obs_dict.get('num_pending_orders', 0)}",
        f"Idle drivers      : {obs_dict.get('num_idle_drivers', 0)}",
        f"Delivered so far  : {obs_dict.get('num_delivered_orders', 0)} "
        f"(on-time rate: {obs_dict.get('on_time_rate', 0.0):.0%})",
        f"Failed orders     : {obs_dict.get('num_failed_orders', 0)}",
        f"Cumulative reward : {obs_dict.get('cumulative_reward', 0.0):.2f}",
        "",
        "IDLE DRIVERS:",
    ]
    lines += idle_drivers if idle_drivers else ["  (none)"]
    lines += ["", "PENDING ORDERS (most urgent first):"]
    lines += pending_lines if pending_lines else ["  (none)"]
    lines += ["", "RECENT HISTORY:"]
    lines += history_lines if history_lines else ["  (no history yet)"]
    lines += ["", "Decide your action now. Return ONLY a JSON object."]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM action decision with robust fallback
# ---------------------------------------------------------------------------

def safe_default_action(obs_dict: Dict) -> Dict:
    """
    Safe fallback action: greedy nearest-driver assignment or wait.
    Used when LLM fails or returns invalid JSON.
    """
    idle_drivers = [
        d for d in obs_dict.get("drivers", []) if d.get("status") == "idle"
    ]
    pending_orders = [
        o for o in obs_dict.get("orders", []) if o.get("status") == "pending"
    ]

    if not idle_drivers or not pending_orders:
        return {"action_type": "wait"}

    assignments: List[Dict[str, int]] = []
    assigned_orders: set = set()

    for driver in idle_drivers:
        best_order = None
        best_dist  = float("inf")
        for order in pending_orders:
            if order["order_id"] in assigned_orders:
                continue
            dx   = driver["x"] - order["pickup_x"]
            dy   = driver["y"] - order["pickup_y"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist  = dist
                best_order = order

        if best_order:
            assignments.append({
                "driver_id": driver["driver_id"],
                "order_id":  best_order["order_id"],
            })
            assigned_orders.add(best_order["order_id"])

    if not assignments:
        return {"action_type": "wait"}
    if len(assignments) == 1:
        return {
            "action_type": "assign",
            "driver_id":   assignments[0]["driver_id"],
            "order_id":    assignments[0]["order_id"],
        }
    return {"action_type": "batch", "assignments": assignments}


def decide_action(
    client:   OpenAI,
    obs_dict: Dict,
    step:     int,
    history:  List[Dict],
) -> Dict:
    """
    Ask the LLM to choose an action. Falls back to greedy assignment if
    the LLM response cannot be parsed after MAX_RETRIES attempts.
    """
    user_msg = build_user_prompt(obs_dict, step, history)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            if not raw:
                raise ValueError("Empty response from LLM")

            # Strip markdown fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

            # Find JSON object in response
            start = raw.find("{")
            end   = raw.rfind("}") + 1
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

            return action

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)

    return safe_default_action(obs_dict)


# ---------------------------------------------------------------------------
# Score calculator — normalized to [0, 1] using reward accumulation
# ---------------------------------------------------------------------------

def compute_normalized_score(rewards: List[float], max_steps: int) -> float:
    """
    Normalized score in [0, 1] based on total reward accumulated.
    Uses the formula: score = sum(rewards) / MAX_TOTAL_REWARD
    where MAX_TOTAL_REWARD is the theoretical maximum achievable reward.
    """
    if not rewards:
        return 0.0

    total_reward = sum(rewards)
    MAX_TOTAL_REWARD = max_steps * 10.0  # Conservative upper bound

    if MAX_TOTAL_REWARD <= 0:
        return 0.0

    score = total_reward / MAX_TOTAL_REWARD

    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

async def run_inference() -> None:
    """Main async entry point."""
    from models import FoodDeliveryAction
    from client import FoodDeliveryEnv

    # Tracking variables — defined before try for finally block access
    all_rewards:  List[float] = []
    steps_taken:  int         = 0
    success:      bool        = False
    score:        float       = 0.0

    log_start(task=TASK, env=IMAGE_NAME, model=MODEL_NAME)

    # Initialise OpenAI-compatible client pointing to HuggingFace router
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history:  List[Dict] = []
    max_steps = MAX_STEPS
    env: FoodDeliveryEnv | None = None

    try:
        # Connect to environment via Docker image
        env = await FoodDeliveryEnv.from_docker_image(IMAGE_NAME)

        # Reset environment
        reset_result = await env.reset()
        obs          = reset_result.observation
        obs_dict     = obs.model_dump()
        max_steps    = obs_dict.get("max_steps", MAX_STEPS)

        done = False

        while not done and steps_taken < max_steps:
            steps_taken += 1
            error_msg: str | None = None

            # Ask LLM (with greedy fallback on failure)
            action_dict = decide_action(llm, obs_dict, steps_taken, history)
            action_type = action_dict.get("action_type", "wait")
            reward      = 0.0

            try:
                # Build typed action and step environment
                action = FoodDeliveryAction(**action_dict)
                result = await env.step(action)

                obs      = result.observation
                obs_dict = obs.model_dump()
                reward   = result.reward or 0.0
                done     = result.done or obs_dict.get("done", False)

                # Collect error from invalid actions
                if not obs_dict.get("last_action_valid", True):
                    error_msg = obs_dict.get("last_action_message", "invalid action")

            except Exception as exc:
                error_msg = str(exc)
                reward    = 0.0
                done      = False

            all_rewards.append(reward)

            # Log step in EXACT required format
            log_step(
                step   = steps_taken,
                action = action_dict,
                reward = reward,
                done   = done,
                error  = error_msg,
            )

            # Update history for prompt context
            history.append({
                "step":        steps_taken,
                "action_type": action_type,
                "reward":      reward,
            })

        # Compute normalized score in [0, 1]
        score   = compute_normalized_score(all_rewards, max_steps)
        success = score > 0.0

    except Exception as exc:
        # Log to stderr only — stdout reserved for strict format
        print(f"[ERROR] {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        success = False
        score   = compute_normalized_score(all_rewards, max_steps)

    finally:
        # Always close environment
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

        # ALWAYS print [END] — even on crash
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = all_rewards,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_inference())
