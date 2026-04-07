"""
Food Delivery Dispatch — LLM-Based Inference Agent.

An LLM (via OpenAI-compatible API) observes the environment state each step
and decides which dispatch action to take.  The agent is prompted with the
full observation (pending orders, idle drivers, traffic, history) and must
return a valid JSON action.

Log format:
    [START] task=<task> env=<image> model=<model>
    [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<n> score=<float> rewards=<list>

Environment variables:
    API_BASE_URL   — OpenAI-compatible API base (default: https://api.openai.com/v1)
    MODEL_NAME     — Model to use        (default: gpt-4o-mini)
    HF_TOKEN       — Hugging Face token  (optional, for private HF spaces)
    IMAGE_NAME     — Docker image name   (default: food_delivery_openenv-env:latest)
    TASK           — easy | medium | hard (default: medium)
    MAX_STEPS      — Override max steps  (optional)
    MAX_RETRIES    — LLM retry attempts  (default: 3)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
IMAGE_NAME   = os.getenv("IMAGE_NAME",   "food_delivery_openenv-env:latest")
TASK         = os.getenv("TASK",         "medium")
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Logging helpers  (exact required format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    action_str = json.dumps(action) if not isinstance(action, str) else action
    err_str    = "null" if error is None else json.dumps(error)
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

    # Summarise drivers
    idle_drivers = [
        f"  Driver {d['driver_id']}: pos=({d['x']:.3f},{d['y']:.3f})"
        for d in obs_dict.get("drivers", [])
        if d.get("status") == "idle"
    ]

    # Summarise pending orders (sorted by urgency)
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

    # Recent reward history (last 5 steps)
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
# LLM action decision
# ---------------------------------------------------------------------------

def decide_action(
    client:   OpenAI,
    obs_dict: Dict,
    step:     int,
    history:  List[Dict],
) -> Dict:
    """
    Ask the LLM to choose an action.  Falls back to greedy assignment if
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

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            action = json.loads(raw)

            # Basic validation
            if "action_type" not in action:
                raise ValueError("Missing action_type in LLM response")

            return action

        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                # Fall back to greedy on final failure
                return _greedy_fallback(obs_dict)
            time.sleep(0.5)

    return _greedy_fallback(obs_dict)


def _greedy_fallback(obs_dict: Dict) -> Dict:
    """
    Greedy fallback: assign each idle driver to the nearest pending order
    by pickup distance.  Returns a batch action (or wait if nothing to do).
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


# ---------------------------------------------------------------------------
# Score calculator (mirrors server-side logic)
# ---------------------------------------------------------------------------

def compute_score(
    rewards:         List[float],
    delivered_all:   int,
    delivered_on_time: int,
    failed:          int,
    total_orders:    int,
    idle_steps:      int,
    max_steps:       int,
) -> float:
    """Normalised score in [0, 1]."""
    if total_orders == 0:
        return 0.0

    delivery_rate  = delivered_all / total_orders
    on_time_rate   = delivered_on_time / max(delivered_all, 1)
    total_reward   = sum(rewards)
    reward_ceiling = total_orders * 20.0
    reward_rate    = max(0.0, min(total_reward / max(reward_ceiling, 1), 1.0))
    idle_ceiling   = max_steps * 10
    efficiency     = 1.0 - min(idle_steps / max(idle_ceiling, 1), 1.0)

    score = (0.50 * delivery_rate
           + 0.25 * on_time_rate
           + 0.15 * reward_rate
           + 0.10 * efficiency)
    return round(max(0.0, min(score, 1.0)), 4)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

async def run_inference() -> None:
    """Main async entry point."""
    from food_delivery_openenv import FoodDeliveryAction, FoodDeliveryEnv

    log_start(task=TASK, env=IMAGE_NAME, model=MODEL_NAME)

    # Initialise OpenAI client
    client_kwargs: Dict[str, Any] = {"base_url": API_BASE_URL}
    api_key = os.getenv("OPENAI_API_KEY", HF_TOKEN or "no-key")
    client_kwargs["api_key"] = api_key
    llm = OpenAI(**client_kwargs)

    # Tracking
    all_rewards:      List[float] = []
    history:          List[Dict]  = []
    steps_taken:      int         = 0
    success:          bool        = False
    score:            float        = 0.0
    idle_steps_total: int          = 0
    delivered_all:    int          = 0
    delivered_ot:     int          = 0
    failed:           int          = 0
    total_orders:     int          = 0
    max_steps:        int          = 200

    env: Optional[FoodDeliveryEnv] = None

    try:
        # Connect to environment via Docker
        env = FoodDeliveryEnv.from_docker_image(IMAGE_NAME)

        # Reset
        reset_result = env.reset()
        obs          = reset_result.observation
        obs_dict     = obs.model_dump()
        max_steps    = obs_dict.get("max_steps", 200)

        done = False

        while not done:
            steps_taken += 1

            # Ask LLM (with greedy fallback)
            action_dict = decide_action(llm, obs_dict, steps_taken, history)
            action_type = action_dict.get("action_type", "wait")
            error_msg: Optional[str] = None

            try:
                # Build typed action
                action = FoodDeliveryAction(**action_dict)
                result = env.step(action)

                obs      = result.observation
                obs_dict = obs.model_dump()
                reward   = result.reward or 0.0
                done     = result.done or obs_dict.get("done", False)

                all_rewards.append(reward)

                # Update tracking from obs
                delivered_all = obs_dict.get("num_delivered_orders", delivered_all)
                failed        = obs_dict.get("num_failed_orders", failed)
                meta          = obs_dict.get("metadata", {})
                delivered_ot  = meta.get("delivered_on_time", delivered_ot)
                total_orders  = len(obs_dict.get("orders", []))

                # Count idle driver steps this step
                idle_this_step = obs_dict.get("num_idle_drivers", 0)
                idle_steps_total += idle_this_step

                if not obs_dict.get("last_action_valid", True):
                    error_msg = obs_dict.get("last_action_message", "invalid action")

            except Exception as exc:
                error_msg = str(exc)
                reward    = 0.0
                all_rewards.append(reward)

            # Append to history
            history.append({
                "step":        steps_taken,
                "action_type": action_type,
                "reward":      reward,
            })

            log_step(
                step   = steps_taken,
                action = action_dict,
                reward = reward,
                done   = done,
                error  = error_msg,
            )

            # Safety: respect max_steps even if done flag is delayed
            if steps_taken >= max_steps:
                break

        # Compute final score
        score = compute_score(
            rewards           = all_rewards,
            delivered_all     = delivered_all,
            delivered_on_time = delivered_ot,
            failed            = failed,
            total_orders      = total_orders,
            idle_steps        = idle_steps_total,
            max_steps         = max_steps,
        )
        success = True

    except Exception as exc:
        error_detail = traceback.format_exc()
        print(f"[ERROR] {exc}\n{error_detail}", file=sys.stderr, flush=True)
        success = False

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

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
