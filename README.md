---
title: Food Delivery Dispatch
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv

---
# Food Delivery Dispatch — OpenEnv RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-orange)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> A **production-grade reinforcement learning environment** for multi-driver food delivery dispatch optimisation.  
> Built with **pure OpenEnv APIs**.  
> An LLM or RL agent acts as the **dispatch controller**, assigning orders to drivers to maximise throughput.

---

## Real-World Problem

Modern food delivery platforms (Zomato, Swiggy, Uber Eats) must continuously assign thousands of incoming orders to a fleet of drivers while accounting for:

- **Geography** — driver and order locations scattered across a city grid
- **Time pressure** — each order has a hard deadline; late deliveries hurt revenue
- **Traffic congestion** — zones that slow driver movement unpredictably
- **Dynamic demand** — new orders arriving continuously throughout a shift
- **Fleet utilisation** — idle drivers are wasted capacity with real costs

This environment models that exact problem as a **multi-step MDP**. At each step the agent issues one of four action types, the simulation advances, and a dense reward signal is returned.

---

## Project Structure

```
food_delivery_dispatch/
├── baseline/
│   └── run_baseline.py                       ← Greedy nearest-driver policy + evaluation
├── server/
│   ├── __init__.py
│   ├── app.py                                ← FastAPI + OpenEnv HTTP/WS server
│   └── food_delivery_dispatch_environment.py  ← Full environment logic (with reward fixes)
├── tasks/
│   ├── easy.py                               ← 2 drivers, 3 orders, no traffic
│   ├── medium.py                             ← 4 drivers, 8 orders + traffic
│   ├── hard.py                               ← 6 drivers, 15 orders + traffic + dynamic spawning
│   ├── grader.py                             ← Scoring formula shared by all tasks
│   └── __init__.py
├── __init__.py                               ← Package exports
├── client.py                                 ← FoodDeliveryEnv client (WebSocket)
├── Dockerfile                                ← Production container
├── inference.py                              ← LLM agent inference script (HuggingFace router)
├── models.py                                 ← Action & Observation (pure OpenEnv)
├── openenv.yaml                              ← OpenEnv manifest (all 3 tasks registered)
├── pyproject.toml                            ← Build config & dependencies
├── requirements.txt                          ← Runtime dependencies
└── runtime.txt                               ← Python version
```

---

## Environment Design

### Task Tiers

Three difficulty levels are provided, each with increasing complexity:

| Task   | Drivers | Orders | Traffic | Dynamic Spawning | Deadline Range | Max Steps | Target Score |
|--------|---------|--------|---------|-----------------|----------------|-----------|-------------|
| easy   | 2       | 3      | ✗       | ✗               | 50-120 steps   | 150       | 0.80+       |
| medium | 4       | 8      | ✓       | ✗               | 30-80 steps    | 200       | 0.60-0.80   |
| hard   | 6       | 15     | ✓       | ✓ (~12%/step)   | 25-70 steps    | 300       | 0.50-0.75   |

### Difficulty Scaling

**EASY** — Designed for initial policy testing:
- Only 2 drivers and 3 orders (trivially small state space)
- No traffic zones, so travel times are predictable
- Generous deadlines (50-120 steps) — no time pressure
- Lenient penalties: inactivity threshold 30 steps, wait penalty 0.1/step

**MEDIUM** — The baseline challenge:
- 4 drivers and 8 orders (moderate complexity)
- 2-3 traffic zones with 1.5x-3.0x slowdowns
- Moderate deadlines requiring some urgency management
- Balanced penalties: inactivity threshold 20 steps, wait penalty 0.2/step

**HARD** — Full production complexity:
- 6 drivers, 15 initial orders, up to 30 total (dynamic arrival)
- Traffic zones slowing drivers unpredictably
- Tight deadlines requiring immediate action
- Strict penalties: inactivity threshold 15 steps, wait penalty 0.3/step, consecutive wait threshold 2

### Simulation Loop

```
reset()
  │
  ▼
step(action)
  ├─ 1. Decode and apply dispatch action (with safety validation)
  ├─ 2. Move all active drivers toward their targets
  ├─ 3. Resolve pickups  (PICKING_UP → DELIVERING + pickup_reward)
  ├─ 4. Resolve deliveries (DELIVERING → IDLE + delivery_reward)
  ├─ 5. Expire overdue orders → FAILED + failure_penalty
  ├─ 6. Spawn new orders (HARD mode only)
  ├─ 7. Accumulate idle driver penalty
  ├─ 8. Apply inactivity penalty (if no delivery in N steps)
  ├─ 9. Apply efficiency bonus (delivery_rate / steps)
  └─ 10. Return FoodDeliveryObservation (reward, done, full state)
```

---

## Action Space

Four structured, serialisable action types:

```python
# 1. Assign one driver to one order
FoodDeliveryAction(action_type="assign", driver_id=0, order_id=2)

# 2. Assign multiple pairs at once (MOST EFFICIENT — preferred)
FoodDeliveryAction(
    action_type="batch",
    assignments=[
        {"driver_id": 0, "order_id": 2},
        {"driver_id": 1, "order_id": 5},
    ]
)

# 3. Reject a pending order (e.g. unreachable before deadline)
FoodDeliveryAction(action_type="reject", order_id=7)

# 4. Do nothing this step — PENALIZED if work is available
FoodDeliveryAction(action_type="wait")
```

**Safety guarantees:**
- Invalid actions (wrong IDs, wrong status) apply a penalty (-3 to -5) but **never crash** the environment
- All observations contain only plain Python types — fully JSON serializable

---

## Reward System

Dense reward signal with new anti-wait penalties:

### Positive Rewards

| Event | Reward |
|-------|--------|
| Valid assignment issued | `+0.5 − 0.5 × driver_distance` |
| Order picked up | `+1.0` |
| On-time delivery | `+10.0` |
| Early delivery bonus | `+0 … +5.0` (scaled by steps ahead) |
| Late delivery | `+3.0 − 2.0 × steps_late` |
| Efficiency bonus | `+delivery_rate × 0.3-0.7` per step |

### Penalties

| Event | Penalty |
|-------|---------|
| Order expired (deadline missed) | `−6.0 to −9.0` (scales with task) |
| Order rejected | `−1.0` |
| Invalid action (bad IDs, wrong status) | `−3.0 to −5.0` (was 0.2 — now much stronger) |
| Idle driver per step | `−0.05 … −2.5` (grows with idle duration) |
| **Useless wait (idle drivers + pending orders)** | `−0.1 to −0.3 × consecutive_waits` |
| **Consecutive waits (after threshold)** | `extra −1.0 to −3.0` |
| **Inactivity (no delivery in N steps)** | `−0.5 to −1.5` (scales with duration) |

### Why These Penalties Matter

The key behavioral problem in LLM agents is **wait-spamming**: choosing `"wait"` when drivers are available and orders are pending. This was addressed by:

1. **Useless wait penalty** — immediate `-0.1 to -0.3` per useless wait
2. **Escalating consecutive wait penalty** — after 2-5 consecutive waits, an additional `-1.0 to -3.0` is applied, compounding rapidly
3. **Inactivity penalty** — if no delivery completes in the last 15-30 steps, a growing penalty applies
4. **Strong invalid action penalty** — raised from `-0.2` to `-5.0` to prevent random guessing

### Normalised Score Formula

```
score = 0.50 × delivery_rate + 0.25 × on_time_rate + 0.15 × reward_rate + 0.10 × efficiency_rate

Where:
  delivery_rate   = delivered / total_orders
  on_time_rate    = on_time / max(delivered, 1)
  reward_rate     = clamp(total_reward / (total_orders × 20), 0, 1)
  efficiency_rate = 1 − clamp(idle_steps / (max_steps × 10), 0, 1)
```

Score is always in **[0.0, 1.0]**.

---

## LLM Agent Behavior Guidance

The `inference.py` script uses a carefully designed system prompt to prevent wait-spamming:

### Key Prompt Features

1. **Explicit warning** about wait penalties and their compounding nature
2. **Clear priority order**: assign > reject > wait
3. **Step-by-step dispatch algorithm** the LLM must follow
4. **Urgency indicators** ( URGENT / SOON / OK) in the observation
5. **Consecutive wait counter** shown to the LLM so it knows when it's compounding penalties
6. **Wait override**: if the LLM returns "wait" when work is available, the code automatically substitutes the greedy assignment

### Safety Override

```python
# If LLM wants to wait when work is available, override it
if action["action_type"] == "wait":
    idle = [d for d in obs_dict.get("drivers", []) if d.get("status") == "idle"]
    pending = [o for o in obs_dict.get("orders", []) if o.get("status") == "pending"]
    if idle and pending:
        return safe_default_action(obs_dict)  # Greedy assignment instead
```

This ensures the agent **never** wastes a step when there's productive work to do.

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Build the image
docker build -t food_delivery_dispatch-env:latest .

# 2. Run with EASY task
docker run --rm -p 8000:8000 -e FOOD_DELIVERY_TASK=easy food_delivery_dispatch-env:latest

# 3. Run with MEDIUM task (default)
docker run --rm -p 8000:8000 food_delivery_dispatch-env:latest

# 4. Run with HARD task
docker run --rm -p 8000:8000 -e FOOD_DELIVERY_TASK=hard food_delivery_dispatch-env:latest
```

### Option B — Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (MEDIUM by default)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# 3. Start with specific task
FOOD_DELIVERY_TASK=easy uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# 4. Interactive API docs
open http://localhost:8000/docs

# 5. Web interface
open http://localhost:8000/web
```

---

## Run Inference (LLM Agent via HuggingFace Router)

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (primary) | — |
| `API_KEY` | Alternative API key | — |
| `API_BASE_URL` | API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `IMAGE_NAME` | Docker image name | `food_delivery_dispatch-env:latest` |
| `TASK` | Task difficulty: `easy`, `medium`, `hard` | `medium` |
| `MAX_RETRIES` | LLM retry attempts on failure | `3` |

### Run Inference

```bash
# Set required environment variables
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export IMAGE_NAME="food_delivery_dispatch-env:latest"

# Run EASY task first (for testing)
TASK=easy python inference.py

# Run MEDIUM task
TASK=medium python inference.py

# Run HARD task
TASK=hard python inference.py
```

### Example Output

```
[START] task=medium env=food_delivery_dispatch-env:latest model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type": "batch", "assignments": [{"driver_id": 0, "order_id": 2}, {"driver_id": 1, "order_id": 5}]} reward=0.38 done=false error=null
[STEP] step=2 action={"action_type": "assign", "driver_id": 2, "order_id": 3} reward=0.42 done=false error=null
[STEP] step=3 action={"action_type": "wait"} reward=-0.10 done=false error=null
...
[STEP] step=87 action={"action_type": "wait"} reward=0.00 done=true error=null
[END] success=true steps=87 score=0.7812 rewards=[0.38, 0.42, -0.10, ...]
```

### Robustness Features

- **Always prints `[END]`** — even if the environment crashes or the LLM fails
- **Greedy fallback** — if LLM returns invalid JSON, uses nearest-driver heuristic
- **Wait override** — if LLM says "wait" when work is available, automatically assigns
- **Invalid action safety** — bad driver/order IDs apply a penalty but never crash
- **Score normalization** — score is always clamped to `[0.0, 1.0]`
- **Low temperature (0.1)** — more deterministic, less random dispatch decisions

---

## Baseline Policy

The greedy baseline in `baseline/run_baseline.py` uses a simple nearest-driver heuristic:

```bash
# Run baseline on all 3 tasks
python -m baseline.run_baseline --episodes 5
```

Expected baseline scores:
- **EASY**: ~0.85-0.95 (trivial with greedy)
- **MEDIUM**: ~0.60-0.75 (greedy handles traffic poorly)
- **HARD**: ~0.50-0.65 (dynamic orders challenge greedy)

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST   | Reset environment, returns initial observation |
| `/step`  | POST   | Execute action, returns next observation + reward |
| `/state` | GET    | Current episode_id and step_count |
| `/health`| GET    | `{"status": "ok"}` |
| `/docs`  | GET    | Interactive Swagger UI |
| `/web`   | GET    | Web interface |
| `/ws`    | WS     | Persistent WebSocket session |

---

## Deploy to Hugging Face Spaces

```bash
# From the project root (where openenv.yaml is located)
openenv push

# With options
openenv push --namespace my-org --private
openenv push --repo-id my-org/food-delivery-dispatch
```

---

## Limitations

- The LLM agent may overuse the "wait" action in certain scenarios, especially under medium and hard tasks.
- While reward shaping discourages idle behavior, long-horizon planning is still limited by the LLM’s step-by-step decision process.
- Performance can vary depending on the underlying model and temperature settings.
- Further improvements can be achieved using trained RL policies or stronger planning-based agents.
- These limitations highlight opportunities for future work in improving agent reasoning, planning, and policy learning.

---

## License

MIT © Food Delivery RL Team
