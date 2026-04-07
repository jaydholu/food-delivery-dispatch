---
title: Food Delivery Dispatch — OpenEnv RL Environment
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - dispatch-optimization
---

# Food Delivery Dispatch — OpenEnv RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-orange)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> A **production-grade reinforcement learning environment** for multi-driver food delivery dispatch optimisation.  
> Built with **pure OpenEnv APIs** — zero Gymnasium / numpy dependencies.  
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
food_delivery_openenv/
├── baseline/
│   └── run_baseline.py                       ← Greedy nearest-driver policy + evaluation
├── server/
│   ├── __init__.py
│   ├── app.py                                ← FastAPI + OpenEnv HTTP/WS server
│   └── food_delivery_openenv_environment.py  ← Full environment logic
├── tasks/
│   ├── easy.py                               ← 2 drivers, 3 orders
│   ├── medium.py                             ← 4 drivers, 8 orders + traffic
│   ├── hard.py                               ← 6 drivers, 15 orders + traffic + dynamic spawning
│   ├── grader.py                             ← Scoring formula shared by all tasks
│   └── __init__.py
├── training/
│   ├── train_ppo.py                          ← PPO training with Stable-Baselines3
│   ├── train_all_tasks.py                    ← PPO training for tasks
│   └── evaluate_ppo.py                       ← PPO evaluation + greedy baseline comparison
├── __init__.py                               ← Package exports
├── client.py                                 ← FoodDeliveryEnv client (WebSocket)
├── Dockerfile                                ← Production container
├── inference.py                              ← LLM agent inference script
├── models.py                                 ← Action & Observation (pure OpenEnv)
├── openenv.yaml                              ← OpenEnv manifest
├── pyproject.toml                            ← Build config & dependencies
├── .gitignore                                ← git ignore files
├── requirements.txt                          ← Runtime dependencies
├── runtime.txt                               ← Python verison
└── uv.lock                             
```

---

## Environment Design

### Task Tiers

| Task   | Drivers | Orders | Traffic | Dynamic Spawning | Max Steps |
|--------|---------|--------|---------|-----------------|-----------|
| easy   | 2       | 3      | ✗       | ✗               | 150       |
| medium | 4       | 8      | ✓       | ✗               | 200       |
| hard   | 6       | 15     | ✓       | ✓ (~12%/step)   | 300       |

### Simulation Loop

```
reset()
  │
  ▼
step(action)
  ├─ 1. Decode and apply dispatch action
  ├─ 2. Move all active drivers toward their targets
  ├─ 3. Resolve pickups  (PICKING_UP → DELIVERING + pickup_reward)
  ├─ 4. Resolve deliveries (DELIVERING → IDLE + delivery_reward)
  ├─ 5. Expire overdue orders → FAILED + failure_penalty
  ├─ 6. Spawn new orders (HARD mode only)
  ├─ 7. Accumulate idle driver penalty
  └─ 8. Return FoodDeliveryObservation (reward, done, full state)
```

---

## Action Space

Four structured, serialisable action types:

```python
# 1. Assign one driver to one order
FoodDeliveryAction(action_type="assign", driver_id=0, order_id=2)

# 2. Assign multiple pairs at once (most efficient)
FoodDeliveryAction(
    action_type="batch",
    assignments=[
        {"driver_id": 0, "order_id": 2},
        {"driver_id": 1, "order_id": 5},
    ]
)

# 3. Reject a pending order (e.g. unreachable before deadline)
FoodDeliveryAction(action_type="reject", order_id=7)

# 4. Do nothing this step
FoodDeliveryAction(action_type="wait")
```

Invalid actions (driver not idle, order not pending) return `last_action_valid=False` in the observation — the environment **never crashes** on a bad action.

---

## Observation Space

Full environment snapshot returned as clean JSON — no numpy arrays:

```json
{
  "episode_id":           "uuid-...",
  "current_step":         12,
  "max_steps":            200,
  "steps_remaining":      188,

  "drivers": [
    { "driver_id": 0, "x": 0.42, "y": 0.71, "status": "idle",
      "idle_steps": 3, "total_deliveries": 1, "speed": 0.055 },
    { "driver_id": 1, "x": 0.18, "y": 0.33, "status": "delivering",
      "assigned_order_id": 3 }
  ],
  "num_idle_drivers":     1,

  "orders": [
    { "order_id": 0, "pickup_x": 0.3, "pickup_y": 0.6,
      "dropoff_x": 0.8, "dropoff_y": 0.2,
      "status": "pending", "deadline": 60,
      "steps_until_deadline": 48, "priority": 0.87,
      "distance_to_nearest_idle_driver": 0.312 }
  ],
  "num_pending_orders":   1,
  "num_delivered_orders": 2,
  "num_failed_orders":    0,

  "traffic_zones": [
    { "center_x": 0.5, "center_y": 0.5, "radius": 0.15,
      "slowdown_multiplier": 2.3, "active": true }
  ],

  "last_reward":          0.38,
  "last_action_valid":    true,
  "last_action_message":  "Assigned driver 0 → order 0 (dist=0.312).",
  "cumulative_reward":    14.7,
  "delivery_rate":        0.667,
  "on_time_rate":         1.0,

  "done":   false,
  "reward": 0.38
}
```

---

## Reward System

Dense reward signal — feedback at every step, not just at delivery:

| Event | Reward |
|-------|--------|
| Valid assignment issued | `+0.5 − 0.5 × driver_distance` |
| Order picked up | `+1.0` |
| On-time delivery | `+10.0` |
| Early delivery bonus | `+0 … +5.0` (scaled by steps ahead) |
| Late delivery | `+3.0 − 2.0 × steps_late` |
| Order expired (deadline missed) | `−8.0` |
| Order rejected | `−1.0` |
| Idle driver per step | `−0.1 … −2.0` (grows with idle duration) |

### Normalised Score Formula

```
score = 0.50 × delivery_rate
      + 0.25 × on_time_rate
      + 0.15 × reward_rate
      + 0.10 × efficiency_rate

Where:
  delivery_rate   = delivered / total_orders
  on_time_rate    = on_time / max(delivered, 1)
  reward_rate     = clamp(total_reward / (total_orders × 20), 0, 1)
  efficiency_rate = 1 − clamp(idle_steps / (max_steps × 10), 0, 1)
```

Score is always in **[0.0, 1.0]**.

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Build the image
docker build -t food_delivery_openenv-env:latest .

# 2. Run the server
docker run --rm -p 8000:8000 food_delivery_openenv-env:latest

# 3. Run with HARD task
docker run --rm -p 8000:8000 -e FOOD_DELIVERY_TASK=hard food_delivery_openenv-env:latest
```

### Option B — Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# 3. Interactive API docs
open http://localhost:8000/docs

# 4. Web interface
open http://localhost:8000/web
```

---

## Run Inference (LLM Agent)

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"   # or HF Inference endpoint
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export IMAGE_NAME="food_delivery_openenv-env:latest"
export TASK="medium"

# Run the inference agent
python inference.py
```

Expected output:

```
[START] task=medium env=food_delivery_openenv-env:latest model=gpt-4o-mini
[STEP] step=1 action={"action_type":"batch","assignments":[{"driver_id":0,"order_id":2},...]} reward=0.38 done=false error=null
[STEP] step=2 action={"action_type":"wait"} reward=-0.40 done=false error=null
...
[END] success=true steps=87 score=0.7812 rewards=[0.38,-0.40,...]
```

### Custom LLM / HuggingFace Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="hf_..."   # same as HF_TOKEN for HF endpoints
python inference.py
```

---

## Use the Client Directly

```python
from food_delivery_openenv import FoodDeliveryAction, FoodDeliveryEnv

# Via Docker (auto-starts container)
env = FoodDeliveryEnv.from_docker_image("food_delivery_openenv-env:latest")
try:
    result = env.reset()
    obs    = result.observation
    print(f"Pending: {obs.num_pending_orders}  Idle drivers: {obs.num_idle_drivers}")

    # Assign driver 0 to order 0
    action = FoodDeliveryAction(action_type="assign", driver_id=0, order_id=0)
    result = env.step(action)
    print(f"Reward: {result.reward:.2f}  Done: {result.done}")
    print(f"Message: {result.observation.last_action_message}")
finally:
    env.close()

# Via running server
with FoodDeliveryEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    while not result.done:
        result = env.step(FoodDeliveryAction(action_type="wait"))
```

---

## Deploy to Hugging Face Spaces

```bash
# From the project root (where openenv.yaml is located)
openenv push

# With options
openenv push --namespace my-org --private
openenv push --repo-id my-org/food-delivery-dispatch
```

After deployment, your space includes:
- **Web Interface** at `/web` — interactive environment explorer
- **API Docs** at `/docs` — full OpenAPI/Swagger UI
- **Health Check** at `/health`
- **WebSocket** at `/ws` — persistent low-latency session

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

## License

MIT © Food Delivery RL Team
